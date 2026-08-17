// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/CodeGen/GlobalISel/MachineIRBuilder.h>
#include <llvm/CodeGen/MIRParser/MIRParser.h>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/CodeGenTypes/LowLevelType.h>
#include <llvm/CodeGenTypes/LowLevelType.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>
#include <vector>

namespace {

// MIR parsing and codegen report failures through LLVMContext::diagnose --
// their return values are only a bare success/failure bit, so the rich reason
// (line, column, message) would otherwise be lost to the default handler's
// stderr print, which is invisible under the Python/nanobind harness. This RAII
// guard installs a handler that captures error-severity diagnostics into `sink`
// for the duration of one parse/codegen call, restoring the previous handler on
// scope exit so the capture buffer (a caller stack local) never outlives the
// handler that points at it.
struct ScopedDiagnosticCapture {
  llvm::LLVMContext &ctx;
  llvm::DiagnosticHandler::DiagnosticHandlerTy prevHandler;
  void *prevContext;

  ScopedDiagnosticCapture(llvm::LLVMContext &ctx, std::string &sink)
      : ctx(ctx), prevHandler(ctx.getDiagnosticHandlerCallBack()),
        prevContext(ctx.getDiagnosticContext()) {
    ctx.setDiagnosticHandlerCallBack(
        [](const llvm::DiagnosticInfo *di, void *context) {
          if (di->getSeverity() == llvm::DS_Error) {
            auto *out = static_cast<std::string *>(context);
            llvm::raw_string_ostream os(*out);
            llvm::DiagnosticPrinterRawOStream printer(os);
            di->print(printer);
          }
        },
        &sink);
  }
  ~ScopedDiagnosticCapture() {
    ctx.setDiagnosticHandlerCallBack(prevHandler, prevContext);
  }
  ScopedDiagnosticCapture(const ScopedDiagnosticCapture &) = delete;
  ScopedDiagnosticCapture &operator=(const ScopedDiagnosticCapture &) = delete;
};

// "<base>" when no diagnostic was captured, else "<base>: <detail>".
std::string withDetail(llvm::StringRef base, const std::string &detail) {
  if (detail.empty())
    return base.str(); // LCOV_EXCL_LINE -- MIRParser always diagnoses on error
  return (base + ": " + detail).str();
}

// Owns everything the inspected MachineFunctions transitively depend on, so
// none of it is freed out from under them: the LLVMContext (pinned by
// `ctxKeepAlive`), the IR Module (whose Functions the MachineFunctions
// reference), and the MachineModuleInfo that owns the MachineFunctions. The MMI
// is owned two ways depending on how it was built: the codegen path
// (run_codegen_to_mir) leaves it inside a MachineModuleInfoWrapperPass owned by
// `pm`; the parse path (parse_mir) owns it directly via `ownedMmi`. `mmi` is
// the query handle either way. The TargetMachine is a separate Python object
// kept alive by a keep_alive.
struct MachineModuleInfo {
  std::shared_ptr<llvm::LLVMContext> ctxKeepAlive;
  std::unique_ptr<llvm::Module> module;
  std::unique_ptr<llvm::legacy::PassManager> pm;     // codegen path
  std::unique_ptr<llvm::MachineModuleInfo> ownedMmi; // parse path
  llvm::MachineModuleInfo *mmi;

  // Print the whole module as .mir text: the IR block, then each function's
  // machine-level block, matching what `llc -stop-after=finalize-isel` emits. A
  // function with no MachineFunction is skipped -- expected for declarations
  // (nothing to lower); a definition only lacks one if codegen failed, which
  // run_codegen_to_mir now reports as an exception rather than reaching here.
  std::string toMIR() {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    llvm::printMIR(os, *module);
    for (llvm::Function &f : *module) {
      if (llvm::MachineFunction *mf = mmi->getMachineFunction(f))
        llvm::printMIR(os, *mmi, *mf);
    }
    return buf;
  }
};

} // namespace

// LowLevelType (LLT) is the generic-MIR type: a target-independent "bag of
// bits" describing a scalar/pointer/vector operand, distinct from the uniqued
// llvm::Type hierarchy. It is a small value type (not context-owned and not
// polymorphic), so it is bound by value with no ownership/keep_alive plumbing.
// Binding it first also proves LLVMCodeGenTypes links into the extension.
void populate_mir(nb::module_ &m) {
  nb::class_<llvm::LLT>(m, "LLT")
      .def_static("scalar", &llvm::LLT::scalar, "size_in_bits"_a)
      .def_static("pointer", &llvm::LLT::pointer, "address_space"_a,
                  "size_in_bits"_a)
      .def_static(
          "fixed_vector",
          [](unsigned numElements, unsigned scalarSizeInBits) {
            return llvm::LLT::fixed_vector(numElements, scalarSizeInBits);
          },
          "num_elements"_a, "scalar_size_in_bits"_a)
      .def_prop_ro("size_in_bits",
                   [](const llvm::LLT &self) {
                     return self.getSizeInBits().getKnownMinValue();
                   })
      .def_prop_ro(
          "scalar_size_in_bits",
          [](const llvm::LLT &self) { return self.getScalarSizeInBits(); })
      .def_prop_ro("num_elements",
                   [](const llvm::LLT &self) { return self.getNumElements(); })
      .def_prop_ro("address_space",
                   [](const llvm::LLT &self) { return self.getAddressSpace(); })
      .def_prop_ro("is_scalar",
                   [](const llvm::LLT &self) { return self.isScalar(); })
      .def_prop_ro("is_pointer",
                   [](const llvm::LLT &self) { return self.isPointer(); })
      .def_prop_ro("is_vector",
                   [](const llvm::LLT &self) { return self.isVector(); })
      .def_prop_ro("is_integer",
                   [](const llvm::LLT &self) { return self.isInteger(); })
      .def_prop_ro("is_float",
                   [](const llvm::LLT &self) { return self.isFloat(); })
      .def_prop_ro("is_valid",
                   [](const llvm::LLT &self) { return self.isValid(); })
      .def(
          "__eq__",
          [](const llvm::LLT &self, const llvm::LLT &other) {
            return self == other;
          },
          nb::is_operator())
      .def(
          "__ne__",
          [](const llvm::LLT &self, const llvm::LLT &other) {
            return self != other;
          },
          nb::is_operator())
      .def("__hash__",
           [](const llvm::LLT &self) {
             return static_cast<Py_ssize_t>(self.getUniqueRAWLLTData());
           })
      .def("__str__",
           [](const llvm::LLT &self) { return eudsl::toString(self); });

  // llvm::Register -- a machine register operand's value (a tagged unsigned:
  // virtual registers are SSA-ish temporaries from ISel, physical are the
  // target's real registers).
  nb::class_<llvm::Register>(m, "Register")
      .def_prop_ro("id", [](llvm::Register &self) { return self.id(); })
      .def_prop_ro("is_valid",
                   [](llvm::Register &self) { return self.isValid(); })
      .def_prop_ro("is_virtual",
                   [](llvm::Register &self) { return self.isVirtual(); })
      .def_prop_ro("is_physical",
                   [](llvm::Register &self) { return self.isPhysical(); })
      .def_prop_ro("virt_reg_index",
                   [](llvm::Register &self) {
                     if (!self.isVirtual())
                       throw nb::value_error("register is not virtual");
                     return self.virtRegIndex();
                   })
      .def(
          "__eq__",
          [](llvm::Register &self, llvm::Register other) {
            return self == other;
          },
          nb::is_operator())
      .def(
          "__ne__",
          [](llvm::Register &self, llvm::Register other) {
            return self != other;
          },
          nb::is_operator())
      .def("__hash__", [](llvm::Register &self) {
        return static_cast<Py_ssize_t>(self.id());
      });

  // llvm::MachineOperand -- one operand of a MachineInstr. is_def/is_use are
  // register-only in LLVM (they assert otherwise), so they are guarded to
  // report false for non-register operands rather than crash.
  nb::class_<llvm::MachineOperand>(m, "MachineOperand")
      .def_prop_ro("is_reg",
                   [](llvm::MachineOperand &self) { return self.isReg(); })
      .def_prop_ro("is_imm",
                   [](llvm::MachineOperand &self) { return self.isImm(); })
      .def_prop_ro("is_def",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isDef();
                   })
      .def_prop_ro("is_use",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isUse();
                   })
      .def_prop_ro("reg",
                   [](llvm::MachineOperand &self) -> llvm::Register {
                     if (!self.isReg())
                       throw nb::value_error("operand is not a register");
                     return self.getReg();
                   })
      .def_prop_ro("imm",
                   [](llvm::MachineOperand &self) {
                     if (!self.isImm())
                       throw nb::value_error("operand is not an immediate");
                     return self.getImm();
                   })
      // Remaining register-operand flags (read side).
      .def_prop_ro("is_debug",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isDebug();
                   })
      .def_prop_ro("is_internal_read",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isInternalRead();
                   })
      .def_prop_ro("is_tied",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isTied();
                   })
      .def_prop_ro(
          "target_flags",
          [](llvm::MachineOperand &self) { return self.getTargetFlags(); })
      // Operand-kind predicates (mirror MachineOperand::isX()).
      .def_prop_ro("is_cimm",
                   [](llvm::MachineOperand &self) { return self.isCImm(); })
      .def_prop_ro("is_fpimm",
                   [](llvm::MachineOperand &self) { return self.isFPImm(); })
      .def_prop_ro("is_mbb",
                   [](llvm::MachineOperand &self) { return self.isMBB(); })
      .def_prop_ro("is_fi",
                   [](llvm::MachineOperand &self) { return self.isFI(); })
      .def_prop_ro("is_cpi",
                   [](llvm::MachineOperand &self) { return self.isCPI(); })
      .def_prop_ro("is_jti",
                   [](llvm::MachineOperand &self) { return self.isJTI(); })
      .def_prop_ro(
          "is_target_index",
          [](llvm::MachineOperand &self) { return self.isTargetIndex(); })
      .def_prop_ro("is_global",
                   [](llvm::MachineOperand &self) { return self.isGlobal(); })
      .def_prop_ro("is_symbol",
                   [](llvm::MachineOperand &self) { return self.isSymbol(); })
      .def_prop_ro(
          "is_block_address",
          [](llvm::MachineOperand &self) { return self.isBlockAddress(); })
      .def_prop_ro("is_reg_mask",
                   [](llvm::MachineOperand &self) { return self.isRegMask(); })
      .def_prop_ro("is_metadata",
                   [](llvm::MachineOperand &self) { return self.isMetadata(); })
      .def_prop_ro(
          "is_predicate",
          [](llvm::MachineOperand &self) { return self.isPredicate(); })
      // Kind-specific getters (guarded; each asserts its kind in LLVM).
      .def_prop_ro(
          "cimm",
          [](llvm::MachineOperand &self) -> const llvm::ConstantInt * {
            if (!self.isCImm())
              throw nb::value_error("operand is not a CImm");
            return self.getCImm();
          },
          nb::rv_policy::reference)
      .def_prop_ro(
          "fpimm",
          [](llvm::MachineOperand &self) -> const llvm::ConstantFP * {
            if (!self.isFPImm())
              throw nb::value_error("operand is not an FPImm");
            return self.getFPImm();
          },
          nb::rv_policy::reference)
      .def_prop_ro(
          "mbb",
          [](llvm::MachineOperand &self) -> llvm::MachineBasicBlock * {
            if (!self.isMBB())
              throw nb::value_error("operand is not a MachineBasicBlock");
            return self.getMBB();
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("index",
                   [](llvm::MachineOperand &self) {
                     if (!self.isFI() && !self.isCPI() && !self.isJTI() &&
                         !self.isTargetIndex())
                       throw nb::value_error("operand has no index");
                     return self.getIndex();
                   })
      .def_prop_ro(
          "global_value",
          [](llvm::MachineOperand &self) -> const llvm::GlobalValue * {
            if (!self.isGlobal())
              throw nb::value_error("operand is not a global address");
            return self.getGlobal();
          },
          nb::rv_policy::reference)
      .def_prop_ro("symbol_name",
                   [](llvm::MachineOperand &self) {
                     if (!self.isSymbol())
                       throw nb::value_error(
                           "operand is not an external symbol");
                     return std::string(self.getSymbolName());
                   })
      .def_prop_ro("offset",
                   [](llvm::MachineOperand &self) {
                     if (!self.isGlobal() && !self.isSymbol() &&
                         !self.isCPI() && !self.isTargetIndex() &&
                         !self.isBlockAddress())
                       throw nb::value_error("operand has no offset");
                     return self.getOffset();
                   })
      .def("__str__",
           [](llvm::MachineOperand &self) { return eudsl::toString(self); });

  // llvm::MachineInstr -- one machine instruction. opcode is the target opcode
  // number; opcode_name resolves its mnemonic via the function's
  // TargetInstrInfo. Non-owning (lives in its MachineBasicBlock).
  nb::class_<llvm::MachineInstr>(m, "MachineInstr")
      .def_prop_ro("opcode",
                   [](llvm::MachineInstr &self) { return self.getOpcode(); })
      .def_prop_ro("opcode_name",
                   [](llvm::MachineInstr &self) {
                     const llvm::MachineFunction *mf = self.getMF();
                     // LCOV_EXCL_START -- block instrs are always attached
                     if (!mf) {
                       throw nb::value_error(
                           "instruction is not attached to a MachineFunction");
                     }
                     // LCOV_EXCL_STOP
                     const llvm::TargetInstrInfo *tii =
                         mf->getSubtarget().getInstrInfo();
                     // LCOV_EXCL_START -- AArch64 always has a TargetInstrInfo
                     if (!tii) {
                       throw nb::value_error("target has no TargetInstrInfo");
                     }
                     // LCOV_EXCL_STOP
                     return tii->getName(self.getOpcode()).str();
                   })
      .def_prop_ro(
          "num_operands",
          [](llvm::MachineInstr &self) { return self.getNumOperands(); })
      .def(
          "operand",
          [](llvm::MachineInstr &self,
             unsigned i) -> const llvm::MachineOperand * {
            if (i >= self.getNumOperands())
              throw nb::index_error("operand index out of range");
            return &self.getOperand(i);
          },
          "index"_a, nb::rv_policy::reference_internal)
      .def_prop_ro(
          "parent",
          [](llvm::MachineInstr &self) -> const llvm::MachineBasicBlock * {
            return self.getParent();
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("num_defs",
                   [](llvm::MachineInstr &self) { return self.getNumDefs(); })
      .def_prop_ro("num_explicit_operands",
                   [](llvm::MachineInstr &self) {
                     return self.getNumExplicitOperands();
                   })
      // Classification predicates (query the MCInstrDesc; mirror MachineInstr).
      .def_prop_ro("is_terminator",
                   [](llvm::MachineInstr &self) { return self.isTerminator(); })
      .def_prop_ro("is_branch",
                   [](llvm::MachineInstr &self) { return self.isBranch(); })
      .def_prop_ro(
          "is_conditional_branch",
          [](llvm::MachineInstr &self) { return self.isConditionalBranch(); })
      .def_prop_ro(
          "is_unconditional_branch",
          [](llvm::MachineInstr &self) { return self.isUnconditionalBranch(); })
      .def_prop_ro(
          "is_indirect_branch",
          [](llvm::MachineInstr &self) { return self.isIndirectBranch(); })
      .def_prop_ro("is_barrier",
                   [](llvm::MachineInstr &self) { return self.isBarrier(); })
      .def_prop_ro("is_call",
                   [](llvm::MachineInstr &self) { return self.isCall(); })
      .def_prop_ro("is_return",
                   [](llvm::MachineInstr &self) { return self.isReturn(); })
      .def_prop_ro("is_copy",
                   [](llvm::MachineInstr &self) { return self.isCopy(); })
      .def_prop_ro("is_phi",
                   [](llvm::MachineInstr &self) { return self.isPHI(); })
      .def_prop_ro(
          "is_implicit_def",
          [](llvm::MachineInstr &self) { return self.isImplicitDef(); })
      .def_prop_ro("may_load",
                   [](llvm::MachineInstr &self) { return self.mayLoad(); })
      .def_prop_ro("may_store",
                   [](llvm::MachineInstr &self) { return self.mayStore(); })
      .def_prop_ro("is_debug_instr",
                   [](llvm::MachineInstr &self) { return self.isDebugInstr(); })
      .def("__str__",
           [](llvm::MachineInstr &self) { return eudsl::toString(self); });

  // llvm::MachineBasicBlock -- a machine basic block. `instructions` is the
  // ordered list of its MachineInstrs.
  nb::class_<llvm::MachineBasicBlock>(m, "MachineBasicBlock")
      .def_prop_ro(
          "name",
          [](llvm::MachineBasicBlock &self) { return self.getName().str(); })
      .def_prop_ro(
          "number",
          [](llvm::MachineBasicBlock &self) { return self.getNumber(); })
      .def_prop_ro(
          "instructions",
          [](llvm::MachineBasicBlock &self) {
            std::vector<llvm::MachineInstr *> instrs;
            for (llvm::MachineInstr &mi : self)
              instrs.push_back(&mi);
            return instrs;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "parent",
          [](llvm::MachineBasicBlock &self) -> llvm::MachineFunction * {
            return self.getParent();
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "is_entry_block",
          [](llvm::MachineBasicBlock &self) { return self.isEntryBlock(); })
      .def_prop_ro(
          "successors",
          [](llvm::MachineBasicBlock &self) {
            std::vector<llvm::MachineBasicBlock *> succs(self.succ_begin(),
                                                         self.succ_end());
            return succs;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "predecessors",
          [](llvm::MachineBasicBlock &self) {
            std::vector<llvm::MachineBasicBlock *> preds(self.pred_begin(),
                                                         self.pred_end());
            return preds;
          },
          nb::rv_policy::reference_internal)
      .def("__str__",
           [](llvm::MachineBasicBlock &self) { return eudsl::toString(self); });

  // llvm::MachineFunction -- the machine-level body of one IR Function after
  // instruction selection. Non-owning: it lives inside the MachineModuleInfo
  // that produced it, so it is returned by pointer with the owning wrapper kept
  // alive (reference_internal).
  nb::class_<llvm::MachineFunction>(m, "MachineFunction")
      .def_prop_ro(
          "name",
          [](llvm::MachineFunction &self) { return self.getName().str(); })
      .def_prop_ro(
          "blocks",
          [](llvm::MachineFunction &self) {
            std::vector<llvm::MachineBasicBlock *> blocks;
            for (llvm::MachineBasicBlock &mbb : self)
              blocks.push_back(&mbb);
            return blocks;
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("num_blocks",
                   [](llvm::MachineFunction &self) { return self.size(); })
      .def_prop_ro(
          "function",
          [](llvm::MachineFunction &self) -> const llvm::Function * {
            return &self.getFunction();
          },
          nb::rv_policy::reference_internal)
      .def(
          "create_generic_vreg",
          [](llvm::MachineFunction &self, llvm::LLT ty) -> llvm::Register {
            return self.getRegInfo().createGenericVirtualRegister(ty);
          },
          "type"_a, "Create a new generic virtual register of the given LLT.")
      .def("__str__",
           [](llvm::MachineFunction &self) { return eudsl::toString(self); });

  // The result of run_codegen_to_mir or parse_mir: owns the MachineFunctions
  // and everything they depend on. `machine_function` resolves one by its IR
  // Function name.
  nb::class_<MachineModuleInfo>(m, "MachineModuleInfo")
      .def(
          "machine_function",
          [](MachineModuleInfo &self,
             const std::string &name) -> llvm::MachineFunction * {
            llvm::Function *f = self.module->getFunction(name);
            if (!f) {
              throw nb::key_error(
                  ("no function named '" + name + "' in the module").c_str());
            }
            llvm::MachineFunction *mf = self.mmi->getMachineFunction(*f);
            if (!mf) {
              throw nb::key_error(
                  ("function '" + name + "' has no MachineFunction").c_str());
            }
            return mf;
          },
          "name"_a, nb::rv_policy::reference_internal)
      .def_prop_ro(
          "machine_functions",
          [](MachineModuleInfo &self) {
            std::vector<llvm::MachineFunction *> mfs;
            for (llvm::Function &f : *self.module) {
              if (llvm::MachineFunction *mf = self.mmi->getMachineFunction(f))
                mfs.push_back(mf);
            }
            return mfs;
          },
          nb::rv_policy::reference_internal,
          "The MachineFunctions in the module (functions without one -- e.g. "
          "declarations -- are skipped).")
      .def(
          "to_mir", [](MachineModuleInfo &self) { return self.toMIR(); },
          "Serialize the whole module (IR block + machine functions) as .mir "
          "text.");

  // Run instruction selection on an IR module and hand back the
  // MachineModuleInfo that owns the resulting MachineFunctions. Consumes the
  // module (like adding it to the JIT): the MachineFunctions reference its
  // Functions, so ownership moves into the returned wrapper. Only the ISel
  // passes are added -- not the object-emission pipeline (addPassesToEmitFile),
  // which would append FreeMachineFunctionPass -- so the MachineFunctions are
  // retained for inspection, the state `llc -stop-after=finalize-isel`
  // produces.
  m.def(
      "run_codegen_to_mir",
      [](eudsl::Module &mod, llvm::TargetMachine &tm) {
        std::shared_ptr<llvm::LLVMContext> ctxKeepAlive =
            mod.context().shared();
        std::unique_ptr<llvm::Module> module = mod.take();
        module->setDataLayout(tm.createDataLayout());

        auto pm = std::make_unique<llvm::legacy::PassManager>();
        llvm::TargetPassConfig *tpc = tm.createPassConfig(*pm);
        pm->add(tpc);
        auto *mmiwp = new llvm::MachineModuleInfoWrapperPass(&tm);
        pm->add(mmiwp);
        // LCOV_EXCL_START -- only a misconfigured target fails to add ISel
        if (tpc->addISelPasses()) {
          throw std::runtime_error(
              "target cannot add instruction-selection passes");
        }
        // LCOV_EXCL_STOP
        tpc->setInitialized();

        // Codegen reports failures (e.g. an un-selectable construct) through
        // the context diagnostic handler, not pm->run()'s return value; capture
        // them so a failed selection surfaces as an exception instead of a
        // silently-empty MachineModuleInfo.
        std::string diag;
        {
          ScopedDiagnosticCapture capture(module->getContext(), diag);
          pm->run(*module);
        }
        // LCOV_EXCL_START -- needs an un-selectable input to trigger
        if (!diag.empty()) {
          throw std::runtime_error(
              withDetail("instruction selection failed", diag));
        }
        // LCOV_EXCL_STOP

        return new MachineModuleInfo{std::move(ctxKeepAlive), std::move(module),
                                     std::move(pm), /*ownedMmi=*/nullptr,
                                     &mmiwp->getMMI()};
      },
      "module"_a, "target_machine"_a, nb::keep_alive<0, 2>());

  // Parse .mir text into a MachineModuleInfo. Mirrors run_codegen_to_mir's
  // result but builds the MachineFunctions by deserialization rather than
  // instruction selection. The context is kept alive by the returned wrapper;
  // the TargetMachine (which the parsed MachineFunctions bind to) by
  // keep_alive.
  m.def(
      "parse_mir",
      [](const std::string &text, eudsl::Context &context,
         llvm::TargetMachine &tm) {
        std::shared_ptr<llvm::LLVMContext> ctxKeepAlive = context.shared();
        // The MIR parser reports syntax/semantic errors through the context
        // diagnostic handler and only returns a bare null/true; capture the
        // real diagnostic so the thrown message carries the line and reason.
        std::string diag;
        ScopedDiagnosticCapture capture(context.get(), diag);
        std::unique_ptr<llvm::MIRParser> parser = llvm::createMIRParser(
            llvm::MemoryBuffer::getMemBuffer(text, "<mir>"), context.get());
        // LCOV_EXCL_START -- createMIRParser only fails on internal error
        if (!parser) {
          throw std::runtime_error("could not create MIR parser");
        }
        // LCOV_EXCL_STOP
        std::unique_ptr<llvm::Module> module = parser->parseIRModule();
        if (!module) {
          throw std::runtime_error(
              withDetail("failed to parse the IR portion of the MIR", diag));
        }
        module->setDataLayout(tm.createDataLayout());
        auto ownedMmi = std::make_unique<llvm::MachineModuleInfo>(&tm);
        if (parser->parseMachineFunctions(*module, *ownedMmi)) {
          throw std::runtime_error(
              withDetail("failed to parse machine functions", diag));
        }
        llvm::MachineModuleInfo *mmi = ownedMmi.get();
        return new MachineModuleInfo{std::move(ctxKeepAlive), std::move(module),
                                     /*pm=*/nullptr, std::move(ownedMmi), mmi};
      },
      "text"_a, "context"_a, "target_machine"_a, nb::keep_alive<0, 3>());

  // Create a fresh, empty MachineFunction to build into: make a void() IR
  // Function stub named `name` (a MachineFunction must attach to one), get its
  // MachineFunction from a freshly owned MachineModuleInfo, and give it a
  // single empty entry block. Consumes the module into the returned wrapper
  // (like run_codegen_to_mir), which owns everything the MachineFunction
  // depends on.
  m.def(
      "create_machine_function",
      [](eudsl::Module &mod, llvm::TargetMachine &tm, const std::string &name) {
        std::shared_ptr<llvm::LLVMContext> ctxKeepAlive =
            mod.context().shared();
        std::unique_ptr<llvm::Module> module = mod.take();
        module->setDataLayout(tm.createDataLayout());

        llvm::FunctionType *fnTy =
            llvm::FunctionType::get(llvm::Type::getVoidTy(module->getContext()),
                                    /*isVarArg=*/false);
        llvm::Function *f = llvm::Function::Create(
            fnTy, llvm::GlobalValue::ExternalLinkage, name, *module);

        auto ownedMmi = std::make_unique<llvm::MachineModuleInfo>(&tm);
        llvm::MachineFunction &mf = ownedMmi->getOrCreateMachineFunction(*f);
        mf.push_back(mf.CreateMachineBasicBlock());

        llvm::MachineModuleInfo *mmi = ownedMmi.get();
        return new MachineModuleInfo{std::move(ctxKeepAlive), std::move(module),
                                     /*pm=*/nullptr, std::move(ownedMmi), mmi};
      },
      "module"_a, "target_machine"_a, "name"_a, nb::keep_alive<0, 2>());

  // llvm::MachineIRBuilder -- the GlobalISel builder for generic (G_*) MIR. The
  // typed helpers take the result type as an LLT (a fresh generic vreg is
  // created for it) and the operands as Registers, returning the def Register
  // so builds chain. Construction positions the builder at the end of the
  // function's entry block.
  nb::class_<llvm::MachineIRBuilder>(m, "MachineIRBuilder")
      .def(
          "__init__",
          [](llvm::MachineIRBuilder *self, llvm::MachineFunction &mf) {
            new (self) llvm::MachineIRBuilder(mf);
            self->setMBB(mf.front());
          },
          "machine_function"_a, nb::keep_alive<1, 2>())
      .def(
          "build_constant",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty,
             int64_t value) -> llvm::Register {
            return self.buildConstant(ty, value).getReg(0);
          },
          "type"_a, "value"_a)
      .def(
          "build_add",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, llvm::Register lhs,
             llvm::Register rhs) -> llvm::Register {
            return self.buildAdd(ty, lhs, rhs).getReg(0);
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_sub",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, llvm::Register lhs,
             llvm::Register rhs) -> llvm::Register {
            return self.buildSub(ty, lhs, rhs).getReg(0);
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_mul",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, llvm::Register lhs,
             llvm::Register rhs) -> llvm::Register {
            return self.buildMul(ty, lhs, rhs).getReg(0);
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_copy",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, llvm::Register src)
              -> llvm::Register { return self.buildCopy(ty, src).getReg(0); },
          "type"_a, "src"_a);
}
