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
#include <llvm/CodeGen/TargetOpcodes.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/CodeGenTypes/LowLevelType.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <nanobind/stl/pair.h>

#include <memory>
#include <string>
#include <utility>
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

// GlobalISel's build helpers validate their operands only with asserts, which
// are compiled out under NDEBUG (the shipped wheel), so bad input would emit
// malformed MIR or fault in a later pass instead of raising. Enforce the same
// preconditions here regardless of build mode: `reg` must be a generic virtual
// register of `b`'s MachineFunction carrying type `ty`. MachineRegisterInfo::
// getType returns an invalid LLT{} for a non-virtual, out-of-bounds, or
// wrong-function register, so a single type compare rejects both a type
// mismatch and a register minted by a different MachineFunction.
void requireVRegOfType(llvm::MachineIRBuilder &b, llvm::Register reg,
                       llvm::LLT ty, const char *role) {
  if (b.getMF().getRegInfo().getType(reg) != ty)
    throw nb::value_error((std::string(role) +
                           " must be a virtual register of this "
                           "MachineFunction with the result type")
                              .c_str());
}

// A MachineBasicBlock / Register crosses the Python boundary as a bare handle
// with no back-link to its function; passing one from a different function
// builds corrupt MIR (cross-linked CFGs, out-of-bounds vreg indexes) that the
// verifier -- gone under NDEBUG -- would otherwise catch. Guard provenance
// against the builder's (or another block's) function.
void requireSameFunction(const llvm::MachineFunction &mf,
                         const llvm::MachineBasicBlock *mbb, const char *role) {
  if (mbb->getParent() != &mf)
    throw nb::value_error(
        (std::string(role) + " belongs to a different MachineFunction")
            .c_str());
}

void requireVReg(llvm::MachineIRBuilder &b, llvm::Register reg,
                 const char *role) {
  if (!b.getMF().getRegInfo().getType(reg).isValid())
    throw nb::value_error(
        (std::string(role) +
         " must be a generic virtual register of this MachineFunction")
            .c_str());
}

// getInstrInfo() can be null for a target without one; real backends always
// have it (these MachineFunctions come from create_machine_function/codegen
// with a real subtarget), so this is purely defensive.
const llvm::TargetInstrInfo &requireTII(const llvm::MachineFunction &mf) {
  const llvm::TargetInstrInfo *tii = mf.getSubtarget().getInstrInfo();
  if (!tii)
    throw nb::value_error("target has no TargetInstrInfo"); // LCOV_EXCL_LINE
  return *tii;
}

// MCInstrInfo::get/getName index an array bounded only by an assert (compiled
// out under NDEBUG), so an out-of-range opcode would read out of bounds -- a
// segfault or, worse, a silently-wrong name. Validate before indexing.
void requireValidOpcode(const llvm::TargetInstrInfo &tii, unsigned opcode) {
  if (opcode >= tii.getNumOpcodes())
    throw nb::index_error("opcode number out of range");
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
      .def(
          "set_branch_target",
          [](llvm::MachineInstr &self, llvm::MachineBasicBlock *mbb) {
            if (self.getNumOperands() == 0 || !self.getOperand(0).isMBB()) {
              throw nb::value_error(
                  "instruction has no branch-target (MBB) operand");
            }
            self.getOperand(0).setMBB(mbb);
          },
          "block"_a,
          "Repoint a branch's target block (operand 0), e.g. a G_BR.")
      .def(
          "add_phi_incoming",
          [](llvm::MachineInstr &self, llvm::Register reg,
             llvm::MachineBasicBlock *mbb) {
            // Appending (value, block) operands only makes sense for a G_PHI;
            // on any other instruction it silently grows it with junk operands
            // (malformed MIR, no verifier under NDEBUG).
            if (self.getOpcode() != llvm::TargetOpcode::G_PHI) {
              throw nb::value_error(
                  "add_phi_incoming requires a G_PHI instruction");
            }
            llvm::MachineFunction &mf = *self.getMF();
            self.addOperand(mf,
                            llvm::MachineOperand::CreateReg(reg,
                                                            /*isDef=*/false));
            self.addOperand(mf, llvm::MachineOperand::CreateMBB(mbb));
          },
          "value"_a, "block"_a,
          "Append a (value, predecessor-block) incoming pair to a G_PHI.")
      .def(
          "add_def",
          [](llvm::MachineInstr &self, llvm::Register reg) {
            self.addOperand(*self.getMF(), llvm::MachineOperand::CreateReg(
                                               reg, /*isDef=*/true));
          },
          "reg"_a, "Append a register def operand.")
      .def(
          "add_use",
          [](llvm::MachineInstr &self, llvm::Register reg) {
            self.addOperand(*self.getMF(), llvm::MachineOperand::CreateReg(
                                               reg, /*isDef=*/false));
          },
          "reg"_a, "Append a register use operand.")
      .def(
          "add_imm",
          [](llvm::MachineInstr &self, int64_t value) {
            self.addOperand(*self.getMF(),
                            llvm::MachineOperand::CreateImm(value));
          },
          "value"_a, "Append an immediate operand.")
      .def(
          "add_mbb",
          [](llvm::MachineInstr &self, llvm::MachineBasicBlock *mbb) {
            self.addOperand(*self.getMF(),
                            llvm::MachineOperand::CreateMBB(mbb));
          },
          "block"_a, "Append a machine-basic-block operand.")
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
      .def(
          "add_successor",
          [](llvm::MachineBasicBlock &self, llvm::MachineBasicBlock *succ) {
            // addSuccessor also calls succ->addPredecessor(this), so a
            // cross-function successor would corrupt two functions' CFGs.
            requireSameFunction(*self.getParent(), succ, "successor");
            self.addSuccessor(succ);
          },
          "successor"_a, "Add a CFG successor edge to another block.")
      .def(
          "replace_successor",
          [](llvm::MachineBasicBlock &self, llvm::MachineBasicBlock *old,
             llvm::MachineBasicBlock *replacement) {
            // replaceSuccessor only asserts `old` is a successor (gone under
            // NDEBUG); without the check it walks past succ_end() and corrupts
            // the CFG. Guard it, and reject a cross-function replacement.
            if (!self.isSuccessor(old)) {
              throw nb::value_error("`old` is not a successor of this block");
            }
            requireSameFunction(*self.getParent(), replacement, "new");
            self.replaceSuccessor(old, replacement);
          },
          "old"_a, "new"_a, "Replace a CFG successor edge with another block.")
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
          "create_generic_virtual_register",
          [](llvm::MachineFunction &self, llvm::LLT ty) -> llvm::Register {
            return self.getRegInfo().createGenericVirtualRegister(ty);
          },
          "type"_a, "Create a new generic virtual register of the given LLT.")
      .def(
          "create_block",
          [](llvm::MachineFunction &self,
             const llvm::BasicBlock *bb) -> llvm::MachineBasicBlock * {
            llvm::MachineBasicBlock *mbb = self.CreateMachineBasicBlock(bb);
            self.push_back(mbb);
            return mbb;
          },
          "basic_block"_a.none() = nb::none(),
          nb::rv_policy::reference_internal,
          "Append a new, empty MachineBasicBlock to the function, optionally "
          "linked to an IR BasicBlock for debug info/naming.")
      .def(
          "opcode",
          [](llvm::MachineFunction &self, const std::string &name) -> unsigned {
            const llvm::TargetInstrInfo &tii = requireTII(self);
            for (unsigned i = 0, e = tii.getNumOpcodes(); i < e; ++i) {
              if (tii.getName(i) == name)
                return i;
            }
            throw nb::key_error(
                ("no target opcode named '" + name + "'").c_str());
          },
          "name"_a,
          "Look up a target opcode number by mnemonic (e.g. \"ADDWrr\").")
      .def(
          "opcode_name",
          [](llvm::MachineFunction &self, unsigned opcode) {
            const llvm::TargetInstrInfo &tii = requireTII(self);
            requireValidOpcode(tii, opcode);
            return tii.getName(opcode).str();
          },
          "opcode"_a, "The mnemonic for a target opcode number.")
      .def(
          "verify",
          [](llvm::MachineFunction &self) {
            return self.verify(/*p=*/nullptr, /*Banner=*/nullptr,
                               /*OS=*/nullptr, /*AbortOnError=*/false);
          },
          "Run the machine verifier; returns True if no problems were found.")
      .def(
          "verify_diagnostic",
          [](llvm::MachineFunction &self) {
            std::string buf;
            llvm::raw_string_ostream os(buf);
            self.verify(/*p=*/nullptr, /*Banner=*/nullptr, &os,
                        /*AbortOnError=*/false);
            return buf;
          },
          "Run the machine verifier and return its report -- an empty string "
          "if the MIR is well-formed, else the verifier's explanation.")
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
  // produces. With global_isel=True the ISel passes are the GlobalISel pipeline
  // (IRTranslator -> Legalizer -> RegBankSelect -> InstructionSelect), so the
  // retained MIR is fully target-selected when selection succeeds;
  // DisableWithDiag keeps a legalization failure from aborting the process, and
  // a post-run scan turns a resulting residual generic op into an exception.
  m.def(
      "run_codegen_to_mir",
      [](eudsl::Module &mod, llvm::TargetMachine &tm, bool globalISel) {
        std::shared_ptr<llvm::LLVMContext> ctxKeepAlive =
            mod.context().shared();
        std::unique_ptr<llvm::Module> module = mod.take();
        module->setDataLayout(tm.createDataLayout());

        // Set both flags on every call (the tm is caller-owned and reused), so
        // the selection mode is a pure function of `globalISel` and doesn't
        // stick from a prior global_isel=True call.
        tm.setGlobalISel(globalISel);
        tm.setGlobalISelAbort(globalISel
                                  ? llvm::GlobalISelAbortMode::DisableWithDiag
                                  : llvm::GlobalISelAbortMode::Enable);

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

        // GlobalISel with DisableWithDiag emits its fallback diagnostic as a
        // warning (not captured above) and leaves residual generic (G_*) ops
        // rather than aborting. Scan for them so an incompletely-selected
        // function is reported instead of returned as if fully selected.
        if (globalISel) {
          for (llvm::Function &f : *module) {
            llvm::MachineFunction *mf = mmiwp->getMMI().getMachineFunction(f);
            if (!mf)
              continue;
            for (llvm::MachineBasicBlock &mbb : *mf) {
              for (llvm::MachineInstr &mi : mbb) {
                // LCOV_EXCL_START -- needs an un-selectable input to trigger
                if (llvm::isPreISelGenericOpcode(mi.getOpcode())) {
                  throw std::runtime_error(
                      "GlobalISel did not fully select the module (a generic "
                      "G_* instruction remains)");
                }
                // LCOV_EXCL_STOP
              }
            }
          }
        }

        return new MachineModuleInfo{std::move(ctxKeepAlive), std::move(module),
                                     std::move(pm), /*ownedMmi=*/nullptr,
                                     &mmiwp->getMMI()};
      },
      "module"_a, "target_machine"_a, "global_isel"_a = false,
      nb::keep_alive<0, 2>());

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
        // Validate before take() so a rejected call leaves the module usable.
        // Function::Create does not fail on a name collision -- it appends a
        // numeric suffix (`f` -> `f.1`), which would then make
        // machine_function(name) throw a confusing "no function named" error.
        if (name.empty())
          throw nb::value_error("function name must not be empty");
        if (mod.get().getFunction(name))
          throw nb::value_error(
              ("module already has a function named '" + name + "'").c_str());

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
            // setMBB(mf.front()) on a block-less MachineFunction dereferences
            // the ilist sentinel (UB, no assert). create_machine_function seeds
            // a block, but this ctor is public and accepts any MachineFunction.
            if (mf.empty())
              throw nb::value_error(
                  "MachineFunction has no basic block to build into");
            new (self) llvm::MachineIRBuilder(mf);
            self->setMBB(mf.front());
          },
          "machine_function"_a, nb::keep_alive<1, 2>())
      .def(
          "build_constant",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty,
             int64_t value) -> llvm::Register {
            // buildConstant derives the width from ty's scalar size; a pointer,
            // scalable-vector, or invalid LLT hits asserting accessors that
            // vanish under NDEBUG and would emit a wrong-width G_CONSTANT.
            if (!(ty.isScalar() || ty.isFixedVector()))
              throw nb::value_error("build_constant requires a scalar or "
                                    "fixed-vector type");
            return self.buildConstant(ty, value).getReg(0);
          },
          "type"_a, "value"_a)
      .def(
          "build_add",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, llvm::Register lhs,
             llvm::Register rhs) -> llvm::Register {
            requireVRegOfType(self, lhs, ty, "lhs");
            requireVRegOfType(self, rhs, ty, "rhs");
            return self.buildAdd(ty, lhs, rhs).getReg(0);
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_sub",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, llvm::Register lhs,
             llvm::Register rhs) -> llvm::Register {
            requireVRegOfType(self, lhs, ty, "lhs");
            requireVRegOfType(self, rhs, ty, "rhs");
            return self.buildSub(ty, lhs, rhs).getReg(0);
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_mul",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, llvm::Register lhs,
             llvm::Register rhs) -> llvm::Register {
            requireVRegOfType(self, lhs, ty, "lhs");
            requireVRegOfType(self, rhs, ty, "rhs");
            return self.buildMul(ty, lhs, rhs).getReg(0);
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_copy",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty,
             llvm::Register src) -> llvm::Register {
            requireVRegOfType(self, src, ty, "src");
            return self.buildCopy(ty, src).getReg(0);
          },
          "type"_a, "src"_a)
      .def_prop_ro(
          "insert_block",
          [](llvm::MachineIRBuilder &self) -> llvm::MachineBasicBlock * {
            return &self.getMBB();
          },
          nb::rv_policy::reference_internal,
          "The block new instructions are appended to.")
      .def(
          "set_block",
          [](llvm::MachineIRBuilder &self, llvm::MachineBasicBlock *mbb) {
            requireSameFunction(self.getMF(), mbb, "block");
            self.setMBB(*mbb);
          },
          "block"_a, "Insert subsequent instructions at the end of `block`.")
      .def_prop_ro(
          "machine_function",
          [](llvm::MachineIRBuilder &self) -> llvm::MachineFunction * {
            return &self.getMF();
          },
          nb::rv_policy::reference_internal,
          "The MachineFunction this builder inserts into.")
      .def(
          "build_icmp",
          [](llvm::MachineIRBuilder &self, llvm::CmpInst::Predicate pred,
             llvm::LLT ty, llvm::Register lhs,
             llvm::Register rhs) -> llvm::Register {
            // buildICmp only asserts these (gone under NDEBUG): an integer
            // predicate, an s1 (or vector-of-s1) result, and same-typed operand
            // vregs of this function.
            if (!llvm::CmpInst::isIntPredicate(pred))
              throw nb::value_error(
                  "build_icmp requires an integer comparison predicate");
            llvm::LLT s1 = llvm::LLT::scalar(1);
            if (ty != s1 && !(ty.isFixedVector() && ty.getElementType() == s1))
              throw nb::value_error(
                  "build_icmp result type must be s1 or a fixed vector of s1");
            requireVReg(self, lhs, "lhs");
            requireVReg(self, rhs, "rhs");
            const llvm::MachineRegisterInfo &mri = self.getMF().getRegInfo();
            if (mri.getType(lhs) != mri.getType(rhs))
              throw nb::value_error("build_icmp operands must have the same "
                                    "type");
            return self.buildICmp(pred, ty, lhs, rhs).getReg(0);
          },
          "predicate"_a, "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_br",
          [](llvm::MachineIRBuilder &self,
             llvm::MachineBasicBlock *dest) -> llvm::MachineInstr * {
            requireSameFunction(self.getMF(), dest, "dest");
            return self.buildBr(*dest).getInstr();
          },
          "dest"_a, nb::rv_policy::reference_internal,
          "Build an unconditional branch (G_BR) to `dest`; returns the "
          "instruction so its target can be repointed.")
      .def(
          "build_brcond",
          [](llvm::MachineIRBuilder &self, llvm::Register cond,
             llvm::MachineBasicBlock *dest) {
            requireVReg(self, cond, "cond");
            requireSameFunction(self.getMF(), dest, "dest");
            self.buildBrCond(cond, *dest);
          },
          "cond"_a, "dest"_a,
          "Build a conditional branch (G_BRCOND) on `cond` to `dest`.")
      .def(
          "build_phi",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty,
             std::vector<std::pair<llvm::Register, llvm::MachineBasicBlock *>>
                 incomings) -> llvm::Register {
            // A G_PHI with no operands is structurally invalid; each incoming
            // value must be a vreg of this function with the phi's type, and
            // each predecessor block must belong to this function.
            if (incomings.empty())
              throw nb::value_error("build_phi requires at least one "
                                    "(value, predecessor-block) pair");
            for (auto &[reg, mbb] : incomings) {
              requireVRegOfType(self, reg, ty, "incoming value");
              requireSameFunction(self.getMF(), mbb, "predecessor block");
            }
            llvm::Register res =
                self.getMRI()->createGenericVirtualRegister(ty);
            llvm::MachineInstrBuilder phi =
                self.buildInstr(llvm::TargetOpcode::G_PHI);
            phi.addDef(res);
            for (auto &[reg, mbb] : incomings) {
              phi.addUse(reg);
              phi.addMBB(mbb);
            }
            return res;
          },
          "type"_a, "incomings"_a,
          "Build a G_PHI from (value, predecessor-block) pairs.")
      .def(
          "build_empty_phi",
          [](llvm::MachineIRBuilder &self,
             llvm::LLT ty) -> llvm::MachineInstr * {
            llvm::Register res =
                self.getMRI()->createGenericVirtualRegister(ty);
            llvm::MachineInstrBuilder phi =
                self.buildInstr(llvm::TargetOpcode::G_PHI);
            phi.addDef(res);
            return phi.getInstr();
          },
          "type"_a, nb::rv_policy::reference_internal,
          "Build a G_PHI with only its def (of type `type`); add incomings "
          "later "
          "with MachineInstr.add_phi_incoming. Its def is operand 0.")
      .def(
          "build_instr",
          [](llvm::MachineIRBuilder &self,
             unsigned opcode) -> llvm::MachineInstr * {
            requireValidOpcode(requireTII(self.getMF()), opcode);
            return self.buildInstr(opcode).getInstr();
          },
          "opcode"_a, nb::rv_policy::reference_internal,
          "Build and insert an empty instruction of the given opcode; append "
          "operands with MachineInstr.add_def/add_use/add_imm/add_mbb. The "
          "BuildMI analogue for target-specific opcodes. Like BuildMI, the "
          "appended operands are not validated against the opcode's "
          "MCInstrDesc (arity/kind/def-first order are the caller's "
          "responsibility).");
}
