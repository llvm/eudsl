// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/CodeGen/MIRParser/MIRParser.h>
#include <llvm/CodeGen/MIRPrinter.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/CodeGenTypes/LowLevelType.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>
#include <vector>

namespace {

// Owns everything the inspected MachineFunctions transitively depend on, so
// none of it is freed out from under them: the IR Module (whose Functions the
// MachineFunctions reference) and the MachineModuleInfo that owns the
// MachineFunctions. The MMI is owned two ways depending on how it was built:
// the codegen path (run_codegen_to_mir) leaves it inside a
// MachineModuleInfoWrapperPass owned by `pm`; the parse path (parse_mir) owns
// it directly via `ownedMmi`. `mmi` is the query handle either way. The
// TargetMachine is a separate Python object kept alive by a keep_alive.
struct MachineModuleInfo {
  std::shared_ptr<llvm::LLVMContext> ctxKeepAlive;
  std::unique_ptr<llvm::Module> module;
  std::unique_ptr<llvm::legacy::PassManager> pm;     // codegen path
  std::unique_ptr<llvm::MachineModuleInfo> ownedMmi; // parse path
  llvm::MachineModuleInfo *mmi;

  // Print the whole module as .mir text: the IR block, then each function's
  // machine-level block, matching what `llc -stop-after` emits.
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
      .def_prop_ro("is_virtual",
                   [](llvm::Register &self) { return self.isVirtual(); })
      .def_prop_ro("is_physical",
                   [](llvm::Register &self) { return self.isPhysical(); });

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
                     const llvm::TargetInstrInfo *tii =
                         self.getMF()->getSubtarget().getInstrInfo();
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
      .def("__str__",
           [](llvm::MachineFunction &self) { return eudsl::toString(self); });

  // The result of run_codegen_to_mir: owns the MachineFunctions and everything
  // they depend on. `machine_function` resolves one by its IR Function name.
  nb::class_<MachineModuleInfo>(m, "MachineModuleInfo")
      .def(
          "machine_function",
          [](MachineModuleInfo &self,
             const std::string &name) -> llvm::MachineFunction * {
            llvm::Function *f = self.module->getFunction(name);
            if (!f)
              throw nb::key_error(
                  ("no function named '" + name + "' in the module").c_str());
            llvm::MachineFunction *mf = self.mmi->getMachineFunction(*f);
            if (!mf)
              throw nb::key_error(
                  ("function '" + name + "' has no MachineFunction").c_str());
            return mf;
          },
          "name"_a, nb::rv_policy::reference_internal)
      .def(
          "to_mir", [](MachineModuleInfo &self) { return self.toMIR(); },
          "Serialize the whole module (IR block + machine functions) as .mir "
          "text.");

  // Run instruction selection on an IR module and hand back the
  // MachineModuleInfo that owns the resulting MachineFunctions. Consumes the
  // module (like adding it to the JIT): the MachineFunctions reference its
  // Functions, so ownership moves into the returned wrapper. Only the ISel
  // passes are added -- not addMachinePasses -- so the MachineFunctions are
  // retained (not freed by FreeMachineFunctionPass) for inspection, the state
  // `llc -stop-after=finalize-isel` produces.
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
        if (tpc->addISelPasses())
          throw std::runtime_error( // LCOV_EXCL_LINE
              "target cannot add instruction-selection passes"); // LCOV_EXCL_LINE
        tpc->setInitialized();
        pm->run(*module);

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
        std::unique_ptr<llvm::MIRParser> parser = llvm::createMIRParser(
            llvm::MemoryBuffer::getMemBuffer(text, "<mir>"), context.get());
        if (!parser)
          throw std::runtime_error(           // LCOV_EXCL_LINE
              "could not create MIR parser"); // LCOV_EXCL_LINE
        std::unique_ptr<llvm::Module> module = parser->parseIRModule();
        if (!module)
          throw std::runtime_error("failed to parse the IR portion of the MIR");
        module->setDataLayout(tm.createDataLayout());
        auto ownedMmi = std::make_unique<llvm::MachineModuleInfo>(&tm);
        if (parser->parseMachineFunctions(*module, *ownedMmi))
          throw std::runtime_error("failed to parse machine functions");
        llvm::MachineModuleInfo *mmi = ownedMmi.get();
        return new MachineModuleInfo{std::move(ctxKeepAlive), std::move(module),
                                     /*pm=*/nullptr, std::move(ownedMmi), mmi};
      },
      "text"_a, "context"_a, "target_machine"_a, nb::keep_alive<0, 3>());
}
