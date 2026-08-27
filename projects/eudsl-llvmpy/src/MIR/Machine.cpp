// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"
#include "MIR/Diagnostics.h"

#include <llvm/ADT/Hashing.h>
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
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetOpcodes.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
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
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace eudsl {
// Defined in PythonCodegen.cpp: the class registered under a scheduler name (an
// invalid object if the name was not registered via register_scheduler), the
// shared MachineSchedRegistry ctor those names share, and install/clear the
// per-thread active class the ctor reads while the pipeline runs.
nb::type_object schedulerClass(const std::string &name);
llvm::MachineSchedRegistry::ScheduleDAGCtor registeredSchedCtor();
void setActiveSchedClass(nb::type_object cls);
void clearActiveSchedClass();

// The regalloc analogues: the RegAllocBase subclass registered under a name,
// the shared harness-pass ctor emit_object points the -regalloc option at, and
// install/clear the per-thread active class the harness reads while it runs.
nb::type_object regallocClass(const std::string &name);
llvm::RegisterRegAlloc::FunctionPassCtor registeredRegAllocCtor();
void setActiveRegAllocClass(nb::type_object cls);
void clearActiveRegAllocClass();
} // namespace eudsl

// Defined in PythonCodegen.cpp.
void populate_python_codegen(nb::module_ &m);

namespace {

// A machine register paired with the MachineFunction that owns it. A virtual
// register's numeric id is only an index into its own function's
// MachineRegisterInfo, so the same id names a *different* register in another
// function. When that id also happens to be a valid, same-typed register in
// the target function, feeding the foreign one in produces well-formed MIR that
// even the verifier cannot flag (it references a register that is legal there);
// only the out-of-bounds / wrong-type foreign ids are detectable, and only when
// asserts are on. Carrying the owner lets the build/consume helpers reject a
// cross-function vreg structurally, by identity, rather than relying on a
// type/index mismatch that may never surface. A physical register is
// target-static (the same MCRegister across every function of a target), so it
// carries a null owner and is accepted anywhere -- matching how physregs are
// used (block live-ins, COPYs to/from $wN).
//
// The vreg<=>owner correspondence is an invariant: a virtual register has a
// non-null owner, a physical register has none. It is not passed in but derived
// by the `owned` factory from the register's own virtual-ness, and the
// constructor is private, so no caller can construct an inconsistent pairing (a
// vreg with no owner, or a physreg with one) -- the illegal states are
// unrepresentable. Equality and hashing consider both id and owner, so two
// same-id registers from different functions are distinct rather than
// colliding; the owner is also checked whenever a register is consumed by the
// builder.
class TypedRegister {
public:
  static TypedRegister owned(llvm::MachineFunction &mf, llvm::Register reg) {
    return TypedRegister(reg, reg.isVirtual() ? &mf : nullptr);
  }
  llvm::Register reg() const { return reg_; }
  llvm::MachineFunction *owner() const { return mf_; }

private:
  TypedRegister(llvm::Register reg, llvm::MachineFunction *mf)
      : reg_(reg), mf_(mf) {}
  llvm::Register reg_;
  llvm::MachineFunction *mf_;
};

// Reject a virtual register minted by a different MachineFunction than `mf`.
// This is the single structural cross-function guard shared by every site that
// consumes a caller-supplied register (the builder helpers, add_reg/def/use,
// G_PHI incomings, the generic `build`), so the check -- and its message --
// live in one place. A physical register (null owner) is target-static and
// passes.
void requireOwnedVReg(const llvm::MachineFunction &mf, const TypedRegister &reg,
                      const char *role) {
  if (reg.reg().isVirtual() && reg.owner() != &mf)
    throw nb::value_error(
        (std::string(role) + " belongs to a different MachineFunction")
            .c_str());
}

// GlobalISel's build helpers validate their operands only with asserts, which
// are compiled out under NDEBUG (the shipped wheel), so bad input would emit
// malformed MIR or fault in a later pass instead of raising. Enforce the same
// preconditions here regardless of build mode: `reg` must be a generic virtual
// register of `b`'s MachineFunction carrying type `ty`. The owner check rejects
// a register minted by a different function; MachineRegisterInfo::getType then
// returns an invalid LLT{} for a non-virtual or out-of-bounds register, so the
// type compare rejects both a type mismatch and a physical register.
void requireVRegOfType(llvm::MachineIRBuilder &b, const TypedRegister &reg,
                       llvm::LLT ty, const char *role) {
  requireOwnedVReg(b.getMF(), reg, role);
  if (b.getMF().getRegInfo().getType(reg.reg()) != ty)
    throw nb::value_error((std::string(role) +
                           " must be a virtual register of this "
                           "MachineFunction with the result type")
                              .c_str());
}

// The Python API lets any MachineBasicBlock be passed to any builder or block,
// with nothing tying the argument to the target function; passing one from a
// different function builds corrupt MIR (cross-linked CFGs) that the verifier
// -- gone under NDEBUG -- would otherwise catch. Guard provenance via
// getParent() against the builder's (or another block's) function. (Registers
// get the same guarantee from TypedRegister's owner.)
void requireSameFunction(const llvm::MachineFunction &mf,
                         const llvm::MachineBasicBlock *mbb, const char *role) {
  if (mbb->getParent() != &mf)
    throw nb::value_error(
        (std::string(role) + " belongs to a different MachineFunction")
            .c_str());
}

void requireVReg(llvm::MachineIRBuilder &b, const TypedRegister &reg,
                 const char *role) {
  requireOwnedVReg(b.getMF(), reg, role);
  if (!b.getMF().getRegInfo().getType(reg.reg()).isValid())
    throw nb::value_error(
        (std::string(role) +
         " must be a generic virtual register of this MachineFunction")
            .c_str());
}

// A block is "terminated" once its last instruction is a barrier terminator
// (G_BR, a return, ...) -- control cannot fall through off the end. A block
// ending in only a conditional G_BRCOND is NOT terminated: it still falls
// through, which is exactly why the standard `build_brcond` + `build_br` pair
// is two terminators in one block. We require *both* isTerminator() and
// isBarrier(): isBarrier() distinguishes the unconditional closers from the
// fall-through conditional, and pairing it with isTerminator() excludes the
// rare barrier-but-not-terminator instruction (e.g. a target TRAP), which does
// not close the block for branch purposes.
bool blockEndsInBarrier(const llvm::MachineBasicBlock &mbb) {
  return !mbb.empty() && mbb.back().isTerminator() && mbb.back().isBarrier();
}

// Appending a terminator to a block that already ends in a barrier builds a
// malformed block (a second, unreachable terminator past the barrier). The
// machine verifier catches this, but only when run explicitly via verify():
// the automatic verification and the build-helper asserts that would flag it
// during construction are gone under NDEBUG, so in a release build the bad
// block is otherwise built silently. The builder only ever inserts at the end
// of its block (setMBB parks the insert point at end(); no mid-block insert
// point is exposed), so back() is reliably where the next instruction lands.
// Reject the double-terminator here.
void requireNotTerminated(llvm::MachineIRBuilder &b, const char *role) {
  if (blockEndsInBarrier(b.getMBB())) {
    throw nb::value_error(
        (std::string(role) +
         " into a block that already ends in a barrier terminator; a block "
         "cannot have a second terminator after an unconditional one")
            .c_str());
  }
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

// getRegisterInfo() can be null for a target without one; real backends always
// have it (these MachineFunctions come from create_machine_function/codegen
// with a real subtarget), so this is purely defensive (mirrors requireTII).
const llvm::TargetRegisterInfo &requireTRI(const llvm::MachineFunction &mf) {
  const llvm::TargetRegisterInfo *tri = mf.getSubtarget().getRegisterInfo();
  if (!tri)
    throw nb::value_error("target has no TargetRegisterInfo"); // LCOV_EXCL_LINE
  return *tri;
}

// Shared register-operand construction for add_def/add_use/add_reg, so the
// "same operation with defaults" invariant lives in one place rather than in
// three parallel CreateReg calls. Rejects flags that LLVM only permits on the
// opposite operand kind (kill on a use, dead on a def) and out-of-range
// subregister indices -- all guarded by asserts in LLVM that vanish under
// NDEBUG, so an otherwise-silent corrupt operand would result.
void addRegOperand(llvm::MachineInstr &mi, const TypedRegister &reg, bool isDef,
                   bool isImp, bool isKill, bool isDead, bool isUndef,
                   bool isEarlyClobber, unsigned subReg, bool isDebug,
                   bool isInternalRead, bool isRenamable) {
  llvm::MachineFunction &mf = *mi.getMF();
  requireOwnedVReg(mf, reg, "register");
  if (isKill && isDef)
    throw nb::value_error(
        "is_kill is only valid on a use operand (is_def=False)");
  if (isDead && !isDef)
    throw nb::value_error(
        "is_dead is only valid on a def operand (is_def=True)");
  if (isRenamable && !reg.reg().isPhysical())
    throw nb::value_error("is_renamable is only valid on a physical register");
  if (subReg) {
    const llvm::TargetRegisterInfo &tri = requireTRI(mf);
    if (subReg >= tri.getNumSubRegIndices())
      throw nb::index_error("sub_reg index out of range");
  }
  mi.addOperand(mf, llvm::MachineOperand::CreateReg(
                        reg.reg(), isDef, isImp, isKill, isDead, isUndef,
                        isEarlyClobber, subReg, isDebug, isInternalRead,
                        isRenamable));
}

// The "current MachineIRBuilder" is tracked on a thread-local stack, mirroring
// the IR builder's `with builder:` / current_builder() mechanism (see the
// thread_local ThreadContextEntry stack in IR/Builder.cpp). `with builder:`
// pushes the builder; current_machine_builder() returns the innermost. The
// entries are MachineIRBuilder pointers; nanobind's instance registry maps a
// returned pointer back to the same Python object, so current_machine_builder
// hands back the object that was entered (identity-stable, which the DSL relies
// on to anchor a MachineValue to its builder). A `with builder:` keeps that
// Python object alive for the block, so the raw pointer never dangles.
static std::vector<llvm::MachineIRBuilder *> &machineBuilderStack() {
  static thread_local std::vector<llvm::MachineIRBuilder *> stack;
  return stack;
}

// The llvm::MachineModuleInfo that owns the MachineFunctions is held one of
// three ways depending on how a MirModule was built. Each construction path is
// its own type, and MirModule::state is a std::variant over them, so "which
// path built this" is the active alternative rather than a runtime null-check
// over three sometimes-null owner pointers. `queryHandle()` recovers the shared
// llvm::MachineModuleInfo* the read accessors need, whichever alternative is
// live.

// run_codegen_to_mir: instruction selection ran in `pm`, which adopted the
// MachineModuleInfoWrapperPass holding the MachineFunctions; `pm` owns it and
// `mmi` queries into it.
struct CodegenOwned {
  std::unique_ptr<llvm::legacy::PassManager> pm;
  llvm::MachineModuleInfo *mmi;
  llvm::MachineModuleInfo *queryHandle() const { return mmi; }
};
// parse_mir: the MachineModuleInfo is owned directly (no PassManager).
struct ParseOwned {
  std::unique_ptr<llvm::MachineModuleInfo> mmi;
  llvm::MachineModuleInfo *queryHandle() const { return mmi.get(); }
};
// create_machine_function: the wrapper is not yet in any PassManager, so
// emit_object can still hand it to one. `tm` (borrowed, kept alive by a
// keep_alive on the factory) is needed to build that emission pipeline.
struct BuildOwned {
  std::unique_ptr<llvm::MachineModuleInfoWrapperPass> mmiwp;
  llvm::TargetMachine *tm;
  llvm::MachineModuleInfo *queryHandle() const { return &mmiwp->getMMI(); }
};
// After emit_object: addPassesToEmitFile adopted the build wrapper into the
// emission `pm`, which we keep so `mmi` stays valid. A BuildOwned transitions
// to this once emitted; it is a distinct type, so a second emit_object cannot
// find a BuildOwned to re-run, and the old "both mmiwp and pm live" state is
// unrepresentable.
struct EmittedOwned {
  std::unique_ptr<llvm::legacy::PassManager> pm;
  llvm::MachineModuleInfo *mmi;
  llvm::MachineModuleInfo *queryHandle() const { return mmi; }
};

// Owns everything the MachineFunctions transitively depend on, so none of it is
// freed out from under them: the LLVMContext (pinned by `ctxKeepAlive`), the IR
// Module (whose Functions the MachineFunctions reference), and -- through
// `state` -- the MachineModuleInfo that owns the MachineFunctions. Constructed
// only through the named factories below, each of which maps a construction
// path to its ownership alternative; the private constructor keeps the
// path/alternative pairing an invariant callers cannot break.
class MirModule {
public:
  static MirModule codegen(std::shared_ptr<llvm::LLVMContext> ctx,
                           std::unique_ptr<llvm::Module> module,
                           std::unique_ptr<llvm::legacy::PassManager> pm,
                           llvm::MachineModuleInfo *mmi) {
    return MirModule(std::move(ctx), std::move(module),
                     CodegenOwned{std::move(pm), mmi});
  }
  static MirModule parsed(std::shared_ptr<llvm::LLVMContext> ctx,
                          std::unique_ptr<llvm::Module> module,
                          std::unique_ptr<llvm::MachineModuleInfo> mmi) {
    return MirModule(std::move(ctx), std::move(module),
                     ParseOwned{std::move(mmi)});
  }
  static MirModule
  building(std::shared_ptr<llvm::LLVMContext> ctx,
           std::unique_ptr<llvm::Module> module,
           std::unique_ptr<llvm::MachineModuleInfoWrapperPass> mmiwp,
           llvm::TargetMachine *tm) {
    return MirModule(std::move(ctx), std::move(module),
                     BuildOwned{std::move(mmiwp), tm});
  }

  llvm::Module &module() { return *module_; }

  // The query handle the read accessors used to get for free as a field: the
  // active alternative knows how to produce its own llvm::MachineModuleInfo*.
  llvm::MachineModuleInfo *mmi() {
    return std::visit([](auto &s) { return s.queryHandle(); }, state_);
  }

  // Print the whole module as .mir text: the IR block, then each function's
  // machine-level block, matching what `llc -stop-after=finalize-isel` emits. A
  // function with no MachineFunction is skipped -- expected for declarations
  // (nothing to lower); a definition only lacks one if codegen failed, which
  // run_codegen_to_mir now reports as an exception rather than reaching here.
  std::string toMIR() {
    llvm::MachineModuleInfo *info = mmi();
    std::string buf;
    llvm::raw_string_ostream os(buf);
    llvm::printMIR(os, *module_);
    for (llvm::Function &f : *module_) {
      if (llvm::MachineFunction *mf = info->getMachineFunction(f))
        llvm::printMIR(os, *info, *mf);
    }
    return buf;
  }

  // Emit a relocatable object for the built (already-selected) MIR by running
  // the back half of codegen. Only valid on the build path, and only once: the
  // BuildOwned is consumed into an EmittedOwned, so "build-path only" and the
  // one-shot are enforced by the variant's active alternative rather than a
  // pair of runtime flags.
  nb::bytes emitObject(std::optional<std::string> scheduler,
                       std::optional<std::string> regalloc) {
    if (std::holds_alternative<EmittedOwned>(state_))
      throw std::runtime_error("object already emitted");
    BuildOwned *build = std::get_if<BuildOwned>(&state_);
    if (!build) {
      throw std::runtime_error(
          "emit_object requires a module built with create_machine_function");
    }
    llvm::TargetMachine *tm = build->tm;
    llvm::MachineModuleInfo *info = &build->mmiwp->getMMI();

    llvm::MachineSchedRegistry::ScheduleDAGCtor schedCtor = nullptr;
    nb::type_object schedClass;
    if (scheduler) {
      schedClass = eudsl::schedulerClass(*scheduler);
      if (!schedClass.is_valid())
        throw std::runtime_error("unknown scheduler: " + *scheduler);
      schedCtor = eudsl::registeredSchedCtor();
    }

    llvm::RegisterRegAlloc::FunctionPassCtor regAllocCtor = nullptr;
    nb::type_object regAllocClass;
    if (regalloc) {
      regAllocClass = eudsl::regallocClass(*regalloc);
      if (!regAllocClass.is_valid())
        throw std::runtime_error("unknown regalloc: " + *regalloc);
      regAllocCtor = eudsl::registeredRegAllocCtor();
    }

    // Verify the hand-built MIR up front so malformed input (the prior PRs'
    // unchecked primitives can produce it: bogus properties, undefined vregs,
    // ...) raises a catchable Python error instead of muddling through the
    // emission pipeline -- whose in-pass verifier and asserts are gone under
    // NDEBUG -- to a garbage object or a fatal codegen abort. The pipeline runs
    // with DisableVerify=true since we verify here.
    std::string report;
    llvm::raw_string_ostream reportOS(report);
    bool ok = true;
    for (llvm::Function &f : *module_) {
      if (llvm::MachineFunction *mf = info->getMachineFunction(f)) {
        ok &= mf->verify(/*p=*/nullptr, /*Banner=*/nullptr, &reportOS,
                         /*AbortOnError=*/false);
      }
    }
    if (!ok)
      throw std::runtime_error(
          eudsl::withDetail("hand-built MIR failed verification", report));

    // Run only the back half of codegen (regalloc, prologue/epilogue, object
    // emission) over the already-selected MachineFunctions:
    // -start-after=finalize-isel skips instruction selection so the hand-built
    // MIR is used as-is. The option is process-global (read by the pass config
    // addPassesToEmitFile builds), so set+restore it; there is no lock, so this
    // relies on the GIL serializing callers (no concurrent/nested/free-threaded
    // codegen).
    auto &opts = llvm::cl::getRegisteredOptions();
    auto it = opts.find("start-after");
    // LCOV_EXCL_START -- start-after is always registered by codegen
    if (it == opts.end()) {
      throw std::runtime_error("the -start-after option is not registered");
    }
    // LCOV_EXCL_STOP
    auto &startAfter = *static_cast<llvm::cl::opt<std::string> *>(it->second);
    std::string saved = startAfter;
    struct Restore {
      llvm::cl::opt<std::string> &opt;
      std::string value;
      ~Restore() { opt = value; }
    } restore{startAfter, saved};
    startAfter = "finalize-isel";

    using SchedCtor = llvm::MachineSchedRegistry::ScheduleDAGCtor;
    using SchedOpt =
        llvm::cl::opt<SchedCtor, false,
                      llvm::RegisterPassParser<llvm::MachineSchedRegistry>>;
    struct RestoreSched {
      SchedOpt *opt = nullptr;
      SchedCtor value = nullptr;
      ~RestoreSched() {
        if (opt)
          *opt = value;
      }
    } restoreSched;
    struct RestoreActiveClass {
      bool active = false;
      ~RestoreActiveClass() {
        if (active)
          eudsl::clearActiveSchedClass();
      }
    } restoreActiveClass;
    if (schedCtor) {
      auto mischedIt = opts.find("misched");
      // LCOV_EXCL_START -- misched is always registered by codegen
      if (mischedIt == opts.end())
        throw std::runtime_error("the -misched option is not registered");
      // LCOV_EXCL_STOP
      auto &misched = *static_cast<SchedOpt *>(mischedIt->second);
      restoreSched.opt = &misched;
      restoreSched.value = misched;
      misched = schedCtor;
      eudsl::setActiveSchedClass(schedClass);
      restoreActiveClass.active = true;
    }

    using RACtor = llvm::RegisterRegAlloc::FunctionPassCtor;
    using RAOpt =
        llvm::cl::opt<RACtor, false,
                      llvm::RegisterPassParser<llvm::RegisterRegAlloc>>;
    struct RestoreRegAlloc {
      RAOpt *opt = nullptr;
      RACtor value = nullptr;
      ~RestoreRegAlloc() {
        if (opt)
          *opt = value;
      }
    } restoreRegAlloc;
    struct RestoreActiveRegAllocClass {
      bool active = false;
      ~RestoreActiveRegAllocClass() {
        if (active)
          eudsl::clearActiveRegAllocClass();
      }
    } restoreActiveRegAllocClass;
    if (regAllocCtor) {
      auto regallocIt = opts.find("regalloc");
      // LCOV_EXCL_START -- regalloc is always registered by codegen
      if (regallocIt == opts.end())
        throw std::runtime_error("the -regalloc option is not registered");
      // LCOV_EXCL_STOP
      auto &regallocOpt = *static_cast<RAOpt *>(regallocIt->second);
      restoreRegAlloc.opt = &regallocOpt;
      restoreRegAlloc.value = regallocOpt;
      regallocOpt = regAllocCtor;
      eudsl::setActiveRegAllocClass(regAllocClass);
      restoreActiveRegAllocClass.active = true;
    }

    // Write straight into a SmallVector (raw_svector_ostream writes through, so
    // no deferred flush surprises when `pm` outlives here).
    llvm::SmallVector<char, 0> buf;
    llvm::raw_svector_ostream os(buf);
    auto pm = std::make_unique<llvm::legacy::PassManager>();
    // addPassesToEmitFile adopts the MMIWrapperPass into `pm` (which we keep,
    // so `info` stays valid); it holds the built MachineFunctions.
    // LCOV_EXCL_START -- AArch64 can always emit an object file
    if (tm->addPassesToEmitFile(
            *pm, os, nullptr, llvm::CodeGenFileType::ObjectFile,
            /*DisableVerify=*/true, build->mmiwp.release())) {
      // The wrapper has already been released into `pm`, so the BuildOwned's
      // mmiwp is now null. Consume it into EmittedOwned before throwing so `pm`
      // (and thus `info`) stays alive and a later query/retry reports a clean
      // error instead of dereferencing the released mmiwp.
      state_ = EmittedOwned{std::move(pm), info};
      throw std::runtime_error("target cannot emit an object file");
    }
    // LCOV_EXCL_STOP
    // The back half (register allocation) requires reserved registers to be
    // frozen; the front of the pipeline normally does this, so do it here for
    // the hand-built MachineFunctions.
    for (llvm::Function &f : *module_) {
      if (llvm::MachineFunction *mf = info->getMachineFunction(f))
        mf->getRegInfo().freezeReservedRegs();
    }
    // A codegen pass reports failure through the context diagnostic handler
    // (DS_Error -> stderr + exit under the default handler), not pm->run()'s
    // value; capture it so it surfaces as an exception. runCodegenPipeline
    // re-raises any Python exception a scheduler override stashed during the
    // run.
    std::string diag;
    {
      eudsl::ScopedDiagnosticCapture capture(module_->getContext(), diag);
      try {
        eudsl::runCodegenPipeline(*pm, *module_);
      } catch (...) {
        // A scheduler override stashed a Python exception, now re-raised. The
        // MMI wrapper already lives in `pm`, so consume into EmittedOwned (as
        // the success path does) before propagating -- otherwise `state_` keeps
        // a released (null) wrapper and `pm`/`info` would dangle.
        state_ = EmittedOwned{std::move(pm), info};
        throw;
      }
    }
    // Consume the BuildOwned into an EmittedOwned: the released wrapper now
    // lives in `pm`, and a second emit_object sees EmittedOwned and refuses.
    state_ = EmittedOwned{std::move(pm), info};
    // LCOV_EXCL_START -- eager verification makes a back-half failure or empty
    // emission unreachable from well-formed hand-built MIR.
    if (!diag.empty())
      throw std::runtime_error(
          eudsl::withDetail("object emission failed", diag));
    if (buf.empty())
      throw std::runtime_error("object emission produced no output");
    // LCOV_EXCL_STOP
    return nb::bytes(buf.data(), buf.size());
  }

private:
  MirModule(
      std::shared_ptr<llvm::LLVMContext> ctx,
      std::unique_ptr<llvm::Module> module,
      std::variant<CodegenOwned, ParseOwned, BuildOwned, EmittedOwned> state)
      : ctxKeepAlive_(std::move(ctx)), module_(std::move(module)),
        state_(std::move(state)) {}

  std::shared_ptr<llvm::LLVMContext> ctxKeepAlive_;
  std::unique_ptr<llvm::Module> module_;
  std::variant<CodegenOwned, ParseOwned, BuildOwned, EmittedOwned> state_;
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
  // target's real registers). Bound as TypedRegister so a virtual register also
  // carries the MachineFunction that minted it (physregs carry none), letting
  // the builder reject a register passed into a different function. Equality
  // and hashing consider both id and owner, so two same-id registers from
  // different functions are distinct -- they can safely coexist as set/dict
  // keys, and a cross-function `==` is False (as well as rejected at build
  // time).
  nb::class_<TypedRegister>(m, "Register")
      .def_prop_ro("id", [](TypedRegister &self) { return self.reg().id(); })
      .def_prop_ro("is_valid",
                   [](TypedRegister &self) { return self.reg().isValid(); })
      .def_prop_ro("is_virtual",
                   [](TypedRegister &self) { return self.reg().isVirtual(); })
      .def_prop_ro("is_physical",
                   [](TypedRegister &self) { return self.reg().isPhysical(); })
      .def_prop_ro("virt_reg_index",
                   [](TypedRegister &self) {
                     if (!self.reg().isVirtual())
                       throw nb::value_error("register is not virtual");
                     return self.reg().virtRegIndex();
                   })
      .def(
          "__eq__",
          [](TypedRegister &self, TypedRegister other) {
            return self.reg() == other.reg() && self.owner() == other.owner();
          },
          nb::is_operator())
      .def(
          "__ne__",
          [](TypedRegister &self, TypedRegister other) {
            return self.reg() != other.reg() || self.owner() != other.owner();
          },
          nb::is_operator())
      .def("__hash__", [](TypedRegister &self) {
        return static_cast<Py_ssize_t>(
            llvm::hash_combine(self.reg().id(), self.owner()));
      });

  // A target register class (e.g. AArch64 GPR32), looked up by name via
  // MachineFunction.reg_class. Opaque handle used when creating typed vregs for
  // already-selected MIR.
  nb::class_<llvm::TargetRegisterClass>(m, "TargetRegisterClass");

  // MachineFunctionProperties::Property -- the flags a MachineFunction carries
  // (set with MachineFunction.set_property). Some mark progress through the
  // codegen pipeline (Legalized, RegBankSelected, Selected) and some are
  // descriptive invariants of the current body (IsSSA, NoPHIs, TracksLiveness,
  // NoVRegs). Building already-selected MIR by hand means setting these to
  // match what the corresponding codegen stage would have produced.
  nb::enum_<llvm::MachineFunctionProperties::Property>(
      m, "MachineFunctionProperty")
      .value("IsSSA", llvm::MachineFunctionProperties::Property::IsSSA)
      .value("NoPHIs", llvm::MachineFunctionProperties::Property::NoPHIs)
      .value("TracksLiveness",
             llvm::MachineFunctionProperties::Property::TracksLiveness)
      .value("NoVRegs", llvm::MachineFunctionProperties::Property::NoVRegs)
      .value("Legalized", llvm::MachineFunctionProperties::Property::Legalized)
      .value("RegBankSelected",
             llvm::MachineFunctionProperties::Property::RegBankSelected)
      .value("Selected", llvm::MachineFunctionProperties::Property::Selected);

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
      .def_prop_ro("is_implicit",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isImplicit();
                   })
      .def_prop_ro("is_kill",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isKill();
                   })
      .def_prop_ro("is_dead",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isDead();
                   })
      .def_prop_ro("is_undef",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isUndef();
                   })
      .def_prop_ro("is_early_clobber",
                   [](llvm::MachineOperand &self) {
                     return self.isReg() && self.isEarlyClobber();
                   })
      .def_prop_ro("is_renamable",
                   [](llvm::MachineOperand &self) {
                     // isRenamable() asserts a physical register (renamable is
                     // a post-RA physreg concept), so guard on that too.
                     return self.isReg() && self.getReg().isPhysical() &&
                            self.isRenamable();
                   })
      .def_prop_ro("sub_reg",
                   [](llvm::MachineOperand &self) -> unsigned {
                     return self.isReg() ? self.getSubReg() : 0;
                   })
      .def_prop_ro("reg",
                   [](llvm::MachineOperand &self) -> TypedRegister {
                     if (!self.isReg())
                       throw nb::value_error("operand is not a register");
                     // Carry provenance so a register inspected off an
                     // instruction can be fed back into its own function's
                     // builder: `owned` records the function for a virtual
                     // register and no owner for a (target-static) physical
                     // one. An operand reachable through this binding always
                     // belongs to an inserted instruction, so getParent() (and
                     // its getMF()) is non-null; a future binding exposing a
                     // detached operand would need to revisit this.
                     return TypedRegister::owned(*self.getParent()->getMF(),
                                                 self.getReg());
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
          [](llvm::MachineInstr &self, TypedRegister reg,
             llvm::MachineBasicBlock *mbb) {
            // Appending (value, block) operands only makes sense for a G_PHI;
            // on any other instruction it silently grows it with junk operands
            // (malformed MIR, no verifier under NDEBUG).
            if (self.getOpcode() != llvm::TargetOpcode::G_PHI) {
              throw nb::value_error(
                  "add_phi_incoming requires a G_PHI instruction");
            }
            llvm::MachineFunction &mf = *self.getMF();
            requireOwnedVReg(mf, reg, "value");
            self.addOperand(mf,
                            llvm::MachineOperand::CreateReg(reg.reg(),
                                                            /*isDef=*/false));
            self.addOperand(mf, llvm::MachineOperand::CreateMBB(mbb));
          },
          "value"_a, "block"_a,
          "Append a (value, predecessor-block) incoming pair to a G_PHI.")
      .def(
          "add_def",
          [](llvm::MachineInstr &self, TypedRegister reg) {
            addRegOperand(self, reg, /*isDef=*/true, /*isImp=*/false,
                          /*isKill=*/false, /*isDead=*/false, /*isUndef=*/false,
                          /*isEarlyClobber=*/false, /*subReg=*/0,
                          /*isDebug=*/false, /*isInternalRead=*/false,
                          /*isRenamable=*/false);
          },
          "reg"_a, "Append a register def operand.")
      .def(
          "add_use",
          [](llvm::MachineInstr &self, TypedRegister reg, bool implicit) {
            addRegOperand(self, reg, /*isDef=*/false, /*isImp=*/implicit,
                          /*isKill=*/false, /*isDead=*/false, /*isUndef=*/false,
                          /*isEarlyClobber=*/false, /*subReg=*/0,
                          /*isDebug=*/false, /*isInternalRead=*/false,
                          /*isRenamable=*/false);
          },
          "reg"_a, "implicit"_a = false,
          "Append a register use operand (implicit=True for an implicit use).")
      .def(
          "add_reg",
          [](llvm::MachineInstr &self, TypedRegister reg, bool isDef,
             bool isImp, bool isKill, bool isDead, bool isUndef,
             bool isEarlyClobber, unsigned subReg, bool isDebug,
             bool isInternalRead, bool isRenamable) {
            addRegOperand(self, reg, isDef, isImp, isKill, isDead, isUndef,
                          isEarlyClobber, subReg, isDebug, isInternalRead,
                          isRenamable);
          },
          "reg"_a, "is_def"_a = false, "implicit"_a = false,
          "is_kill"_a = false, "is_dead"_a = false, "is_undef"_a = false,
          "is_early_clobber"_a = false, "sub_reg"_a = 0, "is_debug"_a = false,
          "is_internal_read"_a = false, "is_renamable"_a = false,
          "Append a register operand, exposing the full MachineOperand "
          "register flag set (def/use, implicit, kill, dead, undef, "
          "early-clobber, sub-register, debug, internal-read, renamable). "
          "is_kill is only valid on a use, is_dead only on a def, and "
          "is_renamable only on a physical register.")
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
          "is_terminated",
          [](llvm::MachineBasicBlock &self) {
            return blockEndsInBarrier(self);
          },
          "Whether this block ends in a barrier terminator (G_BR today; a "
          "return once such a builder exists) so control cannot fall through. "
          "A block ending in only a conditional G_BRCOND is not terminated -- "
          "it falls through.")
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
      .def(
          "add_livein",
          [](llvm::MachineBasicBlock &self, TypedRegister reg) {
            // asMCReg() only asserts physicality (compiled out under NDEBUG),
            // so a virtual register would be truncated into a garbage
            // MCRegister and recorded as a bogus livein. Reject it here.
            if (!reg.reg().isPhysical())
              throw nb::value_error("add_livein requires a physical register");
            self.addLiveIn(reg.reg().asMCReg());
          },
          "reg"_a, "Declare a physical register live-in to this block.")
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
          [](llvm::MachineFunction &self, llvm::LLT ty) -> TypedRegister {
            return TypedRegister::owned(
                self, self.getRegInfo().createGenericVirtualRegister(ty));
          },
          "type"_a, "Create a new generic virtual register of the given LLT.")
      .def(
          "reg_class",
          [](llvm::MachineFunction &self,
             const std::string &name) -> const llvm::TargetRegisterClass * {
            const llvm::TargetRegisterInfo &tri = requireTRI(self);
            for (unsigned i = 0, e = tri.getNumRegClasses(); i < e; ++i) {
              const llvm::TargetRegisterClass *rc = tri.getRegClass(i);
              if (name == tri.getRegClassName(
                              &tri.MCRegisterInfo::getRegClass(rc->getID()))) {
                return rc;
              }
            }
            throw nb::key_error(
                ("no target register class named '" + name + "'").c_str());
          },
          "name"_a, nb::rv_policy::reference,
          "Look up a target register class by name (e.g. \"GPR32\").")
      .def(
          "physreg",
          [](llvm::MachineFunction &self,
             const std::string &name) -> TypedRegister {
            const llvm::TargetRegisterInfo &tri = requireTRI(self);
            for (unsigned i = 1, e = tri.getNumRegs(); i < e; ++i) {
              if (name == tri.getName(i))
                return TypedRegister::owned(self, llvm::Register(i));
            }
            throw nb::key_error(
                ("no physical register named '" + name + "'").c_str());
          },
          "name"_a, "Look up a physical register by name (e.g. \"W0\").")
      .def(
          "create_vreg",
          [](llvm::MachineFunction &self,
             const llvm::TargetRegisterClass *rc) -> TypedRegister {
            return TypedRegister::owned(
                self, self.getRegInfo().createVirtualRegister(rc));
          },
          "reg_class"_a,
          "Create a new virtual register constrained to a register class.")
      .def(
          "set_property",
          [](llvm::MachineFunction &self,
             llvm::MachineFunctionProperties::Property prop) {
            self.getProperties().set(prop);
          },
          "property"_a,
          "Set a MachineFunctionProperties flag. Warning: this asserts, it "
          "does "
          "not request -- the machine verifier trusts these flags and chooses "
          "which invariant classes to enforce from them (e.g. liveness is only "
          "checked when TracksLiveness is set), so setting a flag the MIR does "
          "not actually satisfy can make verify() pass on malformed MIR.")
      .def(
          "has_property",
          [](llvm::MachineFunction &self,
             llvm::MachineFunctionProperties::Property prop) {
            return self.getProperties().hasProperty(prop);
          },
          "property"_a,
          "Whether a MachineFunctionProperties flag is set (read side of "
          "set_property).")
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
  nb::class_<MirModule>(m, "MirModule")
      .def(
          "machine_function",
          [](MirModule &self,
             const std::string &name) -> llvm::MachineFunction * {
            llvm::Function *f = self.module().getFunction(name);
            if (!f) {
              throw nb::key_error(
                  ("no function named '" + name + "' in the module").c_str());
            }
            llvm::MachineFunction *mf = self.mmi()->getMachineFunction(*f);
            if (!mf) {
              throw nb::key_error(
                  ("function '" + name + "' has no MachineFunction").c_str());
            }
            return mf;
          },
          "name"_a, nb::rv_policy::reference_internal)
      .def_prop_ro(
          "machine_functions",
          [](MirModule &self) {
            llvm::MachineModuleInfo *info = self.mmi();
            std::vector<llvm::MachineFunction *> mfs;
            for (llvm::Function &f : self.module()) {
              if (llvm::MachineFunction *mf = info->getMachineFunction(f))
                mfs.push_back(mf);
            }
            return mfs;
          },
          nb::rv_policy::reference_internal,
          "The MachineFunctions in the module (functions without one -- e.g. "
          "declarations -- are skipped).")
      .def("to_mir", &MirModule::toMIR,
           "Serialize the whole module (IR block + machine functions) as .mir "
           "text.")
      .def(
          "emit_object", &MirModule::emitObject, "scheduler"_a = nb::none(),
          "regalloc"_a = nb::none(),
          "Emit a relocatable object file for the built (already-selected) "
          "MIR by running the back half of codegen (regalloc, emission). "
          "Verifies the MIR first, raising if it is malformed. When "
          "`scheduler` names a registered MachineSchedStrategy (see "
          "register_scheduler), the pre-RA MachineScheduler runs it instead of "
          "the target default; an unregistered name raises. When `regalloc` "
          "names a registered RegAllocBase subclass (see register_regalloc), "
          "it "
          "drives register allocation instead of the target default; an "
          "unregistered name raises.");

  // Run instruction selection on an IR module and hand back the MirModule that
  // owns the resulting MachineFunctions. Consumes the
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
          eudsl::ScopedDiagnosticCapture capture(module->getContext(), diag);
          pm->run(*module);
        }
        // LCOV_EXCL_START -- needs an un-selectable input to trigger
        if (!diag.empty()) {
          throw std::runtime_error(
              eudsl::withDetail("instruction selection failed", diag));
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

        return new MirModule(
            MirModule::codegen(std::move(ctxKeepAlive), std::move(module),
                               std::move(pm), &mmiwp->getMMI()));
      },
      "module"_a, "target_machine"_a, "global_isel"_a = false,
      nb::keep_alive<0, 2>());

  // Parse .mir text into a MirModule. Mirrors run_codegen_to_mir's
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
        eudsl::ScopedDiagnosticCapture capture(context.get(), diag);
        std::unique_ptr<llvm::MIRParser> parser = llvm::createMIRParser(
            llvm::MemoryBuffer::getMemBuffer(text, "<mir>"), context.get());
        // LCOV_EXCL_START -- createMIRParser only fails on internal error
        if (!parser) {
          throw std::runtime_error("could not create MIR parser");
        }
        // LCOV_EXCL_STOP
        std::unique_ptr<llvm::Module> module = parser->parseIRModule();
        if (!module) {
          throw std::runtime_error(eudsl::withDetail(
              "failed to parse the IR portion of the MIR", diag));
        }
        module->setDataLayout(tm.createDataLayout());
        auto ownedMmi = std::make_unique<llvm::MachineModuleInfo>(&tm);
        if (parser->parseMachineFunctions(*module, *ownedMmi)) {
          throw std::runtime_error(
              eudsl::withDetail("failed to parse machine functions", diag));
        }
        return new MirModule(MirModule::parsed(
            std::move(ctxKeepAlive), std::move(module), std::move(ownedMmi)));
      },
      "text"_a, "context"_a, "target_machine"_a, nb::keep_alive<0, 3>());

  // Create a fresh MachineFunction to build into. Make an IR Function stub
  // named `name` (a MachineFunction must attach to one) with a trivial body so
  // it is a *definition* -- codegen only processes defined functions, so
  // emit_object would otherwise skip it. The stub's signature and linkage are
  // surfaced as arguments rather than hardcoded; `function_type` defaults to
  // `void()` (a MachineFunction anchored purely for building MIR never reads
  // the IR signature). Hold the MachineFunction in a
  // MachineModuleInfoWrapperPass (not yet in a PassManager) so emit_object can
  // hand it to one. Consumes the module into the returned wrapper.
  m.def(
      "create_machine_function",
      [](eudsl::Module &mod, llvm::TargetMachine &tm, const std::string &name,
         llvm::FunctionType *fnTy, llvm::GlobalValue::LinkageTypes linkage) {
        // Validate before take() so a rejected call leaves the module usable.
        // Function::Create does not fail on a name collision -- it appends a
        // numeric suffix (`f` -> `f.1`), which would then make
        // machine_function(name) throw a confusing "no function named" error.
        if (name.empty())
          throw nb::value_error("function name must not be empty");
        if (mod.get().getFunction(name)) {
          throw nb::value_error(
              ("module already has a function named '" + name + "'").c_str());
        }

        std::shared_ptr<llvm::LLVMContext> ctxKeepAlive =
            mod.context().shared();
        std::unique_ptr<llvm::Module> module = mod.take();
        module->setDataLayout(tm.createDataLayout());

        if (!fnTy)
          fnTy = llvm::FunctionType::get(
              llvm::Type::getVoidTy(module->getContext()), /*isVarArg=*/false);
        llvm::Function *f =
            llvm::Function::Create(fnTy, linkage, name, *module);
        llvm::ReturnInst::Create(
            module->getContext(),
            llvm::BasicBlock::Create(module->getContext(), "entry", f));

        auto mmiwp = std::make_unique<llvm::MachineModuleInfoWrapperPass>(&tm);
        llvm::MachineFunction &mf =
            mmiwp->getMMI().getOrCreateMachineFunction(*f);
        mf.push_back(mf.CreateMachineBasicBlock());

        return new MirModule(MirModule::building(
            std::move(ctxKeepAlive), std::move(module), std::move(mmiwp), &tm));
      },
      "module"_a, "target_machine"_a, "name"_a, "function_type"_a = nullptr,
      "linkage"_a = llvm::GlobalValue::LinkageTypes::ExternalLinkage,
      nb::keep_alive<0, 2>());

  // llvm::MachineIRBuilder -- the GlobalISel builder for generic (G_*) MIR. The
  // typed helpers take the result type as an LLT (a fresh generic vreg is
  // created for it) and the operands as Registers, returning the def Register
  // so builds chain. Construction positions the builder at the end of the
  // function's entry block. Entering the builder as a context manager
  // (`with MachineIRBuilder(mf):`) makes it the current builder for the
  // duration of the block; current_machine_builder() reads it back.
  nb::class_<llvm::MachineIRBuilder>(m, "MachineIRBuilder")
      .def(
          "__init__",
          [](llvm::MachineIRBuilder *self, llvm::MachineFunction &mf) {
            // setMBB(mf.front()) on a block-less MachineFunction dereferences
            // the ilist sentinel (UB, no assert). create_machine_function seeds
            // a block, but this ctor is public and accepts any MachineFunction.
            if (mf.empty()) {
              throw nb::value_error(
                  "MachineFunction has no basic block to build into");
            }
            new (self) llvm::MachineIRBuilder(mf);
            self->setMBB(mf.front());
          },
          "machine_function"_a, nb::keep_alive<1, 2>())
      .def(
          "__enter__",
          [](llvm::MachineIRBuilder *self) -> llvm::MachineIRBuilder * {
            // Make this builder the current one for current_machine_builder().
            machineBuilderStack().push_back(self);
            return self;
          },
          nb::rv_policy::reference)
      .def(
          "__exit__",
          [](llvm::MachineIRBuilder *self, nb::handle, nb::handle, nb::handle) {
            auto &stack = machineBuilderStack();
            if (stack.empty() || stack.back() != self)
              throw nb::value_error("unbalanced MachineIRBuilder enter/exit");
            stack.pop_back();
          },
          "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none())
      .def(
          "build_constant",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty,
             int64_t value) -> TypedRegister {
            // buildConstant derives the width from ty's scalar size; a pointer,
            // scalable-vector, or invalid LLT hits asserting accessors that
            // vanish under NDEBUG and would emit a wrong-width G_CONSTANT.
            if (!(ty.isScalar() || ty.isFixedVector())) {
              throw nb::value_error("build_constant requires a scalar or "
                                    "fixed-vector type");
            }
            return TypedRegister::owned(
                self.getMF(), self.buildConstant(ty, value).getReg(0));
          },
          "type"_a, "value"_a)
      .def(
          "build_add",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, TypedRegister lhs,
             TypedRegister rhs) -> TypedRegister {
            requireVRegOfType(self, lhs, ty, "lhs");
            requireVRegOfType(self, rhs, ty, "rhs");
            return TypedRegister::owned(
                self.getMF(),
                self.buildAdd(ty, lhs.reg(), rhs.reg()).getReg(0));
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_sub",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, TypedRegister lhs,
             TypedRegister rhs) -> TypedRegister {
            requireVRegOfType(self, lhs, ty, "lhs");
            requireVRegOfType(self, rhs, ty, "rhs");
            return TypedRegister::owned(
                self.getMF(),
                self.buildSub(ty, lhs.reg(), rhs.reg()).getReg(0));
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_mul",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty, TypedRegister lhs,
             TypedRegister rhs) -> TypedRegister {
            requireVRegOfType(self, lhs, ty, "lhs");
            requireVRegOfType(self, rhs, ty, "rhs");
            return TypedRegister::owned(
                self.getMF(),
                self.buildMul(ty, lhs.reg(), rhs.reg()).getReg(0));
          },
          "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_copy",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty,
             TypedRegister src) -> TypedRegister {
            requireVRegOfType(self, src, ty, "src");
            return TypedRegister::owned(
                self.getMF(), self.buildCopy(ty, src.reg()).getReg(0));
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
             llvm::LLT ty, TypedRegister lhs,
             TypedRegister rhs) -> TypedRegister {
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
            if (mri.getType(lhs.reg()) != mri.getType(rhs.reg()))
              throw nb::value_error("build_icmp operands must have the same "
                                    "type");
            return TypedRegister::owned(
                self.getMF(),
                self.buildICmp(pred, ty, lhs.reg(), rhs.reg()).getReg(0));
          },
          "predicate"_a, "type"_a, "lhs"_a, "rhs"_a)
      .def(
          "build_br",
          [](llvm::MachineIRBuilder &self,
             llvm::MachineBasicBlock *dest) -> llvm::MachineInstr * {
            requireSameFunction(self.getMF(), dest, "dest");
            requireNotTerminated(self, "build_br");
            return self.buildBr(*dest).getInstr();
          },
          "dest"_a, nb::rv_policy::reference_internal,
          "Build an unconditional branch (G_BR) to `dest`; returns the "
          "instruction so its target can be repointed.")
      .def(
          "build_brcond",
          [](llvm::MachineIRBuilder &self, TypedRegister cond,
             llvm::MachineBasicBlock *dest) {
            requireVReg(self, cond, "cond");
            requireSameFunction(self.getMF(), dest, "dest");
            requireNotTerminated(self, "build_brcond");
            self.buildBrCond(cond.reg(), *dest);
          },
          "cond"_a, "dest"_a,
          "Build a conditional branch (G_BRCOND) on `cond` to `dest`.")
      .def(
          "branch",
          [](llvm::MachineIRBuilder &self,
             llvm::MachineBasicBlock *dest) -> llvm::MachineInstr * {
            requireSameFunction(self.getMF(), dest, "dest");
            requireNotTerminated(self, "branch");
            self.getMBB().addSuccessor(dest);
            return self.buildBr(*dest).getInstr();
          },
          "dest"_a, nb::rv_policy::reference_internal,
          "Unconditional branch to `dest` that also wires the CFG successor "
          "edge from the current block (add_successor + build_br in one step); "
          "returns the G_BR so its target can be repointed. This helper owns "
          "the "
          "edge -- do not also call add_successor(dest) yourself, as neither "
          "dedups and the edge would be wired twice.")
      .def(
          "cond_branch",
          [](llvm::MachineIRBuilder &self, TypedRegister cond,
             llvm::MachineBasicBlock *true_block,
             llvm::MachineBasicBlock *false_block) -> llvm::MachineInstr * {
            requireVReg(self, cond, "cond");
            requireSameFunction(self.getMF(), true_block, "true block");
            requireSameFunction(self.getMF(), false_block, "false block");
            requireNotTerminated(self, "cond_branch");
            // Equal true/false blocks would wire the same successor edge twice
            // (addSuccessor does not dedup) -- a malformed CFG. Such a branch
            // is degenerate anyway (both arms go to the same place); reject it.
            if (true_block == false_block) {
              throw nb::value_error(
                  "cond_branch true_block and false_block must differ; equal "
                  "blocks would wire a duplicate successor edge");
            }
            self.buildBrCond(cond.reg(), *true_block);
            llvm::MachineInstr *falseBr = self.buildBr(*false_block).getInstr();
            self.getMBB().addSuccessor(true_block);
            self.getMBB().addSuccessor(false_block);
            return falseBr;
          },
          "cond"_a, "true_block"_a, "false_block"_a,
          nb::rv_policy::reference_internal,
          "Conditional branch: G_BRCOND on `cond` to `true_block`, fallthrough "
          "G_BR to `false_block`, wiring both CFG successor edges from the "
          "current block (build_brcond + build_br + two add_successor calls in "
          "one step). Returns the fallthrough G_BR so its target can be "
          "repointed. `true_block` and `false_block` must differ, and this "
          "helper owns both edges -- do not also call add_successor for them "
          "yourself, as neither dedups and the edges would be wired twice.")
      .def(
          "build_phi",
          [](llvm::MachineIRBuilder &self, llvm::LLT ty,
             std::vector<std::pair<TypedRegister, llvm::MachineBasicBlock *>>
                 incomings) -> TypedRegister {
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
              phi.addUse(reg.reg());
              phi.addMBB(mbb);
            }
            return TypedRegister::owned(self.getMF(), res);
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
          "responsibility).")
      .def(
          "build",
          [](llvm::MachineIRBuilder &self, unsigned opcode,
             std::vector<std::variant<llvm::LLT, TypedRegister>> dsts,
             std::vector<TypedRegister> srcs) -> llvm::MachineInstr * {
            // The typed analogue of GlobalISel's buildInstr(opcode, DstOps,
            // SrcOps): each dst is either an LLT (a fresh generic vreg of that
            // type is minted for it) or an existing Register to define, and
            // each src is a Register use -- so, unlike the single-def
            // build_add/... helpers, this expresses multiple defs and writing
            // into a caller-provided vreg. buildInstr validates arity/types
            // only with asserts, so bad input aborts the process (our LLVM is
            // built with assertions on) or, where an assert is compiled out,
            // silently builds malformed MIR. Guard the opcode and every
            // caller-supplied register's provenance here so those raise a clean
            // Python error instead; leave the rest (like build_instr) as the
            // caller's responsibility.
            requireValidOpcode(requireTII(self.getMF()), opcode);
            llvm::SmallVector<llvm::DstOp, 1> dstOps;
            for (auto &dst : dsts) {
              if (auto *ty = std::get_if<llvm::LLT>(&dst)) {
                dstOps.emplace_back(*ty);
              } else {
                const TypedRegister &reg = std::get<TypedRegister>(dst);
                requireOwnedVReg(self.getMF(), reg, "dst");
                dstOps.emplace_back(reg.reg());
              }
            }
            llvm::SmallVector<llvm::SrcOp, 2> srcOps;
            for (const TypedRegister &src : srcs) {
              requireOwnedVReg(self.getMF(), src, "src");
              srcOps.emplace_back(src.reg());
            }
            return self.buildInstr(opcode, dstOps, srcOps).getInstr();
          },
          "opcode"_a, "dsts"_a, "srcs"_a, nb::rv_policy::reference_internal,
          "Build an instruction from typed destination and source operands "
          "(the buildInstr(opcode, DstOps, SrcOps) analogue). Each dst is an "
          "LLT (a fresh generic vreg of that type is created and defined) or a "
          "Register to define; each src is a Register use. Returns the "
          "instruction so its operands -- including any minted defs -- can be "
          "read. Like build_instr, operands are not validated against the "
          "opcode's MCInstrDesc.");

  // The innermost MachineIRBuilder entered as a context manager, mirroring the
  // IR module's current_builder(). Raises when there is none.
  m.def(
      "current_machine_builder",
      []() -> llvm::MachineIRBuilder * {
        auto &stack = machineBuilderStack();
        if (stack.empty())
          throw std::runtime_error(
              "no current MachineIRBuilder; enter one with "
              "`with MachineIRBuilder(mf):` (e.g. inside a @machine_function "
              "body)");
        return stack.back();
      },
      nb::rv_policy::reference,
      "The innermost MachineIRBuilder on the thread-local stack.");

  populate_python_codegen(m);
}
