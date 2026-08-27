// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "MIR/AllocationOrder.h"
#include "MIR/Diagnostics.h"
#include "MIR/RegAllocBase.h"
#include "MIR/SplitKit.h"

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/CodeGen/CalcSpillWeights.h>
#include <llvm/CodeGen/LiveDebugVariables.h>
#include <llvm/CodeGen/LiveInterval.h>
#include <llvm/CodeGen/LiveIntervals.h>
#include <llvm/CodeGen/LiveRangeEdit.h>
#include <llvm/CodeGen/LiveRegMatrix.h>
#include <llvm/CodeGen/LiveStacks.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineBlockFrequencyInfo.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineLoopInfo.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/CodeGen/ScheduleDAG.h>
#include <llvm/CodeGen/ScheduleDAGMutation.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/Spiller.h>
#include <llvm/CodeGen/VirtRegMap.h>
#include <llvm/PassRegistry.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <deque>
#include <exception>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace eudsl {
// The single definition of the codegen-error stash declared extern in
// Diagnostics.h; scheduler overrides in this TU stash into it.
thread_local std::exception_ptr pendingCodegenError;
} // namespace eudsl

namespace {

// The Python strategy class emit_object selected for the current run. Set
// (under the GIL) before the pipeline runs and cleared after, so the shared
// registry ctor knows which class to instantiate per MachineFunction.
// GIL-serialized, matching the process-global -misched handling in Machine.cpp.
thread_local nb::type_object activeSchedClass;

// The Python RegAllocBase subclass emit_object selected for the current run,
// set/cleared under the GIL. Mirrors activeSchedClass: the harness pass ctor is
// a non-capturing function pointer, so it reads the class from here.
thread_local nb::type_object activeRegAllocClass;

class PySchedStrategy : public llvm::MachineSchedStrategy {
public:
  NB_TRAMPOLINE(llvm::MachineSchedStrategy, 9);

  llvm::MachineSchedPolicy getPolicy() const override {
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return {};
    try {
      return nb::cast<llvm::MachineSchedPolicy>(
          nb_trampoline.base().attr("get_policy")());
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
      return {};
    }
  }

  bool shouldTrackPressure() const override {
    return getPolicy().ShouldTrackPressure;
  }

  void initialize(llvm::ScheduleDAGMI *dag) override {
    shadow.clear();
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("initialize")(
          nb::cast(dag, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void releaseTopNode(llvm::SUnit *su) override {
    shadow.push_back(su);
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("release_top_node")(
          nb::cast(su, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void releaseBottomNode(llvm::SUnit *su) override {
    shadow.push_back(su);
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("release_bottom_node")(
          nb::cast(su, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  llvm::SUnit *pickNode(bool &isTopNode) override {
    nb::gil_scoped_acquire gil;
    if (!eudsl::pendingCodegenError) {
      try {
        nb::object choice = nb_trampoline.base().attr("pick_node")();
        // None signals "nothing ready" -- end scheduling (LLVM's nullptr).
        if (choice.is_none())
          return nullptr;
        auto [su, isTop] = nb::cast<std::pair<llvm::SUnit *, bool>>(choice);
        isTopNode = isTop;
        return su;
      } catch (...) {
        eudsl::pendingCodegenError = std::current_exception();
      }
    }
    // Stash path (or draining after a prior stash): return a still-ready,
    // not-yet-scheduled node so the unskippable pipeline winds down to
    // runCodegenPipeline's re-raise. LLVM only releases ready nodes; skip any
    // it has since scheduled, and report the node's readiness direction.
    while (!shadow.empty()) {
      llvm::SUnit *su = shadow.front();
      shadow.erase(shadow.begin());
      if (!su->isScheduled) {
        isTopNode = su->isTopReady();
        return su;
      }
    }
    return nullptr;
  }

  void schedNode(llvm::SUnit *su, bool isTopNode) override {
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("sched_node")(
          nb::cast(su, nb::rv_policy::reference), isTopNode);
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  // Optional lifecycle hooks: forwarded only if the Python subclass defines
  // them, otherwise LLVM's no-op default stands. registerRoots fires once the
  // full initial ready set has been released; enterMBB/leaveMBB bracket each
  // block (which may hold several scheduling regions).
  void registerRoots() override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "register_roots"))
      return;
    try {
      self.attr("register_roots")();
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void enterMBB(llvm::MachineBasicBlock *mbb) override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "enter_mbb"))
      return;
    try {
      self.attr("enter_mbb")(nb::cast(mbb, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void leaveMBB() override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "leave_mbb"))
      return;
    try {
      self.attr("leave_mbb")();
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

private:
  // Mirrors the ready nodes LLVM releases, recorded before the fallible Python
  // call. It is the pick_node fallback after a stash: LLVM only releases ready
  // nodes, so a still-unscheduled one is always a legal choice. The raw
  // contract puts the real ready set in Python, out of reach once it has failed
  // -- this is the safety net.
  std::vector<llvm::SUnit *> shadow;
};

// The strategy LLVM owns. nanobind refuses to move a Python-created instance
// into a default-deleter unique_ptr<MachineSchedStrategy> (its C++ subobject is
// inline in the PyObject, so a raw delete is UB) -- which is
// ScheduleDAGMILive's sink. So LLVM owns this plain heap object, which keeps
// the Python instance alive and forwards each virtual into its inline
// PySchedStrategy (which does the work and never throws). The dtor drops the
// Python reference under the GIL.
class OwningPyStrategy : public llvm::MachineSchedStrategy {
public:
  explicit OwningPyStrategy(nb::object pyStrategy)
      : pyStrategy(std::move(pyStrategy)),
        inner(nb::inst_ptr<llvm::MachineSchedStrategy>(this->pyStrategy)) {}

  ~OwningPyStrategy() override {
    nb::gil_scoped_acquire gil;
    pyStrategy.reset();
  }

  llvm::MachineSchedPolicy getPolicy() const override {
    return inner->getPolicy();
  }
  bool shouldTrackPressure() const override {
    return inner->shouldTrackPressure();
  }
  void initialize(llvm::ScheduleDAGMI *dag) override { inner->initialize(dag); }
  void releaseTopNode(llvm::SUnit *su) override { inner->releaseTopNode(su); }
  void releaseBottomNode(llvm::SUnit *su) override {
    inner->releaseBottomNode(su);
  }
  llvm::SUnit *pickNode(bool &isTopNode) override {
    return inner->pickNode(isTopNode);
  }
  void schedNode(llvm::SUnit *su, bool isTopNode) override {
    inner->schedNode(su, isTopNode);
  }
  void registerRoots() override { inner->registerRoots(); }
  void enterMBB(llvm::MachineBasicBlock *mbb) override { inner->enterMBB(mbb); }
  void leaveMBB() override { inner->leaveMBB(); }

private:
  nb::object pyStrategy;
  llvm::MachineSchedStrategy *inner;
};

// Stable storage for registered names: a leaked deque (pure C++, no Python
// refs) whose element c_str() the MachineSchedRegistry node borrows for process
// lifetime.
std::deque<std::string> &schedNames() {
  static auto *names = new std::deque<std::string>();
  return *names;
}

// name -> Python class, held in llvm.mir_strategies._scheduler_classes. Python
// owns it, so the classes are released at interpreter teardown (a C++-held
// nb::object static would pin the subclass types past nanobind's teardown and
// trip its leak checker).
nb::dict schedulerClasses() {
  return nb::cast<nb::dict>(
      nb::module_::import_("llvm.mir_strategies").attr("_scheduler_classes"));
}

// The live MachineSchedRegistry nodes, kept (leaked) so they stay registered
// for process lifetime.
std::vector<std::unique_ptr<llvm::MachineSchedRegistry>> &schedRegistryNodes() {
  static auto *nodes =
      new std::vector<std::unique_ptr<llvm::MachineSchedRegistry>>();
  return *nodes;
}

// Shared registry ctor for every registered name (the ctor is a non-capturing
// function pointer, so it cannot carry the class). emit_object set
// activeSchedClass; construct a fresh instance per MachineFunction and hand
// LLVM an OwningPyStrategy around it.
llvm::ScheduleDAGInstrs *
createRegisteredPyStrategy(llvm::MachineSchedContext *c) {
  nb::gil_scoped_acquire gil;
  // LCOV_EXCL_START -- emit_object always sets the active class before running
  if (!activeSchedClass.is_valid())
    return llvm::createSchedLive(c);
  // LCOV_EXCL_STOP
  try {
    auto strategy = std::make_unique<OwningPyStrategy>(activeSchedClass());
    auto *dag = new llvm::ScheduleDAGMILive(c, std::move(strategy));
    dag->addMutation(llvm::createCopyConstrainDAGMutation(dag->TII, dag->TRI));
    return dag;
  } catch (...) {
    // Constructing the strategy runs the subclass __init__, which can raise.
    // This ctor is called from inside pm.run (libLLVMCodeGen, -fno-exceptions),
    // so stash the error and wind down with the default DAG; runCodegenPipeline
    // re-raises after the run.
    eudsl::pendingCodegenError = std::current_exception();
    return llvm::createSchedLive(c);
  }
}

// A complete pure-C++ RegAllocBase: first-free-or-spill over the target
// allocation order, driven by a spill-weight priority queue. It is both the
// base for the Python trampoline (supplying the fallback each virtual defers
// to) and the standalone allocator the harness runs when a Python __init__
// raises, so the emitted MIR stays valid and the stashed exception can
// re-raise.
class NativeRegAlloc : public llvm::RegAllocBase {
public:
  // spiller() is pure and dereferenced by allocatePhysRegs; the harness injects
  // the concrete Spiller before driving, so this is never called with null.
  llvm::Spiller &spiller() override { return *injectedSpiller; }

  void enqueueImpl(const llvm::LiveInterval *li) override {
    nativeQueue.push({li->weight(), li->reg()});
  }

  const llvm::LiveInterval *dequeue() override {
    // The queue holds (weight, reg) rather than pointers because splitting
    // recreates intervals, so re-fetch and skip registers already assigned or
    // gone (the spiller coalesces snippets).
    while (!nativeQueue.empty()) {
      QueueEntry e = nativeQueue.top();
      nativeQueue.pop();
      if (VRM->hasPhys(e.reg) || !LIS->hasInterval(e.reg))
        continue;
      return &LIS->getInterval(e.reg);
    }
    return nullptr;
  }

  llvm::MCRegister
  selectOrSplit(const llvm::LiveInterval &vreg,
                llvm::SmallVectorImpl<llvm::Register> &splitLVRs) override {
    auto order =
        llvm::AllocationOrder::create(vreg.reg(), *VRM, RegClassInfo, Matrix);
    for (llvm::MCRegister phys : order) {
      if (Matrix->checkInterference(vreg, phys) == llvm::LiveRegMatrix::IK_Free)
        return phys;
    }
    llvm::LiveRangeEdit lre(&vreg, splitLVRs, *mf, *LIS, VRM,
                            /*delegate=*/nullptr, &DeadRemats);
    injectedSpiller->spill(lre);
    return llvm::MCRegister();
  }

  // Public wrappers the harness pass uses to reach the protected driver.
  void pyInit(llvm::VirtRegMap &vrm, llvm::LiveIntervals &lis,
              llvm::LiveRegMatrix &mat, llvm::Spiller &sp,
              llvm::MachineFunction &mfn, llvm::SplitAnalysis *sa,
              llvm::SplitEditor *se) {
    injectedSpiller = &sp;
    mf = &mfn;
    splitAnalysis = sa;
    splitEditor = se;
    init(vrm, lis, mat);
  }
  void pyAllocate() {
    allocatePhysRegs();
    postOptimization();
  }

  // Protected RegAllocBase state, surfaced to the Python helpers.
  llvm::LiveRegMatrix *matrix() { return Matrix; }
  llvm::LiveIntervals *intervals() { return LIS; }
  llvm::VirtRegMap *virtRegMap() { return VRM; }
  llvm::MachineFunction *machineFunction() { return mf; }

  // Physregs, in target allocation order, that Python may try for `li`.
  std::vector<unsigned> allocationOrder(const llvm::LiveInterval &li) {
    auto order =
        llvm::AllocationOrder::create(li.reg(), *VRM, RegClassInfo, Matrix);
    std::vector<unsigned> ids;
    for (llvm::MCRegister r : order)
      ids.push_back(r.id());
    return ids;
  }

  // Spill `li` into the current select_or_split's split-vreg vector.
  void spill(const llvm::LiveInterval &li) {
    llvm::LiveRangeEdit lre(&li, *currentSplit, *mf, *LIS, VRM,
                            /*delegate=*/nullptr, &DeadRemats);
    injectedSpiller->spill(lre);
  }

  llvm::Spiller *injectedSpiller = nullptr;

protected:
  struct QueueEntry {
    float weight;
    llvm::Register reg;
  };
  // Highest spill weight first; break ties on the lower register id so the
  // default order is deterministic.
  struct QueueLess {
    bool operator()(const QueueEntry &a, const QueueEntry &b) const {
      if (a.weight != b.weight)
        return a.weight < b.weight;
      return a.reg.id() > b.reg.id();
    }
  };

  std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueLess>
      nativeQueue;
  llvm::SmallVectorImpl<llvm::Register> *currentSplit = nullptr;
  llvm::MachineFunction *mf = nullptr;
  llvm::SplitAnalysis *splitAnalysis = nullptr;
  llvm::SplitEditor *splitEditor = nullptr;
};

// Trampoline letting Python subclass the allocator. Each virtual calls the
// Python override under the GIL and stashes on raise; when no override exists
// or the run is already winding down after a stash, it defers to the
// NativeRegAlloc base (which always makes forward progress), so the pipeline
// reaches runCodegenPipeline's re-raise with valid MIR.
class PyRegAllocBase : public NativeRegAlloc {
public:
  NB_TRAMPOLINE(NativeRegAlloc, 6);

  // Keep the native fallback queue complete, then forward to an optional
  // Python enqueue.
  void enqueueImpl(const llvm::LiveInterval *li) override {
    NativeRegAlloc::enqueueImpl(li);
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "enqueue"))
      return;
    try {
      self.attr("enqueue")(nb::cast(const_cast<llvm::LiveInterval *>(li),
                                    nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  const llvm::LiveInterval *dequeue() override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (!eudsl::pendingCodegenError && nb::hasattr(self, "dequeue")) {
      try {
        nb::object choice = self.attr("dequeue")();
        if (choice.is_none())
          return nullptr;
        return nb::cast<llvm::LiveInterval *>(choice);
      } catch (...) {
        eudsl::pendingCodegenError = std::current_exception();
      }
    }
    return NativeRegAlloc::dequeue();
  }

  llvm::MCRegister
  selectOrSplit(const llvm::LiveInterval &vreg,
                llvm::SmallVectorImpl<llvm::Register> &splitLVRs) override {
    currentSplit = &splitLVRs;
    nb::gil_scoped_acquire gil;
    if (!eudsl::pendingCodegenError) {
      try {
        nb::object r = nb_trampoline.base().attr("select_or_split")(nb::cast(
            const_cast<llvm::LiveInterval *>(&vreg), nb::rv_policy::reference));
        currentSplit = nullptr;
        // None means Python handled it (spill/split appended new vregs); an int
        // is the chosen physreg for the driver to assign.
        if (r.is_none())
          return llvm::MCRegister();
        return llvm::MCRegister(nb::cast<unsigned>(r));
      } catch (...) {
        eudsl::pendingCodegenError = std::current_exception();
      }
    }
    llvm::MCRegister phys = NativeRegAlloc::selectOrSplit(vreg, splitLVRs);
    currentSplit = nullptr;
    return phys;
  }

  void postOptimization() override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (!eudsl::pendingCodegenError && nb::hasattr(self, "post_optimization")) {
      try {
        self.attr("post_optimization")();
        return;
      } catch (...) {
        eudsl::pendingCodegenError = std::current_exception();
      }
    }
    // No override, or winding down after a stash: run the base cleanup (spiller
    // post-optimize + dead-remat removal) so the emitted MIR stays valid.
    llvm::RegAllocBase::postOptimization();
  }

  // Called by allocatePhysRegs before it drops an interval the spiller left
  // unused; forwarded to an optional Python about_to_remove_interval.
  void aboutToRemoveInterval(const llvm::LiveInterval &li) override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError ||
        !nb::hasattr(self, "about_to_remove_interval"))
      return;
    try {
      self.attr("about_to_remove_interval")(nb::cast(
          const_cast<llvm::LiveInterval *>(&li), nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }
};

// name -> Python RegAllocBase subclass, held in
// llvm.mir_strategies._regalloc_classes (Python owns it so the subclass types
// are released at interpreter teardown, matching schedulerClasses()).
nb::dict regallocClasses() {
  return nb::cast<nb::dict>(
      nb::module_::import_("llvm.mir_strategies").attr("_regalloc_classes"));
}

// Registered allocator names, in registration order.
std::deque<std::string> &regallocNames() {
  static auto *names = new std::deque<std::string>();
  return *names;
}

// select_or_split is the one required override (RegAllocBase's only pure
// heuristic hook without a default); enqueue/dequeue/post_optimization are
// optional and fall back to the native queue / base default. type_object_t
// rejects a non-RegAllocBase class at the call boundary. Keyed on the
// trampoline (the bound value type) rather than llvm::RegAllocBase because that
// class has a protected destructor, which nanobind cannot bind as a value type.
void registerRegAlloc(const std::string &name,
                      nb::type_object_t<PyRegAllocBase> cls) {
  if (!nb::hasattr(cls, "select_or_split")) {
    throw nb::type_error(
        "register allocator class must define select_or_split");
  }
  nb::dict classes = regallocClasses();
  if (!classes.contains(name.c_str()))
    regallocNames().push_back(name);
  classes[name.c_str()] = cls;
}

// The fixed C++ MachineFunctionPass that hosts the Python allocator. It
// occupies the register-allocator slot in the codegen pipeline (selected via
// the -regalloc option), fetches the analyses a RegAllocBase driver needs,
// builds the spiller + SplitAnalysis/SplitEditor, constructs a fresh Python
// allocator per MachineFunction, and drives it through the public
// pyInit/pyAllocate wrappers. Analysis wiring mirrors RABasic (the minimal
// RegAllocBase user); the SplitAnalysis/SplitEditor it injects give Python the
// greedy-style splitting primitives without RAGreedy's extra pass dependencies.
class PyRegAllocDriver : public llvm::MachineFunctionPass {
public:
  static char ID;
  PyRegAllocDriver() : llvm::MachineFunctionPass(ID) {}

  ~PyRegAllocDriver() override {
    if (heldInstance.is_valid()) {
      nb::gil_scoped_acquire gil;
      heldInstance.reset();
    }
  }

  llvm::StringRef getPassName() const override {
    return "eudsl Python register allocator";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &au) const override {
    au.setPreservesCFG();
    au.addRequired<llvm::AAResultsWrapperPass>();
    au.addPreserved<llvm::AAResultsWrapperPass>();
    au.addRequired<llvm::LiveIntervalsWrapperPass>();
    au.addPreserved<llvm::LiveIntervalsWrapperPass>();
    au.addPreserved<llvm::SlotIndexesWrapperPass>();
    au.addRequired<llvm::LiveDebugVariablesWrapperLegacy>();
    au.addPreserved<llvm::LiveDebugVariablesWrapperLegacy>();
    au.addRequired<llvm::LiveStacksWrapperLegacy>();
    au.addPreserved<llvm::LiveStacksWrapperLegacy>();
    au.addRequired<llvm::ProfileSummaryInfoWrapperPass>();
    au.addRequired<llvm::MachineBlockFrequencyInfoWrapperPass>();
    au.addRequired<llvm::MachineDominatorTreeWrapperPass>();
    au.addRequiredID(llvm::MachineDominatorsID);
    au.addRequired<llvm::MachineLoopInfoWrapperPass>();
    au.addRequired<llvm::VirtRegMapWrapperLegacy>();
    au.addPreserved<llvm::VirtRegMapWrapperLegacy>();
    au.addRequired<llvm::LiveRegMatrixWrapperLegacy>();
    au.addPreserved<llvm::LiveRegMatrixWrapperLegacy>();
    llvm::MachineFunctionPass::getAnalysisUsage(au);
  }

  llvm::MachineFunctionProperties getRequiredProperties() const override {
    return llvm::MachineFunctionProperties().set(
        llvm::MachineFunctionProperties::Property::NoPHIs);
  }

  bool runOnMachineFunction(llvm::MachineFunction &mfn) override {
    llvm::VirtRegMap &vrm =
        getAnalysis<llvm::VirtRegMapWrapperLegacy>().getVRM();
    llvm::LiveIntervals &lis =
        getAnalysis<llvm::LiveIntervalsWrapperPass>().getLIS();
    llvm::LiveRegMatrix &mat =
        getAnalysis<llvm::LiveRegMatrixWrapperLegacy>().getLRM();
    llvm::MachineBlockFrequencyInfo &mbfi =
        getAnalysis<llvm::MachineBlockFrequencyInfoWrapperPass>().getMBFI();
    llvm::LiveStacks &livestks =
        getAnalysis<llvm::LiveStacksWrapperLegacy>().getLS();
    llvm::MachineDominatorTree &mdt =
        getAnalysis<llvm::MachineDominatorTreeWrapperPass>().getDomTree();
    llvm::MachineLoopInfo &loops =
        getAnalysis<llvm::MachineLoopInfoWrapperPass>().getLI();
    llvm::ProfileSummaryInfo &psi =
        getAnalysis<llvm::ProfileSummaryInfoWrapperPass>().getPSI();

    nb::gil_scoped_acquire gil;
    // LCOV_EXCL_START -- emit_object always sets the active class before
    // running
    if (!activeRegAllocClass.is_valid())
      return false;
    // LCOV_EXCL_STOP

    // Analyses shared by whichever allocator drives this function.
    llvm::VirtRegAuxInfo vrai(mfn, lis, vrm, loops, mbfi, &psi);
    vrai.calculateSpillWeightsAndHints();
    std::unique_ptr<llvm::Spiller> spiller(
        llvm::createInlineSpiller({lis, livestks, mdt, mbfi}, mfn, vrm, vrai));
    llvm::SplitAnalysis sa(vrm, lis, loops);
    llvm::SplitEditor se(sa, lis, vrm, mdt, mbfi, vrai);

    nb::object obj;
    try {
      obj = activeRegAllocClass();
    } catch (...) {
      // The subclass __init__ raised. Stash it and allocate natively so the MIR
      // is valid and the pipeline reaches runCodegenPipeline's re-raise instead
      // of aborting the rewriter on unallocated vregs.
      eudsl::pendingCodegenError = std::current_exception();
      NativeRegAlloc fallback;
      fallback.pyInit(vrm, lis, mat, *spiller, mfn, &sa, &se);
      fallback.pyAllocate();
      return true;
    }
    auto *base =
        static_cast<PyRegAllocBase *>(nb::inst_ptr<PyRegAllocBase>(obj));
    base->pyInit(vrm, lis, mat, *spiller, mfn, &sa, &se);
    base->pyAllocate();
    // Hold the instance until the pass is destroyed so its C++ subobject (and
    // any Python-recorded witness state) outlives this call; dropped under the
    // GIL in the dtor.
    heldInstance = std::move(obj);
    return true;
  }

private:
  nb::object heldInstance;
};

// The ctor every registered name shares; emit_object points the -regalloc
// option at it (as it does -misched for the scheduler) and sets
// activeRegAllocClass so this knows which subclass to instantiate.
llvm::FunctionPass *createRegisteredPyRegAlloc() {
  return new PyRegAllocDriver();
}

} // namespace

// Declared here (rather than InitializePasses.h) since this pass lives in-tree;
// the INITIALIZE_PASS block below defines it.
namespace llvm {
void initializePyRegAllocDriverPass(PassRegistry &);
} // namespace llvm

char PyRegAllocDriver::ID = 0;

// INITIALIZE_PASS expands to unqualified PassRegistry/PassInfo/callDefaultCtor
// (LLVM's macros assume an in-scope llvm namespace).
using namespace llvm;

INITIALIZE_PASS_BEGIN(PyRegAllocDriver, "eudsl-python-regalloc",
                      "eudsl Python register allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariablesWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(PyRegAllocDriver, "eudsl-python-regalloc",
                    "eudsl Python register allocator", false, false)

namespace eudsl {

// Validate that cls defines the required methods, record it, and (if new) add a
// MachineSchedRegistry node so the pipeline can select it by name. The
// type_object_t parameter makes nanobind reject a non-MachineSchedStrategy
// class at the call boundary. Re-registering a name swaps the class.
void registerScheduler(const std::string &name,
                       nb::type_object_t<llvm::MachineSchedStrategy> cls) {
  static const char *required[] = {"initialize",       "get_policy",
                                   "pick_node",        "sched_node",
                                   "release_top_node", "release_bottom_node"};
  for (const char *method : required) {
    if (!nb::hasattr(cls, method)) {
      throw nb::type_error(
          (std::string("scheduler class must define ") + method).c_str());
    }
  }
  nb::dict classes = schedulerClasses();
  if (!classes.contains(name.c_str())) {
    schedNames().push_back(name);
    const char *cname = schedNames().back().c_str();
    schedRegistryNodes().push_back(std::make_unique<llvm::MachineSchedRegistry>(
        cname, cname, createRegisteredPyStrategy));
  }
  classes[name.c_str()] = cls;
}

// The class registered under `name`, or an invalid object if `name` was not
// registered via register_scheduler.
nb::type_object schedulerClass(const std::string &name) {
  nb::dict classes = schedulerClasses();
  if (classes.contains(name.c_str()))
    return nb::borrow<nb::type_object>(classes[name.c_str()]);
  return nb::type_object();
}

// The ctor every register_scheduler name shares, so emit_object can point
// -misched at it without walking the registry.
llvm::MachineSchedRegistry::ScheduleDAGCtor registeredSchedCtor() {
  return createRegisteredPyStrategy;
}

void setActiveSchedClass(nb::type_object cls) {
  activeSchedClass = std::move(cls);
}
void clearActiveSchedClass() { activeSchedClass = nb::type_object(); }

// The class registered under `name`, or an invalid object if `name` was not
// registered via register_regalloc.
nb::type_object regallocClass(const std::string &name) {
  nb::dict classes = regallocClasses();
  if (classes.contains(name.c_str()))
    return nb::borrow<nb::type_object>(classes[name.c_str()]);
  return nb::type_object();
}

void setActiveRegAllocClass(nb::type_object cls) {
  activeRegAllocClass = std::move(cls);
}
void clearActiveRegAllocClass() { activeRegAllocClass = nb::type_object(); }

// The ctor emit_object points the -regalloc option at, mirroring
// registeredSchedCtor for -misched.
llvm::RegisterRegAlloc::FunctionPassCtor registeredRegAllocCtor() {
  return createRegisteredPyRegAlloc;
}

} // namespace eudsl

void populate_python_codegen(nb::module_ &m) {
  // Per-region scheduling policy a strategy returns from get_policy(). Field
  // names mirror MachineSchedPolicy's members.
  nb::class_<llvm::MachineSchedPolicy>(m, "MachineSchedPolicy")
      .def(nb::init<>())
      .def_rw("should_track_pressure",
              &llvm::MachineSchedPolicy::ShouldTrackPressure)
      .def_rw("should_track_lane_masks",
              &llvm::MachineSchedPolicy::ShouldTrackLaneMasks)
      .def_rw("only_top_down", &llvm::MachineSchedPolicy::OnlyTopDown)
      .def_rw("only_bottom_up", &llvm::MachineSchedPolicy::OnlyBottomUp);

  // One scheduling unit (a MachineInstr plus its dependency edges). A strategy
  // receives these via release_top_node/release_bottom_node and returns one
  // from pick_node.
  nb::class_<llvm::SUnit>(m, "SUnit")
      .def_prop_ro(
          "node_num", [](llvm::SUnit &su) { return su.NodeNum; },
          "Entry number of this node in the DAG's node vector.")
      .def_prop_ro(
          "is_top_ready", [](llvm::SUnit &su) { return su.isTopReady(); },
          "All predecessors scheduled (ready top-down).")
      .def_prop_ro(
          "is_bottom_ready", [](llvm::SUnit &su) { return su.isBottomReady(); },
          "All successors scheduled (ready bottom-up).")
      .def_prop_ro(
          "instr", [](llvm::SUnit &su) { return su.getInstr(); },
          nb::rv_policy::reference_internal,
          "The representative MachineInstr this unit wraps.");

  // The scheduling DAG passed to initialize(dag). Opaque for now -- a strategy
  // receives its nodes via release_top_node/release_bottom_node.
  nb::class_<llvm::ScheduleDAGMI>(m, "ScheduleDAGMI");

  nb::class_<llvm::MachineSchedStrategy, PySchedStrategy>(
      m, "MachineSchedStrategy")
      .def(nb::init<>());

  m.def("register_scheduler", &eudsl::registerScheduler, "name"_a, "cls"_a,
        "Register a MachineSchedStrategy subclass under `name` so "
        "emit_object(scheduler=name) can select it. The class must define "
        "initialize, get_policy, pick_node, sched_node, release_top_node, and "
        "release_bottom_node; re-registering a name replaces it.");

  m.def(
      "registered_schedulers",
      []() {
        return std::vector<std::string>(schedNames().begin(),
                                        schedNames().end());
      },
      "Names registered via register_scheduler, selectable with "
      "emit_object(scheduler=...).");

  // Register the harness pass (and its analysis dependencies) so the legacy
  // PassManager can resolve them when the pipeline runs the allocator slot.
  llvm::initializePyRegAllocDriverPass(*llvm::PassRegistry::getPassRegistry());

  nb::class_<PyRegAllocBase>(m, "RegAllocBase")
      .def(nb::init<>())
      .def("allocation_order", &PyRegAllocBase::allocationOrder, "li"_a,
           "Physregs (as ids) to try for `li`, in target allocation order.")
      .def("spill", &PyRegAllocBase::spill, "li"_a,
           "Spill `li`; new split vregs are appended for re-enqueue.")
      .def_prop_ro(
          "matrix", [](PyRegAllocBase &self) { return self.matrix(); },
          nb::rv_policy::reference_internal,
          "The LiveRegMatrix for interference queries and assignment.")
      .def_prop_ro(
          "lis", [](PyRegAllocBase &self) { return self.intervals(); },
          nb::rv_policy::reference_internal,
          "The LiveIntervals analysis for this function.")
      .def_prop_ro(
          "vrm", [](PyRegAllocBase &self) { return self.virtRegMap(); },
          nb::rv_policy::reference_internal,
          "The VirtRegMap being populated with assignments.")
      .def_prop_ro(
          "machine_function",
          [](PyRegAllocBase &self) { return self.machineFunction(); },
          nb::rv_policy::reference_internal,
          "The MachineFunction being allocated.");

  // A virtual register's live interval: the allocator receives one per
  // select_or_split call and queries/assigns it against the matrix.
  nb::class_<llvm::LiveInterval>(m, "LiveInterval")
      .def_prop_ro("reg",
                   [](const llvm::LiveInterval &li) { return li.reg().id(); })
      .def_prop_ro("weight",
                   [](const llvm::LiveInterval &li) { return li.weight(); })
      .def_prop_ro("is_spillable", [](const llvm::LiveInterval &li) {
        return li.isSpillable();
      });

  nb::class_<llvm::VirtRegMap>(m, "VirtRegMap");
  nb::class_<llvm::Spiller>(m, "Spiller");

  // A program point. Live ranges and the split editor are expressed in terms of
  // these; they order the instructions of a function.
  nb::class_<llvm::SlotIndex>(m, "SlotIndex")
      .def("is_valid", &llvm::SlotIndex::isValid)
      .def(
          "__lt__",
          [](const llvm::SlotIndex &a, const llvm::SlotIndex &b) {
            return a < b;
          },
          "other"_a)
      .def(
          "__eq__",
          [](const llvm::SlotIndex &a, const llvm::SlotIndex &b) {
            return a == b;
          },
          "other"_a)
      .def("get_reg_slot",
           [](const llvm::SlotIndex &i) { return i.getRegSlot(); })
      .def("get_base_index",
           [](const llvm::SlotIndex &i) { return i.getBaseIndex(); })
      .def("get_boundary_index",
           [](const llvm::SlotIndex &i) { return i.getBoundaryIndex(); })
      .def("get_next_index",
           [](const llvm::SlotIndex &i) { return i.getNextIndex(); })
      .def("__repr__", [](const llvm::SlotIndex &i) {
        return i.isValid() ? std::string("SlotIndex(valid)")
                           : std::string("SlotIndex(invalid)");
      });

  nb::class_<llvm::LiveIntervals>(m, "LiveIntervals")
      .def(
          "instruction_index",
          [](llvm::LiveIntervals &l, llvm::MachineInstr *mi) {
            return l.getInstructionIndex(*mi);
          },
          "mi"_a)
      .def(
          "mbb_start_index",
          [](llvm::LiveIntervals &l, llvm::MachineBasicBlock *mbb) {
            return l.getMBBStartIdx(mbb);
          },
          "mbb"_a)
      .def(
          "mbb_end_index",
          [](llvm::LiveIntervals &l, llvm::MachineBasicBlock *mbb) {
            return l.getMBBEndIdx(mbb);
          },
          "mbb"_a)
      .def(
          "has_interval",
          [](llvm::LiveIntervals &l, unsigned reg) {
            return l.hasInterval(llvm::Register(reg));
          },
          "reg"_a)
      .def(
          "interval",
          [](llvm::LiveIntervals &l, unsigned reg) -> llvm::LiveInterval & {
            return l.getInterval(llvm::Register(reg));
          },
          nb::rv_policy::reference_internal, "reg"_a);

  nb::enum_<llvm::LiveRegMatrix::InterferenceKind>(m, "InterferenceKind")
      .value("IK_Free", llvm::LiveRegMatrix::IK_Free)
      .value("IK_VirtReg", llvm::LiveRegMatrix::IK_VirtReg)
      .value("IK_RegUnit", llvm::LiveRegMatrix::IK_RegUnit)
      .value("IK_RegMask", llvm::LiveRegMatrix::IK_RegMask);

  nb::class_<llvm::LiveRegMatrix>(m, "LiveRegMatrix")
      .def(
          "check_interference",
          [](llvm::LiveRegMatrix &mat, const llvm::LiveInterval &li,
             unsigned preg) {
            return mat.checkInterference(li, llvm::MCRegister(preg));
          },
          "li"_a, "physreg"_a)
      .def(
          "is_free",
          [](llvm::LiveRegMatrix &mat, const llvm::LiveInterval &li,
             unsigned preg) {
            return mat.checkInterference(li, llvm::MCRegister(preg)) ==
                   llvm::LiveRegMatrix::IK_Free;
          },
          "li"_a, "physreg"_a)
      .def(
          "assign",
          [](llvm::LiveRegMatrix &mat, const llvm::LiveInterval &li,
             unsigned preg) { mat.assign(li, llvm::MCRegister(preg)); },
          "li"_a, "physreg"_a)
      .def(
          "unassign",
          [](llvm::LiveRegMatrix &mat, const llvm::LiveInterval &li) {
            mat.unassign(li);
          },
          "li"_a);

  m.def("register_regalloc", &registerRegAlloc, "name"_a, "cls"_a,
        "Register a RegAllocBase subclass under `name` so "
        "emit_object(regalloc=name) can select it. The class must define "
        "select_or_split; enqueue, dequeue, and post_optimization are "
        "optional. Re-registering a name replaces it.");

  m.def(
      "registered_regallocs",
      []() {
        return std::vector<std::string>(regallocNames().begin(),
                                        regallocNames().end());
      },
      "Names registered via register_regalloc, selectable with "
      "emit_object(regalloc=...).");
}
