// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "MIR/AllocationOrder.h"
#include "MIR/Diagnostics.h"
#include "MIR/InterferenceCache.h"
#include "MIR/RegAllocBase.h"
#include "MIR/SpillPlacement.h"
#include "MIR/SplitKit.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/CodeGen/CalcSpillWeights.h>
#include <llvm/CodeGen/EdgeBundles.h>
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
#include <llvm/CodeGen/MachineInstrBundle.h>
#include <llvm/CodeGen/MachineLoopInfo.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/CodeGen/ScheduleDAG.h>
#include <llvm/CodeGen/ScheduleDAGMutation.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/Spiller.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/VirtRegMap.h>
#include <llvm/MC/LaneBitmask.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/CommandLine.h>

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <algorithm>
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

// llvm::LiveIntervals::computeVirtRegInterval is private, but computing the
// interval of a register whose def we just rematerialized (and whose empty
// interval LiveRangeEdit::create already made) is exactly what the remat flow
// needs. Reach it with the explicit-instantiation access trick: access control
// is not applied to a pointer-to-member named as a template argument of an
// explicit instantiation (per [temp.explicit], the "access checks that apply to
// template-arguments" carve-out), so instantiating this emits a namespace-scope
// friend that forwards to the private member.
template <auto Member>
struct AccessComputeVirtRegInterval {
  friend bool eudslComputeVirtRegInterval(llvm::LiveIntervals &lis,
                                          llvm::LiveInterval &li) {
    return (lis.*Member)(li);
  }
};
template struct AccessComputeVirtRegInterval<
    &llvm::LiveIntervals::computeVirtRegInterval>;
bool eudslComputeVirtRegInterval(llvm::LiveIntervals &lis,
                                 llvm::LiveInterval &li);

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

  // This LLVM's MachineSchedStrategy has no getPolicy() virtual (the scheduler
  // reads shouldTrackPressure()/shouldTrackLaneMasks() directly), so this is a
  // plain helper the predicates below delegate to, not an override.
  llvm::MachineSchedPolicy getPolicy() const {
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
    return llvm::createGenericSchedLive(c);
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
    return llvm::createGenericSchedLive(c);
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
              llvm::MachineFunction &mfn) {
    injectedSpiller = &sp;
    mf = &mfn;
    init(vrm, lis, mat);
  }
  void pyAllocate() {
    allocatePhysRegs();
    postOptimization();
  }

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
  llvm::MachineFunction *mf = nullptr;
  llvm::Spiller *injectedSpiller = nullptr;
};

// Trampoline letting Python subclass the allocator. Each virtual calls the
// Python override under the GIL and stashes on raise; when no override exists
// or the run is already winding down after a stash, it defers to the
// NativeRegAlloc base (which always makes forward progress), so the pipeline
// reaches runCodegenPipeline's re-raise with valid MIR. It also carries the
// Python-only splitting surface (split analysis/editor, the current split-vreg
// vector, and the edit buffer) that the standalone NativeRegAlloc fallback has
// no use for.
// RAGreedy resolves reverseLocalAssignment / regClassPriorityTrumpsGlobalness
// as `flag.getNumOccurrences() ? flag : TRI->hook()`, where `flag` is one of
// its file-static hidden cl::opts. Those statics aren't visible here, but they
// are the same options registered in the global cl registry, so look them up by
// name to honor a `-greedy-*` override exactly as the allocator would (in an
// embedding with no command-line parsing the count is 0, so this returns the
// target hook).
bool greedyEffectiveFlag(llvm::StringRef flag, bool targetDefault) {
  auto &opts = llvm::cl::getRegisteredOptions();
  auto it = opts.find(flag);
  if (it == opts.end() || !it->second->getNumOccurrences())
    return targetDefault;
  // Only reached when a -greedy-* override was set on LLVM's command line; this
  // embedding never parses those flags, so it is unreachable from tests.
  return *static_cast<llvm::cl::opt<bool> *>(it->second); // LCOV_EXCL_LINE
}

class PyRegAllocBase : public NativeRegAlloc {
public:
  NB_TRAMPOLINE(NativeRegAlloc, 6);

  // Keep the native fallback queue complete, then forward the register id (not
  // the interval, which splitting can invalidate) to an optional Python
  // enqueue.
  void enqueueImpl(const llvm::LiveInterval *li) override {
    NativeRegAlloc::enqueueImpl(li);
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "enqueue"))
      return;
    try {
      self.attr("enqueue")(li->reg().id());
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  const llvm::LiveInterval *dequeue() override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (!eudsl::pendingCodegenError && nb::hasattr(self, "dequeue")) {
      try {
        // Python returns register ids (stable across splitting); re-fetch and
        // skip any already assigned or removed, mirroring the native drain.
        while (true) {
          nb::object choice = self.attr("dequeue")();
          if (choice.is_none())
            return nullptr;
          llvm::Register r(nb::cast<unsigned>(choice));
          if (VRM->hasPhys(r) || !LIS->hasInterval(r))
            continue;
          return &LIS->getInterval(r);
        }
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
        // None means Python handled it via spill/split (new vregs appended); an
        // int is a physreg to assign, validated as a free candidate rather than
        // letting Matrix::assign abort on a bad choice.
        llvm::MCRegister phys;
        if (!r.is_none())
          phys = validatedPhysReg(vreg, nb::cast<unsigned>(r));
        clearSplitContext();
        return phys;
      } catch (...) {
        eudsl::pendingCodegenError = std::current_exception();
      }
    }
    llvm::MCRegister phys = NativeRegAlloc::selectOrSplit(vreg, splitLVRs);
    clearSplitContext();
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

  void pyInit(llvm::VirtRegMap &vrm, llvm::LiveIntervals &lis,
              llvm::LiveRegMatrix &mat, llvm::Spiller &sp,
              llvm::MachineFunction &mfn, llvm::SplitAnalysis *sa,
              llvm::SplitEditor *se, llvm::MachineBlockFrequencyInfo *mbfi,
              llvm::EdgeBundles *eb, llvm::SpillPlacement *spl,
              llvm::VirtRegAuxInfo *vrai, llvm::MachineLoopInfo *ml) {
    splitAnalysis = sa;
    splitEditor = se;
    blockFreqInfo = mbfi;
    edgeBundles = eb;
    spillPlacer = spl;
    auxInfo = vrai;
    loops = ml;
    regCosts = mfn.getSubtarget().getRegisterInfo()->getRegisterCosts(mfn);
    NativeRegAlloc::pyInit(vrm, lis, mat, sp, mfn);
    // Matches RAGreedy::IntfCache.init: MF, the matrix's per-regunit unions,
    // slot indexes, LIS, TRI.
    intfCache.init(mf, Matrix->getLiveUnions(), LIS->getSlotIndexes(), LIS,
                   mf->getSubtarget().getRegisterInfo());
  }

  // Protected driver state, surfaced to the Python helpers. These borrow
  // objects owned by the harness pass frame; the bindings hand them out with
  // rv_policy::reference (not reference_internal) since the allocator does not
  // own them, and they are valid only for the duration of an allocator
  // callback.
  llvm::LiveRegMatrix *matrix() { return Matrix; }
  llvm::LiveIntervals *intervals() { return LIS; }
  llvm::VirtRegMap *virtRegMap() { return VRM; }
  llvm::MachineFunction *machineFunction() { return mf; }
  llvm::SplitAnalysis *splitAnalysisPtr() { return splitAnalysis; }
  llvm::SplitEditor *splitEditorPtr() { return splitEditor; }
  llvm::MachineBlockFrequencyInfo *blockFrequencyInfo() {
    return blockFreqInfo;
  }
  llvm::EdgeBundles *edgeBundlesPtr() { return edgeBundles; }
  llvm::SpillPlacement *spillPlacerPtr() { return spillPlacer; }

  // A cursor into the interference cache, for region-split cost queries. Point
  // it at a physreg with set_interference_physreg, then move_to_block/first/
  // last/has_interference per block. Copyable and refcounts its cache entry;
  // valid only within an allocator callback, do not retain past it.
  llvm::InterferenceCache::Cursor newInterferenceCursor() {
    return llvm::InterferenceCache::Cursor();
  }

  // setPhysReg needs the driver's owned cache, which the free Cursor can't
  // reach; do it through the driver.
  void cursorSetPhysReg(llvm::InterferenceCache::Cursor &cur,
                        unsigned physreg) {
    cur.setPhysReg(intfCache, llvm::MCRegister(physreg));
  }

  // Header block number of the innermost loop containing `mbbNumber`, or
  // std::nullopt if none. Reproduces growRegion's looksLikeLoopIV header check
  // without exposing the loop tree.
  std::optional<int> loopHeaderNumber(unsigned mbbNumber) {
    llvm::MachineBasicBlock *mbb = mf->getBlockNumbered(mbbNumber);
    llvm::MachineLoop *loop = loops->getLoopFor(mbb);
    if (!loop)
      return std::nullopt;
    return loop->getHeader()->getNumber();
  }

  // Slot index of the first non-debug instruction in block `n`, or std::nullopt
  // if the block is empty (addThroughConstraints' abort guard).
  std::optional<llvm::SlotIndex> firstNonDebugInstrIndex(unsigned n) {
    llvm::MachineBasicBlock *mbb = mf->getBlockNumbered(n);
    auto it = mbb->getFirstNonDebugInstr();
    // An empty (all-debug) block cannot be produced by the hand-built test MIR
    // -- the pre-regalloc pipeline requires a terminator -- so this defensive
    // guard is unreachable from tests.
    if (it == mbb->end())
      return std::nullopt; // LCOV_EXCL_LINE
    return LIS->getInstructionIndex(*it);
  }

  // The live-in insertion-point index for block `n` (SkipPHIsLabelsAndDebug for
  // the analyzed interval's reg), matching addThroughConstraints' InsertIdx.
  llvm::SlotIndex throughInsertIndex(unsigned n) {
    llvm::MachineBasicBlock *mbb = mf->getBlockNumbered(n);
    llvm::Register reg = splitAnalysis->getParent().reg();
    auto insertPt = mbb->SkipPHIsLabelsAndDebug(mbb->begin(), reg);
    return insertPt == mbb->end() ? LIS->getMBBEndIdx(mbb)
                                  : LIS->getInstructionIndex(*insertPt);
  }

  llvm::SlotIndex mbbStartIndexByNumber(unsigned n) {
    return LIS->getMBBStartIdx(mf->getBlockNumbered(n));
  }

  // Physregs, in target allocation order, that Python may try for `li`.
  std::vector<unsigned> allocationOrder(const llvm::LiveInterval &li) {
    auto order =
        llvm::AllocationOrder::create(li.reg(), *VRM, RegClassInfo, Matrix);
    std::vector<unsigned> ids;
    for (llvm::MCRegister r : order)
      ids.push_back(r.id());
    return ids;
  }

  // Virtual registers whose live ranges interfere with `li` on `physreg` --
  // those assigned to `physreg` itself or to a physreg that aliases it. The
  // enumeration eviction needs. Querying every reg unit of `physreg` (not a
  // Python-side "physreg -> vreg" shadow) is what makes it alias/subregister-
  // correct, which is what RAGreedy relies on; the same vreg surfacing on
  // several of `physreg`'s units is de-duplicated.
  std::vector<unsigned> interferingVRegs(const llvm::LiveInterval &li,
                                         unsigned physreg) {
    const llvm::TargetRegisterInfo *tri = mf->getSubtarget().getRegisterInfo();
    std::vector<unsigned> ids;
    // This LLVM has no regunits() range; iterate the units with
    // MCRegUnitIterator (a unit is a plain unsigned here).
    for (llvm::MCRegUnitIterator units(llvm::MCRegister(physreg), tri);
         units.isValid(); ++units) {
      for (const llvm::LiveInterval *intf :
           Matrix->query(li, *units).interferingVRegs()) {
        unsigned id = intf->reg().id();
        if (std::find(ids.begin(), ids.end(), id) == ids.end())
          ids.push_back(id);
      }
    }
    return ids;
  }

  // Fixed (physical) reg-unit interference for `physreg` overlapping `li`: the
  // segments of LIS->getRegUnit(unit) for each of `physreg`'s reg units that
  // overlap li's range. calcGapWeights marks gaps covered by these huge_valf --
  // a physreg clobbered mid-interval can't hold the value across the clobber.
  std::vector<llvm::LiveRange::Segment>
  fixedInterferenceSpans(const llvm::LiveInterval &li, unsigned physreg) {
    const llvm::TargetRegisterInfo *tri = mf->getSubtarget().getRegisterInfo();
    llvm::SlotIndex start = li.beginIndex(), stop = li.endIndex();
    std::vector<llvm::LiveRange::Segment> segs;
    for (llvm::MCRegUnitIterator units(llvm::MCRegister(physreg), tri);
         units.isValid(); ++units) {
      const llvm::LiveRange &lr = LIS->getRegUnit(*units);
      for (const llvm::LiveRange::Segment &s : lr) {
        if (s.start < stop && start < s.end) // overlaps li
          segs.push_back(s);
      }
    }
    return segs;
  }

  // Whether `li` is live across any register-mask operand (a call clobber).
  bool checkRegMaskInterferenceLI(const llvm::LiveInterval &li) {
    return Matrix->checkRegMaskInterference(
        const_cast<llvm::LiveInterval &>(li));
  }

  // Whether `physreg` is clobbered by a register mask that `li` crosses.
  bool checkRegMaskInterferencePhys(const llvm::LiveInterval &li,
                                    unsigned physreg) {
    return Matrix->checkRegMaskInterference(
        const_cast<llvm::LiveInterval &>(li), llvm::MCRegister(physreg));
  }

  // The register-mask slot indexes in block `mbbNumber` (tryLocalSplit finds
  // the gaps overlapping these to mark call-clobbered).
  std::vector<llvm::SlotIndex> regMaskSlotsInBlock(unsigned mbbNumber) {
    llvm::ArrayRef<llvm::SlotIndex> rms =
        LIS->getRegMaskSlotsInBlock(mbbNumber);
    return std::vector<llvm::SlotIndex>(rms.begin(), rms.end());
  }

  // Per-use cost of `physreg` (the CostPerUseLimit heuristic RAGreedy uses to
  // decide whether a register is worth allocating). Indexed by physreg id. The
  // cost table is an ArrayRef into the target's static tables (fixed for this
  // function), cached in pyInit.
  unsigned registerCost(unsigned physreg) {
    if (physreg >= regCosts.size())
      throw nb::index_error("physreg id out of range");
    return regCosts[physreg];
  }

  const llvm::TargetRegisterClass *regClass(unsigned reg) {
    return mf->getRegInfo().getRegClass(llvm::Register(reg));
  }

  unsigned numAllocatableRegs(const llvm::TargetRegisterClass *rc) {
    return RegClassInfo.getNumAllocatableRegs(rc);
  }

  // LLVM's fixed number of slot-index positions per instruction (SlotIndex::
  // InstrDist) -- RAGreedy converts range size to instruction count with it.
  unsigned slotIndexInstrDistance() { return llvm::SlotIndex::InstrDist; }

  // Whether the target assigns local ranges in reverse instruction order
  // (RAGreedy::enqueue orders local ranges by this), honoring the
  // -greedy-reverse-local-assignment override the allocator applies.
  bool reverseLocalAssignment() {
    return greedyEffectiveFlag(
        "greedy-reverse-local-assignment",
        mf->getSubtarget().getRegisterInfo()->reverseLocalAssignment());
  }

  // Whether the register class's AllocationPriority outranks globalness in the
  // priority calculation (RAGreedy's RegClassPriorityTrumpsGlobalness),
  // honoring the -greedy-regclass-priority-trumps-globalness override. This
  // LLVM's TargetRegisterInfo has no regClassPriorityTrumpsGlobalness hook (the
  // feature postdates it), so the target default is false.
  bool regClassPriorityTrumpsGlobalness() {
    return greedyEffectiveFlag("greedy-regclass-priority-trumps-globalness",
                               false);
  }

  // This LLVM's TargetRegisterClass has no GlobalPriority field (the feature
  // postdates it), so no class carries global priority.
  bool regClassHasGlobalPriority(const llvm::TargetRegisterClass *) {
    return false;
  }

  // `rc`'s target-assigned allocation priority (RC.AllocationPriority), one of
  // the fields getPriority packs into the enqueue key.
  unsigned regClassAllocationPriority(const llvm::TargetRegisterClass *rc) {
    return rc->AllocationPriority;
  }

  // Whether `reg` has a known physreg preference (a copy hint the framework
  // already resolved) -- getPriority boosts these.
  bool hasKnownPreference(unsigned reg) {
    return VRM->hasKnownPreference(llvm::Register(reg));
  }

  // Whether `reg` is currently assigned to its preferred physreg (a satisfied,
  // unbroken copy hint): canEvictInterferenceBasedOnCost charges BrokenHints
  // when evicting such a range would break that hint.
  bool hasPreferredPhys(unsigned reg) {
    return VRM->hasPreferredPhys(llvm::Register(reg));
  }

  // TargetRegisterClass::getCopyCost -- the per-broken-hint weight the eviction
  // cost model adds to BrokenHints.
  int regClassCopyCost(const llvm::TargetRegisterClass *rc) {
    return rc->getCopyCost();
  }

  // The last / zero slot indexes of the function, for the instruction-order
  // priority of local ranges (getApproxInstrDistance endpoints).
  llvm::SlotIndex lastSlotIndex() {
    return LIS->getSlotIndexes()->getLastIndex();
  }
  llvm::SlotIndex zeroSlotIndex() {
    return LIS->getSlotIndexes()->getZeroIndex();
  }

  bool regClassIsAllocatable(const llvm::TargetRegisterClass *rc) {
    return rc->isAllocatable();
  }

  // Whether `reg`'s whole live interval is contained in a single MBB. RAGreedy
  // routes single-block ranges to local splitting and multi-block ranges to
  // global (region/block) splitting.
  bool intervalIsInOneMBB(unsigned reg) {
    return LIS->intervalIsInOneMBB(LIS->getInterval(llvm::Register(reg)));
  }

  // Whether `reg`'s class is a proper subclass of its allocation superclass
  // (RAGreedy's SingleInstrs input to shouldSplitSingleBlock: a constrained
  // subclass makes even a single-instruction isolation worthwhile).
  bool isProperSubClass(unsigned reg) {
    return RegClassInfo.isProperSubClass(
        mf->getRegInfo().getRegClass(llvm::Register(reg)));
  }

  // Whether the instruction defining/using `li` at `idx` is copy-like
  // (a plain COPY or SUBREG_TO_REG). shouldSplitSingleBlock refuses to isolate
  // a lone copy since it carries no register-class constraint.
  bool isCopyLikeAt(llvm::SlotIndex idx) {
    llvm::MachineInstr *mi = LIS->getInstructionFromIndex(idx);
    if (!mi)
      return false; // LCOV_EXCL_LINE -- a use slot always has an instruction
    const llvm::TargetInstrInfo *tii = mf->getSubtarget().getInstrInfo();
    return tii->isCopyInstr(*mi).has_value() || mi->isSubregToReg();
  }

  // MachineInstr::isCopyLike() for the instruction at `idx` -- exactly the
  // predicate SplitAnalysis::shouldSplitSingleBlock uses (generic COPY or
  // SUBREG_TO_REG only). Unlike isCopyLikeAt, this does NOT match target-
  // specific copies (TII::isCopyInstr), so it is the faithful test there.
  bool isCopyLikeInstrAt(llvm::SlotIndex idx) {
    llvm::MachineInstr *mi = LIS->getInstructionFromIndex(idx);
    if (!mi)
      return false; // LCOV_EXCL_LINE -- a use slot always has an instruction
    return mi->isCopyLike();
  }

  // TargetRegisterInfo::shouldRegionSplitForVirtReg -- a target hook (default
  // true) that tryRegionSplit consults before attempting a region split.
  bool shouldRegionSplitForVirtReg(unsigned reg) {
    const llvm::TargetRegisterInfo *tri = mf->getSubtarget().getRegisterInfo();
    return tri->shouldRegionSplitForVirtReg(
        *mf, LIS->getInterval(llvm::Register(reg)));
  }

  // Whether the instruction at `idx` is a full (non-subreg) copy --
  // tryInstructionSplit skips such uses. This LLVM has no
  // TII::isFullCopyInstr; MachineInstr::isFullCopy is the equivalent (a plain
  // COPY with no sub-register indices).
  bool isFullCopyInstrAt(llvm::SlotIndex idx) {
    llvm::MachineInstr *mi = LIS->getInstructionFromIndex(idx);
    return mi && mi->isFullCopy();
  }

  // RAGreedy::readsLaneSubset: whether the instruction defining/using `li` at
  // `idx` reads only a subset of the lanes live there (so splitting around it
  // can move the rest to a wider class). A verbatim port of the file-static
  // helper (with its getInstReadLaneMask), for tryInstructionSplit's subrange
  // arm on sub-register-liveness targets.
  bool readsLaneSubset(const llvm::LiveInterval &li, llvm::SlotIndex idx) {
    llvm::MachineInstr *mi = LIS->getInstructionFromIndex(idx);
    const llvm::TargetInstrInfo *tii = mf->getSubtarget().getInstrInfo();
    const llvm::TargetRegisterInfo *tri = mf->getSubtarget().getRegisterInfo();
    llvm::MachineRegisterInfo &mri = mf->getRegInfo();
    // Common case: a copy whose source and destination sub-registers match
    // reads the whole value.
    auto destSrc = tii->isCopyInstr(*mi);
    if (destSrc && !mi->isBundled() &&
        destSrc->Destination->getSubReg() == destSrc->Source->getSubReg())
      return false;
    // getInstReadLaneMask: the lanes this instruction reads of li's reg.
    llvm::Register reg = li.reg();
    llvm::LaneBitmask readMask;
    llvm::SmallVector<std::pair<llvm::MachineInstr *, unsigned>, 8> ops;
    (void)llvm::AnalyzeVirtRegInBundle(*mi, reg, &ops);
    for (auto [opMI, opIdx] : ops) {
      const llvm::MachineOperand &mo = opMI->getOperand(opIdx);
      unsigned subReg = mo.getSubReg();
      if (subReg == 0 && mo.isUse()) {
        if (mo.isUndef())
          continue;
        readMask = mri.getMaxLaneMaskForVReg(reg);
        break;
      }
      llvm::LaneBitmask subRegMask = tri->getSubRegIndexLaneMask(subReg);
      if (mo.isDef()) {
        if (!mo.isUndef())
          readMask |= ~subRegMask;
      } else {
        readMask |= subRegMask;
      }
    }
    llvm::LaneBitmask liveAtMask;
    for (const llvm::LiveInterval::SubRange &s : li.subranges()) {
      if (s.liveAt(idx))
        liveAtMask |= s.LaneMask;
    }
    return (readMask & ~(liveAtMask & tri->getCoveringLanes())).any();
  }

  bool isTriviallyRematerializable(llvm::MachineInstr *mi) {
    return mf->getSubtarget().getInstrInfo()->isTriviallyReMaterializable(*mi);
  }

  // Copy-hint registers for `reg` (physregs and/or vregs it is copy-related
  // The register-allocation hints for virtual register `reg`: a (type, [ids])
  // pair. `type` is the hint kind (0 = target-independent copy hints; nonzero
  // = a target-specific hint the target expands), `ids` the hinted physregs.
  // RAGreedy prefers a hinted physreg and counts "broken hints" in eviction
  // cost. `reg` must be a virtual register. (0, []) if it has no hints.
  std::pair<unsigned, std::vector<unsigned>> regAllocationHints(unsigned reg) {
    unsigned type = 0;
    std::vector<unsigned> ids;
    // This LLVM returns a reference to the (kind, regs) pair (empty if the vreg
    // has no hints), not a nullable pointer.
    const auto &hints =
        mf->getRegInfo().getRegAllocationHints(llvm::Register(reg));
    type = hints.first;
    for (llvm::Register h : hints.second)
      ids.push_back(h.id());
    return {type, ids};
  }

  // The single "simple" copy hint for virtual register `reg` (id, or 0 if
  // none). `reg` must be a virtual register.
  unsigned simpleHint(unsigned reg) {
    return mf->getRegInfo().getSimpleHint(llvm::Register(reg)).id();
  }

  // The last callee-saved register aliasing `physreg` (id, or 0 if none) --
  // RAGreedy's isUnusedCalleeSavedReg check, which biases against introducing a
  // CSR spill.
  unsigned lastCalleeSavedAlias(unsigned physreg) {
    return RegClassInfo.getLastCalleeSavedAlias(llvm::MCRegister(physreg)).id();
  }

  // Recompute the spill weight (and hint) of `reg` from its current defs/uses.
  // RAGreedy does this for the vregs produced by a split so their enqueue
  // priority reflects the new, shorter ranges.
  void calculateSpillWeightAndHint(unsigned reg) {
    auxInfo->calculateSpillWeightAndHint(LIS->getInterval(llvm::Register(reg)));
  }

  // Spill `li` into the current select_or_split's split-vreg vector. Returns
  // the ids of the new vregs the spiller produced (reloads/remats), which the
  // framework re-enqueues; RAGreedy marks them RS_Done so they are never split
  // or spilled again.
  std::vector<unsigned> spill(const llvm::LiveInterval &li) {
    if (!currentSplit)
      throw nb::value_error("spill() is only valid inside select_or_split");
    // A LiveRangeEdit registers itself as the MachineRegisterInfo delegate for
    // its lifetime. This LLVM's MRI holds a single delegate (newer LLVM holds a
    // set), so an earlier split attempt's heldEdit must be released before this
    // spill's edit registers, or setDelegate asserts. The split it belonged to
    // has finished by the time a range falls through to spilling.
    heldEdit.reset();
    size_t before = currentSplit->size();
    llvm::LiveRangeEdit lre(&li, *currentSplit, *mf, *LIS, VRM,
                            /*delegate=*/nullptr, &DeadRemats);
    injectedSpiller->spill(lre);
    std::vector<unsigned> produced;
    for (size_t i = before; i < currentSplit->size(); ++i)
      produced.push_back((*currentSplit)[i].id());
    return produced;
  }

  // A LiveRangeEdit over the current split-vreg vector for the split editor to
  // write into. Held so it outlives the SplitEditor reset/open/use/finish calls
  // that reference it; cleared at the end of the select_or_split so it never
  // outlives the loop-local vector it references nor lingers as an MRI delegate
  // into later iterations.
  llvm::LiveRangeEdit *newLiveRangeEdit(const llvm::LiveInterval &li) {
    if (!currentSplit)
      throw nb::value_error(
          "new_live_range_edit() is only valid inside select_or_split");
    // Release any previous edit first: a LiveRangeEdit registers as the MRI
    // delegate in its ctor, and this LLVM's MRI holds a single delegate, so
    // constructing the new one while the old is still alive would assert. (The
    // make_unique below would otherwise build the new edit before destroying
    // the old.)
    heldEdit.reset();
    heldEdit = std::make_unique<llvm::LiveRangeEdit>(
        &li, *currentSplit, *mf, *LIS, VRM, /*delegate=*/nullptr, &DeadRemats);
    return heldEdit.get();
  }

private:
  // A returned physreg must be a free candidate in `vreg`'s allocation order;
  // otherwise the driver's Matrix::assign would abort. Checking membership
  // first keeps the checkInterference call on a valid physreg.
  llvm::MCRegister validatedPhysReg(const llvm::LiveInterval &vreg,
                                    unsigned id) {
    llvm::MCRegister phys(id);
    auto order =
        llvm::AllocationOrder::create(vreg.reg(), *VRM, RegClassInfo, Matrix);
    for (llvm::MCRegister cand : order) {
      if (cand == phys) {
        if (Matrix->checkInterference(vreg, phys) ==
            llvm::LiveRegMatrix::IK_Free)
          return phys;
        break;
      }
    }
    throw nb::value_error("select_or_split returned a physreg that is not a "
                          "free candidate in the allocation order");
  }

  // End-of-select_or_split cleanup: the split-vreg vector and edit buffer both
  // reference allocatePhysRegs's per-iteration loop-local storage, so neither
  // may outlive this call.
  void clearSplitContext() {
    currentSplit = nullptr;
    heldEdit.reset();
  }

  llvm::SmallVectorImpl<llvm::Register> *currentSplit = nullptr;
  llvm::SplitAnalysis *splitAnalysis = nullptr;
  llvm::SplitEditor *splitEditor = nullptr;
  llvm::MachineBlockFrequencyInfo *blockFreqInfo = nullptr;
  llvm::EdgeBundles *edgeBundles = nullptr;
  llvm::SpillPlacement *spillPlacer = nullptr;
  llvm::VirtRegAuxInfo *auxInfo = nullptr;
  llvm::MachineLoopInfo *loops = nullptr;
  llvm::ArrayRef<uint8_t> regCosts;
  std::unique_ptr<llvm::LiveRangeEdit> heldEdit;
  // Per-block interference cache for region-split cost queries (RAGreedy's
  // IntfCache). Owns cursors handed to Python; init'd in pyInit once the
  // matrix/LIS are wired.
  llvm::InterferenceCache intfCache;
};

// name -> Python RegAllocBase subclass, held in
// llvm.mir_strategies._regalloc_classes (Python owns it so the subclass types
// are released at interpreter teardown, matching schedulerClasses()).
nb::dict regallocClasses() {
  return nb::cast<nb::dict>(
      nb::module_::import_("llvm.mir_strategies").attr("_regalloc_classes"));
}

std::deque<std::string> &regallocNames() {
  static std::deque<std::string> names;
  return names;
}

// select_or_split is the one required override: the trampoline calls it
// unconditionally, whereas enqueue/dequeue/post_optimization/
// about_to_remove_interval are hasattr-guarded and fall back to the native
// queue / base default. type_object_t rejects a non-RegAllocBase class at the
// call boundary. Keyed on the trampoline (the bound value type) rather than
// llvm::RegAllocBase because that class has a protected destructor, which
// nanobind cannot bind as a value type.
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
    au.addRequired<llvm::LiveIntervals>();
    au.addPreserved<llvm::LiveIntervals>();
    au.addPreserved<llvm::SlotIndexes>();
    au.addRequired<llvm::LiveDebugVariables>();
    au.addPreserved<llvm::LiveDebugVariables>();
    au.addRequired<llvm::LiveStacks>();
    au.addPreserved<llvm::LiveStacks>();
    au.addRequired<llvm::ProfileSummaryInfoWrapperPass>();
    au.addRequired<llvm::MachineBlockFrequencyInfo>();
    au.addRequired<llvm::MachineDominatorTree>();
    au.addRequiredID(llvm::MachineDominatorsID);
    au.addRequired<llvm::MachineLoopInfo>();
    au.addRequired<llvm::VirtRegMap>();
    au.addPreserved<llvm::VirtRegMap>();
    au.addRequired<llvm::LiveRegMatrix>();
    au.addPreserved<llvm::LiveRegMatrix>();
    au.addRequired<llvm::EdgeBundles>();
    au.addRequired<llvm::SpillPlacement>();
    llvm::MachineFunctionPass::getAnalysisUsage(au);
  }

  llvm::MachineFunctionProperties getRequiredProperties() const override {
    return llvm::MachineFunctionProperties().set(
        llvm::MachineFunctionProperties::Property::NoPHIs);
  }

  bool runOnMachineFunction(llvm::MachineFunction &mfn) override {
    // In this LLVM the classic analysis passes are themselves the analysis
    // result (no WrapperPass/WrapperLegacy split), so getAnalysis<T>() returns
    // the T& directly.
    llvm::VirtRegMap &vrm = getAnalysis<llvm::VirtRegMap>();
    llvm::LiveIntervals &lis = getAnalysis<llvm::LiveIntervals>();
    llvm::LiveRegMatrix &mat = getAnalysis<llvm::LiveRegMatrix>();
    llvm::MachineBlockFrequencyInfo &mbfi =
        getAnalysis<llvm::MachineBlockFrequencyInfo>();
    llvm::LiveStacks &livestks = getAnalysis<llvm::LiveStacks>();
    llvm::MachineDominatorTree &mdt = getAnalysis<llvm::MachineDominatorTree>();
    llvm::MachineLoopInfo &loops = getAnalysis<llvm::MachineLoopInfo>();
    llvm::EdgeBundles &edgeBundles = getAnalysis<llvm::EdgeBundles>();
    llvm::SpillPlacement &spillPlacer = getAnalysis<llvm::SpillPlacement>();
    llvm::AAResults &aa =
        getAnalysis<llvm::AAResultsWrapperPass>().getAAResults();

    nb::gil_scoped_acquire gil;
    // LCOV_EXCL_START -- emit_object always sets the active class before
    // running
    if (!activeRegAllocClass.is_valid())
      return false;
    // LCOV_EXCL_STOP

    // Analyses shared by whichever allocator drives this function. This LLVM's
    // VirtRegAuxInfo takes no ProfileSummaryInfo, the inline spiller takes the
    // owning pass rather than an analysis bundle, and SplitEditor takes an
    // explicit AAResults.
    llvm::VirtRegAuxInfo vrai(mfn, lis, vrm, loops, mbfi);
    vrai.calculateSpillWeightsAndHints();
    std::unique_ptr<llvm::Spiller> spiller(
        llvm::createInlineSpiller(*this, mfn, vrm, vrai));
    llvm::SplitAnalysis sa(vrm, lis, loops);
    llvm::SplitEditor se(sa, aa, lis, vrm, mdt, mbfi, vrai);

    auto runNative = [&] {
      NativeRegAlloc fallback;
      fallback.pyInit(vrm, lis, mat, *spiller, mfn);
      fallback.pyAllocate();
    };

    // LCOV_EXCL_START -- a module from create_machine_function holds exactly
    // one MachineFunction, so this runs once per emit_object and a stash from
    // an earlier function is never already pending. Defensive for a
    // hypothetical multi-function module: don't re-enter Python (which would
    // clobber the pending error); allocate natively so this function's MIR
    // stays valid and the original error still re-raises.
    if (eudsl::pendingCodegenError) {
      runNative();
      return true;
    }
    // LCOV_EXCL_STOP

    nb::object obj;
    try {
      obj = activeRegAllocClass();
    } catch (...) {
      // The subclass __init__ raised; keep the first pending error if any, then
      // allocate natively so the MIR is valid and the pipeline reaches
      // runCodegenPipeline's re-raise instead of aborting on unallocated vregs.
      if (!eudsl::pendingCodegenError)
        eudsl::pendingCodegenError = std::current_exception();
      runNative();
      return true;
    }
    auto *base =
        static_cast<PyRegAllocBase *>(nb::inst_ptr<PyRegAllocBase>(obj));
    base->pyInit(vrm, lis, mat, *spiller, mfn, &sa, &se, &mbfi, &edgeBundles,
                 &spillPlacer, &vrai, &loops);
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
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_DEPENDENCY(EdgeBundles)
INITIALIZE_PASS_DEPENDENCY(SpillPlacement)
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

  nb::class_<llvm::InterferenceCache::Cursor>(m, "InterferenceCursor")
      .def(
          "move_to_block",
          [](llvm::InterferenceCache::Cursor &c, unsigned n) {
            c.moveToBlock(n);
          },
          "mbb_number"_a, "Point the cursor at block `mbb_number`.")
      .def(
          "has_interference",
          [](llvm::InterferenceCache::Cursor &c) {
            return c.hasInterference();
          },
          "Whether the current block has any interference for this physreg.")
      .def(
          "first", [](llvm::InterferenceCache::Cursor &c) { return c.first(); },
          "First interfering SlotIndex in the current block.")
      .def(
          "last", [](llvm::InterferenceCache::Cursor &c) { return c.last(); },
          "Last interfering SlotIndex in the current block.");

  nb::class_<PyRegAllocBase>(m, "RegAllocBase")
      .def(nb::init<>())
      .def("allocation_order", &PyRegAllocBase::allocationOrder, "li"_a,
           "Physregs (as ids) to try for `li`, in target allocation order.")
      .def("interfering_vregs", &PyRegAllocBase::interferingVRegs, "li"_a,
           "physreg"_a,
           "Ids of virtual registers whose live ranges interfere with `li` on "
           "`physreg` -- those assigned to `physreg` or to a physreg aliasing "
           "it. Alias/subregister correct (queries every reg unit of "
           "`physreg`, de-duplicating). These are the eviction candidates.")
      .def("register_cost", &PyRegAllocBase::registerCost, "physreg"_a,
           "Per-use cost of `physreg` (the CostPerUseLimit heuristic; 0 on "
           "targets with uniform register cost).")
      .def(
          "fixed_interference_spans", &PyRegAllocBase::fixedInterferenceSpans,
          "li"_a, "physreg"_a,
          "Fixed (physical) reg-unit interference segments for `physreg` "
          "overlapping `li` -- calcGapWeights marks gaps they cover huge_valf.")
      .def("check_reg_mask_interference",
           &PyRegAllocBase::checkRegMaskInterferenceLI, "li"_a,
           "Whether `li` is live across any register-mask (call clobber).")
      .def("check_reg_mask_interference_phys",
           &PyRegAllocBase::checkRegMaskInterferencePhys, "li"_a, "physreg"_a,
           "Whether `physreg` is clobbered by a register mask `li` crosses.")
      .def("reg_mask_slots_in_block", &PyRegAllocBase::regMaskSlotsInBlock,
           "mbb_number"_a,
           "The register-mask slot indexes in block `mbb_number`.")
      .def("reg_class", &PyRegAllocBase::regClass, nb::rv_policy::reference,
           "reg"_a,
           "The register class of virtual register `reg` (target-static; "
           "borrowed).")
      .def("num_allocatable_regs", &PyRegAllocBase::numAllocatableRegs,
           "reg_class"_a,
           "Number of actually-allocatable registers in `reg_class` (the "
           "register-pressure denominator; reserved registers excluded).")
      .def("slot_index_instr_distance", &PyRegAllocBase::slotIndexInstrDistance,
           "Slot positions per instruction (SlotIndex::InstrDist).")
      .def("reverse_local_assignment", &PyRegAllocBase::reverseLocalAssignment,
           "Whether the target assigns local ranges in reverse order "
           "(honors -greedy-reverse-local-assignment).")
      .def(
          "reg_class_priority_trumps_globalness",
          &PyRegAllocBase::regClassPriorityTrumpsGlobalness,
          "Whether the register class's AllocationPriority outranks globalness "
          "in the priority calculation (honors "
          "-greedy-regclass-priority-trumps-globalness).")
      .def(
          "reg_class_has_global_priority",
          &PyRegAllocBase::regClassHasGlobalPriority, "reg_class"_a,
          "`reg_class`'s GlobalPriority flag -- the first disjunct of RAGreedy "
          "ForceGlobal (the size-based disjunct is computed in enqueue).")
      .def("reg_class_is_allocatable", &PyRegAllocBase::regClassIsAllocatable,
           "reg_class"_a, "Whether `reg_class` is allocatable.")
      .def("reg_class_allocation_priority",
           &PyRegAllocBase::regClassAllocationPriority, "reg_class"_a,
           "`reg_class`'s target allocation priority (getPriority's "
           "AllocationPriority field).")
      .def("has_known_preference", &PyRegAllocBase::hasKnownPreference, "reg"_a,
           "Whether `reg` has a known physreg preference (getPriority boosts "
           "these).")
      .def("has_preferred_phys", &PyRegAllocBase::hasPreferredPhys, "reg"_a,
           "Whether `reg` is assigned to its preferred physreg (a satisfied "
           "copy "
           "hint) -- evicting it breaks that hint (eviction BrokenHints).")
      .def("reg_class_copy_cost", &PyRegAllocBase::regClassCopyCost,
           "reg_class"_a,
           "TargetRegisterClass::getCopyCost -- the per-broken-hint weight in "
           "the "
           "eviction cost model.")
      .def(
          "last_slot_index", &PyRegAllocBase::lastSlotIndex,
          "The last SlotIndex of the function (local-range priority endpoint).")
      .def("zero_slot_index", &PyRegAllocBase::zeroSlotIndex,
           "The zero SlotIndex of the function (reverse local-range endpoint).")
      .def("interval_is_in_one_mbb", &PyRegAllocBase::intervalIsInOneMBB,
           "reg"_a,
           "Whether `reg`'s whole live interval lies in a single block "
           "(local- vs global-split routing in trySplit).")
      .def("is_proper_sub_class", &PyRegAllocBase::isProperSubClass, "reg"_a,
           "Whether `reg`'s class is a proper subclass of its allocation "
           "superclass (shouldSplitSingleBlock's SingleInstrs input).")
      .def("is_copy_like_at", &PyRegAllocBase::isCopyLikeAt, "idx"_a,
           "Whether the instruction at slot `idx` is copy-like including "
           "target-specific copies (TII::isCopyInstr, or SUBREG_TO_REG).")
      .def("is_copy_like_instr_at", &PyRegAllocBase::isCopyLikeInstrAt, "idx"_a,
           "MachineInstr::isCopyLike() at slot `idx` (generic COPY / "
           "SUBREG_TO_REG only) -- the exact test shouldSplitSingleBlock uses.")
      .def("should_region_split_for_virt_reg",
           &PyRegAllocBase::shouldRegionSplitForVirtReg, "reg"_a,
           "TargetRegisterInfo::shouldRegionSplitForVirtReg (default true) -- "
           "tryRegionSplit's target-hook guard.")
      .def("is_full_copy_instr_at", &PyRegAllocBase::isFullCopyInstrAt, "idx"_a,
           "TII::isFullCopyInstr at slot `idx` (tryInstructionSplit skips full "
           "copies).")
      .def(
          "reads_lane_subset", &PyRegAllocBase::readsLaneSubset, "li"_a,
          "idx"_a,
          "RAGreedy::readsLaneSubset -- whether the instruction at `idx` reads "
          "only a subset of `li`'s live lanes (tryInstructionSplit's subrange "
          "arm splits around such uses).")
      .def(
          "is_trivially_rematerializable",
          &PyRegAllocBase::isTriviallyRematerializable, "mi"_a,
          "Whether `mi` (e.g. an interval's defining instruction) is trivially "
          "rematerializable.")
      .def("reg_allocation_hints", &PyRegAllocBase::regAllocationHints, "reg"_a,
           "The (type, [ids]) allocation hints for virtual register `reg`: "
           "`type` is the hint kind (0 = target-independent copy hints), "
           "`ids` the hinted physregs (a hinted physreg is preferred; evicting "
           "a hinted assignment is a 'broken hint' in eviction cost). (0, []) "
           "if none. `reg` must be a virtual register.")
      .def("simple_hint", &PyRegAllocBase::simpleHint, "reg"_a,
           "The single simple copy-hint reg id for virtual register `reg`, or "
           "0 if none. `reg` must be a virtual register.")
      .def("last_callee_saved_alias", &PyRegAllocBase::lastCalleeSavedAlias,
           "physreg"_a,
           "The last callee-saved register aliasing `physreg` (id, or 0) -- "
           "biases against introducing a callee-saved spill.")
      .def("calculate_spill_weight_and_hint",
           &PyRegAllocBase::calculateSpillWeightAndHint, "reg"_a,
           "Recompute `reg`'s spill weight and hint from its current defs/uses "
           "-- for the vregs a split produced, so their enqueue priority "
           "reflects the new ranges.")
      .def("spill", &PyRegAllocBase::spill, "li"_a,
           "Spill `li`; new split vregs are appended for re-enqueue. Only "
           "valid inside select_or_split.")
      .def_prop_ro(
          "matrix", &PyRegAllocBase::matrix, nb::rv_policy::reference,
          "The LiveRegMatrix for interference queries and assignment. Borrowed "
          "and valid only within an allocator callback; do not retain.")
      .def_prop_ro(
          "lis", &PyRegAllocBase::intervals, nb::rv_policy::reference,
          "The LiveIntervals analysis for this function. Borrowed and valid "
          "only within an allocator callback; do not retain.")
      .def_prop_ro(
          "vrm", &PyRegAllocBase::virtRegMap, nb::rv_policy::reference,
          "The VirtRegMap being populated with assignments. Borrowed and valid "
          "only within an allocator callback; do not retain.")
      .def_prop_ro(
          "machine_function", &PyRegAllocBase::machineFunction,
          nb::rv_policy::reference,
          "The MachineFunction being allocated. Borrowed and valid only within "
          "an allocator callback; do not retain.")
      .def_prop_ro(
          "split_analysis", &PyRegAllocBase::splitAnalysisPtr,
          nb::rv_policy::reference,
          "The SplitAnalysis for planning live-range splits. Borrowed and "
          "valid only within an allocator callback; do not retain.")
      .def_prop_ro(
          "split_editor", &PyRegAllocBase::splitEditorPtr,
          nb::rv_policy::reference,
          "The SplitEditor for applying live-range splits (call reset(...) "
          "before open_intv/use_intv). Borrowed and valid only within an "
          "allocator callback; do not retain.")
      .def("new_live_range_edit", &PyRegAllocBase::newLiveRangeEdit,
           nb::rv_policy::reference, "li"_a,
           "A LiveRangeEdit over the current split-vreg vector, for "
           "split_editor.reset. Only valid inside select_or_split; do not "
           "retain past the call.")
      .def_prop_ro(
          "mbfi", &PyRegAllocBase::blockFrequencyInfo, nb::rv_policy::reference,
          "The MachineBlockFrequencyInfo for frequency-weighted cost models. "
          "Borrowed and valid only within an allocator callback; do not "
          "retain.")
      .def_prop_ro(
          "edge_bundles", &PyRegAllocBase::edgeBundlesPtr,
          nb::rv_policy::reference,
          "The EdgeBundles partition of CFG edges, mapping blocks to the "
          "bundles global splitting reasons about. Borrowed and valid only "
          "within an allocator callback; do not retain.")
      .def_prop_ro(
          "spill_placer", &PyRegAllocBase::spillPlacerPtr,
          nb::rv_policy::reference,
          "The SpillPlacement network for choosing global-split boundaries "
          "(the machinery RAGreedy's splitAroundRegion drives). Borrowed and "
          "valid only within an allocator callback; do not retain.")
      .def(
          "new_interference_cursor", &PyRegAllocBase::newInterferenceCursor,
          "A fresh interference-cache cursor (call set_interference_physreg to "
          "point it at a physreg). Valid only within an allocator callback.")
      .def("set_interference_physreg", &PyRegAllocBase::cursorSetPhysReg,
           "cursor"_a, "physreg"_a,
           "Point `cursor` at `physreg`'s per-block interference.")
      .def("loop_header_number", &PyRegAllocBase::loopHeaderNumber,
           "mbb_number"_a,
           "Header block number of the innermost loop containing "
           "`mbb_number`, or None if none.")
      .def("first_nondebug_instr_index",
           &PyRegAllocBase::firstNonDebugInstrIndex, "mbb_number"_a,
           "SlotIndex of the first non-debug instruction in the block, or None "
           "if empty.")
      .def("through_insert_index", &PyRegAllocBase::throughInsertIndex,
           "mbb_number"_a,
           "Live-in insertion-point index for the analyzed interval's reg in "
           "the block.")
      .def("mbb_start_index_by_number", &PyRegAllocBase::mbbStartIndexByNumber,
           "mbb_number"_a, "Start SlotIndex of the block.");

  // A value number: one definition of a virtual register's live interval.
  // Rematerialization is keyed on the VNInfo whose defining instruction is
  // re-cloned at a use.
  nb::class_<llvm::VNInfo>(m, "VNInfo")
      .def_prop_ro("id", [](const llvm::VNInfo &v) { return v.id; })
      .def_prop_ro(
          "def_index", [](const llvm::VNInfo &v) { return v.def; },
          "SlotIndex where this value is defined.")
      .def_prop_ro("is_phi_def",
                   [](const llvm::VNInfo &v) { return v.isPHIDef(); })
      .def_prop_ro("is_unused",
                   [](const llvm::VNInfo &v) { return v.isUnused(); });

  nb::class_<llvm::LiveRange::Segment>(m, "LiveRangeSegment")
      .def_ro("start", &llvm::LiveRange::Segment::start)
      .def_ro("end", &llvm::LiveRange::Segment::end)
      .def_prop_ro(
          "valno", [](const llvm::LiveRange::Segment &s) { return s.valno; },
          nb::rv_policy::reference,
          "The value number this segment carries. Borrowed; do not retain "
          "across interval-recomputing calls.");

  // A virtual register's live interval: the allocator receives one per
  // select_or_split call and queries/assigns it against the matrix.
  nb::class_<llvm::LiveInterval>(m, "LiveInterval")
      .def_prop_ro("reg",
                   [](const llvm::LiveInterval &li) { return li.reg().id(); })
      .def_prop_ro(
          "has_sub_ranges",
          [](const llvm::LiveInterval &li) { return li.hasSubRanges(); },
          "Whether this interval tracks per-sub-register live ranges (only on "
          "targets with sub-register liveness, e.g. AMDGPU).")
      .def_prop_rw(
          "weight", [](const llvm::LiveInterval &li) { return li.weight(); },
          [](llvm::LiveInterval &li, float w) { li.setWeight(w); })
      .def_prop_ro(
          "is_spillable",
          [](const llvm::LiveInterval &li) { return li.isSpillable(); })
      .def_prop_ro(
          "begin_index",
          [](const llvm::LiveInterval &li) {
            // beginIndex/endIndex assert on an empty range; raise a catchable
            // error instead of aborting. Intervals reaching Python from
            // select_or_split are always non-empty, so the guard is defensive.
            if (li.empty())
              throw nb::value_error("live interval is empty"); // LCOV_EXCL_LINE
            return li.beginIndex();
          },
          "Lowest SlotIndex covered by this interval.")
      .def_prop_ro(
          "end_index",
          [](const llvm::LiveInterval &li) {
            if (li.empty())
              throw nb::value_error("live interval is empty"); // LCOV_EXCL_LINE
            return li.endIndex();
          },
          "Highest (exclusive) SlotIndex covered by this interval.")
      .def_prop_ro(
          "size", [](const llvm::LiveInterval &li) { return li.getSize(); },
          "Total size in slot-index units (RAGreedy ranks priority by this).")
      .def(
          "get_vni_at",
          [](llvm::LiveInterval &li, llvm::SlotIndex idx) {
            return li.getVNInfoAt(idx);
          },
          nb::rv_policy::reference, "idx"_a,
          "The value number live at `idx`, or None. Borrowed; do not retain "
          "across shrink_to_uses/eliminate_dead_defs, which recompute the "
          "interval and free its value numbers.")
      .def_prop_ro("num_val_nums", &llvm::LiveInterval::getNumValNums)
      .def(
          "get_val_num_info",
          [](llvm::LiveInterval &li, unsigned i) {
            return li.getValNumInfo(i);
          },
          nb::rv_policy::reference, "i"_a,
          "Value number #`i` of this interval. Borrowed; do not retain across "
          "shrink_to_uses/eliminate_dead_defs (see get_vni_at).")
      .def(
          "segments",
          [](const llvm::LiveInterval &li) {
            return std::vector<llvm::LiveRange::Segment>(li.begin(), li.end());
          },
          "The [start, end) segments of this interval (each with its value "
          "number) -- walk them to place interference per block/gap.");

  // The virtual-to-physical assignment map. get_phys/has_phys let an eviction
  // policy read which physreg an interfering vreg currently holds before
  // unassigning it.
  nb::class_<llvm::VirtRegMap>(m, "VirtRegMap")
      .def(
          "has_phys",
          [](llvm::VirtRegMap &vrm, unsigned reg) {
            return vrm.hasPhys(llvm::Register(reg));
          },
          "reg"_a, "Whether virtual register `reg` currently has a physreg.")
      .def(
          "get_phys",
          [](llvm::VirtRegMap &vrm, unsigned reg) {
            return vrm.getPhys(llvm::Register(reg)).id();
          },
          "reg"_a,
          "The physreg id assigned to `reg`, or 0 if unassigned (check "
          "has_phys first).");
  nb::class_<llvm::Spiller>(m, "Spiller");

  // A block's estimated execution frequency as a fixed-point number scaled by
  // the entry frequency (llvm::BlockFrequency). Comparable and additive so a
  // cost model can weigh and combine frequencies directly.
  nb::class_<llvm::BlockFrequency>(m, "BlockFrequency")
      .def(nb::init<uint64_t>(), "freq"_a)
      .def_static(
          "max",
          // This LLVM exposes the saturation value as the scalar
          // getMaxFrequency(); wrap it back into a BlockFrequency.
          [] {
            return llvm::BlockFrequency(
                llvm::BlockFrequency::getMaxFrequency());
          },
          "The saturation value (maximum possible frequency).")
      .def("get_frequency", &llvm::BlockFrequency::getFrequency,
           "The raw fixed-point frequency value.")
      .def(nb::self < nb::self, "other"_a)
      .def(nb::self <= nb::self, "other"_a)
      .def(nb::self > nb::self, "other"_a)
      .def(nb::self >= nb::self, "other"_a)
      .def(nb::self == nb::self, "other"_a)
      // This LLVM's BlockFrequency has no operator!=, so express it via ==.
      .def(
          "__ne__",
          [](const llvm::BlockFrequency &a, const llvm::BlockFrequency &b) {
            return a.getFrequency() != b.getFrequency();
          },
          "other"_a, nb::is_operator())
      .def(nb::self + nb::self, "other"_a)
      .def(nb::self - nb::self, "other"_a)
      .def("__repr__", [](const llvm::BlockFrequency &f) {
        return "BlockFrequency(" + std::to_string(f.getFrequency()) + ")";
      });

  // Frequency estimates per block, driving frequency-weighted spill/split cost
  // models (the weighting RAGreedy applies). Frequencies are relative to the
  // entry block; block_freq_relative_to_entry_block gives the ratio directly.
  nb::class_<llvm::MachineBlockFrequencyInfo>(m, "MachineBlockFrequencyInfo")
      .def(
          "block_freq",
          [](llvm::MachineBlockFrequencyInfo &mbfi,
             llvm::MachineBasicBlock *mbb) { return mbfi.getBlockFreq(mbb); },
          "mbb"_a,
          "Estimated frequency of `mbb` (compare against other blocks or "
          "entry_freq).")
      .def(
          "block_freq_relative_to_entry_block",
          [](llvm::MachineBlockFrequencyInfo &mbfi,
             llvm::MachineBasicBlock *mbb) {
            return mbfi.getBlockFreqRelativeToEntryBlock(mbb);
          },
          "mbb"_a, "Frequency of `mbb` as a ratio to the entry block (1.0).")
      .def(
          "entry_freq",
          [](llvm::MachineBlockFrequencyInfo &mbfi) {
            // This LLVM's getEntryFreq() returns a raw uint64_t (newer LLVM
            // returns a BlockFrequency); wrap it so the Python surface is a
            // BlockFrequency, matching block_freq() and the newer API.
            return llvm::BlockFrequency(mbfi.getEntryFreq());
          },
          "Frequency of the entry block (the denominator of the relative "
          "frequencies).");

  // CFG edges partitioned into bundles: all edges leaving a block share one
  // bundle, all edges entering it share another. Global region splitting
  // reasons about which bundles keep the value in a register.
  nb::class_<llvm::EdgeBundles>(m, "EdgeBundles")
      .def(
          "get_bundle",
          [](llvm::EdgeBundles &eb, llvm::MachineBasicBlock *mbb, bool out) {
            return eb.getBundle(mbb->getNumber(), out);
          },
          "mbb"_a, "out"_a,
          "Bundle number for `mbb`'s incoming (out=False) or outgoing "
          "(out=True) edges. Taking the block (not a raw number) keeps the "
          "index in range by construction.")
      .def("num_bundles", &llvm::EdgeBundles::getNumBundles,
           "Total number of edge bundles in the CFG.")
      .def(
          "get_bundle_number",
          [](llvm::EdgeBundles &eb, unsigned number, bool out) {
            return eb.getBundle(number, out);
          },
          "mbb_number"_a, "out"_a,
          "Edge bundle for block `mbb_number` (out-edges if `out`).")
      .def(
          "get_blocks",
          [](llvm::EdgeBundles &eb, unsigned bundle) {
            if (bundle >= eb.getNumBundles())
              throw nb::index_error("bundle number out of range");
            llvm::ArrayRef<unsigned> blocks = eb.getBlocks(bundle);
            return std::vector<unsigned>(blocks.begin(), blocks.end());
          },
          "bundle"_a,
          "Block numbers connected to `bundle` (0 <= bundle < "
          "num_bundles).");

  // The bit vector SpillPlacement writes its per-bundle register/spill decision
  // into. prepare() retains it and finish() fills it, so the Python caller must
  // keep it alive across the placement calls.
  nb::class_<llvm::BitVector>(m, "BitVector")
      .def(nb::init<>())
      .def(
          "test", [](llvm::BitVector &bv, unsigned i) { return bv.test(i); },
          "i"_a, "Whether bit `i` is set.")
      .def("count", &llvm::BitVector::count, "Number of set bits.")
      .def("size", &llvm::BitVector::size, "Number of bits.")
      .def(
          "resize", [](llvm::BitVector &bv, unsigned n) { bv.resize(n); },
          "n"_a, "Grow/shrink to `n` bits (new bits cleared).")
      .def(
          "set", [](llvm::BitVector &bv, unsigned i) { bv.set(i); }, "i"_a,
          "Set bit `i`.")
      .def(
          "reset", [](llvm::BitVector &bv, unsigned i) { bv.reset(i); }, "i"_a,
          "Clear bit `i`.")
      .def(
          "set_bits",
          [](llvm::BitVector &bv) {
            std::vector<unsigned> v;
            for (unsigned i : bv.set_bits())
              v.push_back(i);
            return v;
          },
          "Indices of the set bits (the in-register edge bundles after "
          "finish()).");

  // Per-block entry/exit constraints for the spill-placement network.
  nb::enum_<llvm::SpillPlacement::BorderConstraint>(m, "BorderConstraint")
      .value("DontCare", llvm::SpillPlacement::DontCare)
      .value("PrefReg", llvm::SpillPlacement::PrefReg)
      .value("PrefSpill", llvm::SpillPlacement::PrefSpill)
      .value("PrefBoth", llvm::SpillPlacement::PrefBoth)
      .value("MustSpill", llvm::SpillPlacement::MustSpill);

  nb::class_<llvm::SpillPlacement::BlockConstraint>(m, "BlockConstraint")
      .def(nb::init<>())
      .def_rw("number", &llvm::SpillPlacement::BlockConstraint::Number,
              "Basic block number this constraint applies to.")
      // Entry/Exit are bitfields, so bind them through accessors (their address
      // cannot be taken for def_rw).
      .def_prop_rw(
          "entry",
          [](const llvm::SpillPlacement::BlockConstraint &c) {
            return c.Entry;
          },
          [](llvm::SpillPlacement::BlockConstraint &c,
             llvm::SpillPlacement::BorderConstraint v) { c.Entry = v; },
          "Constraint on block entry.")
      .def_prop_rw(
          "exit",
          [](const llvm::SpillPlacement::BlockConstraint &c) { return c.Exit; },
          [](llvm::SpillPlacement::BlockConstraint &c,
             llvm::SpillPlacement::BorderConstraint v) { c.Exit = v; },
          "Constraint on block exit.")
      .def_rw("changes_value",
              &llvm::SpillPlacement::BlockConstraint::ChangesValue,
              "True when the block has a non-PHI def of the live range.");

  // The Hopfield-network spill-placement solver RAGreedy uses to pick global
  // split boundaries: prepare a result vector, add block constraints and
  // transparent links, iterate to convergence, and finish to get the optimal
  // per-bundle register/spill assignment.
  nb::class_<llvm::SpillPlacement>(m, "SpillPlacement")
      .def(
          "prepare",
          [](llvm::SpillPlacement &sp, llvm::BitVector &regBundles) {
            sp.prepare(regBundles);
          },
          "reg_bundles"_a,
          "Reset for a new computation; `reg_bundles` receives the result and "
          "is retained (keep it alive until after finish()).")
      .def(
          "add_constraints",
          [](llvm::SpillPlacement &sp,
             const std::vector<llvm::SpillPlacement::BlockConstraint>
                 &constraints) { sp.addConstraints(constraints); },
          "constraints"_a,
          "Add entry/exit constraints for blocks where the value is live.")
      .def(
          "add_pref_spill",
          [](llvm::SpillPlacement &sp, const std::vector<unsigned> &blocks,
             bool strong) { sp.addPrefSpill(blocks, strong); },
          "blocks"_a, "strong"_a,
          "Bias the listed blocks toward spilling on entry and exit.")
      .def(
          "add_links",
          [](llvm::SpillPlacement &sp, const std::vector<unsigned> &links) {
            sp.addLinks(links);
          },
          "links"_a, "Add transparent (through) blocks by number.")
      .def("scan_active_bundles", &llvm::SpillPlacement::scanActiveBundles,
           "Initial scan of activated bundles; returns whether any prefer a "
           "register.")
      .def("iterate", &llvm::SpillPlacement::iterate,
           "Update the network until convergence.")
      .def(
          "get_recent_positive",
          [](llvm::SpillPlacement &sp) {
            llvm::ArrayRef<unsigned> pos = sp.getRecentPositive();
            return std::vector<unsigned>(pos.begin(), pos.end());
          },
          "Bundles that turned positive during the last scan/iterate.")
      .def("finish", &llvm::SpillPlacement::finish,
           "Compute the optimal placement into the prepared vector; returns "
           "True if a perfect (all-register) solution was found.")
      .def(
          "get_block_frequency",
          [](llvm::SpillPlacement &sp, llvm::MachineBasicBlock *mbb) {
            return sp.getBlockFrequency(mbb->getNumber());
          },
          "mbb"_a,
          "Estimated execution frequency of `mbb` (per function invocation). "
          "Taking the block keeps the index in range by construction.")
      .def(
          "get_block_frequency_by_number",
          [](llvm::SpillPlacement &sp, unsigned number) {
            return sp.getBlockFrequency(number);
          },
          "mbb_number"_a, "Estimated frequency of block `mbb_number`.");

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
      .def(
          "is_earlier_instr",
          [](const llvm::SlotIndex &a, const llvm::SlotIndex &b) {
            return llvm::SlotIndex::isEarlierInstr(a, b);
          },
          "other"_a,
          "Whether this and `other` are on different instructions and this is "
          "earlier (SlotIndex::isEarlierInstr).")
      .def(
          "is_same_instr",
          [](const llvm::SlotIndex &a, const llvm::SlotIndex &b) {
            return llvm::SlotIndex::isSameInstr(a, b);
          },
          "other"_a,
          "Whether this and `other` are on the same instruction "
          "(SlotIndex::isSameInstr).")
      .def("get_reg_slot",
           [](const llvm::SlotIndex &i) { return i.getRegSlot(); })
      .def("get_base_index",
           [](const llvm::SlotIndex &i) { return i.getBaseIndex(); })
      .def("get_boundary_index",
           [](const llvm::SlotIndex &i) { return i.getBoundaryIndex(); })
      .def("get_next_index",
           [](const llvm::SlotIndex &i) { return i.getNextIndex(); })
      .def("distance", &llvm::SlotIndex::distance, "other"_a,
           "Number of slots between this and `other` (a raw distance in the "
           "slot-index space).")
      .def("get_approx_instr_distance", &llvm::SlotIndex::getInstrDistance,
           "other"_a,
           "Approximate number of instructions between this and `other` (what "
           "RAGreedy's size/gap heuristics measure).")
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
          nb::rv_policy::reference_internal, "reg"_a)
      .def(
          "instr_from_index",
          [](llvm::LiveIntervals &l, llvm::SlotIndex idx) {
            return l.getInstructionFromIndex(idx);
          },
          nb::rv_policy::reference, "idx"_a,
          "The MachineInstr at `idx`, or None. Borrowed; do not retain.")
      .def(
          "compute_interval",
          [](llvm::LiveIntervals &l, unsigned reg) {
            if (!l.hasInterval(llvm::Register(reg)))
              throw nb::value_error("register has no live interval to compute");
            return eudslComputeVirtRegInterval(
                l, l.getInterval(llvm::Register(reg)));
          },
          "reg"_a,
          "Compute `reg`'s (already-created) interval from its defs/uses -- "
          "e.g. after rematerializing a def into the empty interval from "
          "LiveRangeEdit.create and redirecting a use onto it. Returns True if "
          "the interval needs splitting.")
      .def(
          "shrink_to_uses",
          [](llvm::LiveIntervals &l, unsigned reg) {
            if (!l.hasInterval(llvm::Register(reg)))
              throw nb::value_error("register has no live interval to shrink");
            llvm::SmallVector<llvm::MachineInstr *, 8> dead;
            l.shrinkToUses(&l.getInterval(llvm::Register(reg)), &dead);
            return std::vector<llvm::MachineInstr *>(dead.begin(), dead.end());
          },
          "reg"_a,
          "Recompute `reg`'s interval from its remaining uses after "
          "redirecting some, marking now-dead defs; returns those dead "
          "instructions (feed them to LiveRangeEdit.eliminate_dead_defs).");

  // Splitting analysis: after analyze(li), it describes how `li` is used per
  // block, guiding where the split editor should open/close intervals.
  nb::class_<llvm::SplitAnalysis> sa(m, "SplitAnalysis");
  nb::class_<llvm::SplitAnalysis::BlockInfo>(sa, "BlockInfo")
      .def_prop_ro(
          "mbb", [](const llvm::SplitAnalysis::BlockInfo &b) { return b.MBB; },
          nb::rv_policy::reference)
      .def_ro("first_instr", &llvm::SplitAnalysis::BlockInfo::FirstInstr)
      .def_ro("last_instr", &llvm::SplitAnalysis::BlockInfo::LastInstr)
      .def_ro("first_def", &llvm::SplitAnalysis::BlockInfo::FirstDef)
      .def_ro("live_in", &llvm::SplitAnalysis::BlockInfo::LiveIn)
      .def_ro("live_out", &llvm::SplitAnalysis::BlockInfo::LiveOut)
      .def(
          "is_one_instr",
          [](const llvm::SplitAnalysis::BlockInfo &b) {
            return b.isOneInstr();
          },
          "Whether the interval touches exactly one instruction in this block "
          "(shouldSplitSingleBlock always splits multi-instruction blocks).");
  sa.def(
        "analyze",
        [](llvm::SplitAnalysis &s, const llvm::LiveInterval &li) {
          s.analyze(&li);
        },
        "li"_a)
      .def("use_blocks",
           [](llvm::SplitAnalysis &s) {
             return std::vector<llvm::SplitAnalysis::BlockInfo>(
                 s.getUseBlocks().begin(), s.getUseBlocks().end());
           })
      .def("num_through_blocks", &llvm::SplitAnalysis::getNumThroughBlocks)
      .def(
          "get_use_slots",
          [](llvm::SplitAnalysis &s) {
            llvm::ArrayRef<llvm::SlotIndex> slots = s.getUseSlots();
            return std::vector<llvm::SlotIndex>(slots.begin(), slots.end());
          },
          "SlotIndexes of the instructions using the analyzed interval "
          "(needed for local/instruction splitting). Valid after analyze().")
      .def(
          "last_split_point",
          [](llvm::SplitAnalysis &s, llvm::MachineBasicBlock *mbb) {
            return s.getLastSplitPoint(mbb);
          },
          "mbb"_a)
      .def(
          "last_split_point_number",
          [](llvm::SplitAnalysis &s, unsigned n) {
            return s.getLastSplitPoint(n);
          },
          "mbb_number"_a, "Last legal split point in block `mbb_number`.")
      .def(
          "first_split_point",
          [](llvm::SplitAnalysis &s, unsigned n) {
            return s.getFirstSplitPoint(n);
          },
          "mbb_number"_a, "First legal split point in block `mbb_number`.")
      .def("num_live_blocks", &llvm::SplitAnalysis::getNumLiveBlocks,
           "Number of blocks where the analyzed interval is live.")
      .def(
          "count_live_blocks",
          [](llvm::SplitAnalysis &s, const llvm::LiveInterval &li) {
            return s.countLiveBlocks(&li);
          },
          "li"_a, "Number of blocks where `li` is live (post-split check).")
      .def(
          "looks_like_loop_iv",
          // This LLVM's SplitAnalysis has no looksLikeLoopIV(); reproduce its
          // check over the public use-block info and loop tree: exactly two use
          // blocks, one of which is a loop latch where the value is live-in,
          // live-out, and defined.
          [](llvm::SplitAnalysis &s) {
            llvm::ArrayRef<llvm::SplitAnalysis::BlockInfo> useBlocks =
                s.getUseBlocks();
            if (useBlocks.size() != 2)
              return false;
            return llvm::any_of(
                useBlocks, [&s](const llvm::SplitAnalysis::BlockInfo &bi) {
                  const llvm::MachineLoop *l = s.Loops.getLoopFor(bi.MBB);
                  return bi.LiveIn && bi.LiveOut && bi.FirstDef && l &&
                         l->isLoopLatch(bi.MBB);
                });
          },
          "Whether the analyzed interval looks like a loop induction "
          "variable.")
      .def(
          "is_original_endpoint",
          [](llvm::SplitAnalysis &s, llvm::SlotIndex idx) {
            return s.isOriginalEndpoint(idx);
          },
          "idx"_a,
          "Whether the original live range was killed or defined at `idx` "
          "(shouldSplitSingleBlock will not isolate an endpoint that an "
          "earlier split created).")
      .def("through_blocks", [](llvm::SplitAnalysis &s) {
        std::vector<unsigned> v;
        for (unsigned b : s.getThroughBlocks().set_bits())
          v.push_back(b);
        return v;
      });

  // A set of subregister lanes (llvm::LaneBitmask), forwarded to
  // rematerialize_at to say which lanes are live at the remat point.
  nb::class_<llvm::LaneBitmask>(m, "LaneBitmask")
      .def(nb::init<uint64_t>(), "mask"_a)
      .def_static("get_all", &llvm::LaneBitmask::getAll,
                  "All lanes (the default when the whole register is live).")
      .def_static("get_none", &llvm::LaneBitmask::getNone, "No lanes.")
      .def("get_as_integer", &llvm::LaneBitmask::getAsInteger)
      .def("none", &llvm::LaneBitmask::none)
      .def("any", &llvm::LaneBitmask::any)
      .def(nb::self == nb::self, "other"_a)
      .def(nb::self != nb::self, "other"_a);

  // The edit buffer the split editor writes new vregs into.
  nb::class_<llvm::LiveRangeEdit> lre(m, "LiveRangeEdit");
  lre.def("new_vregs",
          [](llvm::LiveRangeEdit &e) {
            std::vector<unsigned> v;
            for (llvm::Register r : e.regs())
              v.push_back(r.id());
            return v;
          })
      .def(
          "create", [](llvm::LiveRangeEdit &e) { return e.create().id(); },
          "Create a new vreg (same class as the parent) and append it for "
          "re-enqueue; the destination for a rematerialized def.")
      .def(
          "rematerialize_at",
          [](llvm::LiveRangeEdit &e, llvm::MachineBasicBlock *mbb,
             llvm::MachineInstr *before, unsigned destReg,
             const llvm::LiveRangeEdit::Remat &rm, bool late, unsigned subIdx,
             llvm::MachineInstr *replaceIndexMI, llvm::LaneBitmask usedLanes) {
            const llvm::TargetRegisterInfo *tri =
                mbb->getParent()->getSubtarget().getRegisterInfo();
            // This LLVM's rematerializeAt takes no sub-register index, index-
            // replacement instruction, or used-lane mask; those parameters are
            // accepted for API compatibility but have no effect here.
            (void)subIdx;
            (void)replaceIndexMI;
            (void)usedLanes;
            return e.rematerializeAt(*mbb, before->getIterator(),
                                     llvm::Register(destReg), rm, *tri, late);
          },
          "mbb"_a, "before"_a, "dest_reg"_a, "remat"_a, "late"_a = false,
          "sub_idx"_a = 0, "replace_index_mi"_a.none() = nullptr,
          "used_lanes"_a = llvm::LaneBitmask::getAll(),
          "Clone rm's defining instruction into `dest_reg` just before "
          "`before`; returns the def SlotIndex. `late` inserts after other "
          "defs at the same slot, `sub_idx` writes a subregister, "
          "`replace_index_mi` (if given) is replaced in the index map by the "
          "new instruction, and `used_lanes` is the lane mask live at the "
          "remat "
          "point (forwarded to the target). Liveness is not updated -- call "
          "compute_interval(dest_reg) after.")
      .def(
          "eliminate_dead_defs",
          [](llvm::LiveRangeEdit &e, std::vector<llvm::MachineInstr *> dead,
             std::vector<unsigned> regsBeingSpilled) {
            llvm::SmallVector<llvm::MachineInstr *, 8> deadVec(dead.begin(),
                                                               dead.end());
            llvm::SmallVector<llvm::Register, 4> spilled;
            for (unsigned r : regsBeingSpilled)
              spilled.push_back(llvm::Register(r));
            e.eliminateDeadDefs(deadVec, spilled);
          },
          "dead"_a, "regs_being_spilled"_a = std::vector<unsigned>(),
          "Delete the given now-dead defining instructions and trim their "
          "intervals (e.g. the original def after all its uses were "
          "rematerialized). `regs_being_spilled` lists registers currently "
          "being spilled, which must not be split into new intervals.");

  // Information needed to rematerialize a value at a new location: the value
  // number and (set by the caller) its defining instruction.
  nb::class_<llvm::LiveRangeEdit::Remat>(lre, "Remat")
      .def(nb::init<const llvm::VNInfo *>(), "parent_vni"_a)
      .def_prop_rw(
          "orig_mi",
          [](const llvm::LiveRangeEdit::Remat &r) { return r.OrigMI; },
          [](llvm::LiveRangeEdit::Remat &r, llvm::MachineInstr *mi) {
            r.OrigMI = mi;
          },
          nb::rv_policy::reference,
          "The instruction defining the value (its real expression); set "
          "from lis.instr_from_index(vni.def_index).");

  nb::enum_<llvm::SplitEditor::ComplementSpillMode>(m, "ComplementSpillMode")
      .value("SM_Partition", llvm::SplitEditor::SM_Partition)
      .value("SM_Size", llvm::SplitEditor::SM_Size)
      .value("SM_Speed", llvm::SplitEditor::SM_Speed);

  // Raw live-range splitting primitives (RAGreedy uses these internally); a
  // Python allocator drives them to open/enter/use/leave an interval and
  // finish, producing new vregs for re-enqueue.
  nb::class_<llvm::SplitEditor>(m, "SplitEditor")
      .def("reset", &llvm::SplitEditor::reset, "live_range_edit"_a,
           "mode"_a = llvm::SplitEditor::SM_Partition)
      .def("open_intv", &llvm::SplitEditor::openIntv)
      .def("select_intv", &llvm::SplitEditor::selectIntv, "idx"_a)
      .def("enter_intv_before", &llvm::SplitEditor::enterIntvBefore, "idx"_a)
      .def("enter_intv_after", &llvm::SplitEditor::enterIntvAfter, "idx"_a)
      .def(
          "enter_intv_at_end",
          [](llvm::SplitEditor &s, llvm::MachineBasicBlock *mbb) {
            return s.enterIntvAtEnd(*mbb);
          },
          "mbb"_a)
      .def(
          "use_intv",
          [](llvm::SplitEditor &s, llvm::SlotIndex a, llvm::SlotIndex b) {
            s.useIntv(a, b);
          },
          "start"_a, "end"_a)
      .def(
          "use_intv_mbb",
          [](llvm::SplitEditor &s, llvm::MachineBasicBlock *mbb) {
            s.useIntv(*mbb);
          },
          "mbb"_a)
      .def("leave_intv_after", &llvm::SplitEditor::leaveIntvAfter, "idx"_a)
      .def("leave_intv_before", &llvm::SplitEditor::leaveIntvBefore, "idx"_a)
      .def(
          "leave_intv_at_top",
          [](llvm::SplitEditor &s, llvm::MachineBasicBlock *mbb) {
            return s.leaveIntvAtTop(*mbb);
          },
          "mbb"_a)
      .def("overlap_intv", &llvm::SplitEditor::overlapIntv, "start"_a, "end"_a)
      .def(
          "split_single_block",
          [](llvm::SplitEditor &s, const llvm::SplitAnalysis::BlockInfo &bi) {
            s.splitSingleBlock(bi);
          },
          "block_info"_a,
          "Split the analyzed interval around the uses in a single block "
          "(part of a larger split; does not call finish).")
      .def(
          "split_live_through_block",
          [](llvm::SplitEditor &s, unsigned n, unsigned intvIn,
             llvm::SlotIndex intfIn, unsigned intvOut,
             llvm::SlotIndex intfOut) {
            s.splitLiveThroughBlock(n, intvIn, intfIn, intvOut, intfOut);
          },
          "mbb_number"_a, "intv_in"_a, "intf_in"_a, "intv_out"_a, "intf_out"_a,
          "Split a live-through block: enter in `intv_in`, leave in "
          "`intv_out` (interference-aware placement).")
      .def(
          "split_reg_in_block",
          [](llvm::SplitEditor &s, const llvm::SplitAnalysis::BlockInfo &bi,
             unsigned intvIn,
             llvm::SlotIndex intfIn) { s.splitRegInBlock(bi, intvIn, intfIn); },
          "block_info"_a, "intv_in"_a, "intf_in"_a,
          "Split a block that enters in `intv_in` and leaves on the stack.")
      .def(
          "split_reg_out_block",
          [](llvm::SplitEditor &s, const llvm::SplitAnalysis::BlockInfo &bi,
             unsigned intvOut, llvm::SlotIndex intfOut) {
            s.splitRegOutBlock(bi, intvOut, intfOut);
          },
          "block_info"_a, "intv_out"_a, "intf_out"_a,
          "Split a block that enters on the stack and leaves in `intv_out`.")
      .def(
          "finish",
          [](llvm::SplitEditor &s) {
            // Return the IntvMap: for each new vreg (in LiveRangeEdit order),
            // the open-interval index it landed in (0 = the complement /
            // remainder). RAGreedy reads this to tag non-progress local-split
            // ranges RS_Split2 and to send block-split remainders to spill.
            llvm::SmallVector<unsigned, 8> intvMap;
            s.finish(&intvMap);
            return std::vector<unsigned>(intvMap.begin(), intvMap.end());
          },
          "Apply the queued split and return the IntvMap (new-vreg index -> "
          "open-interval index; 0 is the remainder).");

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
          "li"_a)
      .def(
          "is_phys_reg_used",
          [](llvm::LiveRegMatrix &mat, unsigned physreg) {
            return mat.isPhysRegUsed(llvm::MCRegister(physreg));
          },
          "physreg"_a,
          "Whether any virtual register has been assigned to `physreg` yet "
          "(used by the callee-saved eviction bias).");

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
