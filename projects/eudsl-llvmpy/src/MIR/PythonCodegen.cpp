// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "MIR/AllocationOrder.h"
#include "MIR/Diagnostics.h"
#include "MIR/RegAllocBase.h"

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
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/PassRegistry.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <atomic>
#include <deque>
#include <exception>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace llvm {
// Declared here so PyRegAlloc's ctor can register its PassInfo idempotently;
// defined by the INITIALIZE_PASS block at the end of this file.
void initializePyRegAllocPass(PassRegistry &);
} // namespace llvm

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

// --- Register allocator -----------------------------------------------------
//
// PyRegAlloc is a MachineFunctionPass built on LLVM's RegAllocBase driver (the
// RABasic skeleton), registered in the RegisterRegAlloc registry so
// emit_object(regalloc="name") can select it. Its selectOrSplit assigns the
// first non-interfering physreg, or spills, natively -- unless Python drives
// it: emit_object(select=cb) installs a one-shot callable, or register_regalloc
// installs a class whose select_or_split method a fresh instance runs per
// MachineFunction. This mirrors how LLVM's own allocators are structured (the
// pass *is* the allocator, chosen by name), not a driver-owned strategy object.

// selectOrSplit / spill counters: register allocation is semantics-preserving,
// so the emitted code cannot witness that our pass ran; tests read these.
std::atomic<unsigned> selectOrSplitCount{0};
std::atomic<unsigned> spillCount{0};

// The one-shot select callable (emit_object(select=cb)) and the registered
// allocator class (register_regalloc + emit_object(regalloc="name")). Both are
// per-thread: emit_object installs one under the GIL for a run and clears it
// after, relying on the GIL to serialize callers (as -misched does).
thread_local nb::callable pendingSelectCallback;
thread_local nb::type_object activeRegAllocClass;

// Priority-queue entry: an interval and its dequeue key (higher dequeues
// first). The key is the spill weight by default, or the register_regalloc
// class's priority(li) when it defines one. Register number tie-breaks for a
// deterministic order (matching RABasic's CompSpillWeight default).
struct QueueEntry {
  float key;
  const llvm::LiveInterval *li;
};
struct CompKey {
  bool operator()(const QueueEntry &A, const QueueEntry &B) const {
    return std::tuple(A.key, A.li->reg()) < std::tuple(B.key, B.li->reg());
  }
};

class PyRegAlloc : public llvm::MachineFunctionPass,
                   public llvm::RegAllocBase,
                   // This policy only ever spills VirtReg itself (no
                   // reassignment), so the Matrix/Queue never hold stale
                   // entries and LiveRangeEdit's no-op delegate defaults
                   // suffice (unlike RABasic, which reassigns interferences).
                   private llvm::LiveRangeEdit::Delegate {
  llvm::MachineFunction *MF = nullptr;
  std::unique_ptr<llvm::Spiller> SpillerInstance;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, CompKey> Queue;

  // The callable selectOrSplit routes through, empty for the native policy. The
  // select= callable, or (register_regalloc) a fresh instance's select_or_split
  // bound method, rebuilt per MachineFunction in runOnMachineFunction.
  nb::callable selectCallback;
  // The registered allocator class, when chosen via regalloc="name"; instances
  // are made per function. Invalid for the select= / native paths.
  nb::object allocatorClass;
  // The fresh per-function instance (register_regalloc path), and whether it
  // defines priority(li) so enqueue routes the queue key through it.
  nb::object instance;
  bool usePyPriority = false;

public:
  static char ID;

  PyRegAlloc()
      : llvm::MachineFunctionPass(ID), llvm::RegAllocBase(),
        selectCallback(pendingSelectCallback),
        allocatorClass(activeRegAllocClass) {
    llvm::initializePyRegAllocPass(*llvm::PassRegistry::getPassRegistry());
  }

  llvm::StringRef getPassName() const override {
    return "eudsl register allocator";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
  void releaseMemory() override { SpillerInstance.reset(); }
  llvm::Spiller &spiller() override { return *SpillerInstance; }
  void enqueueImpl(const llvm::LiveInterval *LI) override;

  const llvm::LiveInterval *dequeue() override {
    if (Queue.empty())
      return nullptr;
    const llvm::LiveInterval *LI = Queue.top().li;
    Queue.pop();
    return LI;
  }

  llvm::MCRegister
  selectOrSplit(const llvm::LiveInterval &VirtReg,
                llvm::SmallVectorImpl<llvm::Register> &SplitVRegs) override;

  // Spill VirtReg itself (no reassignment), returning 0 so the driver replaces
  // it with the spill/reload vregs appended to SplitVRegs. Shared by the native
  // path and the callback's spill signal.
  llvm::MCRegister
  spillVirtReg(const llvm::LiveInterval &VirtReg,
               llvm::SmallVectorImpl<llvm::Register> &SplitVRegs);

  bool runOnMachineFunction(llvm::MachineFunction &mf) override;

  llvm::MachineFunctionProperties getRequiredProperties() const override {
    return llvm::MachineFunctionProperties().set(
        llvm::MachineFunctionProperties::Property::NoPHIs);
  }
  llvm::MachineFunctionProperties getClearedProperties() const override {
    return llvm::MachineFunctionProperties().set(
        llvm::MachineFunctionProperties::Property::IsSSA);
  }
};

char PyRegAlloc::ID = 0;

// Push LI with its dequeue key: the class's priority(li) when defined, else the
// spill weight. A raising priority stashes the error and keeps the weight, so
// the queue stays valid and allocation still completes legally (the pipeline
// then winds down to runCodegenPipeline's re-raise).
void PyRegAlloc::enqueueImpl(const llvm::LiveInterval *LI) {
  float key = LI->weight();
  if (usePyPriority && !eudsl::pendingCodegenError) {
    nb::gil_scoped_acquire gil;
    try {
      key = nb::cast<float>(instance.attr("priority")(nb::cast(
          const_cast<llvm::LiveInterval *>(LI), nb::rv_policy::reference)));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }
  Queue.push({key, LI});
}

// Assign the first non-interfering physreg in VirtReg's allocation order, or
// spill. When a callback is installed, present the legal candidate physreg ids
// as a list[int] and honor its return: a candidate id is assigned, None spills,
// anything else raises. On a stashed error, wind down natively.
llvm::MCRegister
PyRegAlloc::selectOrSplit(const llvm::LiveInterval &VirtReg,
                          llvm::SmallVectorImpl<llvm::Register> &SplitVRegs) {
  selectOrSplitCount.fetch_add(1, std::memory_order_relaxed);
  auto Order =
      llvm::AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);

  llvm::SmallVector<llvm::MCRegister, 16> candidateRegs;
  for (llvm::MCRegister PhysReg : Order) {
    assert(PhysReg.isValid());
    if (Matrix->checkInterference(VirtReg, PhysReg) ==
        llvm::LiveRegMatrix::IK_Free)
      candidateRegs.push_back(PhysReg);
  }

  if (selectCallback && !eudsl::pendingCodegenError) {
    // GIL outside the try: the catch stashes with std::current_exception(),
    // which touches Python refcounts and so needs the GIL.
    nb::gil_scoped_acquire gil;
    try {
      nb::list candidates;
      for (llvm::MCRegister PhysReg : candidateRegs)
        candidates.append(PhysReg.id());
      nb::object choice =
          selectCallback(nb::cast(const_cast<llvm::LiveInterval *>(&VirtReg),
                                  nb::rv_policy::reference),
                         candidates);
      if (choice.is_none())
        return spillVirtReg(VirtReg, SplitVRegs);
      unsigned chosenId = 0;
      if (nb::try_cast<unsigned>(choice, chosenId)) {
        for (llvm::MCRegister PhysReg : candidateRegs) {
          if (PhysReg.id() == chosenId)
            return PhysReg;
        }
      }
      throw nb::value_error("selectOrSplit returned a register that is not one "
                            "of the legal candidates");
    } catch (...) {
      // Stash and fall through to a legal native assignment so the pipeline
      // winds down to runCodegenPipeline's re-raise.
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  if (!candidateRegs.empty())
    return candidateRegs.front();
  return spillVirtReg(VirtReg, SplitVRegs);
}

// Spill VirtReg (never an interference -- this policy does no reassignment);
// the driver replaces it with the spill/reload vregs appended to SplitVRegs.
llvm::MCRegister
PyRegAlloc::spillVirtReg(const llvm::LiveInterval &VirtReg,
                         llvm::SmallVectorImpl<llvm::Register> &SplitVRegs) {
  if (!VirtReg.isSpillable())
    return llvm::MCRegister(~0u); // LCOV_EXCL_LINE -- test vregs are spillable
  spillCount.fetch_add(1, std::memory_order_relaxed);
  llvm::LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this,
                          &DeadRemats);
  spiller().spill(LRE);
  return llvm::MCRegister();
}

// Analysis dependency set, verbatim from RABasic::getAnalysisUsage (omitting
// one yields a null-analysis crash); setPreservesCFG and the chain-up are
// required.
void PyRegAlloc::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<llvm::AAResultsWrapperPass>();
  AU.addPreserved<llvm::AAResultsWrapperPass>();
  AU.addRequired<llvm::LiveIntervalsWrapperPass>();
  AU.addPreserved<llvm::LiveIntervalsWrapperPass>();
  AU.addPreserved<llvm::SlotIndexesWrapperPass>();
  AU.addRequired<llvm::LiveDebugVariablesWrapperLegacy>();
  AU.addPreserved<llvm::LiveDebugVariablesWrapperLegacy>();
  AU.addRequired<llvm::LiveStacksWrapperLegacy>();
  AU.addPreserved<llvm::LiveStacksWrapperLegacy>();
  AU.addRequired<llvm::ProfileSummaryInfoWrapperPass>();
  AU.addRequired<llvm::MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequired<llvm::MachineDominatorTreeWrapperPass>();
  AU.addRequiredID(llvm::MachineDominatorsID);
  AU.addRequired<llvm::MachineLoopInfoWrapperPass>();
  AU.addRequired<llvm::VirtRegMapWrapperLegacy>();
  AU.addPreserved<llvm::VirtRegMapWrapperLegacy>();
  AU.addRequired<llvm::LiveRegMatrixWrapperLegacy>();
  AU.addPreserved<llvm::LiveRegMatrixWrapperLegacy>();
  llvm::MachineFunctionPass::getAnalysisUsage(AU);
}

// Wire up the RegAllocBase driver, copied from RABasic::runOnMachineFunction.
bool PyRegAlloc::runOnMachineFunction(llvm::MachineFunction &mf) {
  MF = &mf;
  // register_regalloc path: a fresh instance per function drives selectOrSplit
  // through its select_or_split method (and enqueue through its priority, if
  // defined). The select= path leaves selectCallback as the user callable; the
  // native path leaves it empty.
  if (allocatorClass.is_valid() && !eudsl::pendingCodegenError) {
    nb::gil_scoped_acquire gil;
    try {
      instance = allocatorClass();
      selectCallback =
          nb::borrow<nb::callable>(instance.attr("select_or_split"));
      usePyPriority = nb::hasattr(instance, "priority");
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
      selectCallback = nb::callable();
      usePyPriority = false;
    }
  }
  auto &MBFI =
      getAnalysis<llvm::MachineBlockFrequencyInfoWrapperPass>().getMBFI();
  auto &LiveStks = getAnalysis<llvm::LiveStacksWrapperLegacy>().getLS();
  auto &MDT = getAnalysis<llvm::MachineDominatorTreeWrapperPass>().getDomTree();

  RegAllocBase::init(getAnalysis<llvm::VirtRegMapWrapperLegacy>().getVRM(),
                     getAnalysis<llvm::LiveIntervalsWrapperPass>().getLIS(),
                     getAnalysis<llvm::LiveRegMatrixWrapperLegacy>().getLRM());
  llvm::VirtRegAuxInfo VRAI(
      *MF, *LIS, *VRM, getAnalysis<llvm::MachineLoopInfoWrapperPass>().getLI(),
      MBFI, &getAnalysis<llvm::ProfileSummaryInfoWrapperPass>().getPSI());
  VRAI.calculateSpillWeightsAndHints();

  SpillerInstance.reset(
      createInlineSpiller({*LIS, LiveStks, MDT, MBFI}, *MF, *VRM, VRAI));

  allocatePhysRegs();
  postOptimization();
  releaseMemory();
  return true;
}

llvm::FunctionPass *createPyRegAlloc() { return new PyRegAlloc(); }

llvm::RegisterRegAlloc pythonRegAlloc("eudsl-python",
                                      "eudsl register allocator",
                                      createPyRegAlloc);

// Stable storage for register_regalloc names: a leaked deque whose element
// c_str() the RegisterRegAlloc node borrows for process lifetime.
std::deque<std::string> &regAllocNames() {
  static auto *names = new std::deque<std::string>();
  return *names;
}

// name -> Python class, held in llvm.mir_strategies._regalloc_classes (Python
// owns it so the classes release at teardown, as the scheduler does).
nb::dict regAllocClasses() {
  return nb::cast<nb::dict>(
      nb::module_::import_("llvm.mir_strategies").attr("_regalloc_classes"));
}

// The live RegisterRegAlloc nodes, leaked so they stay registered for process
// lifetime.
std::vector<std::unique_ptr<llvm::RegisterRegAlloc>> &regAllocRegistryNodes() {
  static auto *nodes =
      new std::vector<std::unique_ptr<llvm::RegisterRegAlloc>>();
  return *nodes;
}

} // namespace

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

// Validate that cls defines select_or_split, record it, and (if new) add a
// RegisterRegAlloc node so emit_object(regalloc="name") can select it.
// Re-registering a name swaps the class.
void registerRegAlloc(const std::string &name, nb::type_object cls) {
  if (!nb::hasattr(cls, "select_or_split")) {
    throw nb::type_error(
        "register allocator class must define select_or_split");
  }
  nb::dict classes = regAllocClasses();
  if (!classes.contains(name.c_str())) {
    regAllocNames().push_back(name);
    const char *cname = regAllocNames().back().c_str();
    regAllocRegistryNodes().push_back(std::make_unique<llvm::RegisterRegAlloc>(
        cname, cname, createPyRegAlloc));
  }
  classes[name.c_str()] = cls;
}

// The class registered under `name`, or an invalid object if `name` was not
// registered via register_regalloc (e.g. a built-in like "eudsl-python").
nb::type_object regAllocClass(const std::string &name) {
  nb::dict classes = regAllocClasses();
  if (classes.contains(name.c_str()))
    return nb::borrow<nb::type_object>(classes[name.c_str()]);
  return nb::type_object();
}

void setActiveRegAllocClass(nb::type_object cls) {
  activeRegAllocClass = std::move(cls);
}
void clearActiveRegAllocClass() { activeRegAllocClass = nb::type_object(); }

// Install / clear the per-thread select callback the allocator reads at
// construction; both touch Python refcounts, so the caller holds the GIL.
void setPendingSelectCallback(nb::callable cb) {
  pendingSelectCallback = std::move(cb);
}
void clearPendingSelectCallback() { pendingSelectCallback = nb::callable(); }

// Diagnostic accessors for the selectOrSplit / spill counters.
unsigned pyRegAllocSelectCount() {
  return selectOrSplitCount.load(std::memory_order_relaxed);
}
void resetPyRegAllocSelectCount() {
  selectOrSplitCount.store(0, std::memory_order_relaxed);
}
unsigned pyRegAllocSpillCount() {
  return spillCount.load(std::memory_order_relaxed);
}
void resetPyRegAllocSpillCount() {
  spillCount.store(0, std::memory_order_relaxed);
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

  // The live range of the one virtual register selectOrSplit is assigning. A
  // select callback receives it alongside the legal candidate physregs; the
  // accessors are read-only.
  nb::class_<llvm::LiveInterval>(m, "LiveInterval")
      .def_prop_ro(
          "reg", [](llvm::LiveInterval &li) { return li.reg().id(); },
          "Id of the virtual register this live interval covers.")
      .def_prop_ro(
          "weight", [](llvm::LiveInterval &li) { return li.weight(); },
          "Spill weight; higher means costlier to spill.")
      .def_prop_ro(
          "is_spillable",
          [](llvm::LiveInterval &li) { return li.isSpillable(); },
          "Whether this interval may be spilled (a finite spill weight).");

  m.def("register_regalloc", &eudsl::registerRegAlloc, "name"_a, "cls"_a,
        "Register a register-allocator class under `name` so "
        "emit_object(regalloc=name) can select it. A fresh instance drives "
        "selectOrSplit per MachineFunction via its select_or_split(self, "
        "live_interval, candidates) method (return a candidate physreg id to "
        "assign, or None to spill). It may also define priority(self, "
        "live_interval) -> float to order the allocation queue (highest first; "
        "defaults to spill weight). Re-registering a name replaces it.");

  m.def(
      "registered_regallocs",
      []() {
        return std::vector<std::string>(regAllocNames().begin(),
                                        regAllocNames().end());
      },
      "Names registered via register_regalloc, selectable with "
      "emit_object(regalloc=...).");

  m.def("_regalloc_select_count", &eudsl::pyRegAllocSelectCount,
        "selectOrSplit calls the eudsl allocator has made; tests use it to "
        "verify the allocator ran when selected.");
  m.def("_reset_regalloc_select_count", &eudsl::resetPyRegAllocSelectCount,
        "Reset the eudsl allocator selectOrSplit counter to zero.");
  m.def("_regalloc_spill_count", &eudsl::pyRegAllocSpillCount,
        "Times the eudsl allocator took its spill branch; tests use it to "
        "verify the spill path runs under high register pressure.");
  m.def("_reset_regalloc_spill_count", &eudsl::resetPyRegAllocSpillCount,
        "Reset the eudsl allocator spill counter to zero.");
}

// Register the pass and its analysis dependencies (verbatim from RABasic's
// INITIALIZE_PASS block). The macros spell their helper names unqualified,
// matching RegAllocBasic.cpp's file-scope `using namespace llvm`.
using namespace llvm;
INITIALIZE_PASS_BEGIN(PyRegAlloc, "eudsl-regalloc-python",
                      "eudsl register allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariablesWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescerLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineSchedulerLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(PyRegAlloc, "eudsl-regalloc-python",
                    "eudsl register allocator", false, false)
