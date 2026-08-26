// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The Python-driven codegen bindings and the passes they expose: a pre-RA
// MachineScheduler strategy and a register allocator that hand their choices to
// Python callables, the SUnit binding those callbacks receive, and
// populate_python_codegen which wires the bindings into the module.
//
// PyMachineSchedStrategy is registered in the MachineSchedRegistry as "python"
// so it can be chosen with -misched=python (which MirModule::emit_object drives
// from its `scheduler`/`pick` arguments): the strategy schedules top-down and,
// for each ready set, asks the callable which node to schedule next.
//
// PyRegAlloc is the register allocator, selectable through
// emit_object(regalloc="eudsl-python"). It is a MachineFunctionPass built on
// LLVM's RegAllocBase driver (the same skeleton RABasic uses): the driver seeds
// the priority queue with the live intervals to allocate and repeatedly calls
// selectOrSplit. When a Python select callable is installed, selectOrSplit
// routes each choice through it; otherwise it assigns the first physical
// register in the allocation order that does not interfere, spilling only when
// none is free. That native first-free/spill policy is also the callable's
// required legal fallback. It is registered in the RegisterRegAlloc registry
// under "eudsl-python", which emit_object selects with
// RegisterRegAlloc::setDefault -- read by TargetPassConfig::createRegAllocPass
// when addPassesToEmitFile builds the codegen pipeline. The allocator pulls in
// the two vendored private CodeGen headers (RegAllocBase.h, AllocationOrder.h)
// that RABasic depends on but that the LLVM distro does not ship.

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
#include <llvm/CodeGen/MachineBlockFrequencyInfo.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineLoopInfo.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/CodeGen/ScheduleDAG.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/Spiller.h>
#include <llvm/CodeGen/VirtRegMap.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/PassRegistry.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace llvm {
// Defined by the INITIALIZE_PASS block below; declared here so the constructor
// can register the pass's PassInfo (idempotently) when an instance is created,
// so the legacy pass manager can resolve the analyses selectOrSplit reads.
void initializePyRegAllocPass(PassRegistry &);
} // namespace llvm

namespace nb = nanobind;

namespace {
// The Python callable the python-backed strategy reads at construction. It is
// per-thread: emit_object installs it (under the GIL) for the duration of one
// emission pipeline run and clears it afterward, so every strategy instance the
// run constructs (one per MachineFunction) copies the same callable, and it
// never leaks into a later, callback-less emit. There is no lock -- this relies
// on the GIL serializing emit_object callers, matching the process-global
// -misched / -start-after option handling in Machine.cpp.
thread_local nb::callable pendingPickCallback;

// A pre-RA MachineScheduler strategy that hands the choice to a Python
// callable. It is top-down only with no pressure tracking, a single ready-set
// container fed by releaseTopNode; pickNode marshals the ready SUnits to the
// callable and uses the node it returns. The callable is stored as an
// nb::callable member so its refcount is managed for us (as the IR-pass
// PyFunctionPass does), copied from pendingPickCallback at construction.
class PyMachineSchedStrategy : public llvm::MachineSchedStrategy {
  std::vector<llvm::SUnit *> ReadyQ;
  nb::callable pickCallback;

public:
  explicit PyMachineSchedStrategy(const llvm::MachineSchedContext *)
      : pickCallback(pendingPickCallback) {}
  // Force top-down scheduling and skip register-pressure tracking: this
  // strategy only releases and picks top nodes, so ScheduleDAGMI must not run
  // the bottom-up direction (it would find an empty ready queue).
  llvm::MachineSchedPolicy getPolicy() const override {
    llvm::MachineSchedPolicy Policy;
    Policy.OnlyTopDown = true;
    Policy.ShouldTrackPressure = false;
    return Policy;
  }
  bool shouldTrackPressure() const override { return false; }
  void initialize(llvm::ScheduleDAGMI *) override { ReadyQ.clear(); }
  void releaseTopNode(llvm::SUnit *SU) override { ReadyQ.push_back(SU); }
  void releaseBottomNode(llvm::SUnit *) override {}
  // Present the ready set to the Python callable and schedule the node it
  // picks. The callable receives a list[SUnit] and returns one element; we
  // accept its choice only if it is one of the presented nodes, by pointer
  // identity. With no callable installed (the strategy was selected by name,
  // not via `pick`) we keep the native first-ready choice. A callable that
  // raises, or returns something that is not one of the presented ready nodes,
  // has its exception stashed in eudsl::pendingCodegenError; we fall back to
  // the native first-ready node so this call (and the rest of the unskippable
  // codegen pipeline) returns a legal node, and runCodegenPipeline re-raises
  // the stashed exception after the run.
  llvm::SUnit *pickNode(bool &IsTopNode) override {
    if (ReadyQ.empty())
      return nullptr;
    IsTopNode = true;
    llvm::SUnit *chosen = nullptr;
    // Once a callback has stashed an error, stop invoking Python; the remaining
    // pickNode calls just drain the ready queue in first-ready order so the
    // required pipeline winds down to runCodegenPipeline's re-raise.
    if (pickCallback && !eudsl::pendingCodegenError) {
      // The GIL guard is intentionally outside the try: the catch stashes the
      // exception with std::current_exception(), which for an nb::python_error
      // touches Python refcounts and so must run while the GIL is held. Its
      // construction does not raise, so nothing is lost by leaving it uncaught.
      nb::gil_scoped_acquire gil;
      try {
        nb::list ready;
        for (llvm::SUnit *SU : ReadyQ)
          ready.append(nb::cast(SU, nb::rv_policy::reference));
        nb::object choice = pickCallback(ready);
        llvm::SUnit *returned = nullptr;
        if (nb::try_cast<llvm::SUnit *>(choice, returned)) {
          for (llvm::SUnit *SU : ReadyQ) {
            if (SU == returned) {
              chosen = returned;
              break;
            }
          }
        }
        if (!chosen) {
          throw nb::value_error(
              "scheduler pickNode returned a value that is not one of the "
              "ready nodes");
        }
      } catch (...) {
        // Do not let the exception unwind through LLVM's -fno-exceptions
        // frames; stash it and fall through to the native first-ready node so
        // the pipeline winds down to runCodegenPipeline's re-raise.
        eudsl::pendingCodegenError = std::current_exception();
        chosen = nullptr;
      }
    }
    if (!chosen)
      chosen = ReadyQ.front();
    ReadyQ.erase(std::find(ReadyQ.begin(), ReadyQ.end(), chosen));
    return chosen;
  }
  void schedNode(llvm::SUnit *, bool) override {}
};

llvm::ScheduleDAGInstrs *createPythonSched(llvm::MachineSchedContext *C) {
  return llvm::createSchedLive<PyMachineSchedStrategy>(C);
}

llvm::MachineSchedRegistry
    pythonSchedRegistry("python", "eudsl Python-callable-driven scheduler.",
                        createPythonSched);

// Counts selectOrSplit invocations across all PyRegAlloc instances. It exists
// purely so a test can assert the allocator is actually exercised when selected
// (and not when it is not) -- register allocation is semantics-preserving, so
// the emitted code alone cannot distinguish "the eudsl allocator ran" from
// the target default having run. It is also the fail-if-no-op witness that
// RegisterRegAlloc::setDefault took effect when the pipeline was built.
std::atomic<unsigned> selectOrSplitCount{0};

// Counts the times selectOrSplit takes the spill branch (a vreg could not be
// assigned a free physreg). A companion to selectOrSplitCount so a test can
// prove the spill path -- authored logic here, not in the driver -- actually
// runs when register pressure exceeds the allocatable set.
std::atomic<unsigned> spillCount{0};

// The Python callable the allocator routes selectOrSplit through, when one is
// installed. It is per-thread: emit_object installs it (under the GIL) for the
// duration of one emission pipeline run and clears it afterward, so every
// PyRegAlloc instance the run constructs copies the same callable and it never
// leaks into a later, callback-less emit. There is no lock -- this relies on
// the GIL serializing emit_object callers, matching the process-global
// RegisterRegAlloc default handling in Machine.cpp.
thread_local nb::callable pendingSelectCallback;

// Order the priority queue by ascending spill weight, with the register number
// as a stable tie-breaker for deterministic ordering (copied from RABasic's
// CompSpillWeight).
struct CompSpillWeight {
  bool operator()(const llvm::LiveInterval *A,
                  const llvm::LiveInterval *B) const {
    return std::tuple(A->weight(), A->reg()) <
           std::tuple(B->weight(), B->reg());
  }
};

class PyRegAlloc : public llvm::MachineFunctionPass,
                   public llvm::RegAllocBase,
                   // LiveRangeEdit needs a delegate to notify on erase/shrink
                   // of a vreg during spilling. We inherit it but keep the
                   // base's no-op defaults (unlike RABasic, which overrides
                   // them): RABasic reassigns interfering vregs, so it must
                   // unassign them from the Matrix and re-enqueue shrunk ones.
                   // This policy only ever spills VirtReg itself (it never
                   // touches interferences), so no already-assigned interval is
                   // erased or shrunk and the Matrix/Queue never hold stale
                   // entries -- the spill test JIT-executes correctly with the
                   // defaults, confirming the overrides are unnecessary here.
                   private llvm::LiveRangeEdit::Delegate {
  // The function currently being allocated (set in runOnMachineFunction).
  llvm::MachineFunction *MF = nullptr;

  std::unique_ptr<llvm::Spiller> SpillerInstance;
  std::priority_queue<const llvm::LiveInterval *,
                      std::vector<const llvm::LiveInterval *>, CompSpillWeight>
      Queue;

  // The Python callable driving selectOrSplit, copied from
  // pendingSelectCallback at construction so its refcount is managed for us (as
  // the IR-pass PyFunctionPass and the python scheduler strategy do). Empty
  // when no callback is installed, in which case selectOrSplit uses the native
  // first-free policy.
  nb::callable selectCallback;

public:
  static char ID;

  PyRegAlloc()
      : llvm::MachineFunctionPass(ID), llvm::RegAllocBase(),
        selectCallback(pendingSelectCallback) {
    llvm::initializePyRegAllocPass(*llvm::PassRegistry::getPassRegistry());
  }

  llvm::StringRef getPassName() const override {
    return "eudsl register allocator";
  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  void releaseMemory() override { SpillerInstance.reset(); }

  llvm::Spiller &spiller() override { return *SpillerInstance; }

  void enqueueImpl(const llvm::LiveInterval *LI) override { Queue.push(LI); }

  const llvm::LiveInterval *dequeue() override {
    if (Queue.empty())
      return nullptr;
    const llvm::LiveInterval *LI = Queue.top();
    Queue.pop();
    return LI;
  }

  llvm::MCRegister
  selectOrSplit(const llvm::LiveInterval &VirtReg,
                llvm::SmallVectorImpl<llvm::Register> &SplitVRegs) override;

  // Spill VirtReg itself (the native no-reassignment policy), returning 0 to
  // tell the driver the vreg was replaced by the spill/reload vregs appended to
  // SplitVRegs. Shared by the native path and the callback's spill signal.
  llvm::MCRegister
  spillVirtReg(const llvm::LiveInterval &VirtReg,
               llvm::SmallVectorImpl<llvm::Register> &SplitVRegs);

  bool runOnMachineFunction(llvm::MachineFunction &mf) override;

  // Mirror RABasic: the incoming MIR is post-selection (no PHIs) and the
  // allocator materializes physical registers, clearing SSA form.
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

// The allocation heuristic. When a Python callback is installed, present the
// legal (non-interfering) candidate physregs for VirtReg to it as a list[int]
// of physreg ids and honor its return: an id from that set is assigned; None
// (or any non-id value that is not a candidate) is treated below. This step
// assumes a well-behaved callback -- a return that is neither a candidate id
// nor None falls back to the native policy rather than raising. With no
// callback, or on that fallback, assign the first physical register in
// VirtReg's allocation order that does not interfere; if none is free, spill
// VirtReg. The driver (allocatePhysRegs) calls Matrix->assign for a returned
// physreg, and re-enqueues any new virtual registers appended to SplitVRegs.
llvm::MCRegister
PyRegAlloc::selectOrSplit(const llvm::LiveInterval &VirtReg,
                          llvm::SmallVectorImpl<llvm::Register> &SplitVRegs) {
  selectOrSplitCount.fetch_add(1, std::memory_order_relaxed);
  auto Order =
      llvm::AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);

  // The legal (non-interfering) candidates, in allocation order. Presented to
  // the callback and reused as the native first-free fallback below.
  llvm::SmallVector<llvm::MCRegister, 16> candidateRegs;
  for (llvm::MCRegister PhysReg : Order) {
    assert(PhysReg.isValid());
    if (Matrix->checkInterference(VirtReg, PhysReg) ==
        llvm::LiveRegMatrix::IK_Free)
      candidateRegs.push_back(PhysReg);
  }

  // Once a callback has stashed an error, stop invoking Python; the remaining
  // selectOrSplit calls just assign first-free (or spill) so the required
  // pipeline winds down to runCodegenPipeline's re-raise.
  if (selectCallback && !eudsl::pendingCodegenError) {
    // The GIL guard is intentionally outside the try: the catch stashes the
    // exception with std::current_exception(), which for an nb::python_error
    // touches Python refcounts and so must run while the GIL is held. Its
    // construction does not raise, so nothing is lost by leaving it uncaught.
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
        // A chosen physreg id: honor it only if it is one of the presented
        // candidates (the driver assigns it for us).
        for (llvm::MCRegister PhysReg : candidateRegs) {
          if (PhysReg.id() == chosenId)
            return PhysReg;
        }
      }
      // Not None and not one of the presented candidates: an illegal choice.
      throw nb::value_error("selectOrSplit returned a register that is not one "
                            "of the legal candidates");
    } catch (...) {
      // Do not let the exception unwind through LLVM's -fno-exceptions frames;
      // stash it and fall through to a legal native assignment so the pipeline
      // winds down to runCodegenPipeline's re-raise.
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  if (!candidateRegs.empty())
    return candidateRegs.front();
  return spillVirtReg(VirtReg, SplitVRegs);
}

// No free physreg (or the callback asked to spill): spill VirtReg itself (never
// an interfering vreg -- this policy does no reassignment), which the
// driver replaces with the new spill/reload vregs appended to SplitVRegs. Only
// an unspillable vreg (infinite spill weight) fails to allocate; a spillable
// one always spills to make forward progress.
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

// The analysis dependency set, copied verbatim from RABasic's getAnalysisUsage
// (omitting one yields a null-analysis crash when selectOrSplit / the driver
// reads it). setPreservesCFG and the MachineFunctionPass chain-up are required.
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

// Wire up the RegAllocBase driver, copied from RABasic::runOnMachineFunction
// (analysis-wrapper accessor names as spelled in this LLVM): fetch the
// analyses, init the base, compute spill weights, build the inline spiller,
// then run the allocation driver and clean up.
bool PyRegAlloc::runOnMachineFunction(llvm::MachineFunction &mf) {
  MF = &mf;
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
} // namespace

// Register the pass and its analysis dependencies (verbatim from RABasic's
// INITIALIZE_PASS block, dependency wrapper names as spelled in this LLVM). The
// INITIALIZE_PASS macros spell their helper names (PassInfo, callDefaultCtor,
// ...) unqualified, matching RegAllocBasic.cpp's file-scope `using namespace
// llvm`, so pull the namespace in here for the macro expansion.
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

namespace eudsl {
// The one definition of the codegen-error stash slot declared extern in
// MIR/Diagnostics.h; this TU holds the scheduler's pickNode trampoline that
// stashes into it.
thread_local std::exception_ptr pendingCodegenError;

// Install / clear the per-thread pick callback the python strategy reads at
// construction. Called from emit_object around the emission pipeline run; both
// touch Python refcounts, so the caller holds the GIL.
void setPendingPickCallback(nb::callable cb) {
  pendingPickCallback = std::move(cb);
}
void clearPendingPickCallback() { pendingPickCallback = nb::callable(); }

// Install / clear the per-thread select callback the allocator reads at
// construction. Called from emit_object around the emission pipeline run; both
// touch Python refcounts, so the caller holds the GIL.
void setPendingSelectCallback(nb::callable cb) {
  pendingSelectCallback = std::move(cb);
}
void clearPendingSelectCallback() { pendingSelectCallback = nb::callable(); }

// Diagnostic accessors for the selectOrSplit / spill counters above, called
// from the m.def bindings in populate_python_codegen below.
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
  // llvm::SUnit -- one scheduling unit (a MachineInstr plus its dependency
  // edges) the pre-RA MachineScheduler orders. A python `pick` callback
  // receives the ready SUnits as a list[SUnit] and returns the one to schedule
  // next, matched back by pointer identity. The read-only accessors below
  // expose the node's identity and readiness so the callback can base its
  // choice on the scheduler's state.
  nb::class_<llvm::SUnit>(m, "SUnit")
      .def_prop_ro(
          "node_num", [](llvm::SUnit &su) { return su.NodeNum; },
          "Entry number of this node in the DAG's node vector.")
      .def_prop_ro(
          "is_top_ready", [](llvm::SUnit &su) { return su.isTopReady(); },
          "Whether all predecessors are scheduled (ready for top-down "
          "scheduling).")
      .def_prop_ro(
          "is_bottom_ready", [](llvm::SUnit &su) { return su.isBottomReady(); },
          "Whether all successors are scheduled (ready for bottom-up "
          "scheduling).")
      .def_prop_ro(
          "instr", [](llvm::SUnit &su) { return su.getInstr(); },
          nb::rv_policy::reference_internal,
          "The representative MachineInstr this scheduling unit wraps.");

  // llvm::LiveInterval -- the live range of the one virtual register the
  // register allocator's selectOrSplit is assigning. A python `select` callback
  // receives it alongside the legal candidate physregs so it can base its
  // choice on the vreg's identity and spill cost. The accessors below are
  // read-only.
  nb::class_<llvm::LiveInterval>(m, "LiveInterval")
      .def_prop_ro(
          "reg", [](llvm::LiveInterval &li) { return li.reg().id(); },
          "Id of the virtual register this live interval covers.")
      .def_prop_ro(
          "weight", [](llvm::LiveInterval &li) { return li.weight(); },
          "The spill weight computed for this interval; a higher weight means "
          "it is costlier to spill.")
      .def_prop_ro(
          "is_spillable",
          [](llvm::LiveInterval &li) { return li.isSpillable(); },
          "Whether this interval may be spilled (a finite spill weight).");

  m.def(
      "registered_schedulers",
      []() {
        std::vector<std::string> names;
        for (llvm::MachineSchedRegistry *node =
                 llvm::MachineSchedRegistry::getList();
             node; node = node->getNext())
          names.emplace_back(node->getName().str());
        return names;
      },
      "Names of the pre-RA MachineScheduler strategies registered in this "
      "extension, selectable via emit_object(scheduler=...).");

  m.def("_regalloc_select_count", &eudsl::pyRegAllocSelectCount,
        "Number of selectOrSplit calls the eudsl register allocator has "
        "made; used by tests to verify the allocator actually runs when "
        "selected via emit_object(regalloc=\"eudsl-python\").");
  m.def("_reset_regalloc_select_count", &eudsl::resetPyRegAllocSelectCount,
        "Reset the eudsl register allocator selectOrSplit counter to zero.");
  m.def("_regalloc_spill_count", &eudsl::pyRegAllocSpillCount,
        "Number of times the eudsl register allocator took its spill branch; "
        "used by a test to verify the spill path runs under high register "
        "pressure.");
  m.def("_reset_regalloc_spill_count", &eudsl::resetPyRegAllocSpillCount,
        "Reset the eudsl register allocator spill counter to zero.");
}
