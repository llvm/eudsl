// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A native, trivial-but-correct register allocator, selectable through
// MirModule::emit_object(regalloc="eudsl-trivial"). PyRegAlloc is a
// MachineFunctionPass built on LLVM's RegAllocBase driver (the same skeleton
// RABasic uses): the driver seeds the priority queue with the live intervals to
// allocate and repeatedly calls selectOrSplit; our selectOrSplit assigns the
// first physical register in the allocation order that does not interfere,
// spilling only when none is free. It is a correct baseline, not an optimizing
// allocator.
//
// It is registered in the RegisterRegAlloc registry under "eudsl-trivial", and
// emit_object selects it with RegisterRegAlloc::setDefault, which
// TargetPassConfig::createRegAllocPass reads when addPassesToEmitFile builds
// the codegen pipeline.
//
// This TU is deliberately separate from the nanobind bindings (mirroring
// TrivialScheduler.cpp) and is built with assertions enabled (-UNDEBUG) to
// match the prebuilt LLVM it links against; see the note in CMakeLists.txt. It
// also includes the two vendored private CodeGen headers (RegAllocBase.h,
// AllocationOrder.h) that RABasic depends on.

#include "MIR/AllocationOrder.h"
#include "MIR/RegAllocBase.h"

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/CodeGen/CalcSpillWeights.h>
#include <llvm/CodeGen/LiveDebugVariables.h>
#include <llvm/CodeGen/LiveIntervals.h>
#include <llvm/CodeGen/LiveRangeEdit.h>
#include <llvm/CodeGen/LiveRegMatrix.h>
#include <llvm/CodeGen/LiveStacks.h>
#include <llvm/CodeGen/MachineBlockFrequencyInfo.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineLoopInfo.h>
#include <llvm/CodeGen/Passes.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/CodeGen/Spiller.h>
#include <llvm/CodeGen/VirtRegMap.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/PassRegistry.h>

#include <atomic>
#include <memory>
#include <queue>
#include <tuple>
#include <vector>

namespace llvm {
// Defined by the INITIALIZE_PASS block below; declared here so the constructor
// can register the pass's PassInfo (idempotently) when an instance is created,
// so the legacy pass manager can resolve the analyses selectOrSplit reads.
void initializePyRegAllocPass(PassRegistry &);
} // namespace llvm

namespace {
// Counts selectOrSplit invocations across all PyRegAlloc instances. It exists
// purely so a test can assert the allocator is actually exercised when selected
// (and not when it is not) -- register allocation is semantics-preserving, so
// the emitted code alone cannot distinguish "the trivial allocator ran" from
// the target default having run. It is also the fail-if-no-op witness that
// RegisterRegAlloc::setDefault took effect when the pipeline was built.
std::atomic<unsigned> selectOrSplitCount{0};

// Counts the times selectOrSplit takes the spill branch (a vreg could not be
// assigned a free physreg). A companion to selectOrSplitCount so a test can
// prove the spill path -- authored logic here, not in the driver -- actually
// runs when register pressure exceeds the allocatable set.
std::atomic<unsigned> spillCount{0};

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

class PyRegAlloc
    : public llvm::MachineFunctionPass,
      public llvm::RegAllocBase,
      // LiveRangeEdit needs a delegate to notify on erase/shrink of
      // a vreg during spilling. We inherit it but keep the base's
      // no-op defaults (unlike RABasic, which overrides them):
      // RABasic reassigns interfering vregs, so it must unassign
      // them from the Matrix and re-enqueue shrunk ones. This
      // trivial policy only ever spills VirtReg itself (it never
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

public:
  static char ID;

  PyRegAlloc() : llvm::MachineFunctionPass(ID), llvm::RegAllocBase() {
    llvm::initializePyRegAllocPass(*llvm::PassRegistry::getPassRegistry());
  }

  llvm::StringRef getPassName() const override {
    return "eudsl trivial register allocator";
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

// The allocation heuristic. Assign the first physical register in VirtReg's
// allocation order that does not interfere; if none is free, spill VirtReg
// (unless it is unspillable, which is an allocation failure). The driver
// (allocatePhysRegs) calls Matrix->assign for a returned physreg, and
// re-enqueues any new virtual registers appended to SplitVRegs.
llvm::MCRegister
PyRegAlloc::selectOrSplit(const llvm::LiveInterval &VirtReg,
                          llvm::SmallVectorImpl<llvm::Register> &SplitVRegs) {
  selectOrSplitCount.fetch_add(1, std::memory_order_relaxed);
  auto Order =
      llvm::AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  for (llvm::MCRegister PhysReg : Order) {
    assert(PhysReg.isValid());
    if (Matrix->checkInterference(VirtReg, PhysReg) ==
        llvm::LiveRegMatrix::IK_Free) {
      return PhysReg;
    }
  }
  // No free physreg: spill VirtReg itself (never an interfering vreg -- this
  // trivial policy does no reassignment), which the driver replaces with the
  // new spill/reload vregs appended to SplitVRegs. Only an unspillable vreg
  // (infinite spill weight) fails to allocate; a spillable one always spills to
  // make forward progress.
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

llvm::RegisterRegAlloc trivialRegAlloc("eudsl-trivial",
                                       "eudsl trivial register allocator",
                                       createPyRegAlloc);
} // namespace

// Register the pass and its analysis dependencies (verbatim from RABasic's
// INITIALIZE_PASS block, dependency wrapper names as spelled in this LLVM). The
// INITIALIZE_PASS macros spell their helper names (PassInfo, callDefaultCtor,
// ...) unqualified, matching RegAllocBasic.cpp's file-scope `using namespace
// llvm`, so pull the namespace in here for the macro expansion.
using namespace llvm;
INITIALIZE_PASS_BEGIN(PyRegAlloc, "eudsl-regalloc-trivial",
                      "eudsl trivial register allocator", false, false)
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
INITIALIZE_PASS_END(PyRegAlloc, "eudsl-regalloc-trivial",
                    "eudsl trivial register allocator", false, false)

namespace eudsl {
// Diagnostic accessors for the selectOrSplit counter above, called from the
// nanobind bindings in PythonCodegen.cpp (a separate translation unit).
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
