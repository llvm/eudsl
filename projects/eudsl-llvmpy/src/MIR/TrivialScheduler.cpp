// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A minimal pre-RA MachineScheduler strategy registered under the name
// "trivial", so it can be chosen with -misched=trivial (which
// MirModule::emit_object exposes as its `scheduler` argument). It schedules
// top-down and always picks the first ready node, doing no reordering beyond
// what dependency order forces -- a correct baseline, not an optimizing one.
//
// This TU is deliberately separate from the nanobind bindings and is built with
// assertions enabled (-UNDEBUG) to match the prebuilt LLVM it links against;
// see the note in CMakeLists.txt.

#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/ScheduleDAG.h>

#include <atomic>
#include <vector>

namespace {
// Counts pickNode invocations across all TrivialTopDownStrategy instances. It
// exists purely so a test can assert the strategy is actually exercised when
// selected (and not when it is not) -- scheduling is semantics-preserving, so
// the emitted code alone cannot distinguish "trivial ran" from a no-op.
std::atomic<unsigned> trivialPickCount{0};

class TrivialTopDownStrategy : public llvm::MachineSchedStrategy {
  std::vector<llvm::SUnit *> ReadyQ;

public:
  explicit TrivialTopDownStrategy(const llvm::MachineSchedContext *) {}
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
  // Each node is released exactly once, when its predecessors are all
  // scheduled, so a node sitting in the ready queue is never already scheduled;
  // return the first-ready one (front of the queue) and drop it.
  llvm::SUnit *pickNode(bool &IsTopNode) override {
    trivialPickCount.fetch_add(1, std::memory_order_relaxed);
    if (ReadyQ.empty())
      return nullptr;
    IsTopNode = true;
    llvm::SUnit *SU = ReadyQ.front();
    ReadyQ.erase(ReadyQ.begin());
    return SU;
  }
  void schedNode(llvm::SUnit *, bool) override {}
};

llvm::ScheduleDAGInstrs *createTrivialSched(llvm::MachineSchedContext *C) {
  return llvm::createSchedLive<TrivialTopDownStrategy>(C);
}

llvm::MachineSchedRegistry
    trivialSchedRegistry("trivial", "eudsl trivial first-ready scheduler.",
                         createTrivialSched);
} // namespace

namespace eudsl {
// Diagnostic accessors for the pickNode counter above, called from the nanobind
// bindings in PythonCodegen.cpp (a separate translation unit).
unsigned trivialSchedulerPickCount() {
  return trivialPickCount.load(std::memory_order_relaxed);
}
void resetTrivialSchedulerPickCount() {
  trivialPickCount.store(0, std::memory_order_relaxed);
}
} // namespace eudsl
