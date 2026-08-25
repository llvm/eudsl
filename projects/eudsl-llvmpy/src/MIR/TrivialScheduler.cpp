// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Two pre-RA MachineScheduler strategies, both registered in the
// MachineSchedRegistry so they can be chosen with -misched=<name> (which
// MirModule::emit_object exposes as its `scheduler` argument):
//
//   "trivial" -- schedules top-down and always picks the first ready node,
//     doing no reordering beyond what dependency order forces; a correct
//     baseline, not an optimizing one.
//   "python"  -- same top-down shape, but pickNode delegates the choice to a
//     user-provided Python callable (see PyMachineSchedStrategy). emit_object
//     selects it via its `pick` argument and installs the callable.
//
// This TU is deliberately separate from the nanobind bindings and is built with
// assertions enabled (-UNDEBUG) to match the prebuilt LLVM it links against;
// see the note in CMakeLists.txt.

#include "MIR/Diagnostics.h"

#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/ScheduleDAG.h>

#include <nanobind/nanobind.h>

#include <algorithm>
#include <atomic>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace eudsl {
// Definition of the codegen error stash declared in Diagnostics.h (extern there
// so the two MIR TUs share one instance). The pickNode trampoline below writes
// it; runCodegenPipeline (in Diagnostics.h, used by emit_object) reads and
// re-raises it.
thread_local std::exception_ptr pendingCodegenError;
} // namespace eudsl

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

// The Python callable the python-backed strategy reads at construction. It is
// per-thread: emit_object installs it (under the GIL) for the duration of one
// emission pipeline run and clears it afterward, so every strategy instance the
// run constructs (one per MachineFunction) copies the same callable, and it
// never leaks into a later, callback-less emit. There is no lock -- this relies
// on the GIL serializing emit_object callers, matching the process-global
// -misched / -start-after option handling in Machine.cpp.
thread_local nb::callable pendingPickCallback;

// A pre-RA MachineScheduler strategy that hands the choice to a Python
// callable. It keeps the trivial strategy's shape -- top-down only, no pressure
// tracking, a single ready-set container fed by releaseTopNode -- but pickNode
// marshals the ready SUnits to the callable and uses the node it returns. The
// callable is stored as an nb::callable member so its refcount is managed for
// us (as the IR-pass PyFunctionPass does), copied from pendingPickCallback at
// construction.
class PyMachineSchedStrategy : public llvm::MachineSchedStrategy {
  std::vector<llvm::SUnit *> ReadyQ;
  nb::callable pickCallback;

public:
  explicit PyMachineSchedStrategy(const llvm::MachineSchedContext *)
      : pickCallback(pendingPickCallback) {}
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
  // not via `pick`) we keep the native first-ready choice so the schedule stays
  // legal. A callable that raises, or returns something that is not one of the
  // presented ready nodes, has its exception stashed in
  // eudsl::pendingCodegenError; we then fall back to the native first-ready
  // node so this call (and the rest of the unskippable codegen pipeline)
  // returns a legal node, and runCodegenPipeline re-raises the stashed
  // exception after the run.
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
} // namespace

namespace eudsl {
// Install / clear the per-thread pick callback the python strategy reads at
// construction. Called from emit_object (a nanobind TU) around the emission
// pipeline run; both touch Python refcounts, so the caller holds the GIL.
void setPendingPickCallback(nb::callable cb) {
  pendingPickCallback = std::move(cb);
}
void clearPendingPickCallback() { pendingPickCallback = nb::callable(); }
} // namespace eudsl

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
