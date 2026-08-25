// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A pre-RA MachineScheduler strategy that hands the pickNode choice to a Python
// callable, registered in the MachineSchedRegistry as "python" so it can be
// chosen with -misched=python (which MirModule::emit_object drives from its
// `pick` argument): the strategy schedules top-down and, for each ready set,
// asks the callable which node to schedule next. This TU also holds the SUnit
// binding and populate_python_codegen.

#include "IR/Common.h"
#include "MIR/Diagnostics.h"

#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/ScheduleDAG.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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
} // namespace

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
}
