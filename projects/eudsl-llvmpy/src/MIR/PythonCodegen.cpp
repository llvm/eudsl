// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/ScheduleDAG.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

namespace eudsl {
// Defined in TrivialScheduler.cpp: a diagnostic counter of pickNode calls made
// by the "trivial" strategy, so a test can prove it actually runs when
// selected.
unsigned trivialSchedulerPickCount();
void resetTrivialSchedulerPickCount();
// Defined in TrivialRegAlloc.cpp: a diagnostic counter of selectOrSplit calls
// made by the "eudsl-trivial" allocator, so a test can prove it actually runs
// when selected (the fail-if-no-op witness that setDefault took effect).
unsigned pyRegAllocSelectCount();
void resetPyRegAllocSelectCount();
// Also in TrivialRegAlloc.cpp: a diagnostic counter of the spill branch, so a
// test can prove the allocator's spill path runs under high register pressure.
unsigned pyRegAllocSpillCount();
void resetPyRegAllocSpillCount();
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

  // Diagnostic hooks (leading underscore): the trivial scheduler is
  // semantics-preserving, so tests use this counter to confirm it is exercised
  // when selected and left unused otherwise.
  m.def("_trivial_scheduler_pick_count", &eudsl::trivialSchedulerPickCount,
        "Number of pickNode calls the trivial MachineScheduler strategy has "
        "made; used by tests to verify the strategy actually runs.");
  m.def("_reset_trivial_scheduler_pick_count",
        &eudsl::resetTrivialSchedulerPickCount,
        "Reset the trivial MachineScheduler pickNode counter to zero.");
  m.def("_regalloc_select_count", &eudsl::pyRegAllocSelectCount,
        "Number of selectOrSplit calls the trivial register allocator has "
        "made; used by tests to verify the allocator actually runs when "
        "selected via emit_object(regalloc=\"eudsl-trivial\").");
  m.def("_reset_regalloc_select_count", &eudsl::resetPyRegAllocSelectCount,
        "Reset the trivial register allocator selectOrSplit counter to zero.");
  m.def("_regalloc_spill_count", &eudsl::pyRegAllocSpillCount,
        "Number of times the trivial register allocator took its spill branch; "
        "used by a test to verify the spill path runs under high register "
        "pressure.");
  m.def("_reset_regalloc_spill_count", &eudsl::resetPyRegAllocSpillCount,
        "Reset the trivial register allocator spill counter to zero.");
}
