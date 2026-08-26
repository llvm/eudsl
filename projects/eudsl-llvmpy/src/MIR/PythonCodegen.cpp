// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Python-subclassable pre-RA MachineScheduler strategy.
// mir.MachineSchedStrategy binds llvm::MachineSchedStrategy via a trampoline;
// register_scheduler adds a MachineSchedRegistry node so
// emit_object(scheduler="name") can select it. LLVM owns an OwningPyStrategy
// adaptor that forwards into the Python instance, since nanobind won't move a
// Python-created instance into a default-deleter unique_ptr.

#include "IR/Common.h"
#include "MIR/Diagnostics.h"

#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/ScheduleDAG.h>

#include <nanobind/nanobind.h>

#include <exception>

namespace nb = nanobind;

namespace eudsl {
// The single definition of the codegen-error stash declared extern in
// Diagnostics.h; scheduler overrides in this TU stash into it.
thread_local std::exception_ptr pendingCodegenError;
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
}
