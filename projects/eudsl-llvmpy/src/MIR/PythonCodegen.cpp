// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

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
} // namespace eudsl

void populate_python_codegen(nb::module_ &m) {
  // llvm::SUnit -- one scheduling unit (a MachineInstr plus its dependency
  // edges) the pre-RA MachineScheduler orders. Bound opaque: a python `pick`
  // callback receives the ready SUnits as a list[SUnit] and returns the one to
  // schedule next, matched back by pointer identity. No fields are exposed yet.
  nb::class_<llvm::SUnit>(m, "SUnit");

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
}
