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

#include <nanobind/nanobind.h>

#include <exception>

namespace nb = nanobind;

namespace eudsl {
// The single definition of the codegen-error stash declared extern in
// Diagnostics.h; scheduler overrides in this TU stash into it.
thread_local std::exception_ptr pendingCodegenError;
} // namespace eudsl

void populate_python_codegen(nb::module_ &m) { (void)m; }
