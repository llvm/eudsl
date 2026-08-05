// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>

#include "IR/Errors.h"

namespace nb = nanobind;

void populate_context(nb::module_ &m);
void populate_casters(nb::module_ &m);
void populate_attributes(nb::module_ &m);
void populate_types(nb::module_ &m);
void populate_values(nb::module_ &m);
void populate_instructions(nb::module_ &m);
void populate_constants(nb::module_ &m);
void populate_metadata(nb::module_ &m);
void populate_builder(nb::module_ &m);
void populate_passes(nb::module_ &m);
void populate_target(nb::module_ &m);
void populate_linker(nb::module_ &m);
void populate_jit(nb::module_ &m);
void populate_intrinsics(nb::module_ &m);

NB_MODULE(eudslllvm_ext, m) {
  m.doc() = "Hand-written nanobind bindings over the LLVM C++ IR API.";
  eudsl::registerExceptions(m);
  populate_context(m);
  populate_casters(m);
  populate_attributes(m);
  nb::module_ types = m.def_submodule("types");
  populate_types(types);
  populate_values(m);
  populate_instructions(m);
  populate_constants(m);
  populate_metadata(m);
  populate_builder(m);
  populate_passes(m);
  populate_target(m);
  populate_linker(m);
  populate_jit(m);
  populate_intrinsics(m);
}
