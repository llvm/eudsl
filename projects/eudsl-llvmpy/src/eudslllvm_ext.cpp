// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void populate_context(nb::module_ &m);
void populate_types(nb::module_ &m);
void populate_values(nb::module_ &m);
void populate_instructions(nb::module_ &m);
void populate_constants(nb::module_ &m);
void populate_builder(nb::module_ &m);

NB_MODULE(eudslllvm_ext, m) {
  m.doc() = "Hand-written nanobind bindings over the LLVM C++ IR API.";
  populate_context(m);
  nb::module_ types = m.def_submodule("types");
  populate_types(types);
  populate_values(m);
  populate_instructions(m);
  populate_constants(m);
  populate_builder(m);
}
