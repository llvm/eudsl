// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Stateless support for the Python value-caster layer (see llvm/dsl/casters.py),
// which is analogous to MLIR's register_value_caster / PyValue::maybeDownCast.
//
// The C++ type_hook (Kinds.h) already downcasts a returned llvm::Value* to its
// concrete *bound C++* class (Instruction, ConstantInt, ...). The Python caster
// layer adds a second, user-extensible step: re-wrapping the same Value* as a
// Python subclass (e.g. the DSL's ArithValue).
//
// This file deliberately holds NO Python references at static scope -- the
// caster registry lives in Python -- so nothing leaks at interpreter shutdown.
// The one primitive C++ must provide is nb::inst_reference, which binds an
// existing C++ pointer into a chosen Python type with its lifetime tied to a
// parent object (nanobind offers no Python-level equivalent).

#include "IR/Common.h"

#include <llvm/IR/Value.h>

void populate_casters(nb::module_ &m) {
  m.def(
      "_wrap_value_as",
      [](llvm::Value *v, nb::handle pyType, nb::handle parent) -> nb::object {
        return nb::steal(nb::detail::nb_inst_reference(
            (PyTypeObject *)pyType.ptr(), (void *)v, parent.ptr()));
      },
      "value"_a, "py_type"_a, "parent"_a = nb::none(),
      "Re-wrap an existing Value* as the given Python (sub)type, tying its "
      "lifetime to `parent`. Backs llvm.dsl.casters.maybe_downcast.");
}
