// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Value-caster registry in C++, analogous to MLIR's PyGlobals::valueCasterMap.
//
// The C++ type_hook (Kinds.h) already downcasts a returned Value* to its
// concrete bound C++ class (Instruction, ConstantInt, ...). This layer adds a
// user-extensible second step: re-wrapping the same Value* as a Python subclass
// (e.g. the DSL's ArithValue) keyed on the LLVM Type::TypeID.

#include "IR/Common.h"

#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <unordered_map>

namespace {
std::unordered_map<unsigned, nb::object> &casterMap() {
  static std::unordered_map<unsigned, nb::object> map;
  return map;
}
} // namespace

void populate_casters(nb::module_ &m) {
  m.def(
      "register_value_caster",
      [](unsigned typeId, nb::object caster) {
        casterMap()[typeId] = std::move(caster);
      },
      "type_id"_a, "caster"_a,
      "Register a Python type to wrap Values whose Type has the given TypeID.");

  m.def(
      "maybe_downcast",
      [](nb::object value, nb::handle parent) -> nb::object {
        auto *v = nb::cast<llvm::Value *>(value);
        auto it = casterMap().find((unsigned)v->getType()->getTypeID());
        if (it == casterMap().end())
          return value;
        nb::handle p = parent.is_none() ? value : parent;
        return nb::steal(nb::detail::nb_inst_reference(
            (PyTypeObject *)it->second.ptr(), (void *)v, p.ptr()));
      },
      "value"_a, "parent"_a = nb::none(),
      "Re-wrap a Value as its registered caster subclass, if any.");

  m.def("_clear_casters", []() { casterMap().clear(); });

  nb::module_::import_("atexit").attr("register")(
      nb::cpp_function([]() { casterMap().clear(); }));
}
