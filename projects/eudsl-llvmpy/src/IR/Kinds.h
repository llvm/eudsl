// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Downcasting support for llvm::Type. llvm::Type is deliberately
// non-polymorphic (Type.h has no virtual destructor), so nanobind's RTTI-based
// downcast cannot see through a base pointer. nanobind::detail::type_hook is
// the documented hook for this case. The llvm::Value analogue (valueTypeInfo
// and type_hook<llvm::Value>) lives with the Value bindings, the PR that
// registers the leaf Value classes it names.
//
// INVARIANT: every std::type_info returned here must name a class registered
// with nanobind, otherwise the conversion raises "Unable to convert function
// return value to a Python type". typeTypeInfo only names classes registered in
// this PR.

#pragma once

#include <llvm/IR/Type.h>

#include <nanobind/nanobind.h>

#include <cassert>
#include <typeinfo>

namespace eudsl {
const std::type_info *typeTypeInfo(llvm::Type::TypeID id);
} // namespace eudsl

template <> struct nanobind::detail::type_hook<llvm::Type> {
  static const std::type_info *get(llvm::Type *t) {
    // nanobind resolves a null pointer to None before consulting the hook, so
    // a non-null argument is an invariant here.
    assert(t && "type_hook<llvm::Type> invoked with a null pointer");
    return eudsl::typeTypeInfo(t->getTypeID());
  }
};
