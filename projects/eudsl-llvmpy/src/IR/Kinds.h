// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Downcasting support. llvm::Value and llvm::Type are deliberately
// non-polymorphic (Value.h documents the non-virtual destructor), so nanobind's
// RTTI-based downcast cannot see through a base pointer.
// nanobind::detail::type_hook is the documented hook for this case.
//
// INVARIANT: every std::type_info returned here must name a class registered
// with nanobind, otherwise the conversion raises "Unable to convert function
// return value to a Python type". valueTypeInfo() guards each return with pick()
// so a not-yet-registered class falls back to base Value, keeping every commit
// green while later tasks register the leaves.

#pragma once

#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <nanobind/nanobind.h>

#include <typeinfo>

namespace eudsl {
const std::type_info *typeTypeInfo(llvm::Type::TypeID id);
const std::type_info *valueTypeInfo(unsigned valueID);
} // namespace eudsl

template <> struct nanobind::detail::type_hook<llvm::Type> {
  static const std::type_info *get(llvm::Type *t) {
    // The hook IS consulted for null pointers (a null-returning accessor that
    // nanobind renders as None), so map null to the base type.
    return t ? eudsl::typeTypeInfo(t->getTypeID()) : &typeid(llvm::Type);
  }
};

template <> struct nanobind::detail::type_hook<llvm::Value> {
  static const std::type_info *get(llvm::Value *v) {
    return v ? eudsl::valueTypeInfo(v->getValueID()) : &typeid(llvm::Value);
  }
};
