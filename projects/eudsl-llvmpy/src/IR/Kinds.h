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
// return value to a Python type". valueTypeInfo() guards each return with
// pick() so a not-yet-registered class falls back to base Value, keeping every
// commit green while later tasks register the leaves.

#pragma once

#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <nanobind/nanobind.h>

#include <type_traits>
#include <typeinfo>

namespace eudsl {
const std::type_info *typeTypeInfo(llvm::Type::TypeID id);
const std::type_info *valueTypeInfo(unsigned valueID);
} // namespace eudsl

// type_hook is keyed on the *static* pointer type being converted, so a hook on
// llvm::Value alone would not fire when a binding returns e.g. an
// llvm::Instruction* or llvm::User*. These SFINAE partial specializations cover
// every class in the Value and Type hierarchies, downcasting from any base. The
// hook IS consulted for null pointers (an accessor such as Function.entry_block
// returns a null BasicBlock* that nanobind renders as None), so map null to the
// static type T and let nanobind produce None.
template <typename T>
struct nanobind::detail::type_hook<
    T, std::enable_if_t<std::is_base_of_v<llvm::Value, T>, int>> {
  static const std::type_info *get(T *v) {
    return v ? eudsl::valueTypeInfo(static_cast<llvm::Value *>(v)->getValueID())
             : &typeid(T);
  }
};

template <typename T>
struct nanobind::detail::type_hook<
    T, std::enable_if_t<std::is_base_of_v<llvm::Type, T>, int>> {
  static const std::type_info *get(T *t) {
    return t ? eudsl::typeTypeInfo(static_cast<llvm::Type *>(t)->getTypeID())
             : &typeid(T);
  }
};
