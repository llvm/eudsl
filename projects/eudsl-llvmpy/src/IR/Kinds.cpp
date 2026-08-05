// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Kinds.h"

#include <llvm/IR/DerivedTypes.h>

namespace eudsl {

const std::type_info *pick(const std::type_info *concrete,
                           const std::type_info *base) {
  return nanobind::detail::nb_type_lookup(concrete) ? concrete : base;
}

const std::type_info *typeTypeInfo(llvm::Type::TypeID id) {
  switch (id) {
  case llvm::Type::IntegerTyID:
    return &typeid(llvm::IntegerType);
  case llvm::Type::FunctionTyID:
    return &typeid(llvm::FunctionType);
  case llvm::Type::PointerTyID:
    return &typeid(llvm::PointerType);
  case llvm::Type::StructTyID:
    return &typeid(llvm::StructType);
  case llvm::Type::ArrayTyID:
    return &typeid(llvm::ArrayType);
  case llvm::Type::FixedVectorTyID:
  case llvm::Type::ScalableVectorTyID:
    return &typeid(llvm::VectorType);
  default:
    // Void, Label, Metadata, Token, the float kinds, TypedPointer and
    // TargetExtType have no bound subclass; they stay llvm::Type.
    return &typeid(llvm::Type);
  }
}

} // namespace eudsl
