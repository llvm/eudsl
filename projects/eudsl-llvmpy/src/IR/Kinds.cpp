// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Kinds.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>

namespace eudsl {

const std::type_info *pick(const std::type_info *concrete,
                           const std::type_info *base) {
  return nanobind::detail::nb_type_lookup(concrete) ? concrete : base;
}

const std::type_info *valueTypeInfo(unsigned id) {
  const std::type_info *base = &typeid(llvm::Value);

  if (id >= llvm::Value::InstructionVal) {
    switch (id - llvm::Value::InstructionVal) {
#define HANDLE_INST(num, opcode, Class)                                        \
  case num:                                                                    \
    return pick(&typeid(llvm::Class), base);
#include "llvm/IR/Instruction.def"
    default:
      return &typeid(llvm::Instruction);
    }
  }

  switch (id) {
// MemoryUse/MemoryDef/MemoryPhi are MemorySSA classes, not IR Value subclasses
// reachable from a Module; map their enum slots to base Value.
#define HANDLE_MEMORY_VALUE(Name)                                              \
  case llvm::Value::Name##Val:                                                 \
    return base;
#define HANDLE_VALUE(Name)                                                     \
  case llvm::Value::Name##Val:                                                 \
    return pick(&typeid(llvm::Name), base);
#include "llvm/IR/Value.def"
  default:
    return base;
  }
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
    return &typeid(llvm::FixedVectorType);
  case llvm::Type::ScalableVectorTyID:
    return &typeid(llvm::ScalableVectorType);
  default:
    // Void, Label, Metadata, Token, the float kinds, TypedPointer and
    // TargetExtType have no bound subclass; they stay llvm::Type.
    return &typeid(llvm::Type);
  }
}

} // namespace eudsl
