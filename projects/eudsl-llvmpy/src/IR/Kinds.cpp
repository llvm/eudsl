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

// `id` is an llvm::Value::getValueID() result, which is an `unsigned` rather
// than the Value::ValueTy enum: instruction IDs are `InstructionVal + opcode`
// and run past the ValueTy enumerators, so `unsigned` is what the LLVM API
// hands back here.
const std::type_info *valueTypeInfo(unsigned id) {
  const std::type_info *base = &typeid(llvm::Value);

  // A .def-generated dispatch over every LLVM value kind / instruction opcode.
  // The case bodies ARE exercised (the tests convert many Values and assert
  // their concrete Python class); llvm-cov confirms nonzero hits on the `return
  // pick(...)` lines. Only the `#define`/`#include` preprocessor lines get
  // spurious zero-count records — no test can "execute" a directive — and the
  // default arms are unreachable (the switches are total). Those are excluded.
  if (id >= llvm::Value::InstructionVal) {
    switch (id - llvm::Value::InstructionVal) {
      // LCOV_EXCL_START
#define HANDLE_INST(num, opcode, Class)                                        \
  case num:                                                                    \
    return pick(&typeid(llvm::Class), base);
#include "llvm/IR/Instruction.def"
    default:
      return &typeid(llvm::Instruction);
      // LCOV_EXCL_STOP
    }
  }

  switch (id) {
    // MemoryUse/MemoryDef/MemoryPhi are MemorySSA classes, never IR Module
    // values, so those enum slots never occur here.
    // LCOV_EXCL_START
#define HANDLE_MEMORY_VALUE(Name)                                              \
  case llvm::Value::Name##Val:                                                 \
    return base;
#define HANDLE_VALUE(Name)                                                     \
  case llvm::Value::Name##Val:                                                 \
    return pick(&typeid(llvm::Name), base);
#include "llvm/IR/Value.def"
  default:
    return base;
    // LCOV_EXCL_STOP
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
