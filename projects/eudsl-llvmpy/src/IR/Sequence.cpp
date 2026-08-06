// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Sequence.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Use.h>
#include <llvm/IR/User.h>
#include <llvm/IR/Value.h>

void populate_sequences(nb::module_ &m) {
  eudsl::bindSequence<llvm::Function>(m, "FunctionSequence");
  eudsl::bindSequence<llvm::BasicBlock>(m, "BasicBlockSequence");
  eudsl::bindSequence<llvm::Argument>(m, "ArgumentSequence");
  eudsl::bindSequence<llvm::Instruction>(m, "InstructionSequence");
  eudsl::bindSequence<llvm::Value>(m, "ValueSequence");
  eudsl::bindSequence<llvm::User>(m, "UserSequence");
  eudsl::bindSequence<llvm::Use>(m, "UseSequence");
}
