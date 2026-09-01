// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>

#include <vector>

void populate_intrinsics(nb::module_ &m) {
  m.def(
      "lookup_intrinsic_id",
      [](const std::string &name) {
        return (unsigned)llvm::Function::lookupIntrinsicID(name);
      },
      "name"_a);
  m.def(
      "intrinsic_is_overloaded",
      [](unsigned id) {
        return llvm::Intrinsic::isOverloaded((llvm::Intrinsic::ID)id);
      },
      "id"_a);
  m.def(
      "get_intrinsic_declaration",
      [](eudsl::Module &mod, unsigned id,
         std::vector<llvm::Type *> overloadTypes) -> llvm::Function * {
        return llvm::Intrinsic::getDeclaration(
            &mod.get(), (llvm::Intrinsic::ID)id, overloadTypes);
      },
      "module"_a, "id"_a, "overload_types"_a = std::vector<llvm::Type *>{},
      nb::rv_policy::reference, nb::keep_alive<0, 1>());
}
