// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/User.h>
#include <llvm/IR/Value.h>

#include <vector>

void populate_values(nb::module_ &m) {
  nb::class_<llvm::Value>(m, "Value")
      .def_prop_rw(
          "name", [](llvm::Value &self) { return self.getName().str(); },
          [](llvm::Value &self, const std::string &n) { self.setName(n); })
      .def_prop_ro("type", &llvm::Value::getType, nb::rv_policy::reference)
      .def_prop_ro("num_uses",
                   [](llvm::Value &self) { return self.getNumUses(); })
      .def_prop_ro("users",
                   [](llvm::Value &self) {
                     return std::vector<llvm::User *>(self.user_begin(),
                                                      self.user_end());
                   })
      .def("replace_all_uses_with", &llvm::Value::replaceAllUsesWith, "value"_a)
      .def("__str__", [](llvm::Value &self) { return eudsl::toString(self); })
      .def("__eq__",
           [](llvm::Value &self, nb::handle other) {
             llvm::Value *o;
             if (!nb::try_cast<llvm::Value *>(other, o))
               return false;
             return &self == o;
           })
      .def("__hash__", [](llvm::Value &self) {
        return static_cast<Py_ssize_t>(
            reinterpret_cast<std::uintptr_t>(&self));
      });

  nb::class_<llvm::User, llvm::Value>(m, "User")
      .def_prop_ro("num_operands", &llvm::User::getNumOperands)
      .def("operand", &llvm::User::getOperand, "index"_a,
           nb::rv_policy::reference)
      .def_prop_ro("operands", [](llvm::User &self) {
        std::vector<llvm::Value *> out;
        for (unsigned i = 0, n = self.getNumOperands(); i < n; ++i)
          out.push_back(self.getOperand(i));
        return out;
      });

  // Structural spine of the Value hierarchy. Leaf method bindings are added by
  // Tasks 9-11, which reopen these classes via reopen<T>(). Registering them
  // here lets the Value type_hook name them without raising.
  nb::class_<llvm::Constant, llvm::User>(m, "Constant");
  nb::class_<llvm::GlobalValue, llvm::Constant>(m, "GlobalValue");
  nb::class_<llvm::GlobalObject, llvm::GlobalValue>(m, "GlobalObject");
  nb::class_<llvm::Instruction, llvm::User>(m, "Instruction");
}
