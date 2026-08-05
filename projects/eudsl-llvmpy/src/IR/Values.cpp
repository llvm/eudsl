// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
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
      .def_prop_ro("type", &llvm::Value::getType, nb::rv_policy::reference_internal)
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
           nb::rv_policy::reference_internal)
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
  nb::class_<llvm::Instruction, llvm::User>(m, "Instruction")
      .def_prop_ro("num_successors",
                   [](llvm::Instruction &self) {
                     return self.getNumSuccessors();
                   })
      .def("successor", &llvm::Instruction::getSuccessor, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("is_terminator",
                   [](llvm::Instruction &self) { return self.isTerminator(); })
      .def_prop_ro(
          "parent", [](llvm::Instruction &self) { return self.getParent(); },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::Argument, llvm::Value>(m, "Argument")
      .def_prop_ro("arg_no", &llvm::Argument::getArgNo)
      .def_prop_ro(
          "parent", [](llvm::Argument &self) { return self.getParent(); },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::BasicBlock, llvm::Value>(m, "BasicBlock")
      .def_static(
          "create",
          [](eudsl::Context &ctx, const std::string &name,
             llvm::Function *parent) {
            return llvm::BasicBlock::Create(ctx.get(), name, parent);
          },
          "context"_a, "name"_a = "", "parent"_a = nullptr,
          nb::rv_policy::reference)
      .def_prop_ro(
          "parent", [](llvm::BasicBlock &self) { return self.getParent(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "terminator",
          [](llvm::BasicBlock &self) { return self.getTerminatorOrNull(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro("instructions", [](llvm::BasicBlock &self) {
        std::vector<llvm::Instruction *> out;
        for (llvm::Instruction &i : self)
          out.push_back(&i);
        return out;
      });

  nb::class_<llvm::Function, llvm::GlobalObject>(m, "Function")
      .def_static(
          "create",
          [](llvm::FunctionType *ft, const std::string &name,
             eudsl::Module &mod) {
            return llvm::Function::Create(
                ft, llvm::GlobalValue::ExternalLinkage, name, mod.get());
          },
          "function_type"_a, "name"_a, "module"_a, nb::rv_policy::reference,
          nb::keep_alive<0, 3>())
      .def_prop_ro("function_type", &llvm::Function::getFunctionType,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("return_type", &llvm::Function::getReturnType,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("is_var_arg", &llvm::Function::isVarArg)
      .def_prop_ro("is_declaration", &llvm::Function::isDeclaration)
      .def_prop_ro("num_args", &llvm::Function::arg_size)
      .def("arg", &llvm::Function::getArg, "index"_a, nb::rv_policy::reference_internal)
      .def_prop_ro("args",
                   [](llvm::Function &self) {
                     std::vector<llvm::Argument *> out;
                     for (llvm::Argument &a : self.args())
                       out.push_back(&a);
                     return out;
                   })
      .def_prop_ro("basic_blocks",
                   [](llvm::Function &self) {
                     std::vector<llvm::BasicBlock *> out;
                     for (llvm::BasicBlock &b : self)
                       out.push_back(&b);
                     return out;
                   })
      .def_prop_ro(
          "entry_block",
          [](llvm::Function &self) -> llvm::BasicBlock * {
            if (self.empty())
              return nullptr;
            return &self.getEntryBlock();
          },
          nb::rv_policy::reference_internal)
      .def(
          "append_basic_block",
          [](llvm::Function &self, const std::string &name) {
            return llvm::BasicBlock::Create(self.getContext(), name, &self);
          },
          "name"_a = "", nb::rv_policy::reference_internal);
}
