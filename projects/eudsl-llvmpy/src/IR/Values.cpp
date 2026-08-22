// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"
#include "IR/Sequence.h"

#include <llvm/IR/Argument.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalObject.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Use.h>
#include <llvm/IR/User.h>
#include <llvm/IR/Value.h>

#include <nanobind/make_iterator.h>

#include <algorithm>
#include <vector>

/// The Python wrapper of the module that owns `v`'s storage, or none when `v`
/// has no owning module (a context-owned value such as a constant) or when no
/// live wrapper for that module can be resolved. A sub-object view stores this
/// so holding the view -- or an element it yields -- keeps the owning module
/// alive.
static nb::object ownerObjectFor(const llvm::Value *v) {
  const llvm::Module *m = nullptr;
  if (auto *inst = llvm::dyn_cast<llvm::Instruction>(v))
    m = inst->getModule();
  else if (auto *arg = llvm::dyn_cast<llvm::Argument>(v))
    m = arg->getParent() ? arg->getParent()->getParent() : nullptr;
  else if (auto *gv = llvm::dyn_cast<llvm::GlobalValue>(v))
    m = gv->getParent();
  else if (auto *bb = llvm::dyn_cast<llvm::BasicBlock>(v))
    m = bb->getModule();
  // moduleWrapperFor(nullptr) is null, so a value with no module folds to a
  // none owner without a separate branch.
  eudsl::Module *wrapper = eudsl::moduleWrapperFor(m);
  nb::object obj = wrapper ? nb::find(*wrapper) : nb::none();
  return obj.is_valid() ? obj : nb::none();
}

void populate_values(nb::module_ &m) {
  // A def-use edge: which User uses a value, and at which operand index.
  nb::class_<llvm::Use>(m, "Use")
      .def_prop_ro(
          "user", [](llvm::Use &self) { return self.getUser(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro("operand_number", &llvm::Use::getOperandNo);

  nb::class_<llvm::Value>(m, "Value")
      .def_prop_rw(
          "name", [](llvm::Value &self) { return self.getName().str(); },
          [](llvm::Value &self, const std::string &n) { self.setName(n); })
      .def_prop_ro("type", &llvm::Value::getType, nb::rv_policy::reference)
      .def_prop_ro("num_uses",
                   [](llvm::Value &self) { return self.getNumUses(); })
      .def_prop_ro(
          "users",
          [](llvm::Value &self) {
            llvm::Value *v = &self;
            eudsl::Sequence<llvm::User> seq;
            seq.length = [v] {
              return static_cast<std::size_t>(v->getNumUses());
            };
            seq.at = [v](std::size_t i) {
              auto it = v->user_begin();
              std::advance(it, i);
              return *it;
            };
            seq.owner = ownerObjectFor(v);
            seq.ownerOf = [](llvm::User *e) { return ownerObjectFor(e); };
            return seq;
          },
          nb::keep_alive<0, 1>())
      .def_prop_ro(
          "uses",
          [](llvm::Value &self) {
            llvm::Value *v = &self;
            eudsl::Sequence<llvm::Use> seq;
            seq.length = [v] {
              return static_cast<std::size_t>(v->getNumUses());
            };
            seq.at = [v](std::size_t i) {
              auto it = v->use_begin();
              std::advance(it, i);
              return &*it;
            };
            seq.owner = ownerObjectFor(v);
            // A Use lives in its User's operand list, so it is owned by the
            // User's module.
            seq.ownerOf = [](llvm::Use *e) {
              return ownerObjectFor(e->getUser());
            };
            return seq;
          },
          nb::keep_alive<0, 1>())
      .def("replace_all_uses_with", &llvm::Value::replaceAllUsesWith, "value"_a)
      .def(
          "replace_all_uses_except",
          [](llvm::Value &self, llvm::Value *newValue,
             std::vector<llvm::User *> exceptions) {
            // replaceUsesWithIf asserts the types match and would otherwise
            // abort the interpreter; reject the mismatch as a Python error.
            if (newValue->getType() != self.getType()) {
              throw nb::value_error(
                  "replacement value type does not match the value's type");
            }
            self.replaceUsesWithIf(newValue, [&](llvm::Use &u) {
              return std::find(exceptions.begin(), exceptions.end(),
                               u.getUser()) == exceptions.end();
            });
          },
          "value"_a, "exceptions"_a)
      .def("__str__", [](llvm::Value &self) { return eudsl::toString(self); })
      .def(
          "__eq__",
          [](llvm::Value &self, llvm::Value *other) { return &self == other; },
          nb::is_operator())
      .def("__hash__", [](llvm::Value &self) {
        return static_cast<Py_ssize_t>(reinterpret_cast<std::uintptr_t>(&self));
      });

  nb::class_<llvm::User, llvm::Value>(m, "User")
      .EUDSL_CAST_CTOR(llvm::User, llvm::Value)
      .def_prop_ro("num_operands", &llvm::User::getNumOperands)
      .def("operand", &llvm::User::getOperand, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro(
          "operands",
          [](llvm::User &self) {
            llvm::User *u = &self;
            eudsl::Sequence<llvm::Value> seq;
            seq.length = [u] {
              return static_cast<std::size_t>(u->getNumOperands());
            };
            seq.at = [u](std::size_t i) {
              return u->getOperand(static_cast<unsigned>(i));
            };
            seq.owner = ownerObjectFor(u);
            seq.ownerOf = [](llvm::Value *e) { return ownerObjectFor(e); };
            return seq;
          },
          nb::keep_alive<0, 1>())
      .def("__len__",
           [](llvm::User &self) {
             return static_cast<Py_ssize_t>(self.getNumOperands());
           })
      .def(
          "__getitem__",
          [](llvm::User &self, Py_ssize_t i) -> llvm::Value * {
            Py_ssize_t n = static_cast<Py_ssize_t>(self.getNumOperands());
            if (i < 0)
              i += n;
            if (i < 0 || i >= n)
              throw nb::index_error("index out of range");
            return self.getOperand(static_cast<unsigned>(i));
          },
          nb::rv_policy::reference_internal)
      .def(
          "__iter__",
          [](llvm::User &self) {
            return nb::make_iterator<nb::rv_policy::reference>(
                nb::type<llvm::User>(), "OperandIterator",
                self.value_op_begin(), self.value_op_end(),
                nb::keep_alive<0, 1>());
          },
          nb::keep_alive<0, 1>());

  // Structural spine of the Value hierarchy. These base classes are registered
  // bare so the concrete subclasses bound in Instructions.cpp / Constants.cpp
  // can name them as their nanobind base, and so the Value type_hook can name
  // them without raising.
  nb::class_<llvm::Constant, llvm::User>(m, "Constant");
  nb::class_<llvm::GlobalValue, llvm::Constant>(m, "GlobalValue");
  nb::class_<llvm::GlobalObject, llvm::GlobalValue>(m, "GlobalObject");

  nb::class_<llvm::Instruction, llvm::User>(m, "Instruction")
      .EUDSL_CAST_CTOR(llvm::Instruction, llvm::Value)
      .def_prop_ro(
          "num_successors",
          [](llvm::Instruction &self) { return self.getNumSuccessors(); })
      .def("successor", &llvm::Instruction::getSuccessor, "index"_a,
           nb::rv_policy::reference_internal)
      .def("set_successor", &llvm::Instruction::setSuccessor, "index"_a,
           "block"_a)
      .def_prop_ro("is_terminator",
                   [](llvm::Instruction &self) { return self.isTerminator(); })
      .def_prop_ro(
          "parent", [](llvm::Instruction &self) { return self.getParent(); },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::Argument, llvm::Value>(m, "Argument")
      .EUDSL_CAST_CTOR(llvm::Argument, llvm::Value)
      .def_prop_ro("arg_no", &llvm::Argument::getArgNo)
      .def_prop_ro(
          "parent", [](llvm::Argument &self) { return self.getParent(); },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::BasicBlock, llvm::Value>(m, "BasicBlock")
      .EUDSL_CAST_CTOR(llvm::BasicBlock, llvm::Value)
      .def_static(
          "create",
          [](const std::string &name, llvm::Function *parent,
             eudsl::Context *context) {
            return llvm::BasicBlock::Create(eudsl::currentOr(context).get(),
                                            name, parent);
          },
          "name"_a = "", "parent"_a = nullptr, "context"_a.none() = nb::none(),
          nb::rv_policy::reference, nb::keep_alive<0, 2>())
      .def_prop_ro(
          "parent", [](llvm::BasicBlock &self) { return self.getParent(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "terminator",
          [](llvm::BasicBlock &self) { return self.getTerminatorOrNull(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "instructions",
          [](llvm::BasicBlock &self) {
            llvm::BasicBlock *b = &self;
            eudsl::Sequence<llvm::Instruction> seq;
            seq.length = [b] { return static_cast<std::size_t>(b->size()); };
            seq.at = [b](std::size_t i) {
              auto it = b->begin();
              std::advance(it, i);
              return &*it;
            };
            seq.owner = ownerObjectFor(b);
            seq.ownerOf = [](llvm::Instruction *e) {
              return ownerObjectFor(e);
            };
            return seq;
          },
          nb::keep_alive<0, 1>())
      .def("__len__",
           [](llvm::BasicBlock &self) {
             return static_cast<Py_ssize_t>(self.size());
           })
      .def(
          "__getitem__",
          [](llvm::BasicBlock &self, Py_ssize_t i) {
            std::vector<llvm::Instruction *> out;
            for (llvm::Instruction &inst : self)
              out.push_back(&inst);
            return eudsl::nthOrThrow(out, i);
          },
          nb::rv_policy::reference_internal)
      .def(
          "__iter__",
          [](llvm::BasicBlock &self) {
            return nb::make_iterator<nb::rv_policy::reference>(
                nb::type<llvm::BasicBlock>(), "InstructionIterator",
                self.begin(), self.end(), nb::keep_alive<0, 1>());
          },
          nb::keep_alive<0, 1>());

  nb::class_<llvm::Function, llvm::GlobalObject>(m, "Function")
      .EUDSL_CAST_CTOR(llvm::Function, llvm::Value)
      .def_static(
          "create",
          [](llvm::FunctionType *ft, const std::string &name,
             eudsl::Module &mod, llvm::GlobalValue::LinkageTypes linkage) {
            return llvm::Function::Create(ft, linkage, name, mod.get());
          },
          "function_type"_a, "name"_a, "module"_a,
          "linkage"_a = llvm::GlobalValue::LinkageTypes::ExternalLinkage,
          nb::rv_policy::reference, nb::keep_alive<0, 3>())
      .def_prop_ro("function_type", &llvm::Function::getFunctionType,
                   nb::rv_policy::reference)
      .def_prop_ro("return_type", &llvm::Function::getReturnType,
                   nb::rv_policy::reference)
      .def_prop_ro("is_var_arg", &llvm::Function::isVarArg)
      .def_prop_ro("is_declaration", &llvm::Function::isDeclaration)
      .def_prop_ro("num_args", &llvm::Function::arg_size)
      .def("arg", &llvm::Function::getArg, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro(
          "args",
          [](llvm::Function &self) {
            llvm::Function *f = &self;
            eudsl::Sequence<llvm::Argument> seq;
            seq.length = [f] {
              return static_cast<std::size_t>(f->arg_size());
            };
            seq.at = [f](std::size_t i) {
              return f->getArg(static_cast<unsigned>(i));
            };
            seq.owner = ownerObjectFor(f);
            seq.ownerOf = [](llvm::Argument *e) { return ownerObjectFor(e); };
            return seq;
          },
          nb::keep_alive<0, 1>())
      .def_prop_ro(
          "basic_blocks",
          [](llvm::Function &self) {
            llvm::Function *f = &self;
            eudsl::Sequence<llvm::BasicBlock> seq;
            seq.length = [f] { return static_cast<std::size_t>(f->size()); };
            seq.at = [f](std::size_t i) {
              auto it = f->begin();
              std::advance(it, i);
              return &*it;
            };
            seq.owner = ownerObjectFor(f);
            seq.ownerOf = [](llvm::BasicBlock *e) { return ownerObjectFor(e); };
            return seq;
          },
          nb::keep_alive<0, 1>())
      .def("__len__",
           [](llvm::Function &self) {
             return static_cast<Py_ssize_t>(self.size());
           })
      .def(
          "__getitem__",
          [](llvm::Function &self, Py_ssize_t i) {
            std::vector<llvm::BasicBlock *> out;
            for (llvm::BasicBlock &b : self)
              out.push_back(&b);
            return eudsl::nthOrThrow(out, i);
          },
          nb::rv_policy::reference_internal)
      .def(
          "__iter__",
          [](llvm::Function &self) {
            return nb::make_iterator<nb::rv_policy::reference>(
                nb::type<llvm::Function>(), "BasicBlockIterator", self.begin(),
                self.end(), nb::keep_alive<0, 1>());
          },
          nb::keep_alive<0, 1>())
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
          "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "walk",
          [](llvm::Function &self) {
            // Every instruction in the function, in block then program order --
            // the LLVM analogue of MLIR's op.walk() over a single region.
            return nb::make_iterator<nb::rv_policy::reference>(
                nb::type<llvm::Function>(), "WalkIterator",
                llvm::inst_begin(self), llvm::inst_end(self),
                nb::keep_alive<0, 1>());
          },
          nb::keep_alive<0, 1>())
      .def_prop_rw("linkage", &llvm::Function::getLinkage,
                   &llvm::Function::setLinkage)
      .def_prop_rw("visibility", &llvm::Function::getVisibility,
                   &llvm::Function::setVisibility)
      .def_prop_rw(
          "calling_conv",
          [](llvm::Function &self) {
            return static_cast<CallingConvEnum>(self.getCallingConv());
          },
          [](llvm::Function &self, CallingConvEnum cc) {
            self.setCallingConv(static_cast<llvm::CallingConv::ID>(cc));
          })
      .def(
          "add_fn_attr",
          [](llvm::Function &self, const std::string &name,
             const std::string &value) { self.addFnAttr(name, value); },
          "name"_a, "value"_a = "")
      .def(
          "has_fn_attr",
          [](llvm::Function &self, const std::string &name) {
            return self.hasFnAttribute(name);
          },
          "name"_a)
      .def(
          "fn_attr_value",
          [](llvm::Function &self, const std::string &name) {
            return self.getFnAttribute(name).getValueAsString().str();
          },
          "name"_a);
}
