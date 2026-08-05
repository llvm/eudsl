// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>

#include <vector>

namespace {
// Owns a saved insertion point so `with builder.at_end_of(bb)` positions the
// builder on __enter__ (restoring is unnecessary for the DSL's usage, which
// always sets a fresh insertion point per block).
struct InsertGuard {
  llvm::IRBuilder<> *builder;
  llvm::BasicBlock *block;
};
} // namespace

void populate_builder(nb::module_ &m) {
  using B = llvm::IRBuilder<>;

  nb::class_<InsertGuard>(m, "_InsertGuard")
      .def("__enter__",
           [](InsertGuard &g) { g.builder->SetInsertPoint(g.block); })
      .def(
          "__exit__",
          [](InsertGuard &, nb::object, nb::object, nb::object) {},
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none());

  nb::class_<B>(m, "IRBuilder")
      .def(
          "__init__",
          [](B *self, eudsl::Context &ctx) { new (self) B(ctx.get()); },
          "context"_a, nb::keep_alive<1, 2>())
      .def(
          "set_insert_point",
          [](B &self, llvm::BasicBlock *bb) { self.SetInsertPoint(bb); },
          "block"_a)
      .def(
          "at_end_of",
          [](B &self, llvm::BasicBlock *bb) { return InsertGuard{&self, bb}; },
          "block"_a, nb::keep_alive<0, 1>())
      .def_prop_ro(
          "insert_block", [](B &self) { return self.GetInsertBlock(); },
          nb::rv_policy::reference_internal)
      .def(
          "ret",
          [](B &self, llvm::Value *v) -> llvm::Value * {
            return v ? self.CreateRet(v) : self.CreateRetVoid();
          },
          "value"_a = nullptr, nb::rv_policy::reference_internal)
      .def(
          "br",
          [](B &self, llvm::BasicBlock *dest) -> llvm::Value * {
            return self.CreateBr(dest);
          },
          "dest"_a, nb::rv_policy::reference_internal)
      .def(
          "cond_br",
          [](B &self, llvm::Value *c, llvm::BasicBlock *t,
             llvm::BasicBlock *f) -> llvm::Value * {
            return self.CreateCondBr(c, t, f);
          },
          "cond"_a, "true_dest"_a, "false_dest"_a,
          nb::rv_policy::reference_internal)
#define EUDSL_BIN(pyName, method)                                              \
  .def(                                                                        \
      pyName,                                                                  \
      [](B &self, llvm::Value *l, llvm::Value *r, const std::string &name)     \
          -> llvm::Value * { return self.method(l, r, name); },                \
      "lhs"_a, "rhs"_a, "name"_a = "", nb::rv_policy::reference_internal)
          EUDSL_BIN("add", CreateAdd) EUDSL_BIN("fadd", CreateFAdd)
              EUDSL_BIN("sub", CreateSub) EUDSL_BIN("fsub", CreateFSub)
                  EUDSL_BIN("mul", CreateMul) EUDSL_BIN("fmul", CreateFMul)
                      EUDSL_BIN("sdiv", CreateSDiv)
                          EUDSL_BIN("udiv", CreateUDiv)
                              EUDSL_BIN("fdiv", CreateFDiv)
#undef EUDSL_BIN
      .def(
          "icmp",
          [](B &self, llvm::CmpInst::Predicate p, llvm::Value *l, llvm::Value *r,
             const std::string &name) -> llvm::Value * {
            return self.CreateICmp(p, l, r, name);
          },
          "predicate"_a, "lhs"_a, "rhs"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "fcmp",
          [](B &self, llvm::CmpInst::Predicate p, llvm::Value *l, llvm::Value *r,
             const std::string &name) -> llvm::Value * {
            return self.CreateFCmp(p, l, r, name);
          },
          "predicate"_a, "lhs"_a, "rhs"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "alloca",
          [](B &self, llvm::Type *ty, const std::string &name) -> llvm::Value * {
            return self.CreateAlloca(ty, nullptr, name);
          },
          "type"_a, "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "load",
          [](B &self, llvm::Type *ty, llvm::Value *ptr,
             const std::string &name) -> llvm::Value * {
            return self.CreateLoad(ty, ptr, name);
          },
          "type"_a, "ptr"_a, "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "store",
          [](B &self, llvm::Value *v, llvm::Value *ptr) -> llvm::Value * {
            return self.CreateStore(v, ptr);
          },
          "value"_a, "ptr"_a, nb::rv_policy::reference_internal)
      .def(
          "gep",
          [](B &self, llvm::Type *ty, llvm::Value *ptr,
             std::vector<llvm::Value *> idxs,
             const std::string &name) -> llvm::Value * {
            return self.CreateGEP(ty, ptr, idxs, name);
          },
          "type"_a, "ptr"_a, "indices"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "call",
          [](B &self, llvm::Function *fn, std::vector<llvm::Value *> args,
             const std::string &name) -> llvm::Value * {
            return self.CreateCall(fn->getFunctionType(), fn, args, name);
          },
          "fn"_a, "args"_a, "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "phi",
          [](B &self, llvm::Type *ty, const std::string &name) {
            return self.CreatePHI(ty, 0, name);
          },
          "type"_a, "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "i64_const",
          [](B &self, int64_t v) -> llvm::Value * { return self.getInt64(v); },
          "value"_a, nb::rv_policy::reference_internal)
      .def(
          "i32_const",
          [](B &self, int32_t v) -> llvm::Value * { return self.getInt32(v); },
          "value"_a, nb::rv_policy::reference_internal)
      .def(
          "extract_value",
          [](B &self, llvm::Value *agg, unsigned idx,
             const std::string &name) -> llvm::Value * {
            return self.CreateExtractValue(agg, {idx}, name);
          },
          "aggregate"_a, "index"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "insert_value",
          [](B &self, llvm::Value *agg, llvm::Value *val, unsigned idx,
             const std::string &name) -> llvm::Value * {
            return self.CreateInsertValue(agg, val, {idx}, name);
          },
          "aggregate"_a, "value"_a, "index"_a, "name"_a = "",
          nb::rv_policy::reference_internal);
}
