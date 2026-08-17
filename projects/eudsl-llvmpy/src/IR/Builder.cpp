// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>

#include <iterator>
#include <vector>

namespace {
using RawIP = llvm::IRBuilderBase::InsertPoint;

// MLIR-style insertion point (cf. PyInsertionPoint in MLIR's IRCore.cpp), adapted
// for LLVM's builder. Wraps the raw llvm insert point (block + iterator) plus an
// optional explicit builder; `with InsertPoint(bb, builder=b):` positions `b`
// there and restores it on exit. When `builder` is omitted the current builder
// is resolved from the thread-local stack (an enclosing `with builder:` or the
// enclosing InsertPoint's builder).
struct InsertPoint {
  RawIP ip;
  nb::object builder; // explicit builder, or None -> resolve the current one
};

// One entry on the thread-local stack, mirroring MLIR's PyThreadContextEntry.
// A `with builder:` pushes a builder-only entry; `with InsertPoint(...):` pushes
// an entry that also records the InsertPoint object and the builder's prior
// insert point (restored on exit). current_builder() / InsertPoint.current walk
// the stack for the innermost entry of each kind (cf. getDefault* in MLIR).
// (eudsl::Context's LLVMContext current-stack is separate; see Ownership.cpp.)
struct ThreadContextEntry {
  nb::object builder;
  nb::object insertPoint; // none for a bare `with builder:` entry
  RawIP previous;         // the builder's insert point to restore (InsertPoint)
};

static std::vector<ThreadContextEntry> &threadContextStack() {
  static thread_local std::vector<ThreadContextEntry> stack;
  return stack;
}

// The innermost builder on the stack (an enclosing `with builder:` or the
// builder an enclosing InsertPoint repositioned), or none. Every entry (both
// kinds) records a builder, so the innermost is simply the top of the stack.
static nb::object defaultBuilder() {
  auto &stack = threadContextStack();
  return stack.empty() ? nb::none() : stack.back().builder;
}
} // namespace

void populate_builder(nb::module_ &m) {
  using B = llvm::IRBuilder<>;

  nb::class_<B>(m, "IRBuilder")
      .def(
          "__init__",
          [](B *self, eudsl::Context *context) {
            new (self) B(eudsl::currentOr(context).get());
          },
          "context"_a.none() = nb::none(), nb::keep_alive<1, 2>())
      .def(
          "set_insert_point",
          [](B &self, llvm::BasicBlock *bb) { self.SetInsertPoint(bb); },
          "block"_a)
      .def("__enter__",
           [](nb::object self) -> nb::object {
             // Make this builder the current one; InsertPoint(...) without an
             // explicit builder resolves it from here.
             threadContextStack().push_back({self, nb::none(), RawIP()});
             return self;
           })
      .def(
          "__exit__",
          [](nb::object self, nb::handle, nb::handle, nb::handle) {
            // Symmetric with InsertPoint.__exit__: the top must be this
            // builder's own frame (a builder frame has no insertPoint).
            auto &stack = threadContextStack();
            if (stack.empty() || !stack.back().insertPoint.is_none() ||
                !stack.back().builder.is(self)) {
              throw nb::value_error("unbalanced IRBuilder enter/exit");
            }
            stack.pop_back();
          },
          "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none())
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
          [](B &self, llvm::Value *c, llvm::BasicBlock *t, llvm::BasicBlock *f)
              -> llvm::Value * { return self.CreateCondBr(c, t, f); },
          "cond"_a, "true_dest"_a, "false_dest"_a,
          nb::rv_policy::reference_internal)
#define EUDSL_BIN(pyName, method)                                              \
  .def(                                                                        \
      pyName,                                                                  \
      [](B &self, llvm::Value *l, llvm::Value *r, const std::string &name)     \
          -> llvm::Value * { return self.method(l, r, name); },                \
      "lhs"_a, "rhs"_a, "name"_a = "", nb::rv_policy::reference_internal)
      // clang-format off
      EUDSL_BIN("add", CreateAdd)
      EUDSL_BIN("fadd", CreateFAdd)
      EUDSL_BIN("sub", CreateSub)
      EUDSL_BIN("fsub", CreateFSub)
      EUDSL_BIN("mul", CreateMul)
      EUDSL_BIN("fmul", CreateFMul)
      EUDSL_BIN("sdiv", CreateSDiv)
      EUDSL_BIN("udiv", CreateUDiv)
      EUDSL_BIN("fdiv", CreateFDiv)
      EUDSL_BIN("srem", CreateSRem)
      EUDSL_BIN("urem", CreateURem)
      EUDSL_BIN("frem", CreateFRem)
      EUDSL_BIN("shl", CreateShl)
      EUDSL_BIN("lshr", CreateLShr)
      EUDSL_BIN("ashr", CreateAShr)
      EUDSL_BIN("and_", CreateAnd)
      EUDSL_BIN("or_", CreateOr)
      EUDSL_BIN("xor", CreateXor)
      // clang-format on
#undef EUDSL_BIN
      // Unary ops: (value, name) -> value.
#define EUDSL_UNARY(pyName, method)                                            \
  .def(                                                                        \
      pyName,                                                                  \
      [](B &self, llvm::Value *v, const std::string &name) -> llvm::Value * {  \
        return self.method(v, name);                                           \
      },                                                                       \
      "value"_a, "name"_a = "", nb::rv_policy::reference_internal)
      // clang-format off
      EUDSL_UNARY("neg", CreateNeg)
      EUDSL_UNARY("fneg", CreateFNeg)
      EUDSL_UNARY("not_", CreateNot)
      EUDSL_UNARY("ptrtoaddr", CreatePtrToAddr)
      // clang-format on
#undef EUDSL_UNARY
      // Casts: (value, dest_type, name) -> value.
#define EUDSL_CAST(pyName, method)                                             \
  .def(                                                                        \
      pyName,                                                                  \
      [](B &self, llvm::Value *v, llvm::Type *ty,                              \
         const std::string &name) -> llvm::Value * {                          \
        return self.method(v, ty, name);                                       \
      },                                                                       \
      "value"_a, "dest_type"_a, "name"_a = "", nb::rv_policy::reference_internal)
      // clang-format off
      EUDSL_CAST("trunc", CreateTrunc)
      EUDSL_CAST("zext", CreateZExt)
      EUDSL_CAST("sext", CreateSExt)
      EUDSL_CAST("fptoui", CreateFPToUI)
      EUDSL_CAST("fptosi", CreateFPToSI)
      EUDSL_CAST("uitofp", CreateUIToFP)
      EUDSL_CAST("sitofp", CreateSIToFP)
      EUDSL_CAST("fptrunc", CreateFPTrunc)
      EUDSL_CAST("fpext", CreateFPExt)
      EUDSL_CAST("ptrtoint", CreatePtrToInt)
      EUDSL_CAST("inttoptr", CreateIntToPtr)
      EUDSL_CAST("bitcast", CreateBitCast)
      EUDSL_CAST("addrspacecast", CreateAddrSpaceCast)
      // clang-format on
#undef EUDSL_CAST
      .def(
          "icmp",
          [](B &self, llvm::CmpInst::Predicate p, llvm::Value *l,
             llvm::Value *r, const std::string &name) -> llvm::Value * {
            return self.CreateICmp(p, l, r, name);
          },
          "predicate"_a, "lhs"_a, "rhs"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "fcmp",
          [](B &self, llvm::CmpInst::Predicate p, llvm::Value *l,
             llvm::Value *r, const std::string &name) -> llvm::Value * {
            return self.CreateFCmp(p, l, r, name);
          },
          "predicate"_a, "lhs"_a, "rhs"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "alloca",
          [](B &self, llvm::Type *ty, const std::string &name)
              -> llvm::Value * { return self.CreateAlloca(ty, nullptr, name); },
          "type"_a, "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "load",
          [](B &self, llvm::Type *ty, llvm::Value *ptr, const std::string &name)
              -> llvm::Value * { return self.CreateLoad(ty, ptr, name); },
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
          nb::rv_policy::reference_internal)
      .def(
          "select",
          [](B &self, llvm::Value *cond, llvm::Value *t, llvm::Value *f,
             const std::string &name) -> llvm::Value * {
            return self.CreateSelect(cond, t, f, name);
          },
          "cond"_a, "true_value"_a, "false_value"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "freeze",
          [](B &self, llvm::Value *v, const std::string &name)
              -> llvm::Value * { return self.CreateFreeze(v, name); },
          "value"_a, "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "extract_element",
          [](B &self, llvm::Value *vec, llvm::Value *idx,
             const std::string &name) -> llvm::Value * {
            return self.CreateExtractElement(vec, idx, name);
          },
          "vector"_a, "index"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "insert_element",
          [](B &self, llvm::Value *vec, llvm::Value *elt, llvm::Value *idx,
             const std::string &name) -> llvm::Value * {
            return self.CreateInsertElement(vec, elt, idx, name);
          },
          "vector"_a, "element"_a, "index"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "shuffle_vector",
          [](B &self, llvm::Value *v1, llvm::Value *v2, std::vector<int> mask,
             const std::string &name) -> llvm::Value * {
            return self.CreateShuffleVector(v1, v2, mask, name);
          },
          "v1"_a, "v2"_a, "mask"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "va_arg",
          [](B &self, llvm::Value *list, llvm::Type *ty,
             const std::string &name) -> llvm::Value * {
            return self.CreateVAArg(list, ty, name);
          },
          "arg_list"_a, "type"_a, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "unreachable",
          [](B &self) -> llvm::Value * { return self.CreateUnreachable(); },
          nb::rv_policy::reference_internal)
      .def(
          "switch_",
          [](B &self, llvm::Value *v, llvm::BasicBlock *dest,
             unsigned numCases) -> llvm::SwitchInst * {
            return self.CreateSwitch(v, dest, numCases);
          },
          "value"_a, "default_dest"_a, "num_cases"_a = 10,
          nb::rv_policy::reference_internal)
      .def(
          "indirect_br",
          [](B &self, llvm::Value *addr,
             unsigned numDests) -> llvm::IndirectBrInst * {
            return self.CreateIndirectBr(addr, numDests);
          },
          "address"_a, "num_dests"_a = 10, nb::rv_policy::reference_internal)
      .def(
          "resume",
          [](B &self, llvm::Value *exn)
              -> llvm::Value * { return self.CreateResume(exn); },
          "exn"_a, nb::rv_policy::reference_internal)
      .def(
          "fence",
          [](B &self, llvm::AtomicOrdering ordering,
             bool singleThread) -> llvm::Value * {
            // FenceInst is void-typed, so (unlike the neighboring atomic
            // emitters, which produce a named value) there is no result to name.
            llvm::SyncScope::ID ssid = singleThread
                                           ? llvm::SyncScope::SingleThread
                                           : llvm::SyncScope::System;
            return self.CreateFence(ordering, ssid);
          },
          "ordering"_a, "single_thread"_a = false,
          nb::rv_policy::reference_internal)
      .def(
          "atomic_rmw",
          [](B &self, llvm::AtomicRMWInst::BinOp op, llvm::Value *ptr,
             llvm::Value *val, llvm::AtomicOrdering ordering, bool singleThread,
             const std::string &name) -> llvm::Value * {
            llvm::SyncScope::ID ssid = singleThread
                                           ? llvm::SyncScope::SingleThread
                                           : llvm::SyncScope::System;
            // MaybeAlign() -> IRBuilder derives the natural alignment.
            llvm::Value *v = self.CreateAtomicRMW(op, ptr, val,
                                                  llvm::MaybeAlign(), ordering,
                                                  ssid);
            if (!name.empty())
              v->setName(name);
            return v;
          },
          "op"_a, "ptr"_a, "value"_a, "ordering"_a, "single_thread"_a = false,
          "name"_a = "", nb::rv_policy::reference_internal)
      .def(
          "atomic_cmpxchg",
          [](B &self, llvm::Value *ptr, llvm::Value *cmp, llvm::Value *newValue,
             llvm::AtomicOrdering success, llvm::AtomicOrdering failure,
             bool singleThread, const std::string &name) -> llvm::Value * {
            // The AtomicCmpXchgInst ctor asserts these; check first so a bad
            // ordering raises a catchable Python error instead of aborting.
            if (!llvm::AtomicCmpXchgInst::isValidSuccessOrdering(success)) {
              throw nb::value_error(
                  "invalid cmpxchg success ordering (must not be NotAtomic or "
                  "Unordered)");
            }
            if (!llvm::AtomicCmpXchgInst::isValidFailureOrdering(failure)) {
              throw nb::value_error(
                  "invalid cmpxchg failure ordering (must not be NotAtomic, "
                  "Unordered, Release, or AcquireRelease)");
            }
            llvm::SyncScope::ID ssid = singleThread
                                           ? llvm::SyncScope::SingleThread
                                           : llvm::SyncScope::System;
            llvm::Value *v = self.CreateAtomicCmpXchg(
                ptr, cmp, newValue, llvm::MaybeAlign(), success, failure, ssid);
            if (!name.empty())
              v->setName(name);
            return v;
          },
          "ptr"_a, "cmp"_a, "new_value"_a, "success_ordering"_a,
          "failure_ordering"_a, "single_thread"_a = false, "name"_a = "",
          nb::rv_policy::reference_internal)
      .def(
          "call_intrinsic",
          [](B &self, unsigned intrinsicId, std::vector<llvm::Type *> types,
             std::vector<llvm::Value *> args,
             const std::string &name) -> llvm::Value * {
            llvm::Value *v = self.CreateIntrinsic(
                static_cast<llvm::Intrinsic::ID>(intrinsicId), types, args);
            if (!name.empty())
              v->setName(name);
            return v;
          },
          "intrinsic_id"_a, "overload_types"_a, "args"_a, "name"_a = "",
          nb::rv_policy::reference_internal);

  // --- DSL current-builder/function state + MLIR-style InsertPoint ---
  // The current builder is the innermost one on the thread-local stack (an
  // enclosing `with builder:` or the builder an InsertPoint repositioned).
  // current_function derives from where that builder is positioned (an insert
  // point lives in a block, which belongs to a function), so no separate
  // function state is tracked.

  m.def(
      "current_builder",
      []() -> nb::object {
        nb::object b = defaultBuilder();
        if (b.is_none()) {
          throw std::runtime_error(
              "no current IRBuilder; use `with builder:` or "
              "`with InsertPoint(block, builder=b):`");
        }
        return b;
      },
      "The innermost builder on the thread-local stack.");
  m.def(
      "current_function",
      []() -> llvm::Function * {
        nb::object b = defaultBuilder();
        llvm::BasicBlock *bb =
            b.is_none() ? nullptr : nb::cast<B *>(b)->GetInsertBlock();
        llvm::Function *fn = bb ? bb->getParent() : nullptr;
        if (!fn)
          throw std::runtime_error("no current function");
        return fn;
      },
      nb::rv_policy::reference,
      "The function containing the current builder's insertion block.");

  // MLIR-style InsertPoint (see PyInsertionPoint in MLIR's IRCore.cpp). Wraps the
  // llvm insert point (block + iterator) plus an optional explicit builder; on
  // __enter__ it positions that builder here (resolving the current one when
  // omitted) and restores it on __exit__.
  nb::class_<InsertPoint>(m, "InsertPoint")
      .def(
          "__init__",
          [](InsertPoint *self, nb::handle blockOrBefore, nb::object builder) {
            llvm::BasicBlock *bb;
            llvm::Instruction *inst;
            if (nb::try_cast(blockOrBefore, bb)) {
              new (self) InsertPoint{RawIP(bb, bb->end()), std::move(builder)};
            } else if (nb::try_cast(blockOrBefore, inst)) {
              new (self) InsertPoint{RawIP(inst->getParent(), inst->getIterator()),
                                     std::move(builder)};
            } else {
              throw nb::type_error(
                  "InsertPoint expects a BasicBlock (insert at end) or an "
                  "Instruction (insert before it)");
            }
          },
          "block_or_before"_a, "builder"_a = nb::none())
      .def_static(
          "at_block_begin",
          [](llvm::BasicBlock *bb, nb::object builder) {
            return InsertPoint{RawIP(bb, bb->getFirstInsertionPt()),
                               std::move(builder)};
          },
          "block"_a, "builder"_a = nb::none(),
          "Insert at the start of a block (after any phis).")
      .def_static(
          "at_block_terminator",
          [](llvm::BasicBlock *bb, nb::object builder) {
            llvm::Instruction *term = bb->getTerminatorOrNull();
            if (!term)
              throw nb::value_error("block has no terminator");
            return InsertPoint{RawIP(bb, term->getIterator()),
                               std::move(builder)};
          },
          "block"_a, "builder"_a = nb::none(),
          "Insert before a block's terminator.")
      .def_static(
          "after",
          [](llvm::Instruction *inst, nb::object builder) {
            return InsertPoint{RawIP(inst->getParent(),
                                     std::next(inst->getIterator())),
                               std::move(builder)};
          },
          "instruction"_a, "builder"_a = nb::none(),
          "Insert immediately after an instruction.")
      .def_prop_ro(
          "block", [](InsertPoint &self) { return self.ip.getBlock(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro("is_set", [](InsertPoint &self) { return self.ip.isSet(); })
      .def("__enter__",
           [](nb::object self) -> nb::object {
             InsertPoint &ipObj = nb::cast<InsertPoint &>(self);
             nb::object builderObj =
                 ipObj.builder.is_none() ? defaultBuilder() : ipObj.builder;
             if (builderObj.is_none()) {
               throw nb::value_error(
                   "no current IRBuilder; pass builder= or enter a builder "
                   "(`with builder:`) first");
             }
             B *b = nb::cast<B *>(builderObj);
             threadContextStack().push_back({builderObj, self, b->saveIP()});
             b->restoreIP(ipObj.ip);
             return self;
           })
      .def(
          "__exit__",
          [](nb::object self, nb::handle, nb::handle, nb::handle) {
            auto &stack = threadContextStack();
            if (stack.empty() || !stack.back().insertPoint.is(self))
              throw nb::value_error("unbalanced InsertPoint enter/exit");
            ThreadContextEntry frame = stack.back();
            stack.pop_back();
            nb::cast<B *>(frame.builder)->restoreIP(frame.previous);
          },
          "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none())
      .def_prop_ro_static("current", [](nb::handle) -> nb::object {
        auto &stack = threadContextStack();
        for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
          if (!it->insertPoint.is_none())
            return it->insertPoint;
        }
        throw nb::value_error("no current InsertPoint");
      });
}
