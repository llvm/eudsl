// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/SymbolTable.h"
#include "TestSupport.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *IR = R"(
@g = global i32 1
@c = internal constant i32 2
@tls = thread_local global i32 0
@0 = private constant i32 3
@llvm.used = appending global [1 x ptr] [ptr @g], section "llvm.metadata"
@a = alias i32, ptr @g
@ifn = ifunc void (), ptr @resolver

declare void @used()
declare void @unused()
declare extern_weak void @weakref()
declare void @llvm.donothing()
define available_externally void @ae_used() {
  ret void
}
define available_externally void @ae_unused() {
  ret void
}
define ptr @resolver() {
  ret ptr @f
}
define internal void @f() {
  call void @used()
  call void @weakref()
  call void @ae_used()
  call void @llvm.donothing()
  %v = load i32, ptr @0
  ret void
}
)";

std::vector<std::string> names(ArrayRef<GlobalValue *> Symbols) {
  std::vector<std::string> Out;
  for (GlobalValue *GV : Symbols)
    Out.push_back(GV->getName().str());
  return Out;
}

/// (name string, address operand) of every table entry.
std::vector<std::pair<std::string, GlobalValue *>>
entries(GlobalVariable &Table) {
  std::vector<std::pair<std::string, GlobalValue *>> Out;
  auto *Arr = cast<ArrayType>(Table.getValueType());
  for (unsigned I = 0, E = Arr->getNumElements(); I != E; ++I) {
    Constant *Entry = Table.getInitializer()->getAggregateElement(I);
    StringRef Name;
    EXPECT_TRUE(getConstantStringInfo(Entry->getAggregateElement(0u), Name));
    Out.emplace_back(Name.str(),
                     cast<GlobalValue>(Entry->getAggregateElement(1u)));
  }
  return Out;
}

/// How often llvm.compiler.used lists GV.
unsigned compilerUsedCount(Module &M, GlobalValue *GV) {
  GlobalVariable *Used = M.getNamedGlobal("llvm.compiler.used");
  if (!Used)
    return 0;
  unsigned N = 0;
  for (const Use &U : Used->getInitializer()->operands())
    N += U.get() == GV;
  return N;
}

TEST(SymbolTable, CollectsWhatTierUpCodeCanReference) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  EXPECT_EQ(names(collectSymbols(*M)),
            (std::vector<std::string>{"used", "weakref", "ae_used", "resolver",
                                      "f", "g", "c", "a"}));
}

TEST(SymbolTable, EmitsNameAddressPairs) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  GlobalVariable *Table = emitSymbolTable(*M, collectSymbols(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));

  EXPECT_EQ(Table->getName(), SymbolTableName);
  EXPECT_TRUE(Table->isConstant());
  EXPECT_EQ(compilerUsedCount(*M, Table), 1u);
  auto E = entries(*Table);
  std::vector<std::string> Names;
  for (auto &[Name, GV] : E) {
    Names.push_back(Name);
    EXPECT_EQ(GV->getName(), Name);
  }
  EXPECT_EQ(Names, (std::vector<std::string>{"used", "weakref", "ae_used",
                                             "resolver", "f", "g", "c", "a"}));
}

TEST(SymbolTable, ExtendingAddsTheNamesNotAlreadyThere) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  GlobalVariable *Base = emitSymbolTable(*M, collectSymbols(*M));
  // "used" is already in the table, "unused" exists but was not collected,
  // and "orc_pgo_extra" is named twice.
  GlobalVariable *Table = extendSymbolTable(
      *M, *Base, {"orc_pgo_extra", "used", "unused", "orc_pgo_extra"});
  ASSERT_FALSE(verifyModule(*M, &errs()));

  // The new table takes the old one's name and place in llvm.compiler.used.
  EXPECT_EQ(Table->getName(), SymbolTableName);
  EXPECT_EQ(M->getNamedGlobal(SymbolTableName), Table);
  EXPECT_EQ(compilerUsedCount(*M, Table), 1u);
  auto E = entries(*Table);
  std::vector<std::string> Names;
  for (auto &[Name, GV] : E) {
    Names.push_back(Name);
    EXPECT_EQ(GV->getName(), Name);
  }
  EXPECT_EQ(Names, (std::vector<std::string>{"used", "weakref", "ae_used",
                                             "resolver", "f", "g", "c", "a",
                                             "orc_pgo_extra", "unused"}));
  Function *Extra = M->getFunction("orc_pgo_extra");
  ASSERT_NE(Extra, nullptr);
  EXPECT_TRUE(Extra->isDeclaration());
  EXPECT_TRUE(Extra->hasExternalLinkage());
  EXPECT_EQ(E[8].second, Extra);
  EXPECT_EQ(E[9].second, M->getFunction("unused"));
}

TEST(SymbolTable, PlatformSymbolsCoverWhatCodegenCallsWithoutADeclaration) {
  std::vector<std::string> Elf =
      platformSymbols(Triple("aarch64-unknown-linux-gnu"));
  std::vector<std::string> Darwin =
      platformSymbols(Triple("arm64-apple-macosx"));
  std::set<std::string> ElfSet(Elf.begin(), Elf.end());
  std::set<std::string> DarwinSet(Darwin.begin(), Darwin.end());
  // No duplicates.
  EXPECT_EQ(ElfSet.size(), Elf.size());
  EXPECT_EQ(DarwinSet.size(), Darwin.size());
  for (const char *Name : {"_Unwind_Resume",
                           "memcpy",
                           "memmove",
                           "memset",
                           "__stack_chk_fail",
                           "sin",
                           "sinf",
                           "sinl",
                           "cos",
                           "pow",
                           "powf",
                           "powl",
                           "exp",
                           "exp2",
                           "exp2f",
                           "log",
                           "log2",
                           "log10",
                           "log10l",
                           "fmod",
                           "fmodf",
                           "fmodl",
                           "floor",
                           "fma",
                           "atan2",
                           "modfl"}) {
    EXPECT_TRUE(ElfSet.count(Name)) << Name;
    EXPECT_TRUE(DarwinSet.count(Name)) << Name;
  }
  for (const char *Name : {"bzero", "memset_pattern16", "__stack_chk_guard"}) {
    EXPECT_FALSE(ElfSet.count(Name)) << Name;
    EXPECT_TRUE(DarwinSet.count(Name)) << Name;
  }
  // 29 libm functions in three precisions, 5 libc and unwinder names, and 5
  // more on Darwin.
  EXPECT_EQ(Elf.size(), 29u * 3 + 5);
  EXPECT_EQ(Darwin.size(), 29u * 3 + 5 + 5);
}

TEST(SymbolTable, AnEmptyTableIsValid) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, "define void @f() {\n  ret void\n}\n");
  GlobalVariable *Table = emitSymbolTable(*M, {});
  EXPECT_EQ(cast<ArrayType>(Table->getValueType())->getNumElements(), 0u);
  EXPECT_FALSE(verifyModule(*M, &errs()));
  Table = extendSymbolTable(*M, *Table, {});
  EXPECT_EQ(cast<ArrayType>(Table->getValueType())->getNumElements(), 0u);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST(SymbolTable, KeepsEntriesAliveThroughOptimization) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, R"(
@k = internal constant i32 5
define internal i32 @dead(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
)");
  emitSymbolTable(*M, collectSymbols(*M));
  test::runO2(*M);
  ASSERT_NE(M->getFunction("dead"), nullptr);
  EXPECT_FALSE(M->getFunction("dead")->isDeclaration());
  ASSERT_NE(M->getNamedGlobal("k"), nullptr);
  auto E = entries(*M->getNamedGlobal(SymbolTableName));
  ASSERT_EQ(E.size(), 2u);
  EXPECT_EQ(E[0].first, "dead");
  EXPECT_EQ(E[1].first, "k");
}

TEST(SymbolTable, ExtendedNamesUseWhatOptimizationDeclared) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, R"(
declare double @pow(double, double)
define double @f(double %x) {
  %r = call double @pow(double 2.0, double %x)
  ret double %r
}
)");
  GlobalVariable *Table = emitSymbolTable(*M, collectSymbols(*M));
  EXPECT_EQ(M->getFunction("exp2"), nullptr);
  // pow(2.0, x) becomes exp2(x), which O2 declares.
  test::runO2(*M);
  Function *Exp2 = M->getFunction("exp2");
  ASSERT_NE(Exp2, nullptr);
  Table = extendSymbolTable(*M, *Table, {"exp2", "sin"});
  ASSERT_FALSE(verifyModule(*M, &errs()));
  auto E = entries(*Table);
  ASSERT_EQ(E.size(), 4u);
  EXPECT_EQ(E[0].first, "pow");
  EXPECT_EQ(E[1].first, "f");
  EXPECT_EQ(E[2].first, "exp2");
  EXPECT_EQ(E[2].second, Exp2);
  EXPECT_EQ(Exp2->getFunctionType()->getNumParams(), 1u);
  EXPECT_EQ(Exp2->getReturnType(), Type::getDoubleTy(Ctx));
  EXPECT_EQ(E[3].first, "sin");
}

TEST(SymbolTable, TheMemsetPattern16StubHasTheLibraryPrototype) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, "define void @f() {\n  ret void\n}\n");
  GlobalVariable *Table = emitSymbolTable(*M, collectSymbols(*M));
  extendSymbolTable(*M, *Table, {"memset_pattern16", "other"});
  ASSERT_FALSE(verifyModule(*M, &errs()));
  // void (ptr, ptr, size_t)
  FunctionType *Pattern = M->getFunction("memset_pattern16")->getFunctionType();
  EXPECT_TRUE(Pattern->getReturnType()->isVoidTy());
  ASSERT_EQ(Pattern->getNumParams(), 3u);
  EXPECT_TRUE(Pattern->getParamType(0)->isPointerTy());
  EXPECT_TRUE(Pattern->getParamType(1)->isPointerTy());
  EXPECT_EQ(Pattern->getParamType(2), M->getDataLayout().getIntPtrType(Ctx));
  // Any other stub is void ().
  EXPECT_EQ(M->getFunction("other")->getFunctionType(),
            FunctionType::get(Type::getVoidTy(Ctx), false));
}

} // namespace
