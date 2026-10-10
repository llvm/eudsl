// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestSupport.h"
#include "orc-pgo/CounterLayout.h"
#include "orc-pgo/Instrument.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *IR = R"(
define i32 @branchy(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %pos, label %neg
pos:
  ret i32 1
neg:
  ret i32 -1
}
define i32 @sw(i32 %x) {
entry:
  switch i32 %x, label %d [ i32 0, label %a
                            i32 1, label %b ]
a:
  ret i32 10
b:
  ret i32 20
d:
  ret i32 30
}
define i32 @sel(i32 %x) {
  %c = icmp ne i32 %x, 0
  %s = select i1 %c, i32 1, i32 2
  ret i32 %s
}
)";

/// Instruments IR (checking the array has N elements), JITs it, lets Run call
/// into it, and returns the first N counters.
std::vector<uint64_t> runCounted(StringRef Text, size_t N,
                                 function_ref<void(orc::LLJIT &)> Run) {
  auto J = test::jitIR(Text, [&](Module &M) {
    GlobalVariable *GV = instrumentModule(M, CounterLayout::compute(M));
    EXPECT_EQ(cast<ArrayType>(GV->getValueType())->getNumElements(), N);
    EXPECT_FALSE(verifyModule(M, &errs()));
  });
  Run(*J);
  auto *C = test::lookupFn<uint64_t>(*J, CountersName);
  return std::vector<uint64_t>(C, C + N);
}

struct Accesses {
  unsigned Loads = 0;
  unsigned Stores = 0;
};

/// Atomic loads/stores in F; every counter increment contributes one of each.
Accesses countAccesses(const Function &F) {
  Accesses A;
  for (const Instruction &I : instructions(F)) {
    if (const auto *L = dyn_cast<LoadInst>(&I)) {
      A.Loads += L->isAtomic();
    } else if (const auto *S = dyn_cast<StoreInst>(&I)) {
      A.Stores += S->isAtomic();
    }
  }
  return A;
}

/// A pointer reduced to its base and accumulated constant byte offset.
struct StrippedPtr {
  const Value *Base;
  APInt Offset;
};

StrippedPtr stripPtr(const Value *Ptr, const DataLayout &DL) {
  APInt Off(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
  const Value *Base = Ptr->stripAndAccumulateConstantOffsets(
      DL, Off, /*AllowNonInbounds=*/false);
  return {Base, Off};
}

/// Checks that the load addresses Counters[Idx] for a constant Idx.
void expectConstantIndex(const LoadInst *L, const GlobalVariable *Counters,
                         uint64_t Idx) {
  StrippedPtr P = stripPtr(L->getPointerOperand(), L->getModule()->getDataLayout());
  EXPECT_EQ(P.Base, Counters);
  EXPECT_EQ(P.Offset.getZExtValue(), Idx * 8);
}

/// Checks that I is the store of a `load atomic monotonic; add 1; store atomic
/// monotonic` increment of an i64 and returns the load (nullptr on failure).
LoadInst *expectIncrement(Instruction *I) {
  auto *St = dyn_cast_or_null<StoreInst>(I);
  if (!St) {
    ADD_FAILURE() << "increment must end in a store";
    return nullptr;
  }
  EXPECT_TRUE(St->isAtomic());
  EXPECT_EQ(St->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(St->getAlign(), Align(8));
  auto *Add = dyn_cast<BinaryOperator>(St->getValueOperand());
  if (!Add || Add->getOpcode() != Instruction::Add) {
    ADD_FAILURE() << "stored value must be an add";
    return nullptr;
  }
  auto *Old = dyn_cast<LoadInst>(Add->getOperand(0));
  if (!Old) {
    ADD_FAILURE() << "add must increment the loaded value";
    return nullptr;
  }
  auto *One = dyn_cast<ConstantInt>(Add->getOperand(1));
  EXPECT_TRUE(One && One->getType()->isIntegerTy(64) && One->isOne());
  EXPECT_TRUE(Old->isAtomic());
  EXPECT_EQ(Old->getOrdering(), AtomicOrdering::Monotonic);
  EXPECT_EQ(Old->getAlign(), Align(8));
  EXPECT_TRUE(Old->getType()->isIntegerTy(64));
  const DataLayout &DL = St->getModule()->getDataLayout();
  StrippedPtr OldPtr = stripPtr(Old->getPointerOperand(), DL);
  StrippedPtr StPtr = stripPtr(St->getPointerOperand(), DL);
  EXPECT_EQ(OldPtr.Base, StPtr.Base);
  EXPECT_EQ(OldPtr.Offset, StPtr.Offset);
  // load, add, store are adjacent.
  EXPECT_EQ(Old->getNextNode(), Add);
  EXPECT_EQ(Add->getNextNode(), St);
  return Old;
}

/// Checks that the load addresses Counters[select(C, First, First + 1)] where
/// C is Cond, or freeze(Cond) if Frozen.
void expectSelectedIndex(const LoadInst *L, const GlobalVariable *Counters,
                         const Value *Cond, uint64_t First, bool Frozen) {
  ASSERT_NE(L, nullptr);
  auto *GEP = dyn_cast<GetElementPtrInst>(L->getPointerOperand());
  ASSERT_NE(GEP, nullptr);
  EXPECT_TRUE(GEP->isInBounds());
  EXPECT_EQ(GEP->getPointerOperand(), Counters);
  ASSERT_EQ(GEP->getNumIndices(), 2u);
  EXPECT_TRUE(cast<ConstantInt>(GEP->getOperand(1))->isZero());
  auto *Sel = dyn_cast<SelectInst>(GEP->getOperand(2));
  ASSERT_NE(Sel, nullptr);
  if (Frozen) {
    auto *Fr = dyn_cast<FreezeInst>(Sel->getCondition());
    ASSERT_NE(Fr, nullptr);
    EXPECT_EQ(Fr->getOperand(0), Cond);
  } else {
    EXPECT_EQ(Sel->getCondition(), Cond);
  }
  EXPECT_EQ(cast<ConstantInt>(Sel->getTrueValue())->getZExtValue(), First);
  EXPECT_EQ(cast<ConstantInt>(Sel->getFalseValue())->getZExtValue(), First + 1);
}

TEST(Instrument, CreatesCounterArray) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  CounterLayout L = CounterLayout::compute(*M);
  GlobalVariable *GV = instrumentModule(*M, L);
  EXPECT_EQ(GV->getName(), CountersName);
  EXPECT_EQ(GV->getVisibility(), GlobalValue::HiddenVisibility);
  EXPECT_EQ(cast<ArrayType>(GV->getValueType())->getNumElements(), 10u);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST(Instrument, ArrayIsZeroInitializedMutableI64) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  EXPECT_EQ(M->getNamedGlobal(CountersName), GV);
  EXPECT_TRUE(cast<ArrayType>(GV->getValueType())->getElementType()->isIntegerTy(64));
  EXPECT_FALSE(GV->isConstant());
  EXPECT_EQ(GV->getLinkage(), GlobalValue::ExternalLinkage);
  EXPECT_EQ(GV->getAlign(), Align(8));
  ASSERT_TRUE(GV->hasInitializer());
  EXPECT_TRUE(GV->getInitializer()->isNullValue());
}

TEST(Instrument, CountsEntriesAndEdges) {
  auto J = test::jitIR(IR, [](Module &M) {
    instrumentModule(M, CounterLayout::compute(M));
  });
  auto *Branchy = test::lookupFn<int(int)>(*J, "branchy");
  auto *Sw = test::lookupFn<int(int)>(*J, "sw");
  auto *Sel = test::lookupFn<int(int)>(*J, "sel");
  for (int X : {5, 5, 5, -1})
    Branchy(X);
  for (int X : {7, 7, 7, 0, 1, 1})
    Sw(X);
  for (int X : {1, 0, 0})
    Sel(X);
  auto *C = test::lookupFn<uint64_t>(*J, CountersName);
  std::vector<uint64_t> Got(C, C + 10);
  // branchy: entry, true, false | sw: entry, default, case0, case1 | sel: entry, true, false
  EXPECT_EQ(Got, (std::vector<uint64_t>{4, 3, 1, 6, 3, 1, 2, 3, 1, 2}));
}

TEST(Instrument, EmptyLayoutStillCreatesArray) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, "declare void @f()");
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  EXPECT_EQ(cast<ArrayType>(GV->getValueType())->getNumElements(), 0u);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST(Instrument, UntakenSitesStayZero) {
  // Only the true side of the branch and select, and the first case, are ever
  // taken; every other edge counter must still read zero.
  auto Got = runCounted(IR, 10, [](orc::LLJIT &J) {
    test::lookupFn<int(int)>(J, "branchy")(1);
    test::lookupFn<int(int)>(J, "sw")(0);
    test::lookupFn<int(int)>(J, "sel")(1);
  });
  EXPECT_EQ(Got, (std::vector<uint64_t>{1, 1, 0, 1, 0, 1, 0, 1, 1, 0}));
}

TEST(Instrument, BranchIncrementPrecedesBranchAndTestsItsCondition) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
define i32 @branchy(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %pos, label %neg
pos:
  ret i32 1
neg:
  ret i32 -1
})");
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function &F = *M->getFunction("branchy");
  BasicBlock &Entry = F.getEntryBlock();
  EXPECT_EQ(F.size(), 3u);
  auto *Br = cast<CondBrInst>(Entry.getTerminator());
  // Entry increment is first; the site increment is the last thing before br.
  LoadInst *SiteLoad = expectIncrement(Br->getPrevNode());
  expectSelectedIndex(SiteLoad, GV, Br->getCondition(), 1, /*Frozen=*/false);
  auto *First = dyn_cast<LoadInst>(&Entry.front());
  ASSERT_NE(First, nullptr);
  EXPECT_NE(First, SiteLoad);
  EXPECT_EQ(expectIncrement(First->getNextNode()->getNextNode()), First);
  expectConstantIndex(First, GV, 0);
  // Nothing else in the function touches memory: 2 increments only.
  EXPECT_EQ(countAccesses(F).Loads, 2u);
  EXPECT_EQ(countAccesses(F).Stores, 2u);
}

TEST(Instrument, SelectIncrementPrecedesSelectAndTestsItsCondition) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
define i32 @sel(i32 %x) {
  %c = icmp ne i32 %x, 0
  %s = select i1 %c, i32 1, i32 2
  ret i32 %s
})");
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function &F = *M->getFunction("sel");
  auto *Sel = cast<SelectInst>(F.getEntryBlock().getTerminator()->getPrevNode());
  LoadInst *SiteLoad = expectIncrement(Sel->getPrevNode());
  expectSelectedIndex(SiteLoad, GV, Sel->getCondition(), 1, /*Frozen=*/true);
  EXPECT_EQ(countAccesses(F).Loads, 2u);
  EXPECT_EQ(countAccesses(F).Stores, 2u);
}

TEST(Instrument, EntryIncrementFollowsLeadingAllocas) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
define i32 @allocas(i32 %x) {
entry:
  %a = alloca i32
  %b = alloca i64
  store i32 %x, ptr %a
  %v = load i32, ptr %a
  ret i32 %v
})");
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  BasicBlock &Entry = M->getFunction("allocas")->getEntryBlock();
  auto It = Entry.begin();
  ASSERT_TRUE(isa<AllocaInst>(*It++));
  ASSERT_TRUE(isa<AllocaInst>(*It++));
  auto *Load = dyn_cast<LoadInst>(&*It);
  ASSERT_NE(Load, nullptr);
  EXPECT_TRUE(Load->isAtomic());
  expectConstantIndex(Load, GV, 0);
  Instruction *Store = Load->getNextNode()->getNextNode();
  EXPECT_EQ(expectIncrement(Store), Load);
  // The original body follows the increment, unchanged.
  auto *Orig = dyn_cast<StoreInst>(Store->getNextNode());
  ASSERT_NE(Orig, nullptr);
  EXPECT_FALSE(Orig->isAtomic());
  EXPECT_EQ(Orig->getValueOperand(), M->getFunction("allocas")->getArg(0));
  EXPECT_EQ(Entry.size(), 2u + 3u + 3u);
}

TEST(Instrument, SiteAsFirstInstructionFollowsEntryIncrement) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
define i32 @firstbr(i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  ret i32 1
f:
  ret i32 2
}
define i32 @firstsel(i1 %c) {
  %s = select i1 %c, i32 1, i32 2
  ret i32 %s
})");
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));

  // firstbr: entry increment (counter 0), br increment (counters 1/2), br.
  {
    BasicBlock &Entry = M->getFunction("firstbr")->getEntryBlock();
    auto *Br = cast<CondBrInst>(Entry.getTerminator());
    auto *EntryLoad = dyn_cast<LoadInst>(&Entry.front());
    ASSERT_NE(EntryLoad, nullptr);
    expectConstantIndex(EntryLoad, GV, 0);
    Instruction *EntryStore = EntryLoad->getNextNode()->getNextNode();
    EXPECT_EQ(expectIncrement(EntryStore), EntryLoad);
    auto *Idx = dyn_cast<SelectInst>(EntryStore->getNextNode());
    ASSERT_NE(Idx, nullptr);
    EXPECT_EQ(Idx->getCondition(), Br->getCondition());
    expectSelectedIndex(expectIncrement(Br->getPrevNode()), GV, Br->getCondition(),
                        1, /*Frozen=*/false);
    EXPECT_EQ(Entry.size(), 3u + 5u + 1u);
  }

  // firstsel: entry increment (counter 3), select increment (counters 4/5), select.
  {
    BasicBlock &Entry = M->getFunction("firstsel")->getEntryBlock();
    auto *Sel = cast<SelectInst>(Entry.getTerminator()->getPrevNode());
    auto *EntryLoad = dyn_cast<LoadInst>(&Entry.front());
    ASSERT_NE(EntryLoad, nullptr);
    expectConstantIndex(EntryLoad, GV, 3);
    Instruction *EntryStore = EntryLoad->getNextNode()->getNextNode();
    EXPECT_EQ(expectIncrement(EntryStore), EntryLoad);
    EXPECT_TRUE(isa<FreezeInst>(EntryStore->getNextNode()));
    expectSelectedIndex(expectIncrement(Sel->getPrevNode()), GV, Sel->getCondition(),
                        4, /*Frozen=*/true);
    EXPECT_EQ(Entry.size(), 3u + 6u + 1u + 1u);
  }
}

TEST(Instrument, SitesAsFirstInstructionAreCounted) {
  auto Got = runCounted(R"(
define i32 @firstbr(i1 zeroext %c) {
entry:
  br i1 %c, label %t, label %f
t:
  ret i32 1
f:
  ret i32 2
}
define i32 @firstsel(i1 zeroext %c) {
  %s = select i1 %c, i32 1, i32 2
  ret i32 %s
})", 6, [](orc::LLJIT &J) {
    auto *Br = test::lookupFn<int(bool)>(J, "firstbr");
    auto *Sel = test::lookupFn<int(bool)>(J, "firstsel");
    for (bool X : {true, true, true, false})
      EXPECT_EQ(Br(X), X ? 1 : 2);
    for (bool X : {true, false, false})
      EXPECT_EQ(Sel(X), X ? 1 : 2);
  });
  EXPECT_EQ(Got, (std::vector<uint64_t>{4, 3, 1, 3, 1, 2}));
}

TEST(Instrument, EntryIncrementPrecedesDynamicAlloca) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
define i32 @dyn(i32 %n) {
entry:
  %s = alloca i32
  %a = alloca i32, i32 %n
  store i32 %n, ptr %a
  %v = load i32, ptr %a
  ret i32 %v
})");
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  BasicBlock &Entry = M->getFunction("dyn")->getEntryBlock();
  auto It = Entry.begin();
  auto *Static = dyn_cast<AllocaInst>(&*It++);
  ASSERT_NE(Static, nullptr);
  EXPECT_TRUE(Static->isStaticAlloca());
  auto *Load = dyn_cast<LoadInst>(&*It);
  ASSERT_NE(Load, nullptr);
  expectConstantIndex(Load, GV, 0);
  Instruction *Store = Load->getNextNode()->getNextNode();
  EXPECT_EQ(expectIncrement(Store), Load);
  auto *Dynamic = dyn_cast<AllocaInst>(Store->getNextNode());
  ASSERT_NE(Dynamic, nullptr);
  EXPECT_FALSE(Dynamic->isStaticAlloca());
  EXPECT_EQ(Entry.size(), 1u + 3u + 4u);
}

TEST(Instrument, SwitchOnNarrowAndWideConditions) {
  auto Got = runCounted(R"(
define i32 @sw8(i8 signext %x) {
entry:
  switch i8 %x, label %d [ i8 3, label %a
                           i8 -2, label %b ]
a:
  ret i32 1
b:
  ret i32 2
d:
  ret i32 3
}
define i32 @sw64(i64 %x) {
entry:
  switch i64 %x, label %d [ i64 4294967296, label %a
                            i64 -1, label %b ]
a:
  ret i32 1
b:
  ret i32 2
d:
  ret i32 3
})", 8, [](orc::LLJIT &J) {
    auto *Sw8 = test::lookupFn<int(int8_t)>(J, "sw8");
    auto *Sw64 = test::lookupFn<int(int64_t)>(J, "sw64");
    for (int8_t X : {3, 3, -2, 9})
      Sw8(X);
    // 0 is what 4294967296 truncates to; it must take the default.
    for (int64_t X : {int64_t(1) << 32, int64_t(1) << 32, int64_t(-1), int64_t(0),
                      int64_t(7), int64_t(7)})
      Sw64(X);
  });
  // sw8: entry, default, 3, -2 | sw64: entry, default, 2^32, -1
  EXPECT_EQ(Got, (std::vector<uint64_t>{4, 1, 2, 1, 6, 3, 2, 1}));
}

TEST(Instrument, CountsSurviveO2Pipeline) {
  auto J = test::jitIR(R"(
define i32 @prog(i32 %x, i32 %n) {
entry:
  %pos = icmp sgt i32 %x, 0
  br i1 %pos, label %loop, label %neg
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc2, %loop ]
  %odd = and i32 %i, 1
  %isodd = icmp ne i32 %odd, 0
  %d = select i1 %isodd, i32 3, i32 1
  %acc2 = add i32 %acc, %d
  %next = add i32 %i, 1
  %more = icmp slt i32 %next, %n
  br i1 %more, label %loop, label %after
after:
  switch i32 %acc2, label %other [ i32 1, label %one
                                   i32 4, label %four ]
one:
  ret i32 100
four:
  ret i32 200
other:
  ret i32 %acc2
neg:
  ret i32 -1
})", [](Module &M) {
    instrumentModule(M, CounterLayout::compute(M));
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2).run(M, MAM);
    EXPECT_FALSE(verifyModule(M, &errs()));
  });
  auto *Prog = test::lookupFn<int(int, int)>(*J, "prog");
  // acc after n iterations of +1, +3, +1, ...: n=1 -> 1, n=2 -> 4, n=3 -> 5.
  EXPECT_EQ(Prog(1, 1), 100);
  EXPECT_EQ(Prog(1, 2), 200);
  EXPECT_EQ(Prog(1, 3), 5);
  EXPECT_EQ(Prog(0, 5), -1);
  auto *C = test::lookupFn<uint64_t>(*J, CountersName);
  std::vector<uint64_t> Got(C, C + 10);
  // entry | x>0 T/F | select odd/even | back edge T / exit | switch default/1/4
  EXPECT_EQ(Got, (std::vector<uint64_t>{4, 3, 1, 2, 4, 3, 3, 1, 1, 1}));
}

TEST(Instrument, FunctionWithAllocasStillRuns) {
  auto Got = runCounted(R"(
define i32 @allocas(i32 %x) {
entry:
  %a = alloca i32
  store i32 %x, ptr %a
  %v = load i32, ptr %a
  ret i32 %v
})", 1, [](orc::LLJIT &J) {
    EXPECT_EQ(test::lookupFn<int(int)>(J, "allocas")(41), 41);
    EXPECT_EQ(test::lookupFn<int(int)>(J, "allocas")(42), 42);
  });
  EXPECT_EQ(Got, (std::vector<uint64_t>{2}));
}

TEST(Instrument, SwitchArmsSharingDestinationHaveDistinctCounters) {
  constexpr const char *Text = R"(
define i32 @shared(i32 %x) {
entry:
  switch i32 %x, label %a [ i32 0, label %a
                            i32 1, label %a ]
a:
  ret i32 %x
}
define i32 @mixed(i32 %x) {
entry:
  switch i32 %x, label %b [ i32 0, label %a
                            i32 1, label %b
                            i32 2, label %a ]
a:
  ret i32 1
b:
  ret i32 2
})";
  auto Got = runCounted(Text, 9, [](orc::LLJIT &J) {
    auto *Shared = test::lookupFn<int(int)>(J, "shared");
    auto *Mixed = test::lookupFn<int(int)>(J, "mixed");
    for (int X : {5, 5, 5, 0, 1, 1})
      Shared(X);
    for (int X : {0, 1, 1, 2, 2, 2, 9, 9, 9, 9})
      Mixed(X);
  });
  // shared: entry, default, case0, case1 | mixed: entry, default, case0, case1, case2
  EXPECT_EQ(Got, (std::vector<uint64_t>{6, 3, 1, 2, 10, 4, 1, 2, 3}));
}

TEST(Instrument, SwitchCountersFollowSlotsNotCaseValues) {
  auto Got = runCounted(R"(
define i32 @sw(i32 %x) {
entry:
  switch i32 %x, label %d [ i32 100, label %a
                            i32 -5, label %b
                            i32 7, label %c ]
a:
  ret i32 1
b:
  ret i32 2
c:
  ret i32 3
d:
  ret i32 4
})", 5, [](orc::LLJIT &J) {
    auto *Sw = test::lookupFn<int(int)>(J, "sw");
    for (int X : {100, -5, -5, 7, 7, 7, 0, 0, 0, 0, 1})
      Sw(X);
  });
  // entry, default, 100, -5, 7: slot order, not value order.
  EXPECT_EQ(Got, (std::vector<uint64_t>{11, 5, 1, 2, 3}));
}

TEST(Instrument, EachSiteUsesItsOwnCounters) {
  auto Got = runCounted(R"(
define i32 @multi(i32 %x) {
entry:
  %c = icmp slt i32 %x, 10
  %s = select i1 %c, i32 1, i32 2
  %small = icmp eq i32 %s, 1
  br i1 %small, label %mid, label %big
mid:
  switch i32 %x, label %dflt [ i32 3, label %three ]
three:
  ret i32 3
dflt:
  ret i32 0
big:
  ret i32 99
}
define i32 @after(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %pos, label %neg
pos:
  ret i32 1
neg:
  ret i32 -1
})", 10, [](orc::LLJIT &J) {
    auto *Multi = test::lookupFn<int(int)>(J, "multi");
    auto *After = test::lookupFn<int(int)>(J, "after");
    for (int X : {3, 3, 5, 20, 20, 20, 20})
      Multi(X);
    for (int X : {1, 1, -1})
      After(X);
  });
  // multi: entry, select T/F, br T/F, switch default/case3 | after: entry, T/F
  EXPECT_EQ(Got, (std::vector<uint64_t>{7, 3, 4, 3, 4, 1, 2, 3, 2, 1}));
}

TEST(Instrument, CountsLoopBackEdge) {
  auto Got = runCounted(R"(
define i32 @loop(i32 %n) {
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  %next = add i32 %i, 1
  %c = icmp slt i32 %next, %n
  br i1 %c, label %body, label %exit
exit:
  ret i32 %next
})", 3, [](orc::LLJIT &J) {
    auto *Loop = test::lookupFn<int(int)>(J, "loop");
    EXPECT_EQ(Loop(5), 5);
    EXPECT_EQ(Loop(3), 3);
  });
  // entry: 2 calls; back edge taken 4 + 2 times; exited twice.
  EXPECT_EQ(Got, (std::vector<uint64_t>{2, 6, 2}));
}

TEST(Instrument, SkippedFunctionsGetNoIncrements) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
declare i32 @external(i1)
define available_externally i32 @avail(i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  ret i32 1
f:
  ret i32 2
}
define i32 @__orc_pgo_helper(i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  ret i32 1
f:
  ret i32 2
}
define i32 @real(i1 %c) {
  ret i32 0
})");
  CounterLayout L = CounterLayout::compute(*M);
  GlobalVariable *GV = instrumentModule(*M, L);
  EXPECT_FALSE(verifyModule(*M, &errs()));
  EXPECT_EQ(cast<ArrayType>(GV->getValueType())->getNumElements(), 1u);
  for (const char *Name : {"avail", "__orc_pgo_helper"}) {
    Function &F = *M->getFunction(Name);
    EXPECT_EQ(F.size(), 3u) << Name;
    EXPECT_EQ(F.getInstructionCount(), 3u) << Name;
    EXPECT_EQ(countAccesses(F).Loads, 0u) << Name;
    EXPECT_EQ(countAccesses(F).Stores, 0u) << Name;
  }
  EXPECT_TRUE(M->getFunction("external")->isDeclaration());
  Function &Real = *M->getFunction("real");
  EXPECT_EQ(Real.getInstructionCount(), 4u);
  EXPECT_EQ(countAccesses(Real).Loads, 1u);
  EXPECT_EQ(countAccesses(Real).Stores, 1u);
}

TEST(Instrument, UnprofiledInstructionsGetNoIncrements) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
define <2 x i32> @vec(i32 %x, <2 x i1> %v) {
  %t = select <2 x i1> %v, <2 x i32> zeroinitializer, <2 x i32> <i32 1, i32 1>
  ret <2 x i32> %t
}
define void @onlydefault(i32 %x) {
entry:
  switch i32 %x, label %d []
d:
  ret void
}
define void @uncond() {
entry:
  br label %next
next:
  ret void
})");
  GlobalVariable *GV = instrumentModule(*M, CounterLayout::compute(*M));
  EXPECT_FALSE(verifyModule(*M, &errs()));
  EXPECT_EQ(cast<ArrayType>(GV->getValueType())->getNumElements(), 3u);
  for (const char *Name : {"vec", "onlydefault", "uncond"}) {
    EXPECT_EQ(countAccesses(*M->getFunction(Name)).Loads, 1u) << Name;
    EXPECT_EQ(countAccesses(*M->getFunction(Name)).Stores, 1u) << Name;
  }
}

TEST(Instrument, EveryIncrementIsMonotonicAtomicLoadAddStore) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  instrumentModule(*M, CounterLayout::compute(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  unsigned Increments = 0;
  for (Function &F : *M) {
    for (Instruction &I : instructions(F)) {
      EXPECT_FALSE(isa<AtomicRMWInst>(I));
      EXPECT_FALSE(isa<AtomicCmpXchgInst>(I));
      EXPECT_FALSE(isa<FenceInst>(I));
      if (isa<StoreInst>(I)) {
        EXPECT_NE(expectIncrement(&I), nullptr);
        ++Increments;
      }
    }
  }
  // Entry counters (3) plus branchy (1 site), sw (1 site), sel (1 site).
  EXPECT_EQ(Increments, 6u);
}

} // namespace
