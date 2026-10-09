// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestSupport.h"
#include "orc-pgo/CounterLayout.h"

#include "llvm/IR/Instructions.h"
#include "gtest/gtest.h"

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

define i32 @sel(i32 %x, <2 x i1> %v) {
  %c = icmp ne i32 %x, 0
  %s = select i1 %c, i32 1, i32 2
  %t = select <2 x i1> %v, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer
  ret i32 %s
}

define void @onlydefault(i32 %x) {
entry:
  switch i32 %x, label %d []
d:
  ret void
}

declare void @ext()
define available_externally void @ae() { ret void }
define void @__orc_pgo_helper() { ret void }
)";

TEST(CounterLayout, InternalNames) {
  EXPECT_TRUE(isInternalName("__orc_pgo_counters"));
  EXPECT_TRUE(isInternalName("__orc_pgo"));
  EXPECT_FALSE(isInternalName("orc_pgo"));
  EXPECT_FALSE(isInternalName("main"));
}

TEST(CounterLayout, AssignsDeterministicPositions) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  CounterLayout L = CounterLayout::compute(*M);

  ASSERT_EQ(L.functions().size(), 4u); // branchy, sw, sel, onlydefault
  EXPECT_EQ(L.numCounters(), 3u + 4u + 3u + 1u);

  const FunctionCounters &B = L.functions()[0];
  EXPECT_EQ(B.F->getName(), "branchy");
  EXPECT_EQ(B.FunctionIndex, 0u);
  EXPECT_EQ(B.EntryCounter, 0u);
  ASSERT_EQ(B.Sites.size(), 1u);
  EXPECT_TRUE(isa<CondBrInst>(B.Sites[0].Inst));
  EXPECT_EQ(B.Sites[0].FirstCounter, 1u);
  EXPECT_EQ(B.Sites[0].NumCounters, 2u);

  const FunctionCounters &S = L.functions()[1];
  EXPECT_EQ(S.EntryCounter, 3u);
  ASSERT_EQ(S.Sites.size(), 1u);
  EXPECT_EQ(S.Sites[0].FirstCounter, 4u);
  EXPECT_EQ(S.Sites[0].NumCounters, 3u);

  const FunctionCounters &Sel = L.functions()[2];
  EXPECT_EQ(Sel.EntryCounter, 7u);
  ASSERT_EQ(Sel.Sites.size(), 1u); // vector select is not profiled
  EXPECT_TRUE(isa<SelectInst>(Sel.Sites[0].Inst));
  EXPECT_EQ(Sel.Sites[0].FirstCounter, 8u);

  const FunctionCounters &OD = L.functions()[3];
  EXPECT_EQ(OD.EntryCounter, 10u);
  EXPECT_TRUE(OD.Sites.empty()); // single-successor switch
}

TEST(CounterLayout, LookupAndSkippedFunctions) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  CounterLayout L = CounterLayout::compute(*M);
  EXPECT_EQ(L.lookup(*M->getFunction("sw"))->FunctionIndex, 1u);
  EXPECT_EQ(L.lookup(*M->getFunction("ext")), nullptr);
  EXPECT_EQ(L.lookup(*M->getFunction("ae")), nullptr);
  EXPECT_EQ(L.lookup(*M->getFunction("__orc_pgo_helper")), nullptr);
}

TEST(CounterLayout, SameIRSameLayout) {
  LLVMContext C1, C2;
  auto M1 = test::parseIR(C1, IR);
  auto M2 = test::parseIR(C2, IR);
  CounterLayout L1 = CounterLayout::compute(*M1);
  CounterLayout L2 = CounterLayout::compute(*M2);
  ASSERT_EQ(L1.numCounters(), L2.numCounters());
  for (size_t I = 0; I < L1.functions().size(); ++I) {
    EXPECT_EQ(L1.functions()[I].F->getName(), L2.functions()[I].F->getName());
    EXPECT_EQ(L1.functions()[I].EntryCounter, L2.functions()[I].EntryCounter);
  }
}

} // namespace
