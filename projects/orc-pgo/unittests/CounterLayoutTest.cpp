// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestSupport.h"
#include "orc-pgo/CounterLayout.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

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

// Several sites in one function. %mid comes before %early in block layout but
// after it in CFG order, so its switch must be numbered before %early's select.
constexpr const char *MultiSiteIR = R"(
define i32 @multi(i32 %x, i1 %c) {
entry:
  %s = select i1 %c, i32 1, i32 2
  br i1 %c, label %late, label %early
mid:
  switch i32 %x, label %d [ i32 0, label %d
                            i32 1, label %a
                            i32 2, label %a
                            i32 3, label %late ]
early:
  %e = select i1 %c, i32 3, i32 4
  br label %mid
a:
  ret i32 %s
d:
  ret i32 %e
late:
  ret i32 1
}
)";

// Skipped functions interleaved with eligible ones.
constexpr const char *InterleavedIR = R"(
declare void @a()
define void @f(i1 %c) {
  br i1 %c, label %t, label %t
t:
  ret void
}
define void @__orc_pgo_x() { ret void }
define available_externally void @g() { ret void }
define void @h() { ret void }
)";

// Multi-successor terminators that are deliberately not profiled.
constexpr const char *UnprofiledIR = R"(
declare i32 @__gxx_personality_v0(...)
declare void @may_throw()

define void @inv() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw() to label %ok unwind label %lp
ok:
  ret void
lp:
  %l = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %l
}

define void @ibr(ptr %p) {
entry:
  indirectbr ptr %p, [label %a, label %b]
a:
  ret void
b:
  ret void
}

define void @cbr() {
entry:
  callbr void asm "", "!i"() to label %a [label %b]
a:
  ret void
b:
  ret void
}
)";

constexpr const char *ArmsIR = R"(
define void @dup(i32 %x) {
entry:
  switch i32 %x, label %a [ i32 0, label %a
                            i32 1, label %a ]
a:
  ret void
}

define void @four(i32 %x) {
entry:
  switch i32 %x, label %d [ i32 0, label %a
                            i32 1, label %b
                            i32 2, label %a
                            i32 3, label %b ]
a:
  ret void
b:
  ret void
d:
  ret void
}

define void @same(i1 %c) {
entry:
  br i1 %c, label %t, label %t
t:
  ret void
}

define <2 x i32> @vecops(i1 %c, <2 x i32> %x, <2 x i32> %y) {
  %s = select i1 %c, <2 x i32> %x, <2 x i32> %y
  ret <2 x i32> %s
}
)";

unsigned blockIndex(const BasicBlock &BB) {
  unsigned I = 0;
  for (const BasicBlock &B : *BB.getParent()) {
    if (&B == &BB)
      break;
    ++I;
  }
  return I;
}

unsigned instIndex(const Instruction &Inst) {
  unsigned I = 0;
  for (const Instruction &J : *Inst.getParent()) {
    if (&J == &Inst)
      break;
    ++I;
  }
  return I;
}

/// Same indices for every function and site, and each site at the same
/// position (block, instruction, opcode) in its function.
void expectSameLayout(const CounterLayout &L1, const CounterLayout &L2) {
  ASSERT_EQ(L1.numCounters(), L2.numCounters());
  ASSERT_EQ(L1.functions().size(), L2.functions().size());
  for (size_t I = 0; I < L1.functions().size(); ++I) {
    const FunctionCounters &A = L1.functions()[I];
    const FunctionCounters &B = L2.functions()[I];
    EXPECT_EQ(A.F->getName(), B.F->getName());
    EXPECT_EQ(A.FunctionIndex, B.FunctionIndex);
    EXPECT_EQ(A.EntryCounter, B.EntryCounter);
    ASSERT_EQ(A.Sites.size(), B.Sites.size()) << A.F->getName().str();
    for (size_t S = 0; S < A.Sites.size(); ++S) {
      const CounterSite &SA = A.Sites[S];
      const CounterSite &SB = B.Sites[S];
      EXPECT_EQ(SA.FirstCounter, SB.FirstCounter);
      EXPECT_EQ(SA.NumCounters, SB.NumCounters);
      EXPECT_EQ(SA.Inst->getOpcode(), SB.Inst->getOpcode());
      EXPECT_EQ(blockIndex(*SA.Inst->getParent()), blockIndex(*SB.Inst->getParent()));
      EXPECT_EQ(instIndex(*SA.Inst), instIndex(*SB.Inst));
    }
  }
}

unsigned entryTerminatorCounters(Module &M, StringRef Fn) {
  return numSiteCounters(*M.getFunction(Fn)->getEntryBlock().getTerminator());
}

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
  EXPECT_EQ(S.F->getName(), "sw");
  EXPECT_EQ(S.FunctionIndex, 1u);
  EXPECT_EQ(S.EntryCounter, 3u);
  ASSERT_EQ(S.Sites.size(), 1u);
  EXPECT_TRUE(isa<SwitchInst>(S.Sites[0].Inst));
  EXPECT_EQ(S.Sites[0].FirstCounter, 4u);
  EXPECT_EQ(S.Sites[0].NumCounters, 3u);

  const FunctionCounters &Sel = L.functions()[2];
  EXPECT_EQ(Sel.F->getName(), "sel");
  EXPECT_EQ(Sel.FunctionIndex, 2u);
  EXPECT_EQ(Sel.EntryCounter, 7u);
  ASSERT_EQ(Sel.Sites.size(), 1u); // vector select is not profiled
  EXPECT_TRUE(isa<SelectInst>(Sel.Sites[0].Inst));
  EXPECT_EQ(Sel.Sites[0].FirstCounter, 8u);
  EXPECT_EQ(Sel.Sites[0].NumCounters, 2u);

  const FunctionCounters &OD = L.functions()[3];
  EXPECT_EQ(OD.F->getName(), "onlydefault");
  EXPECT_EQ(OD.FunctionIndex, 3u);
  EXPECT_EQ(OD.EntryCounter, 10u);
  EXPECT_TRUE(OD.Sites.empty()); // single-successor switch
}

TEST(CounterLayout, SitesAreContiguousInLayoutOrder) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, MultiSiteIR);
  CounterLayout L = CounterLayout::compute(*M);

  ASSERT_EQ(L.functions().size(), 1u);
  const FunctionCounters &FC = L.functions()[0];
  EXPECT_EQ(FC.EntryCounter, 0u);
  ASSERT_EQ(FC.Sites.size(), 4u);

  EXPECT_TRUE(isa<SelectInst>(FC.Sites[0].Inst));
  EXPECT_EQ(FC.Sites[0].FirstCounter, 1u);
  EXPECT_EQ(FC.Sites[0].NumCounters, 2u);

  EXPECT_TRUE(isa<CondBrInst>(FC.Sites[1].Inst));
  EXPECT_EQ(FC.Sites[1].FirstCounter, 3u);
  EXPECT_EQ(FC.Sites[1].NumCounters, 2u);

  EXPECT_TRUE(isa<SwitchInst>(FC.Sites[2].Inst));
  EXPECT_EQ(FC.Sites[2].Inst->getParent()->getName(), "mid");
  EXPECT_EQ(FC.Sites[2].FirstCounter, 5u);
  EXPECT_EQ(FC.Sites[2].NumCounters, 5u);

  EXPECT_TRUE(isa<SelectInst>(FC.Sites[3].Inst));
  EXPECT_EQ(FC.Sites[3].Inst->getParent()->getName(), "early");
  EXPECT_EQ(FC.Sites[3].FirstCounter, 10u);
  EXPECT_EQ(FC.Sites[3].NumCounters, 2u);

  EXPECT_EQ(L.numCounters(), 12u);
}

TEST(CounterLayout, SkippedFunctionsDoNotShiftIndices) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, InterleavedIR);
  CounterLayout L = CounterLayout::compute(*M);

  ASSERT_EQ(L.functions().size(), 2u);
  const FunctionCounters &F = L.functions()[0];
  EXPECT_EQ(F.F->getName(), "f");
  EXPECT_EQ(F.FunctionIndex, 0u);
  EXPECT_EQ(F.EntryCounter, 0u);
  ASSERT_EQ(F.Sites.size(), 1u);
  EXPECT_EQ(F.Sites[0].FirstCounter, 1u);

  const FunctionCounters &H = L.functions()[1];
  EXPECT_EQ(H.F->getName(), "h");
  EXPECT_EQ(H.FunctionIndex, 1u);
  EXPECT_EQ(H.EntryCounter, 3u);
  EXPECT_EQ(L.numCounters(), 4u);
}

// invoke, indirectbr and callbr have several successors but get no counters:
// an invoke's unwind edge is cold by default (and edges into landing pads
// can't be split), and indirectbr/callbr are rare with edges that generally
// can't be split. Counting them later changes every subsequent index, so it
// should be a deliberate change to this test.
TEST(CounterLayout, UnprofiledTerminators) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, UnprofiledIR);
  for (Function &F : *M)
    for (Instruction &I : instructions(F))
      EXPECT_EQ(numSiteCounters(I), 0u) << I.getOpcodeName();

  CounterLayout L = CounterLayout::compute(*M);
  ASSERT_EQ(L.functions().size(), 3u);
  for (const FunctionCounters &FC : L.functions())
    EXPECT_TRUE(FC.Sites.empty()) << FC.F->getName().str();
  EXPECT_EQ(L.numCounters(), 3u);
}

// Counters are per successor slot, not per unique destination block: switch
// branch_weights metadata carries one weight per slot (default first), and
// arms that share a block still need distinct counts.
TEST(CounterLayout, CountsArmsNotUniqueSuccessors) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, ArmsIR);
  EXPECT_EQ(entryTerminatorCounters(*M, "dup"), 3u);
  EXPECT_EQ(entryTerminatorCounters(*M, "four"), 5u);
  EXPECT_EQ(entryTerminatorCounters(*M, "same"), 2u);

  // An i1 condition counts even when the operands are vectors.
  Instruction &VecSel = M->getFunction("vecops")->getEntryBlock().front();
  ASSERT_TRUE(isa<SelectInst>(VecSel));
  EXPECT_EQ(numSiteCounters(VecSel), 2u);
}

TEST(CounterLayout, LookupAndSkippedFunctions) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  CounterLayout L = CounterLayout::compute(*M);
  for (const FunctionCounters &FC : L.functions())
    EXPECT_EQ(L.lookup(*FC.F), &FC);
  EXPECT_EQ(L.lookup(*M->getFunction("sw"))->FunctionIndex, 1u);
  EXPECT_EQ(L.lookup(*M->getFunction("ext")), nullptr);
  EXPECT_EQ(L.lookup(*M->getFunction("ae")), nullptr);
  EXPECT_EQ(L.lookup(*M->getFunction("__orc_pgo_helper")), nullptr);
}

TEST(CounterLayout, EmptyModule) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, "declare void @ext()");
  CounterLayout L = CounterLayout::compute(*M);
  EXPECT_TRUE(L.functions().empty());
  EXPECT_EQ(L.numCounters(), 0u);
  EXPECT_EQ(L.lookup(*M->getFunction("ext")), nullptr);
}

TEST(CounterLayout, SameIRSameLayout) {
  LLVMContext C1, C2;
  auto M1 = test::parseIR(C1, IR);
  auto M2 = test::parseIR(C2, IR);
  CounterLayout L1 = CounterLayout::compute(*M1);
  CounterLayout L2 = CounterLayout::compute(*M2);
  expectSameLayout(L1, L2);
  // Lookup is per module: a same-named function from the other copy is unknown.
  EXPECT_EQ(L1.lookup(*M2->getFunction("sw")), nullptr);
}

TEST(CounterLayout, BitcodeRoundTripPreservesLayout) {
  std::string Text = std::string(IR) + MultiSiteIR + InterleavedIR + UnprofiledIR + ArmsIR;
  LLVMContext C1, C2;
  auto M1 = test::parseIR(C1, Text);

  SmallVector<char, 0> Buf;
  raw_svector_ostream OS(Buf);
  WriteBitcodeToFile(*M1, OS);
  auto M2 = cantFail(
      parseBitcodeFile(MemoryBufferRef(StringRef(Buf.data(), Buf.size()), "roundtrip"), C2));

  CounterLayout L1 = CounterLayout::compute(*M1);
  CounterLayout L2 = CounterLayout::compute(*M2);
  ASSERT_GT(L1.numCounters(), 0u);
  expectSameLayout(L1, L2);
}

} // namespace
