// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/TierUp.h"
#include "SimulatedAot.h"
#include "orc-pgo/ProfileBuilder.h"
#include "orc-pgo/Snapshot.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *ProgramIR = R"(
source_filename = "tierup.c"

@table = constant [2 x i32] [i32 3, i32 5]
@state = global i32 0
@valias = alias i32, ptr @state
@falias = alias i32 (i32), ptr @far
@llvm.used = appending global [1 x ptr] [ptr @far], section "llvm.metadata"

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare extern_weak i32 @weakref(i32)

define i32 @far(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
define i32 @leaf(i32 %x) {
  %p = getelementptr [2 x i32], ptr @table, i64 0, i64 1
  %v = load i32, ptr %p
  %r = call i32 @far(i32 %v)
  %s = add i32 %r, %x
  ret i32 %s
}
define i32 @jumpy(i32 %x) {
entry:
  indirectbr ptr blockaddress(@jumpy, %target), [label %target]
target:
  ret i32 %x
}
define i32 @mid(i32 %x) {
  %a = call i32 @leaf(i32 %x)
  %b = call i32 @jumpy(i32 %a)
  ret i32 %b
}
define i32 @hot(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %pos, label %neg
pos:
  %r = call i32 @mid(i32 %x)
  store i32 %r, ptr @state
  %s = call i32 @falias(i32 %r)
  ret i32 %s
neg:
  %w = load i32, ptr @valias
  ret i32 %w
}
define i32 @callit(ptr %f, i32 %x) {
  %r = call i32 %f(i32 %x)
  ret i32 %r
}
define void @copy(ptr %d, ptr %s, i64 %n) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 %n, i1 false)
  ret void
}
define internal i32 @helper(i32 %x) noinline {
  %r = mul i32 %x, 3
  ret i32 %r
}
define i32 @useshelper(i32 %x) {
  %r = call i32 @helper(i32 %x)
  ret i32 %r
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 4}
)";

std::unique_ptr<MemoryBuffer>
profileOf(ArrayRef<OrcPgoFunctionRecord> Records) {
  return cantFail(buildIndexedProfile(Records));
}

std::unique_ptr<MemoryBuffer> profileOf(const test::SimulatedAot &S) {
  return profileOf(ArrayRef(S.Image->Functions, S.Image->NumFunctions));
}

std::unique_ptr<Module> snapshotOf(const test::SimulatedAot &S,
                                   LLVMContext &Ctx) {
  return cantFail(
      parseSnapshot(StringRef(S.Image->Snapshot, S.Image->SnapshotSize), Ctx));
}

bool comparesWith(Function &F, const Value *V) {
  for (Instruction &I : instructions(F))
    if (auto *Cmp = dyn_cast<ICmpInst>(&I))
      if (Cmp->getOperand(0) == V || Cmp->getOperand(1) == V)
        return true;
  return false;
}

bool callsDirectly(Function &F, const Function *Callee) {
  for (Instruction &I : instructions(F))
    if (auto *CB = dyn_cast<CallBase>(&I))
      if (CB->getCalledFunction() == Callee)
        return true;
  return false;
}

/// Whether F branches on argument ArgNo being Size.
bool testsArgAgainst(Function &F, unsigned ArgNo, uint64_t Size) {
  Argument *N = F.getArg(ArgNo);
  for (Instruction &I : instructions(F)) {
    if (auto *Cmp = dyn_cast<ICmpInst>(&I)) {
      auto *C = dyn_cast<ConstantInt>(Cmp->getOperand(1));
      if (Cmp->getOperand(0) == N && C && C->getZExtValue() == Size)
        return true;
    }
    if (auto *SI = dyn_cast<SwitchInst>(&I)) {
      if (SI->getCondition() == N &&
          SI->findCaseValue(ConstantInt::get(cast<IntegerType>(N->getType()),
                                             Size)) != SI->case_default()) {
        return true;
      }
    }
  }
  return false;
}

bool copiesConstantSize(Function &F, uint64_t Size) {
  for (Instruction &I : instructions(F))
    if (auto *MC = dyn_cast<MemCpyInst>(&I))
      if (auto *Len = dyn_cast<ConstantInt>(MC->getLength());
          Len && Len->getZExtValue() == Size)
        return true;
  return false;
}

TEST(TierUp, ShapesAnAnnotatedBatchModule) {
  auto S = test::simulateAot(ProgramIR);
  auto *Hot = S.sym<int(int)>("hot");
  for (int I = 0; I < 30; ++I)
    ASSERT_EQ(Hot(2), 9);
  for (int I = 0; I < 10; ++I)
    ASSERT_EQ(Hot(-1), 8);

  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  Expected<TierUpBatch> Batch =
      prepareTierUpModule(*M, *profileOf(S), {"hot"}, 2);
  ASSERT_EQ(test::errorText(Batch), "");
  EXPECT_EQ(Batch->Names, std::vector<std::string>{"hot.orc_pgo.tier"});
  EXPECT_EQ(Batch->PromotedIndirectCalls, 0u);
  ASSERT_FALSE(verifyModule(*M, &errs()));

  Function *Root = M->getFunction("hot.orc_pgo.tier");
  ASSERT_NE(Root, nullptr);
  EXPECT_EQ(M->getFunction("hot"), nullptr);
  EXPECT_TRUE(Root->hasExternalLinkage());
  EXPECT_FALSE(Root->isDeclaration());
  EXPECT_EQ(Root->getEntryCount(), std::optional<uint64_t>(40));
  SmallVector<uint32_t> Weights;
  ASSERT_TRUE(
      extractBranchWeights(*Root->getEntryBlock().getTerminator(), Weights));
  EXPECT_EQ(Weights, (SmallVector<uint32_t>{30, 10}));

  for (StringRef Name : {"mid", "leaf"}) {
    Function *F = M->getFunction(Name);
    EXPECT_TRUE(F->hasAvailableExternallyLinkage()) << Name.str();
    EXPECT_FALSE(F->isDSOLocal()) << Name.str();
    EXPECT_EQ(F->getEntryCount(), std::optional<uint64_t>(30)) << Name.str();
  }
  for (StringRef Name :
       {"far", "jumpy", "callit", "copy", "helper", "useshelper", "falias"}) {
    Function *F = M->getFunction(Name);
    ASSERT_NE(F, nullptr) << Name.str();
    EXPECT_TRUE(F->isDeclaration()) << Name.str();
    EXPECT_TRUE(F->hasExternalLinkage()) << Name.str();
    EXPECT_FALSE(F->isDSOLocal()) << Name.str();
  }
  EXPECT_TRUE(M->getFunction("weakref")->hasExternalWeakLinkage());
  EXPECT_EQ(M->getNamedAlias("falias"), nullptr);
  EXPECT_EQ(M->getNamedAlias("valias"), nullptr);
  EXPECT_TRUE(M->getNamedGlobal("valias")->isDeclaration());
  EXPECT_TRUE(M->getNamedGlobal("table")->hasAvailableExternallyLinkage());
  EXPECT_TRUE(M->getNamedGlobal("state")->isDeclaration());
  EXPECT_EQ(M->getNamedGlobal("llvm.used"), nullptr);
  EXPECT_NE(M->getModuleFlag("wchar_size"), nullptr);
  EXPECT_NE(M->getProfileSummary(/*IsCS=*/false), nullptr);
}

TEST(TierUp, DepthZeroCopiesNoCallees) {
  auto S = test::simulateAot(ProgramIR);
  S.sym<int(int)>("hot")(2);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  ASSERT_EQ(test::errorText(
                prepareTierUpModule(*M, *profileOf(S), {"hot"}, 0).takeError()),
            "");
  EXPECT_TRUE(M->getFunction("mid")->isDeclaration());
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

TEST(TierUp, LocalRootsAreFoundByTheirPgoName) {
  auto S = test::simulateAot(ProgramIR);
  for (int I = 0; I < 5; ++I)
    ASSERT_EQ(S.sym<int(int)>("useshelper")(2), 6);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  Expected<TierUpBatch> Batch =
      prepareTierUpModule(*M, *profileOf(S), {"tierup.c;helper"}, 0);
  ASSERT_EQ(test::errorText(Batch), "");
  EXPECT_EQ(Batch->Names, std::vector<std::string>{"helper.orc_pgo.tier"});
  Function *Root = M->getFunction("helper.orc_pgo.tier");
  EXPECT_TRUE(Root->hasExternalLinkage());
  EXPECT_EQ(Root->getEntryCount(), std::optional<uint64_t>(5));
}

TEST(TierUp, AnUnknownRootIsAnError) {
  auto S = test::simulateAot(ProgramIR);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  EXPECT_EQ(
      test::errorText(
          prepareTierUpModule(*M, *profileOf(S), {"nope"}, 2).takeError()),
      "the snapshot has no function with PGO name 'nope'");
}

TEST(TierUp, AnnotationErrorsFailTheBatch) {
  auto S = test::simulateAot(ProgramIR);
  S.sym<int(int)>("hot")(2);
  std::vector<OrcPgoFunctionRecord> Records(
      S.Image->Functions, S.Image->Functions + S.Image->NumFunctions);
  for (OrcPgoFunctionRecord &R : Records)
    if (StringRef(R.Name) == "hot")
      R.CfgHash ^= 1;
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  std::string Error = test::errorText(
      prepareTierUpModule(*M, *profileOf(Records), {"hot"}, 2).takeError());
  EXPECT_NE(Error.find("hash mismatch"), std::string::npos) << Error;
}

TEST(TierUp, AHotIndirectCallBecomesAGuardedDirectCall) {
  auto S = test::simulateAot(ProgramIR);
  auto *CallIt = S.sym<int(int (*)(int), int)>("callit");
  auto *Far = S.sym<int(int)>("far");
  for (int I = 0; I < 2000; ++I)
    ASSERT_EQ(CallIt(Far, I), I + 1);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  Expected<TierUpBatch> Batch =
      prepareTierUpModule(*M, *profileOf(S), {"callit"}, 0);
  ASSERT_EQ(test::errorText(Batch), "");
  EXPECT_EQ(Batch->PromotedIndirectCalls, 1u);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function *Root = M->getFunction("callit.orc_pgo.tier");
  Function *Target = M->getFunction("far");
  EXPECT_TRUE(comparesWith(*Root, Target));
  EXPECT_TRUE(callsDirectly(*Root, Target));

  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  optimizeTierUp(*M, *TM);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  EXPECT_TRUE(comparesWith(*M->getFunction("callit.orc_pgo.tier"), Target));
}

// A target of indirect calls that is also a root. The program holds the
// address of its thunk (here in a table; in general in vtables and in
// pointers it has handed out), so a promoted compare has to be against that.
constexpr const char *IcpIR = R"(
source_filename = "icp.c"
@fptrs = constant [1 x ptr] [ptr @far]
define i32 @far(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
define i32 @callit(ptr %f, i32 %x) {
  %r = call i32 %f(i32 %x)
  ret i32 %r
}
define ptr @addr() {
  ret ptr @far
}
)";

TEST(TierUp, APromotedCompareIsAgainstTheProgramsAddressOfARootTarget) {
  auto S = test::simulateAot(IcpIR);
  auto *CallIt = S.sym<int(int (*)(int), int)>("callit");
  auto *Far = S.sym<int(int)>("far");
  for (int I = 0; I < 2000; ++I)
    ASSERT_EQ(CallIt(Far, I), I + 1);
  S.sym<void *()>("addr")();
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  Expected<TierUpBatch> Batch =
      prepareTierUpModule(*M, *profileOf(S), {"callit", "far", "addr"}, 0);
  ASSERT_EQ(test::errorText(Batch), "");
  EXPECT_EQ(Batch->Names,
            (std::vector<std::string>{"callit.orc_pgo.tier", "far.orc_pgo.tier",
                                      "addr.orc_pgo.tier"}));
  EXPECT_EQ(Batch->PromotedIndirectCalls, 1u);
  ASSERT_FALSE(verifyModule(*M, &errs()));

  // `far` is the thunk's name: a declaration, since the tier-up body is
  // far.orc_pgo.tier.
  Function *Thunk = M->getFunction("far");
  Function *Tier = M->getFunction("far.orc_pgo.tier");
  ASSERT_NE(Thunk, nullptr);
  ASSERT_NE(Tier, nullptr);
  EXPECT_TRUE(Thunk->isDeclaration());
  EXPECT_TRUE(Thunk->hasExternalLinkage());
  EXPECT_FALSE(Tier->isDeclaration());
  EXPECT_EQ(M->getNamedGlobal("fptrs")->getInitializer()->getOperand(0), Thunk);
  Function *Addr = M->getFunction("addr.orc_pgo.tier");
  auto *Ret = cast<ReturnInst>(Addr->getEntryBlock().getTerminator());
  EXPECT_EQ(Ret->getReturnValue(), Thunk);

  Function *Root = M->getFunction("callit.orc_pgo.tier");
  EXPECT_TRUE(comparesWith(*Root, Thunk));
  EXPECT_FALSE(comparesWith(*Root, Tier));
  EXPECT_TRUE(callsDirectly(*Root, Tier)); // the guarded, direct call
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  optimizeTierUp(*M, *TM);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  EXPECT_TRUE(comparesWith(*M->getFunction("callit.orc_pgo.tier"), Thunk));
  EXPECT_FALSE(comparesWith(*M->getFunction("callit.orc_pgo.tier"), Tier));
}

TEST(TierUp, ARootsCallsStayOnItsTierUpDefinition) {
  // Only address uses go to the original name: a call, even a recursive one,
  // is to the tier-up body.
  constexpr const char *RecursiveIR = R"(
source_filename = "rec.c"
define i32 @fact(i32 %n) {
entry:
  %c = icmp sle i32 %n, 1
  br i1 %c, label %base, label %rec
base:
  ret i32 1
rec:
  %m = sub i32 %n, 1
  %r = call i32 @fact(i32 %m)
  %p = mul i32 %r, %n
  ret i32 %p
}
)";
  auto S = test::simulateAot(RecursiveIR);
  S.sym<int(int)>("fact")(5);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  ASSERT_EQ(
      test::errorText(
          prepareTierUpModule(*M, *profileOf(S), {"fact"}, 0).takeError()),
      "");
  Function *Tier = M->getFunction("fact.orc_pgo.tier");
  EXPECT_TRUE(callsDirectly(*Tier, Tier));
  EXPECT_EQ(M->getFunction("fact"), nullptr);
}

TEST(TierUp, PromotionsOutsideTheBatchAreNotCounted) {
  // callit's call site is promoted by the whole-module ICP run, but callit is
  // neither a root nor a copy of this batch.
  auto S = test::simulateAot(ProgramIR);
  auto *CallIt = S.sym<int(int (*)(int), int)>("callit");
  auto *Far = S.sym<int(int)>("far");
  for (int I = 0; I < 2000; ++I)
    ASSERT_EQ(CallIt(Far, I), I + 1);
  S.sym<int(int)>("hot")(2);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  Expected<TierUpBatch> Batch =
      prepareTierUpModule(*M, *profileOf(S), {"hot"}, 0);
  ASSERT_EQ(test::errorText(Batch), "");
  EXPECT_EQ(Batch->PromotedIndirectCalls, 0u);
}

constexpr const char *WeakIR = R"(
source_filename = "weak.c"
define weak i32 @wk(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %a, label %b
a:
  ret i32 1
b:
  ret i32 2
}
)";

TEST(TierUp, AStaleHashOfAWeakRootIsCaughtByMustBeAnnotated) {
  // LLVM does not diagnose a hash mismatch of a weak function; the root then
  // has no entry count, which must still fail the batch.
  auto S = test::simulateAot(WeakIR);
  S.sym<int(int)>("wk")(1);
  std::vector<OrcPgoFunctionRecord> Records(
      S.Image->Functions, S.Image->Functions + S.Image->NumFunctions);
  for (OrcPgoFunctionRecord &R : Records)
    R.CfgHash ^= 1;
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  std::string Error = test::errorText(
      prepareTierUpModule(*M, *profileOf(Records), {"wk"}, 0).takeError());
  EXPECT_NE(Error.find("no profile applied to wk"), std::string::npos) << Error;
}

TEST(TierUp, AHotMemcpySizeIsSpecialized) {
  auto S = test::simulateAot(ProgramIR);
  auto *Copy = S.sym<void(char *, const char *, uint64_t)>("copy");
  char Dst[16], Src[16] = {};
  for (int I = 0; I < 2000; ++I)
    Copy(Dst, Src, 16);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  ASSERT_EQ(
      test::errorText(
          prepareTierUpModule(*M, *profileOf(S), {"copy"}, 0).takeError()),
      "");
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  optimizeTierUp(*M, *TM);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function *Root = M->getFunction("copy.orc_pgo.tier");
  EXPECT_TRUE(testsArgAgainst(*Root, 2, 16));
  EXPECT_TRUE(copiesConstantSize(*Root, 16));
}

// Four independent scalar adds: only the SLP vectorizer, which clang runs at
// -O3 and PassBuilder does not by default, combines them into a vector add.
constexpr const char *SlpIR = R"(
source_filename = "slp.c"
define void @slp(ptr noalias %a, ptr noalias %b, ptr noalias %c) {
  %b1 = getelementptr i32, ptr %b, i64 1
  %b2 = getelementptr i32, ptr %b, i64 2
  %b3 = getelementptr i32, ptr %b, i64 3
  %c1 = getelementptr i32, ptr %c, i64 1
  %c2 = getelementptr i32, ptr %c, i64 2
  %c3 = getelementptr i32, ptr %c, i64 3
  %a1 = getelementptr i32, ptr %a, i64 1
  %a2 = getelementptr i32, ptr %a, i64 2
  %a3 = getelementptr i32, ptr %a, i64 3
  %vb0 = load i32, ptr %b
  %vb1 = load i32, ptr %b1
  %vb2 = load i32, ptr %b2
  %vb3 = load i32, ptr %b3
  %vc0 = load i32, ptr %c
  %vc1 = load i32, ptr %c1
  %vc2 = load i32, ptr %c2
  %vc3 = load i32, ptr %c3
  %s0 = add i32 %vb0, %vc0
  %s1 = add i32 %vb1, %vc1
  %s2 = add i32 %vb2, %vc2
  %s3 = add i32 %vb3, %vc3
  store i32 %s0, ptr %a
  store i32 %s1, ptr %a1
  store i32 %s2, ptr %a2
  store i32 %s3, ptr %a3
  ret void
}
)";

TEST(TierUp, O3RunsTheSlpVectorizer) {
  auto S = test::simulateAot(SlpIR);
  auto *Slp = S.sym<void(int *, const int *, const int *)>("slp");
  int A[4], B[4] = {1, 2, 3, 4}, C[4] = {10, 20, 30, 40};
  for (int I = 0; I < 100; ++I)
    Slp(A, B, C);
  EXPECT_EQ(A[3], 44);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  ASSERT_EQ(test::errorText(
                prepareTierUpModule(*M, *profileOf(S), {"slp"}, 0).takeError()),
            "");
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  optimizeTierUp(*M, *TM);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function *Root = M->getFunction("slp.orc_pgo.tier");
  ASSERT_NE(Root, nullptr);
  unsigned VectorAdds = 0;
  for (const Instruction &I : instructions(*Root))
    VectorAdds +=
        I.getOpcode() == Instruction::Add && I.getType()->isVectorTy();
  EXPECT_GE(VectorAdds, 1u);
}

TEST(TierUp, O3LeavesNoCallGraphProfile) {
  // clang's pipeline adds the "CG Profile" module flag, which makes the
  // backend emit a .llvm.call-graph-profile section that JITLink cannot link
  // on x86-64 ELF. hot calls through counted edges, so the pass would add it.
  auto S = test::simulateAot(ProgramIR);
  auto *Hot = S.sym<int(int)>("hot");
  for (int I = 0; I < 30; ++I)
    ASSERT_EQ(Hot(2), 9);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  ASSERT_EQ(test::errorText(
                prepareTierUpModule(*M, *profileOf(S), {"hot"}, 0).takeError()),
            "");
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  optimizeTierUp(*M, *TM);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  EXPECT_EQ(M->getModuleFlag("CG Profile"), nullptr);
}

// The linker may replace a weak definition by another one, so a tier-up
// module must keep calling and reading the program's.
constexpr const char *InterposableIR = R"(
source_filename = "interpose.c"
@wc = weak constant i32 7
@woc = weak_odr constant i32 8
@sc = constant i32 9
define weak i32 @wf(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
define weak_odr i32 @wo(i32 %x) {
  %r = add i32 %x, 2
  ret i32 %r
}
define i32 @sf(i32 %x) {
  %r = add i32 %x, 3
  ret i32 %r
}
define i32 @hot(i32 %x) {
  %a = call i32 @wf(i32 %x)
  %b = call i32 @wo(i32 %a)
  %c = call i32 @sf(i32 %b)
  %p = load i32, ptr @wc
  %q = load i32, ptr @woc
  %r = load i32, ptr @sc
  %s1 = add i32 %c, %p
  %s2 = add i32 %s1, %q
  %s3 = add i32 %s2, %r
  ret i32 %s3
}
)";

TEST(TierUp, InterposableFunctionsAndConstantsStayDeclarations) {
  auto S = test::simulateAot(InterposableIR);
  ASSERT_EQ(S.sym<int(int)>("hot")(1), 1 + 1 + 2 + 3 + 7 + 8 + 9);
  LLVMContext Ctx;
  auto M = snapshotOf(S, Ctx);
  ASSERT_EQ(test::errorText(
                prepareTierUpModule(*M, *profileOf(S), {"hot"}, 2).takeError()),
            "");
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function *WeakFn = M->getFunction("wf");
  EXPECT_TRUE(WeakFn->isDeclaration());
  EXPECT_TRUE(WeakFn->hasExternalLinkage());
  GlobalVariable *Weak = M->getNamedGlobal("wc");
  EXPECT_TRUE(Weak->isDeclaration());
  EXPECT_TRUE(Weak->hasExternalLinkage());
  // weak_odr definitions are all the same, so they are copied like strong ones.
  for (StringRef Name : {"wo", "sf"})
    EXPECT_TRUE(M->getFunction(Name)->hasAvailableExternallyLinkage())
        << Name.str();
  for (StringRef Name : {"woc", "sc"})
    EXPECT_TRUE(M->getNamedGlobal(Name)->hasAvailableExternallyLinkage())
        << Name.str();

  // Nothing of them is folded into the root.
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  optimizeTierUp(*M, *TM);
  Function *Root = M->getFunction("hot.orc_pgo.tier");
  EXPECT_TRUE(callsDirectly(*Root, M->getFunction("wf")));
  bool LoadsWeak = false;
  for (Instruction &I : instructions(*Root))
    if (auto *L = dyn_cast<LoadInst>(&I))
      LoadsWeak |= L->getPointerOperand() == Weak;
  EXPECT_TRUE(LoadsWeak);
}

TEST(TierUp, ModuleInlineAsmIsNotCopied) {
  // The program's object has assembled it already; assembling it again in
  // every batch would define its symbols twice.
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, R"(
module asm ".globl orc_pgo_asm_symbol"
module asm "orc_pgo_asm_symbol:"
define void @f() {
  ret void
}
)");
  ASSERT_TRUE(M->hasModuleInlineAsm());
  ASSERT_EQ(test::errorText(
                prepareTierUpModule(
                    *M, *profileOf(ArrayRef<OrcPgoFunctionRecord>()), {}, 2)
                    .takeError()),
            "");
  EXPECT_FALSE(M->hasModuleInlineAsm());
}

// Without roots every global is reduced; this covers the kinds the JIT'd
// fixture can't hold (ifuncs, thread-locals, comdats).
constexpr const char *GlobalsIR = R"(
source_filename = "globals.c"
$cd = comdat any
@k = constant i32 1
@kpriv = private unnamed_addr constant [3 x i8] c"hi\00"
@cdk = constant i32 3, comdat($cd)
@v = global i32 0
@tlsk = thread_local constant i32 2
@labels = constant [1 x ptr] [ptr blockaddress(@target, %bb)]
@va = alias i32, ptr @v
@fa = alias void (), ptr @f
@ifn = ifunc void (), ptr @resolver
@llvm.used = appending global [1 x ptr] [ptr @f], section "llvm.metadata"

declare extern_weak void @weak()
declare hidden void @hiddendecl()

define ptr @resolver() {
  ret ptr @f
}
define internal void @f() {
  call void @hiddendecl()
  ret void
}
define void @target() {
entry:
  br label %bb
bb:
  ret void
}
)";

TEST(TierUp, GlobalsAreCopiedWhenConstantAndRecreatableAndImportedOtherwise) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, GlobalsIR);
  ASSERT_EQ(test::errorText(
                prepareTierUpModule(
                    *M, *profileOf(ArrayRef<OrcPgoFunctionRecord>()), {}, 2)
                    .takeError()),
            "");
  ASSERT_FALSE(verifyModule(*M, &errs()));

  for (StringRef Name : {"k", "kpriv", "cdk"}) {
    GlobalVariable *GV = M->getNamedGlobal(Name);
    EXPECT_TRUE(GV->hasAvailableExternallyLinkage()) << Name.str();
    EXPECT_FALSE(GV->isDSOLocal()) << Name.str();
    EXPECT_EQ(GV->getComdat(), nullptr) << Name.str();
  }
  for (StringRef Name : {"v", "tlsk", "labels", "va"}) {
    GlobalVariable *GV = M->getNamedGlobal(Name);
    ASSERT_NE(GV, nullptr) << Name.str();
    EXPECT_TRUE(GV->isDeclaration()) << Name.str();
    EXPECT_TRUE(GV->hasExternalLinkage()) << Name.str();
  }
  EXPECT_EQ(M->getNamedAlias("va"), nullptr);
  EXPECT_EQ(M->getNamedAlias("fa"), nullptr);
  EXPECT_EQ(M->getNamedIFunc("ifn"), nullptr);
  EXPECT_EQ(M->getNamedGlobal("llvm.used"), nullptr);
  for (StringRef Name : {"fa", "ifn", "resolver", "f", "target"}) {
    Function *F = M->getFunction(Name);
    ASSERT_NE(F, nullptr) << Name.str();
    EXPECT_TRUE(F->isDeclaration()) << Name.str();
    EXPECT_TRUE(F->hasExternalLinkage()) << Name.str();
  }
  EXPECT_TRUE(M->getFunction("weak")->hasExternalWeakLinkage());
  Function *Hidden = M->getFunction("hiddendecl");
  EXPECT_TRUE(Hidden->hasDefaultVisibility());
  EXPECT_FALSE(Hidden->isDSOLocal());
}

TEST(TierUp, BindingRedirectsOnlyCallsToEarlierTieredFunctions) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = test::parseIR(Ctx, R"(
declare i32 @tiered(i32)
declare i32 @other(i32)
declare i32 @addressonly(i32)
define available_externally i32 @copy(i32 %x) {
  ret i32 %x
}
define i32 @user(i32 %x, ptr %out) personality ptr @__gxx_personality_v0 {
  %a = call i32 @tiered(i32 %x)
  %b = call i32 @other(i32 %a)
  %c = invoke i32 @tiered(i32 %b) to label %ok unwind label %bad
ok:
  store ptr @tiered, ptr %out
  %d = call i32 @copy(i32 %c)
  %e = call i32 @pass(ptr @addressonly, ptr @tiered)
  ret i32 %e
bad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}
declare i32 @pass(ptr, ptr)
declare i32 @__gxx_personality_v0(...)
)");
  Function *User = M->getFunction("user");
  StringSet<> Tiered;
  Tiered.insert("tiered.orc_pgo.tier");
  Tiered.insert("copy.orc_pgo.tier");
  Tiered.insert("addressonly.orc_pgo.tier");
  Tiered.insert("neverseen.orc_pgo.tier");

  // The two calls to @tiered (one an invoke) and the call to @copy.
  EXPECT_EQ(bindTieredCalls(*M, Tiered), 3u);
  EXPECT_FALSE(verifyModule(*M, &errs()));

  Function *Tier = M->getFunction("tiered.orc_pgo.tier");
  ASSERT_NE(Tier, nullptr);
  EXPECT_TRUE(Tier->isDeclaration());
  EXPECT_EQ(Tier->getFunctionType(),
            M->getFunction("tiered")->getFunctionType());
  EXPECT_EQ(Tier->getLinkage(), GlobalValue::ExternalLinkage);
  Function *CopyTier = M->getFunction("copy.orc_pgo.tier");
  ASSERT_NE(CopyTier, nullptr);
  // The copy keeps its body (it may have been inlined elsewhere).
  EXPECT_FALSE(M->getFunction("copy")->isDeclaration());

  unsigned ToTier = 0, ToCopyTier = 0, ToOther = 0;
  for (Instruction &I : instructions(*User)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB)
      continue;
    ToTier += CB->getCalledFunction() == Tier;
    ToCopyTier += CB->getCalledFunction() == CopyTier;
    ToOther += CB->getCalledFunction() == M->getFunction("other");
  }
  EXPECT_EQ(ToTier, 2u);
  EXPECT_EQ(ToCopyTier, 1u);
  EXPECT_EQ(ToOther, 1u);

  // An address stays the original's: stored, or passed as an argument. A
  // function that is only used that way gets no declaration.
  unsigned AddressUses = 0;
  for (Use &U : M->getFunction("tiered")->uses()) {
    EXPECT_FALSE(isa<CallBase>(U.getUser()) &&
                 cast<CallBase>(U.getUser())->isCallee(&U));
    ++AddressUses;
  }
  EXPECT_EQ(AddressUses, 2u);
  EXPECT_EQ(M->getFunction("addressonly.orc_pgo.tier"), nullptr);
  EXPECT_EQ(M->getFunction("neverseen.orc_pgo.tier"), nullptr);

  // Nothing is left to redirect.
  EXPECT_EQ(bindTieredCalls(*M, Tiered), 0u);
}

TEST(TierUp, BindingAfterOptimizationKeepsInliningACopyAndRedirectsWhatIsLeft) {
  auto S = test::simulateAot(ProgramIR);
  LLVMContext Ctx;
  std::unique_ptr<Module> M = snapshotOf(S, Ctx);
  std::unique_ptr<MemoryBuffer> Profile = profileOf(S);
  // @far was tiered up by an earlier batch.
  ASSERT_FALSE(
      errorToBool(prepareTierUpModule(*M, *Profile, {"leaf"}, /*InlineDepth=*/1)
                      .takeError()));
  Function *Far = M->getFunction("far");
  ASSERT_NE(Far, nullptr);
  ASSERT_TRUE(Far->hasAvailableExternallyLinkage());
  // Unoptimized, the root calls its callee copy ...
  Function *Leaf = M->getFunction("leaf.orc_pgo.tier");
  ASSERT_NE(Leaf, nullptr);
  ASSERT_TRUE(callsDirectly(*Leaf, Far));
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  optimizeTierUp(*M, *TM);
  // ... which the optimizer inlines, so there is no call left to bind.
  EXPECT_FALSE(callsDirectly(*Leaf, Far));
  StringSet<> Tiered;
  Tiered.insert("far.orc_pgo.tier");
  EXPECT_EQ(bindTieredCalls(*M, Tiered), 0u);
  EXPECT_EQ(M->getFunction("far.orc_pgo.tier"), nullptr);
}

} // namespace
