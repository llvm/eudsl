// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Thunks.h"
#include "TestSupport.h"
#include "orc-pgo/Analyses.h"
#include "orc-pgo/Instrument.h"

#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *ShapesIR = R"(
source_filename = "thunks.c"

@tls = thread_local global i32 0

declare i32 @pers(...)

define i32 @plain(i32 %x) !dbg !4 {
  %r = add i32 %x, 1, !dbg !7
  ret i32 %r, !dbg !7
}
define internal i32 @local(i32 %x) {
  %r = mul i32 %x, 3
  ret i32 %r
}
define hidden signext i8 @attrs(ptr noundef nonnull %p, i8 zeroext %c) memory(argmem: read) nosync nounwind personality ptr @pers {
  %v = load i8, ptr %p
  %s = add i8 %v, %c
  ret i8 %s
}
define void @noret() {
  ret void
}
define i32 @va(i32 %n, ...) {
  ret i32 %n
}
define void @naked() naked {
  unreachable
}
define i32 @prefixed() prefix i32 1 {
  ret i32 0
}
define i32 @prologued() prologue i32 1 {
  ret i32 0
}
define void @inalloca(ptr inalloca(i32) %p) {
  ret void
}
define void @prealloc(ptr preallocated(i32) %p) {
  ret void
}
define void @byvalarg(ptr byval(i64) %p) {
  ret void
}
define i32 @inregarg(i32 inreg %x) {
  ret i32 %x
}
define i32 @noprof() noprofile {
  ret i32 0
}
define i32 @usestls() {
  %v = load i32, ptr @tls
  ret i32 %v
}
define available_externally i32 @availext(i32 %x) {
  ret i32 %x
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "thunks.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "plain", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 1, scope: !4)
)";

std::unique_ptr<Module> instrumented(LLVMContext &Ctx, StringRef IR) {
  auto M = test::parseHostIR(Ctx, IR);
  EXPECT_EQ(test::errorText(runPgoGen(*M)), "");
  return M;
}

TEST(Thunks, OnlyFunctionsGenInstrumentedAreEligible) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, ShapesIR);
  EXPECT_FALSE(isThunkEligible(*M->getFunction("plain")));
}

TEST(Thunks, EligibilityFollowsTheSpec) {
  LLVMContext Ctx;
  auto M = instrumented(Ctx, ShapesIR);
  for (StringRef Name : {"plain", "local", "attrs", "noret"})
    EXPECT_TRUE(isThunkEligible(*M->getFunction(Name))) << Name.str();
  for (StringRef Name :
       {"va", "naked", "prefixed", "prologued", "inalloca", "prealloc",
        "byvalarg", "inregarg", "noprof", "usestls", "availext", "pers"})
    EXPECT_FALSE(isThunkEligible(*M->getFunction(Name))) << Name.str();
}

TEST(Thunks, CandidatesCarryGensNameAndAreMarkedNoipa) {
  LLVMContext Ctx;
  auto M = instrumented(Ctx, ShapesIR);
  std::vector<ThunkCandidate> C = prepareThunkCandidates(*M);
  std::vector<std::string> Names;
  for (const ThunkCandidate &T : C) {
    Names.push_back(T.PgoName);
    EXPECT_TRUE(T.function()->hasFnAttribute(Attribute::NoIPA)) << T.PgoName;
  }
  EXPECT_EQ(Names, (std::vector<std::string>{"plain", "thunks.c;local", "attrs",
                                             "noret"}));
  EXPECT_FALSE(M->getFunction("va")->hasFnAttribute(Attribute::NoIPA));
}

constexpr const char *InferenceIR = R"(
source_filename = "infer.c"
@g = global i32 0
declare void @ext()
define i32 @leaf(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
define i32 @user(i32 %x) {
  %r = call i32 @leaf(i32 %x)
  ret i32 %r
}
define void @throws() {
  call void @ext()
  ret void
}
define internal i32 @unusedarg(i32 %x, i32 %unused) {
  %r = add i32 %x, 1
  ret i32 %r
}
define i32 @varuser(i32 %n, ...) {
  %r = call i32 @leaf(i32 %n)
  ret i32 %r
}
define i32 @pureuser(i32 %n, ...) {
  %r = add i32 %n, 1
  ret i32 %r
}
define i32 @declared(ptr %p) memory(argmem: read) nosync nounwind {
  %r = load i32, ptr %p
  ret i32 %r
}
define i32 @callsunused(i32 %x) {
  %r = call i32 @unusedarg(i32 %x, i32 7)
  ret i32 %r
}
)";

/// The build's order: infer, Gen, then drop what Gen invalidates.
std::unique_ptr<Module> inferredAndInstrumented(LLVMContext &Ctx) {
  auto M = test::parseHostIR(Ctx, InferenceIR);
  DeclaredAttributes Declared = inferFunctionAttributes(*M);
  EXPECT_EQ(test::errorText(runPgoGen(*M)), "");
  dropInvalidatedAttributes(*M, Declared);
  return M;
}

TEST(Thunks, AttributesAreInferredBeforeNoipaHidesThem) {
  LLVMContext Ctx;
  auto M = inferredAndInstrumented(Ctx);
  std::vector<ThunkCandidate> C = prepareThunkCandidates(*M);
  EXPECT_EQ(C.size(), 6u);
  for (StringRef Name : {"leaf", "user", "unusedarg", "callsunused"}) {
    Function *F = M->getFunction(Name);
    EXPECT_TRUE(F->hasFnAttribute(Attribute::NoIPA)) << Name.str();
    EXPECT_TRUE(F->doesNotThrow()) << Name.str();
    EXPECT_TRUE(F->willReturn()) << Name.str();
    EXPECT_TRUE(F->doesNotRecurse()) << Name.str();
    EXPECT_TRUE(F->doesNotFreeMemory()) << Name.str();
  }
  // Calling @ext, which may unwind, @throws infers nothing.
  Function *Throws = M->getFunction("throws");
  EXPECT_FALSE(Throws->doesNotThrow());
  EXPECT_FALSE(Throws->willReturn());
}

TEST(Thunks, GenInvalidatesInferredMemoryAndSync) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, InferenceIR);
  DeclaredAttributes Declared = inferFunctionAttributes(*M);
  // Inference finds @pureuser neither reading nor writing, and not
  // synchronizing, which holds of tier-up code (it has no counters) ...
  Function *Pure = M->getFunction("pureuser");
  EXPECT_TRUE(Pure->doesNotAccessMemory());
  EXPECT_TRUE(Pure->hasFnAttribute(Attribute::NoSync));
  ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
  dropInvalidatedAttributes(*M, Declared);
  // ... but not of the AOT module, whose counters are written and whose
  // thunks load their slot with an acquire.
  for (Function &F : *M) {
    if (F.getName() == "declared" || F.isIntrinsic())
      continue;
    EXPECT_FALSE(F.hasFnAttribute(Attribute::NoSync)) << F.getName().str();
    EXPECT_FALSE(F.hasFnAttribute(Attribute::Memory)) << F.getName().str();
  }
  EXPECT_TRUE(Pure->doesNotThrow());
  // What the source asserted stays.
  Function *Source = M->getFunction("declared");
  EXPECT_TRUE(Source->hasFnAttribute(Attribute::NoSync));
  EXPECT_EQ(Source->getMemoryEffects(),
            MemoryEffects::argMemOnly(ModRefInfo::Ref));
}

TEST(Thunks, InferredAttributesSurviveThunkifyExceptMemoryAndSync) {
  LLVMContext Ctx;
  auto M = inferredAndInstrumented(Ctx);
  thunkify(*M, prepareThunkCandidates(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function *Thunk = M->getFunction("leaf");
  Function *Body = M->getFunction("leaf.orc_pgo.aot");
  for (Function *F : {Thunk, Body}) {
    EXPECT_TRUE(F->doesNotThrow());
    EXPECT_TRUE(F->willReturn());
    EXPECT_TRUE(F->doesNotRecurse());
    EXPECT_FALSE(F->hasFnAttribute(Attribute::Memory));
    EXPECT_FALSE(F->hasFnAttribute(Attribute::NoSync));
  }
}

TEST(Thunks, NoipaStillStopsDeadArgumentElimination) {
  for (bool Prepare : {false, true}) {
    LLVMContext Ctx;
    auto M = inferredAndInstrumented(Ctx);
    if (Prepare)
      prepareThunkCandidates(*M);
    PgoAnalysisManagers AM(*M);
    ModulePassManager MPM;
    MPM.addPass(DeadArgumentEliminationPass());
    MPM.run(*M, AM.MAM);
    // Without noipa @unusedarg loses %unused, with it the signature stays.
    EXPECT_EQ(M->getFunction("unusedarg")->arg_size(), Prepare ? 2u : 1u);
  }
}

/// The dispatch slot Load reads. Slot 0's address folds to the table itself.
uint64_t slotIndex(const LoadInst &Load, const GlobalVariable &Dispatch) {
  const DataLayout &DL = Load.getModule()->getDataLayout();
  APInt Offset(DL.getIndexTypeSizeInBits(Load.getPointerOperandType()), 0);
  const Value *Base =
      Load.getPointerOperand()->stripAndAccumulateConstantOffsets(
          DL, Offset, /*AllowNonInbounds=*/true);
  EXPECT_EQ(Base, &Dispatch);
  return Offset.getZExtValue() / DL.getPointerSize();
}

TEST(Thunks, ThunksLoadTheirSlotAndMustTailCallIt) {
  LLVMContext Ctx;
  auto M = instrumented(Ctx, ShapesIR);
  ThunkTable T = thunkify(*M, prepareThunkCandidates(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));

  ASSERT_EQ(T.Thunks.size(), 4u);
  EXPECT_EQ(T.Dispatch->getName(), DispatchTableName);
  EXPECT_TRUE(T.Dispatch->hasInternalLinkage());
  auto *Slots = cast<ConstantArray>(T.Dispatch->getInitializer());
  for (unsigned I = 0; I < 4; ++I) {
    const Thunk &Th = T.Thunks[I];
    EXPECT_EQ(Slots->getOperand(I), Th.Body);
    EXPECT_EQ(Th.Body->getName(), (Th.Entry->getName() + AotSuffix).str());
    EXPECT_TRUE(Th.Body->hasInternalLinkage());
    EXPECT_TRUE(Th.Body->hasDefaultVisibility());

    ASSERT_EQ(Th.Entry->size(), 1u);
    BasicBlock &Entry = Th.Entry->getEntryBlock();
    auto *Load = cast<LoadInst>(&Entry.front());
    EXPECT_EQ(Load->getOrdering(), AtomicOrdering::Acquire);
    EXPECT_EQ(slotIndex(*Load, *T.Dispatch), I);
    auto *Call = cast<CallInst>(Load->getNextNode());
    EXPECT_TRUE(Call->isMustTailCall());
    EXPECT_EQ(Call->getCalledOperand(), Load);
    EXPECT_EQ(Call->arg_size(), Th.Entry->arg_size());
    auto *Ret = cast<ReturnInst>(Call->getNextNode());
    EXPECT_EQ(Ret->getReturnValue(),
              Th.Entry->getReturnType()->isVoidTy() ? nullptr : Call);
  }
}

TEST(Thunks, AbiAttributesStayOnTheThunkAndItsCall) {
  LLVMContext Ctx;
  auto M = instrumented(Ctx, ShapesIR);
  thunkify(*M, prepareThunkCandidates(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  Function *Thunk = M->getFunction("attrs");
  Function *Body = M->getFunction("attrs.orc_pgo.aot");
  auto *Call = cast<CallInst>(Thunk->getEntryBlock().front().getNextNode());
  EXPECT_TRUE(Call->paramHasAttr(0, Attribute::NonNull));
  EXPECT_TRUE(Call->paramHasAttr(0, Attribute::NoUndef));
  EXPECT_TRUE(Call->paramHasAttr(1, Attribute::ZExt));
  EXPECT_TRUE(
      Call->getAttributes().getRetAttrs().hasAttribute(Attribute::SExt));
  EXPECT_TRUE(Thunk->hasHiddenVisibility());
  EXPECT_TRUE(Thunk->hasFnAttribute(Attribute::NoUnwind));
  // The thunk reads memory (its slot) and synchronizes with the runtime.
  EXPECT_FALSE(Thunk->hasFnAttribute(Attribute::Memory));
  EXPECT_FALSE(Thunk->hasFnAttribute(Attribute::NoSync));
  EXPECT_FALSE(Thunk->hasPersonalityFn());
  EXPECT_TRUE(Body->hasFnAttribute(Attribute::Memory));
  EXPECT_TRUE(Body->hasFnAttribute(Attribute::NoSync));
  EXPECT_EQ(Body->getPersonalityFn(), M->getFunction("pers"));
}

TEST(Thunks, TheBodyKeepsTheDebugInfoAndArgumentNames) {
  LLVMContext Ctx;
  auto M = instrumented(Ctx, ShapesIR);
  DISubprogram *SP = M->getFunction("plain")->getSubprogram();
  ASSERT_NE(SP, nullptr);
  thunkify(*M, prepareThunkCandidates(*M));
  ASSERT_FALSE(verifyModule(*M, &errs()));
  EXPECT_EQ(M->getFunction("plain")->getSubprogram(), nullptr);
  Function *Body = M->getFunction("plain.orc_pgo.aot");
  EXPECT_EQ(Body->getSubprogram(), SP);
  EXPECT_EQ(Body->getArg(0)->getName(), "x");
}

TEST(Thunks, CandidatesDeletedBeforeThunkifyGetNoSlot) {
  LLVMContext Ctx;
  auto M = instrumented(Ctx, ShapesIR);
  std::vector<ThunkCandidate> C = prepareThunkCandidates(*M);
  ASSERT_EQ(C.size(), 4u); // plain, local, attrs, noret
  EXPECT_EQ(C[1].function(), M->getFunction("local"));
  // As if optimization had removed the unused internal function.
  M->getFunction("local")->eraseFromParent();
  EXPECT_EQ(C[1].function(), nullptr);
  ThunkTable T = thunkify(*M, C);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  ASSERT_EQ(T.Thunks.size(), 3u);
  EXPECT_EQ(T.Thunks[0].Entry, M->getFunction("plain"));
  EXPECT_EQ(T.Thunks[1].Entry, M->getFunction("attrs"));
  EXPECT_EQ(T.Thunks[2].Entry, M->getFunction("noret"));
  EXPECT_EQ(cast<ArrayType>(T.Dispatch->getValueType())->getNumElements(), 3u);
  for (unsigned I = 0; I < 3; ++I)
    EXPECT_EQ(cast<ConstantArray>(T.Dispatch->getInitializer())->getOperand(I),
              T.Thunks[I].Body);
}

TEST(Thunks, NoCandidatesGiveAnEmptyTable) {
  LLVMContext Ctx;
  auto M =
      instrumented(Ctx, "define i32 @va(i32 %n, ...) {\n  ret i32 %n\n}\n");
  ThunkTable T = thunkify(*M, prepareThunkCandidates(*M));
  EXPECT_TRUE(T.Thunks.empty());
  EXPECT_EQ(cast<ArrayType>(T.Dispatch->getValueType())->getNumElements(), 0u);
  EXPECT_FALSE(verifyModule(*M, &errs()));
}

constexpr const char *DispatchIR = R"(
source_filename = "dispatch.c"
define i32 @add1(i32 %x) noinline {
  %r = add i32 %x, 1
  ret i32 %r
}
define i32 @twiceadd1(i32 %x) {
  %a = call i32 @add1(i32 %x)
  %b = call i32 @add1(i32 %a)
  ret i32 %b
}
)";

int times100(int X) { return X * 100; }

TEST(Thunks, CallsGoThroughTheSlot) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = instrumented(*Ctx, DispatchIR);
  // The AOT pipeline's order: candidates, lowering, then thunks.
  std::vector<ThunkCandidate> C = prepareThunkCandidates(*M);
  ASSERT_EQ(test::errorText(lowerPgo(*M).takeError()), "");
  ThunkTable T = thunkify(*M, C);
  // Exported only so the test can find the table in the JIT.
  T.Dispatch->setLinkage(GlobalValue::ExternalLinkage);
  ASSERT_FALSE(verifyModule(*M, &errs()));
  auto J = test::jitModule(std::move(M), std::move(Ctx));
  auto *Add1 = test::lookupSym<int(int)>(*J, "add1");
  auto *TwiceAdd1 = test::lookupSym<int(int)>(*J, "twiceadd1");
  void **Slots = test::lookupSym<void *>(*J, DispatchTableName);

  EXPECT_EQ(Add1(4), 5);
  EXPECT_EQ(TwiceAdd1(4), 6);
  void *Aot = Slots[0];
  EXPECT_NE(Aot, reinterpret_cast<void *>(Add1));
  Slots[0] = reinterpret_cast<void *>(&times100);
  EXPECT_EQ(Add1(3), 300);
  EXPECT_EQ(TwiceAdd1(1),
            10000); // both calls in the caller reach the new target
  Slots[0] = Aot;
  EXPECT_EQ(Add1(3), 4);
}

} // namespace
