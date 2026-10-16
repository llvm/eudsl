// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Instrument.h"
#include "TestSupport.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/Snapshot.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *ShapesIR = R"(
source_filename = "a.c"
define i32 @leaf(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
define internal i32 @loc(i32 %x) {
  %r = mul i32 %x, 3
  ret i32 %r
}
define i32 @useloc(i32 %x) {
  %r = call i32 @loc(i32 %x)
  ret i32 %r
}
define i32 @sel(i32 %x) {
  %c = icmp sgt i32 %x, 0
  %s = select i1 %c, i32 1, i32 2
  ret i32 %s
}
define i32 @loop(i32 %n) {
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %i1 = add i32 %i, 1
  %more = icmp slt i32 %i1, %n
  br i1 %more, label %body, label %exit
exit:
  ret i32 %i1
}
; Like @loop, but the loop exits to a block that doesn't return. (Rotation
; gives @loop a guard, so its exit is split and doesn't return either; see
; CountersOfLoopsWhoseExitReturnsAreNotPromoted.)
define i32 @loop2(i32 %n) {
entry:
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %i1 = add i32 %i, 1
  %more = icmp slt i32 %i1, %n
  br i1 %more, label %body, label %after
after:
  %r = mul i32 %i1, 2
  br label %exit
exit:
  ret i32 %r
}
define i32 @diamond(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %t, label %f
t:
  br label %m
f:
  br label %m
m:
  %r = phi i32 [ 1, %t ], [ 2, %f ]
  ret i32 %r
}
)";

constexpr const char *ValueSitesIR = R"(
source_filename = "v.c"
define void @ta() {
  ret void
}
define void @tb() {
  ret void
}
define void @tc() {
  ret void
}
define void @dispatch(ptr %fp) {
  call void %fp()
  ret void
}
define void @copy(ptr %d, ptr %s, i64 %n) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 %n, i1 false)
  ret void
}
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
)";

// O2 inlines @leaf into @caller's loop, together with its entry counter.
constexpr const char *InlinedLoopIR = R"(
source_filename = "l.c"
define i64 @leaf(i64 %x) {
  %r = add i64 %x, 1
  ret i64 %r
}
define i64 @caller(i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc2, %loop ]
  %acc2 = call i64 @leaf(i64 %acc)
  %next = add i64 %i, 1
  %more = icmp ult i64 %next, %n
  br i1 %more, label %loop, label %done
done:
  ret i64 %acc2
}
)";

struct Instrumented {
  std::unique_ptr<orc::LLJIT> J;
  ArrayRef<OrcPgoFunctionRecord> Table;

  const OrcPgoFunctionRecord &rec(StringRef Name) const {
    for (const OrcPgoFunctionRecord &R : Table)
      if (Name == R.Name)
        return R;
    ADD_FAILURE() << "no record for " << Name.str();
    return Table.front();
  }
  ArrayRef<uint64_t> counters(StringRef Name) const {
    const OrcPgoFunctionRecord &R = rec(Name);
    return ArrayRef(R.Counters, R.NumCounters);
  }
  template <typename Fn>
  Fn *fn(StringRef Name) const {
    return test::lookupSym<Fn>(*J, Name);
  }
};

/// Gen and lowerPgo (targeting TM, if given), optionally O2 (the AOT pipeline's
/// order), then the JIT.
Instrumented instrument(StringRef IR, bool RunO2 = false,
                        TargetMachine *TM = nullptr) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = test::parseHostIR(*Ctx, IR);
  EXPECT_EQ(test::errorText(runPgoGen(*M)), "");
  Expected<LoweredPgo> L = lowerPgo(*M, TM);
  EXPECT_EQ(test::errorText(L), "");
  if (RunO2)
    test::runO2(*M);
  EXPECT_FALSE(verifyModule(*M, &errs()));
  Instrumented I;
  I.J = test::jitModule(std::move(M), std::move(Ctx));
  auto *Pool = test::lookupSym<void *>(*I.J, VNodePoolName);
  setValueNodePool(Pool[0], Pool[1]);
  auto *Table = test::lookupSym<OrcPgoFunctionRecord>(*I.J, FunctionTableName);
  uint32_t N = *test::lookupSym<uint32_t>(*I.J, NumFunctionsName);
  I.Table = ArrayRef(Table, N);
  return I;
}

TEST(Instrument, TableHasOneRecordPerInstrumentedFunction) {
  Instrumented I = instrument(ShapesIR);
  std::set<std::string> Names;
  for (const OrcPgoFunctionRecord &R : I.Table)
    Names.insert(R.Name);
  // Local functions are named "<source_filename>;<name>", as in LLVM PGO.
  EXPECT_EQ(Names, (std::set<std::string>{"leaf", "a.c;loc", "useloc", "sel",
                                          "loop", "loop2", "diamond"}));
}

TEST(Instrument, CounterZeroIsTheEntryCount) {
  Instrumented I = instrument(ShapesIR);
  auto *Leaf = I.fn<int(int)>("leaf");
  auto *UseLoc = I.fn<int(int)>("useloc");
  for (int X : {1, 2, 3})
    EXPECT_EQ(Leaf(X), X + 1);
  EXPECT_EQ(UseLoc(4), 12);
  EXPECT_EQ(UseLoc(5), 15);
  EXPECT_EQ(I.counters("leaf"), ArrayRef<uint64_t>({3}));
  EXPECT_EQ(I.counters("useloc"), ArrayRef<uint64_t>({2}));
  EXPECT_EQ(I.counters("a.c;loc"), ArrayRef<uint64_t>({2}));
}

TEST(Instrument, SelectsGetAStepCounterForTheTrueSide) {
  Instrumented I = instrument(ShapesIR);
  auto *Sel = I.fn<int(int)>("sel");
  for (int X : {5, 5, -1})
    Sel(X);
  EXPECT_EQ(I.counters("sel"), ArrayRef<uint64_t>({3, 2}));
}

TEST(Instrument, SelfLoopBackEdgeIsCounted) {
  Instrumented I = instrument(ShapesIR);
  auto *Loop = I.fn<int(int)>("loop");
  EXPECT_EQ(Loop(3), 3); // 2 back edges
  EXPECT_EQ(Loop(1), 1); // none
  EXPECT_EQ(I.counters("loop"), ArrayRef<uint64_t>({2, 2}));
}

TEST(Instrument, DiamondCountsEntryAndOneInnerEdge) {
  Instrumented I = instrument(ShapesIR);
  auto *Diamond = I.fn<int(int)>("diamond");
  for (int X : {5, 5, 5, -1})
    Diamond(X);
  ArrayRef<uint64_t> C = I.counters("diamond");
  ASSERT_EQ(C.size(), 2u);
  EXPECT_EQ(C[0], 4u);
  // Which inner edge is off the spanning tree is LLVM's choice; it is taken
  // 3 times (x > 0 side) or once (x <= 0 side).
  EXPECT_TRUE(C[1] == 3 || C[1] == 1) << C[1];
}

TEST(Instrument, TableHashIsTheInstrumentationHash) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = test::parseHostIR(*Ctx, ShapesIR);
  ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
  EXPECT_FALSE(verifyModule(*M, &errs()));
  std::map<std::string, uint64_t> HashByName;
  for (Function &F : *M)
    for (Instruction &Inst : instructions(F))
      if (auto *Inc = dyn_cast<InstrProfCntrInstBase>(&Inst))
        HashByName[getPGOFuncNameVarInitializer(Inc->getName()).str()] =
            Inc->getHash()->getZExtValue();
  ASSERT_EQ(HashByName.size(), 7u);
  Expected<LoweredPgo> L = lowerPgo(*M);
  ASSERT_EQ(test::errorText(L), "");
  EXPECT_FALSE(verifyModule(*M, &errs()));
  auto J = test::jitModule(std::move(M), std::move(Ctx));
  auto *Table = test::lookupSym<OrcPgoFunctionRecord>(*J, FunctionTableName);
  uint32_t N = *test::lookupSym<uint32_t>(*J, NumFunctionsName);
  ASSERT_EQ(N, 7u);
  for (const OrcPgoFunctionRecord &R : ArrayRef(Table, N))
    EXPECT_EQ(R.CfgHash, HashByName.at(R.Name)) << R.Name;
}

TEST(Instrument, AddressIsTheFunction) {
  Instrumented I = instrument(ShapesIR);
  EXPECT_EQ(I.rec("leaf").Address,
            reinterpret_cast<const void *>(I.fn<int(int)>("leaf")));
}

TEST(Instrument, ValueSitesAreCountedPerKind) {
  Instrumented I = instrument(ValueSitesIR);
  const OrcPgoFunctionRecord &D = I.rec("dispatch");
  const OrcPgoFunctionRecord &C = I.rec("copy");
  EXPECT_EQ(D.NumValueSites[IPVK_IndirectCallTarget], 1u);
  EXPECT_EQ(D.NumValueSites[IPVK_MemOPSize], 0u);
  EXPECT_EQ(C.NumValueSites[IPVK_IndirectCallTarget], 0u);
  EXPECT_EQ(C.NumValueSites[IPVK_MemOPSize], 1u);
  EXPECT_EQ(I.rec("ta").NumValueSites[IPVK_IndirectCallTarget], 0u);
}

TEST(Instrument, IndirectCallTargetsAreRecorded) {
  Instrumented I = instrument(ValueSitesIR);
  auto *Dispatch = I.fn<void(void (*)())>("dispatch");
  auto *A = I.fn<void()>("ta");
  auto *B = I.fn<void()>("tb");
  auto *C = I.fn<void()>("tc");
  for (void (*T)() : {A, A, B, A, C, A, B, A})
    Dispatch(T);
  std::vector<InstrProfValueData> V =
      readValueSite(I.rec("dispatch"), IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(V.size(), 3u);
  EXPECT_EQ(V[0].Value, reinterpret_cast<uint64_t>(A));
  EXPECT_EQ(V[0].Count, 5u);
  EXPECT_EQ(V[1].Value, reinterpret_cast<uint64_t>(B));
  EXPECT_EQ(V[1].Count, 2u);
  EXPECT_EQ(V[2].Value, reinterpret_cast<uint64_t>(C));
  EXPECT_EQ(V[2].Count, 1u);
}

TEST(Instrument, MemopSizesAreBucketed) {
  Instrumented I = instrument(ValueSitesIR);
  auto *Copy = I.fn<void(char *, const char *, uint64_t)>("copy");
  char Dst[600], Src[600] = {};
  for (uint64_t N : {8, 8, 100, 8, 100, 600})
    Copy(Dst, Src, N);
  std::vector<InstrProfValueData> V =
      readValueSite(I.rec("copy"), IPVK_MemOPSize, 0);
  ASSERT_EQ(V.size(), 3u);
  EXPECT_EQ(std::make_pair(V[0].Value, V[0].Count),
            std::make_pair(uint64_t(8), uint64_t(3)));
  EXPECT_EQ(std::make_pair(V[1].Value, V[1].Count),
            std::make_pair(uint64_t(65), uint64_t(2)));
  EXPECT_EQ(std::make_pair(V[2].Value, V[2].Count),
            std::make_pair(uint64_t(513), uint64_t(1)));
}

/// (block index, instruction index, intrinsic, counter index) of every
/// instrprof intrinsic, per function.
using Placement =
    std::map<std::string,
             std::vector<std::tuple<unsigned, unsigned, unsigned, uint64_t>>>;

Placement placement(Module &M) {
  Placement P;
  for (Function &F : M) {
    unsigned B = 0;
    for (BasicBlock &BB : F) {
      unsigned N = 0;
      for (Instruction &Inst : BB) {
        if (auto *Inc = dyn_cast<InstrProfInstBase>(&Inst)) {
          uint64_t Index = 0;
          if (auto *VP = dyn_cast<InstrProfValueProfileInst>(Inc))
            Index = VP->getIndex()->getZExtValue();
          else if (auto *Cnt = dyn_cast<InstrProfCntrInstBase>(Inc))
            Index = Cnt->getIndex()->getZExtValue();
          P[F.getName().str()].emplace_back(B, N, Inc->getIntrinsicID(), Index);
        }
        ++N;
      }
      ++B;
    }
  }
  return P;
}

TEST(Instrument, GenIsDeterministicAcrossASnapshotReparse) {
  for (StringRef IR : {ShapesIR, ValueSitesIR, InlinedLoopIR}) {
    LLVMContext C1, C2;
    auto M1 = test::parseHostIR(C1, IR);
    Expected<std::unique_ptr<Module>> M2 =
        parseSnapshot(writeSnapshot(*M1), C2);
    ASSERT_EQ(test::errorText(M2), "");
    ASSERT_EQ(test::errorText(runPgoGen(*M1)), "");
    ASSERT_EQ(test::errorText(runPgoGen(**M2)), "");
    EXPECT_FALSE(verifyModule(*M1, &errs()));
    EXPECT_FALSE(verifyModule(**M2, &errs()));
    Placement P1 = placement(*M1);
    EXPECT_FALSE(P1.empty());
    EXPECT_EQ(P1, placement(**M2));
  }
}

/// Number of stores to a __profc_ counter array inside a loop of F.
unsigned counterStoresInLoops(Function &F) {
  DominatorTree DT(F);
  LoopInfo LI(DT);
  unsigned N = 0;
  for (BasicBlock &BB : F) {
    if (!LI.getLoopFor(&BB))
      continue;
    for (Instruction &Inst : BB)
      if (auto *St = dyn_cast<StoreInst>(&Inst))
        if (auto *GV = dyn_cast<GlobalVariable>(
                getUnderlyingObject(St->getPointerOperand())))
          N += GV->getName().starts_with("__profc_");
  }
  return N;
}

/// Gen and lowerPgo on ShapesIR, without the JIT.
std::unique_ptr<Module> lowered(LLVMContext &Ctx) {
  auto M = test::parseHostIR(Ctx, ShapesIR);
  EXPECT_EQ(test::errorText(runPgoGen(*M)), "");
  EXPECT_EQ(test::errorText(lowerPgo(*M).takeError()), "");
  EXPECT_FALSE(verifyModule(*M, &errs()));
  return M;
}

TEST(Instrument, LoopCountersArePromotedOutOfLoops) {
  // Clang lowers with counter promotion: a loop's counter lives in a register
  // and is stored once, on the loop's exit.
  LLVMContext Ctx;
  auto M = lowered(Ctx);
  EXPECT_EQ(counterStoresInLoops(*M->getFunction("loop2")), 0u);
}

TEST(Instrument, CountersOfLoopsWhoseExitReturnsAreNotPromoted) {
  // LLVM's -skip-ret-exit-block: promotion is suppressed for a loop whose
  // exit block returns, so its counter is still stored in the loop. @retloop
  // is in do-while form, which post-PGO loop rotation leaves alone, so its
  // exit stays a dedicated, returning block; @contloop's exit continues.
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, R"(
source_filename = "r.c"
define i32 @retloop(i32 %n) {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  br label %latch
latch:
  %i1 = add i32 %i, 1
  %more = icmp slt i32 %i1, %n
  br i1 %more, label %header, label %exit
exit:
  ret i32 %i1
}
define i32 @contloop(i32 %n) {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  br label %latch
latch:
  %i1 = add i32 %i, 1
  %more = icmp slt i32 %i1, %n
  br i1 %more, label %header, label %after
after:
  %r = mul i32 %i1, 2
  br label %exit
exit:
  ret i32 %r
}
)");
  ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
  ASSERT_EQ(test::errorText(lowerPgo(*M).takeError()), "");
  EXPECT_FALSE(verifyModule(*M, &errs()));
  EXPECT_EQ(counterStoresInLoops(*M->getFunction("retloop")), 1u);
  EXPECT_EQ(counterStoresInLoops(*M->getFunction("contloop")), 0u);
}

TEST(Instrument, PromotedCountersAreExactAfterTheLoop) {
  Instrumented I = instrument(ShapesIR);
  auto *Loop2 = I.fn<int(int)>("loop2");
  EXPECT_EQ(Loop2(3), 6); // 2 back edges
  EXPECT_EQ(Loop2(1), 2); // none
  EXPECT_EQ(I.counters("loop2"), ArrayRef<uint64_t>({2, 2}));
}

// A rotated while loop gets a guard in its entry block. Rotation only
// duplicates a loop header of at most 16 instructions, as costed by
// TargetTransformInfo. Without a target every instruction costs 1, so this
// header (2 phis, 6 adds, extractelement, trunc, icmp, br) is small enough. On
// a baseline CPU (x86-64's SSE2, AArch64's NEON) each <64 x i64> add legalizes
// into 32 128-bit adds, so the header costs well over 16 and the loop is left
// alone. A host CPU could be wide enough (e.g. AVX-512) to rotate it.
constexpr const char *WideHeaderIR = R"(
source_filename = "w.c"
define i32 @w(i32 %n, <64 x i64> %v) {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %acc = phi <64 x i64> [ %v, %entry ], [ %a5, %body ]
  %a0 = add <64 x i64> %acc, %v
  %a1 = add <64 x i64> %a0, %v
  %a2 = add <64 x i64> %a1, %v
  %a3 = add <64 x i64> %a2, %v
  %a4 = add <64 x i64> %a3, %v
  %a5 = add <64 x i64> %a4, %v
  %e = extractelement <64 x i64> %a5, i32 0
  %t = trunc i64 %e to i32
  %c = icmp slt i32 %t, %n
  br i1 %c, label %body, label %exit
body:
  %i1 = add i32 %i, 1
  br label %header
exit:
  ret i32 %i
}
)";

/// Whether the entry block of @w ends in a conditional branch, i.e. loop
/// rotation put a guard there.
bool rotatedWithGuard(TargetMachine *TM) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, WideHeaderIR);
  EXPECT_EQ(test::errorText(runPgoGen(*M)), "");
  EXPECT_EQ(test::errorText(lowerPgo(*M, TM).takeError()), "");
  EXPECT_FALSE(verifyModule(*M, &errs()));
  return isa<CondBrInst>(M->getFunction("w")->getEntryBlock().getTerminator());
}

/// A TargetMachine for the host triple with its baseline CPU and no extra
/// features, so costs don't depend on the machine the test runs on.
std::unique_ptr<TargetMachine> makeBaselineTargetMachine() {
  InitializeNativeTarget();
  Triple TT(sys::getProcessTriple());
  std::string Err;
  const Target *T = TargetRegistry::lookupTarget(TT, Err);
  EXPECT_TRUE(T) << Err;
  StringRef CPU = TT.isX86() ? "x86-64" : "generic";
  return std::unique_ptr<TargetMachine>(
      T->createTargetMachine(TT, CPU, "", TargetOptions(), std::nullopt));
}

TEST(Instrument, LoopRotationCostsInstructionsWithTheGivenTarget) {
  EXPECT_TRUE(rotatedWithGuard(nullptr));
  auto TM = makeBaselineTargetMachine();
  EXPECT_FALSE(rotatedWithGuard(TM.get()));
}

TEST(Instrument, CountersAreExactWithAHostTargetMachine) {
  auto TM = test::makeHostTargetMachine();
  Instrumented I = instrument(ShapesIR, /*RunO2=*/false, TM.get());
  auto *Loop2 = I.fn<int(int)>("loop2");
  EXPECT_EQ(Loop2(3), 6); // 2 back edges
  EXPECT_EQ(Loop2(1), 2); // none
  EXPECT_EQ(I.counters("loop2"), ArrayRef<uint64_t>({2, 2}));
}

TEST(Instrument, CountersAreExactAfterO2InlinesThem) {
  Instrumented I = instrument(InlinedLoopIR, /*RunO2=*/true);
  auto *Caller = I.fn<uint64_t(uint64_t)>("caller");
  EXPECT_EQ(Caller(1000), 1000u);
  // @leaf was inlined into the loop; its copy still counts for @leaf.
  EXPECT_EQ(I.counters("leaf"), ArrayRef<uint64_t>({1000}));
  EXPECT_EQ(I.counters("caller")[0], 1u);
}

TEST(Instrument, LoweredModulesNeedNoCompilerRtProfileRuntime) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, ValueSitesIR);
  ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
  Expected<LoweredPgo> L = lowerPgo(*M);
  ASSERT_EQ(test::errorText(L), "");
  EXPECT_FALSE(verifyModule(*M, &errs()));
  // __llvm_profile_runtime is defined by the module itself ...
  GlobalVariable *Hook = M->getGlobalVariable("__llvm_profile_runtime");
  ASSERT_NE(Hook, nullptr);
  EXPECT_FALSE(Hook->isDeclaration());
  // ... and value sites call orc-pgo's callbacks, not compiler-rt's.
  EXPECT_EQ(M->getFunction("__llvm_profile_instrument_target"), nullptr);
  EXPECT_EQ(M->getFunction("__llvm_profile_instrument_memop"), nullptr);
  EXPECT_NE(M->getFunction("__orc_pgo_instrument_target"), nullptr);
  EXPECT_NE(M->getFunction("__orc_pgo_instrument_memop"), nullptr);
}

// Linux JITs ELF, where ORC sees more of what Gen and lowering emit than on
// MachO (comdats, e.g.), so link the lowered modules as ELF on every host.
TEST(Instrument, LoweredModulesLinkAsElf) {
  for (auto [IR, NumFunctions] :
       {std::pair<StringRef, uint32_t>{ShapesIR, 7}, {ValueSitesIR, 5}}) {
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = test::parseIRFor(*Ctx, IR, test::hostElfTriple());
    ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
    ASSERT_EQ(test::errorText(lowerPgo(*M).takeError()), "");
    // On ELF, Gen emits the version variable in a comdat any, and lowering
    // keeps LLVM's form; ORC's IR layer calls it weak and JITLink strong, so
    // jitModule makes it weak.
    GlobalVariable *Version =
        M->getGlobalVariable("__llvm_profile_raw_version");
    ASSERT_NE(Version, nullptr);
    EXPECT_TRUE(Version->hasExternalLinkage());
    ASSERT_TRUE(Version->hasComdat());
    EXPECT_EQ(Version->getComdat()->getSelectionKind(), Comdat::Any);
    auto J = test::jitModule(test::makeJIT(M->getTargetTriple()), std::move(M),
                             std::move(Ctx));
    // Linking, without running any of it. (Asserting LLVMs abort here if the
    // IR's symbol flags differ from the object's.)
    EXPECT_EQ(*test::lookupSym<uint32_t>(*J, NumFunctionsName), NumFunctions);
  }
}

TEST(Instrument, AnExistingProfileRuntimeHookIsKept) {
  LLVMContext Ctx;
  auto M =
      test::parseHostIR(Ctx, std::string(ShapesIR) +
                                 "\n@__llvm_profile_runtime = global i32 7\n");
  ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
  ASSERT_EQ(test::errorText(lowerPgo(*M).takeError()), "");
  EXPECT_FALSE(verifyModule(*M, &errs()));
  auto *Hook = M->getGlobalVariable("__llvm_profile_runtime");
  EXPECT_EQ(cast<ConstantInt>(Hook->getInitializer())->getZExtValue(), 7u);
}

TEST(Instrument, RunPgoGenReportsOptionSetupFailures) {
  test::RequiredPositionalOption Positional;
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, ShapesIR);
  EXPECT_EQ(test::errorText(runPgoGen(*M)), Positional.ConfigureError);
}

std::string dataSection(const Module &M) {
  return getInstrProfSectionName(IPSK_data,
                                 M.getTargetTriple().getObjectFormat());
}

TEST(Instrument, LowerPgoRejectsUndecodableRecords) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, "@__profd_bogus = global i32 0");
  M->getGlobalVariable("__profd_bogus")->setSection(dataSection(*M));
  Expected<LoweredPgo> L = lowerPgo(*M);
  EXPECT_EQ(test::errorText(L), "unsupported __profd record __profd_bogus");
}

constexpr const char *FakeRecordIR = R"(
@__profd_fake = global { i64, i64, i64, i64, i64, ptr, ptr, i32, [3 x i16], i16, i32 }
  { i64 NAMEREF, i64 0, i64 0, i64 0, i64 0, ptr null, ptr null, i32 0,
    [3 x i16] zeroinitializer, i16 0, i32 0 }
define i32 @leaf(i32 %x) {
  ret i32 %x
}
)";

std::unique_ptr<Module> withFakeRecord(LLVMContext &Ctx, uint64_t NameRef) {
  std::string IR = FakeRecordIR;
  IR.replace(IR.find("NAMEREF"), 7, std::to_string(NameRef));
  auto M = test::parseHostIR(Ctx, IR);
  M->getGlobalVariable("__profd_fake")->setSection(dataSection(*M));
  return M;
}

TEST(Instrument, LowerPgoRejectsRecordsWithUnknownNames) {
  LLVMContext Ctx;
  auto M = withFakeRecord(Ctx, 12345);
  ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
  Expected<LoweredPgo> L = lowerPgo(*M);
  EXPECT_EQ(test::errorText(L), "no PGO name for __profd record __profd_fake");
}

TEST(Instrument, LowerPgoRejectsRecordsWithoutCounters) {
  LLVMContext Ctx;
  auto M = withFakeRecord(Ctx, IndexedInstrProf::ComputeHash("leaf"));
  ASSERT_EQ(test::errorText(runPgoGen(*M)), "");
  Expected<LoweredPgo> L = lowerPgo(*M);
  EXPECT_EQ(test::errorText(L), "no counters for __profd record __profd_fake");
}

} // namespace
