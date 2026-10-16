// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Pipeline.h"
#include "SimulatedAot.h"
#include "orc-pgo/Analyses.h"
#include "orc-pgo/Annotate.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/Instrument.h"
#include "orc-pgo/PgoOptions.h"
#include "orc-pgo/ProfData.h"
#include "orc-pgo/ProfileBuilder.h"
#include "orc-pgo/Snapshot.h"
#include "orc-pgo/SymbolTable.h"
#include "orc-pgo/Thunks.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/PreISelIntrinsicLowering.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <set>
#include <string>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *ProgramIR = R"(
source_filename = "aot.c"

@g = global i32 0

declare extern_weak void @weakref()

define i32 @add1(i32 %x) noinline {
entry:
  %c = icmp sgt i32 %x, 100
  br i1 %c, label %big, label %small
big:
  ret i32 0
small:
  %r = add i32 %x, 1
  ret i32 %r
}
define internal i32 @twice(i32 %x) noinline {
  %r = mul i32 %x, 2
  ret i32 %r
}
define i32 @calltwice(i32 %x) {
  %r = call i32 @twice(i32 %x)
  ret i32 %r
}
define i32 @callind(ptr %f, i32 %x) {
  %r = call i32 %f(i32 %x)
  ret i32 %r
}
define i32 @va(i32 %n, ...) {
  ret i32 %n
}
define void @bump() {
  %v = load i32, ptr @g
  %n = add i32 %v, 1
  store i32 %n, ptr @g
  ret void
}
define i32 @hasweak() {
  %c = icmp ne ptr @weakref, null
  %r = zext i1 %c to i32
  ret i32 %r
}
define i32 @main() {
  call void @bump()
  ret i32 0
}
)";

std::string printed(const Module &M) {
  std::string S;
  raw_string_ostream OS(S);
  M.print(OS, nullptr);
  return S;
}

/// What platformSymbols must return, written out: the libc and unwinder names,
/// those only Darwin has, then each libm function in three precisions.
std::vector<std::string> expectedPlatformSymbols(bool Darwin) {
  std::vector<std::string> Out = {"_Unwind_Resume", "memcpy", "memmove",
                                  "memset", "__stack_chk_fail"};
  if (Darwin)
    Out.insert(Out.end(), {"bzero", "memset_pattern4", "memset_pattern8",
                           "memset_pattern16", "__stack_chk_guard"});
  for (const char *Name :
       {"sin",  "cos",  "tan",   "asin",  "acos", "atan",      "atan2", "sinh",
        "cosh", "tanh", "pow",   "exp",   "exp2", "log",       "log2",  "log10",
        "fmod", "ceil", "floor", "trunc", "rint", "nearbyint", "round", "fma",
        "fmin", "fmax", "ldexp", "frexp", "modf"})
    for (const char *Suffix : {"", "f", "l"})
      Out.push_back(std::string(Name) + Suffix);
  return Out;
}

bool hostIsDarwin() {
  return test::makeHostTargetMachine()->getTargetTriple().isOSDarwin();
}

/// The names in M's symbol table.
std::vector<std::string> tableNames(const Module &M) {
  const GlobalVariable *Table = M.getNamedGlobal(SymbolTableName);
  std::vector<std::string> Names;
  const Constant *Init = Table->getInitializer();
  for (unsigned I = 0,
                E = cast<ArrayType>(Table->getValueType())->getNumElements();
       I != E; ++I) {
    StringRef Name;
    EXPECT_TRUE(getConstantStringInfo(
        Init->getAggregateElement(I)->getAggregateElement(0u), Name));
    Names.push_back(Name.str());
  }
  return Names;
}

struct Built {
  LLVMContext InputCtx;
  std::unique_ptr<Module> Input;
  std::unique_ptr<TargetMachine> TM;
  AotModule Aot;

  explicit Built(StringRef IR)
      : Built(IR, test::makeHostTargetMachine(Reloc::PIC_)) {}
  Built(StringRef IR, std::unique_ptr<TargetMachine> BuildTM)
      : Input(test::parseIR(InputCtx, IR)), TM(std::move(BuildTM)) {
    Expected<AotModule> A = buildAotModule(*Input, *TM);
    if (!A)
      report_fatal_error(A.takeError());
    Aot = std::move(*A);
    EXPECT_FALSE(verifyModule(*Aot.M, &errs()));
  }
  Module &M() { return *Aot.M; }
};

TEST(Pipeline, TheInputGetsTheTargetAndNamesForUnnamedGlobals) {
  Built B("@0 = global i32 0\ndefine i32 @f() {\n  ret i32 0\n}\n");
  EXPECT_EQ(B.Input->getTargetTriple(), B.TM->getTargetTriple());
  EXPECT_EQ(B.Input->getDataLayout(), B.TM->createDataLayout());
  for (const GlobalVariable &GV : B.Input->globals())
    EXPECT_TRUE(GV.hasName());
}

TEST(Pipeline, TheEmbeddedSnapshotIsTheInputModule) {
  Built B(ProgramIR);
  auto *Image =
      cast<ConstantStruct>(B.M().getNamedGlobal(ImageName)->getInitializer());
  auto *Snapshot = cast<GlobalVariable>(Image->getOperand(2));
  StringRef Bytes =
      cast<ConstantDataArray>(Snapshot->getInitializer())->getRawDataValues();
  EXPECT_EQ(cast<ConstantInt>(Image->getOperand(3))->getZExtValue(),
            Bytes.size());
  LLVMContext Ctx;
  auto Parsed = parseSnapshot(Bytes, Ctx);
  ASSERT_EQ(test::errorText(Parsed), "");
  (*Parsed)->setModuleIdentifier(B.Input->getModuleIdentifier());
  EXPECT_EQ(printed(**Parsed), printed(*B.Input));
}

TEST(Pipeline, EligibleFunctionsBecomeThunks) {
  Built B(ProgramIR);
  for (StringRef Name :
       {"add1", "twice", "calltwice", "callind", "bump", "hasweak", "main"}) {
    Function *Body = B.M().getFunction((Name + AotSuffix).str());
    ASSERT_NE(Body, nullptr) << Name.str();
    EXPECT_TRUE(Body->hasInternalLinkage());
  }
  EXPECT_EQ(B.M().getFunction("va.orc_pgo.aot"), nullptr);
  auto *Load =
      cast<LoadInst>(&B.M().getFunction("add1")->getEntryBlock().front());
  EXPECT_EQ(Load->getOrdering(), AtomicOrdering::Acquire);
  EXPECT_TRUE(cast<CallInst>(Load->getNextNode())->isMustTailCall());
}

TEST(Pipeline, MainCallsStartFirst) {
  Built B(ProgramIR);
  Function *Start = B.M().getFunction(StartName);
  ASSERT_NE(Start, nullptr);
  ASSERT_EQ(Start->getNumUses(), 1u);
  auto *Call = cast<CallInst>(*Start->user_begin());
  EXPECT_EQ(Call->getFunction()->getName(), "main");
  EXPECT_EQ(&Call->getFunction()->getEntryBlock().front(), Call);
}

TEST(Pipeline, AModuleWithoutMainHasNoStartCall) {
  Built B("define i32 @f() {\n  ret i32 0\n}\n");
  EXPECT_EQ(B.M().getFunction(StartName), nullptr);
}

TEST(Pipeline, AConstructorRegistersTheImage) {
  Built B(ProgramIR);
  Function *Register = B.M().getFunction(RegisterImageName);
  ASSERT_NE(Register, nullptr);
  ASSERT_EQ(Register->getNumUses(), 1u);
  auto *Call = cast<CallInst>(*Register->user_begin());
  EXPECT_EQ(Call->getArgOperand(0), B.M().getNamedGlobal(ImageName));
  auto *Ctors = cast<ConstantArray>(
      B.M().getNamedGlobal("llvm.global_ctors")->getInitializer());
  ASSERT_EQ(Ctors->getNumOperands(), 1u);
  EXPECT_EQ(cast<ConstantStruct>(Ctors->getOperand(0))->getOperand(1),
            Call->getFunction());
}

TEST(Pipeline, TheImageTypeHasTheCLayout) {
  Built B(ProgramIR);
  auto *Ty = cast<StructType>(B.M().getNamedGlobal(ImageName)->getValueType());
  const DataLayout &DL = B.M().getDataLayout();
  const StructLayout *L = DL.getStructLayout(Ty);
  EXPECT_EQ(L->getSizeInBytes(), sizeof(OrcPgoImage));
  const size_t Offsets[] = {offsetof(OrcPgoImage, Version),
                            offsetof(OrcPgoImage, Fingerprint),
                            offsetof(OrcPgoImage, Snapshot),
                            offsetof(OrcPgoImage, SnapshotSize),
                            offsetof(OrcPgoImage, Functions),
                            offsetof(OrcPgoImage, NumFunctions),
                            offsetof(OrcPgoImage, VNodePool),
                            offsetof(OrcPgoImage, Dispatch),
                            offsetof(OrcPgoImage, DispatchFunctions),
                            offsetof(OrcPgoImage, NumDispatch),
                            offsetof(OrcPgoImage, Symbols),
                            offsetof(OrcPgoImage, NumSymbols)};
  ASSERT_EQ(Ty->getNumElements(), std::size(Offsets));
  for (unsigned I = 0; I < std::size(Offsets); ++I)
    EXPECT_EQ(L->getElementOffset(I), Offsets[I]) << "field " << I;
  auto *FpTy = cast<StructType>(Ty->getElementType(1));
  const StructLayout *FL = DL.getStructLayout(FpTy);
  EXPECT_EQ(FL->getSizeInBytes(), sizeof(OrcPgoFingerprint));
  EXPECT_EQ(FL->getElementOffset(1),
            offsetof(OrcPgoFingerprint, RawProfileVersion));
  EXPECT_EQ(FL->getElementOffset(2), offsetof(OrcPgoFingerprint, OptionsHash));
}

TEST(Pipeline, TheImageDescribesTheProgram) {
  auto S = test::simulateAot(ProgramIR);
  const OrcPgoImage &I = *S.Image;
  EXPECT_EQ(I.Version, ImageVersion);
  EXPECT_STREQ(I.Fingerprint.LLVMVersion, LLVM_VERSION_STRING);
  EXPECT_EQ(I.Fingerprint.RawProfileVersion, 0x050000000000000Bull);
  EXPECT_EQ(I.Fingerprint.OptionsHash, pgoOptionsFingerprint());

  std::set<std::string> Records;
  for (const OrcPgoFunctionRecord &R : ArrayRef(I.Functions, I.NumFunctions))
    Records.insert(R.Name);
  EXPECT_EQ(Records, (std::set<std::string>{"add1", "aot.c;twice", "calltwice",
                                            "callind", "va", "bump", "hasweak",
                                            "main"}));

  std::vector<std::string> Slots;
  for (uint64_t Slot = 0; Slot != I.NumDispatch; ++Slot) {
    Slots.push_back(I.Functions[I.DispatchFunctions[Slot]].Name);
    EXPECT_NE(I.Dispatch[Slot], nullptr);
  }
  EXPECT_EQ(Slots,
            (std::vector<std::string>{"add1", "aot.c;twice", "calltwice",
                                      "callind", "bump", "hasweak", "main"}));

  std::vector<std::string> Symbols;
  for (const OrcPgoSymbol &Sym : ArrayRef(I.Symbols, I.NumSymbols))
    Symbols.push_back(Sym.Name);
  // The program's symbols, then the platform's: 29 libm functions in three
  // precisions, 5 libc and unwinder names, and 5 more on Darwin.
  std::vector<std::string> Platform = expectedPlatformSymbols(hostIsDarwin());
  ASSERT_EQ(Platform.size(), 29u * 3 + 5 + (hostIsDarwin() ? 5 : 0));
  ASSERT_EQ(Symbols.size(), 10u + Platform.size());
  EXPECT_EQ(std::vector<std::string>(Symbols.begin(), Symbols.begin() + 10),
            (std::vector<std::string>{"weakref", "add1", "twice", "calltwice",
                                      "callind", "va", "bump", "hasweak",
                                      "main", "g"}));
  EXPECT_EQ(std::vector<std::string>(Symbols.begin() + 10, Symbols.end()),
            Platform);

  LLVMContext Ctx;
  auto Snap = parseSnapshot(StringRef(I.Snapshot, I.SnapshotSize), Ctx);
  ASSERT_EQ(test::errorText(Snap), "");
  EXPECT_NE((*Snap)->getFunction("add1"), nullptr);
  EXPECT_EQ((*Snap)->getFunction("add1.orc_pgo.aot"), nullptr);
  EXPECT_EQ((*Snap)->getNamedGlobal("__llvm_profile_raw_version"), nullptr);
}

int times100(int X) { return X * 100; }

TEST(Pipeline, CallsDispatchThroughTheSlots) {
  auto S = test::simulateAot(ProgramIR);
  auto *Add1 = S.sym<int(int)>("add1");
  unsigned Slot = S.slot("add1");
  EXPECT_EQ(Add1(5), 6);
  void *Aot = S.Image->Dispatch[Slot];
  EXPECT_NE(Aot, reinterpret_cast<void *>(Add1));
  S.Image->Dispatch[Slot] = reinterpret_cast<void *>(&times100);
  EXPECT_EQ(Add1(3), 300);
  S.Image->Dispatch[Slot] = Aot;
  EXPECT_EQ(Add1(3), 4);
}

TEST(Pipeline, CountersCountWhileTheProgramRuns) {
  auto S = test::simulateAot(ProgramIR);
  auto *Add1 = S.sym<int(int)>("add1");
  for (int X = 0; X < 10; ++X)
    EXPECT_EQ(Add1(X), X + 1);
  const OrcPgoFunctionRecord &R = S.record("add1");
  EXPECT_EQ(R.NumCounters, 2u);
  EXPECT_EQ(R.Counters[0], 10u);
  EXPECT_EQ(S.sym<int(int)>("calltwice")(4), 8);
  EXPECT_EQ(S.record("calltwice").Counters[0], 1u);
  EXPECT_EQ(S.record("aot.c;twice").Counters[0], 1u);
}

int ExtraTarget;

TEST(Pipeline, TheSymbolTableResolvesLiveAddresses) {
  auto S = test::simulateAot(ProgramIR, {"orc_pgo_test_extra"},
                             {{"orc_pgo_test_extra", &ExtraTarget}});
  // The program's 10, the caller's 1 and the platform's 29 * 3 + 5 (+ 5 on
  // Darwin).
  EXPECT_EQ(S.Image->NumSymbols, 11u + 29 * 3 + 5 + (hostIsDarwin() ? 5 : 0));
  // Tier-up code can link against the process's libc, libm and unwinder.
  for (const char *Name :
       {"_Unwind_Resume", "memcpy", "memset", "sin", "powf", "fmodl"}) {
    ASSERT_NE(S.symbol(Name), nullptr) << Name;
    EXPECT_NE(S.symbol(Name)->Address, nullptr) << Name;
  }
  ASSERT_NE(S.symbol("orc_pgo_test_extra"), nullptr);
  EXPECT_EQ(S.symbol("orc_pgo_test_extra")->Address, &ExtraTarget);
  ASSERT_NE(S.symbol("weakref"), nullptr);
  EXPECT_EQ(S.symbol("weakref")->Address, nullptr);
  EXPECT_EQ(S.sym<int()>("hasweak")(), 0);
  EXPECT_EQ(S.symbol("g")->Address, S.sym<int>("g"));
  // Calls from tier-up code must reach the thunk, not the AOT body.
  EXPECT_EQ(S.symbol("add1")->Address,
            reinterpret_cast<const void *>(S.sym<int(int)>("add1")));
}

TEST(Pipeline, MainCallsStartBeforeItsBody) {
  auto S = test::simulateAot(ProgramIR);
  unsigned Before = test::simulatedStartCalls();
  int *G = S.sym<int>("g");
  EXPECT_EQ(*G, 0);
  EXPECT_EQ(S.sym<int()>("main")(), 0);
  EXPECT_EQ(test::simulatedStartCalls(), Before + 1);
  EXPECT_EQ(*G, 1);
}

TEST(Pipeline, TheValueNodePoolIsWired) {
  auto S = test::simulateAot(ProgramIR);
  void *const *Pool = S.Image->VNodePool;
  ASSERT_NE(Pool[0], nullptr);
  EXPECT_EQ(static_cast<char *>(Pool[1]) - static_cast<char *>(Pool[0]),
            static_cast<ptrdiff_t>(10 * sizeof(prof::ValueProfNode)));
  auto *Add1 = S.sym<int(int)>("add1");
  auto *CallInd = S.sym<int(int (*)(int), int)>("callind");
  for (int I = 0; I < 3; ++I)
    EXPECT_EQ(CallInd(Add1, 7), 8);
  std::vector<InstrProfValueData> V =
      readValueSite(S.record("callind"), IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(V.size(), 1u);
  EXPECT_EQ(V[0].Value, reinterpret_cast<uint64_t>(Add1));
  EXPECT_EQ(V[0].Count, 3u);
  // Records name the address indirect calls observe: the thunk if there is one.
  EXPECT_EQ(S.record("add1").Address, reinterpret_cast<const void *>(Add1));
  EXPECT_EQ(S.record("va").Address,
            reinterpret_cast<const void *>(S.sym<int(int, ...)>("va")));
}

constexpr const char *AbiIR = R"(
source_filename = "abi.c"
%big = type { i64, i64, i64, i64, i64 }
define void @sretf(ptr sret(%big) %o, i64 %a) {
  %p = getelementptr %big, ptr %o, i32 0, i32 4
  store i64 %a, ptr %p
  ret void
}
define i64 @byvalf(ptr byval(%big) %b) {
  %p = getelementptr %big, ptr %b, i32 0, i32 4
  %v = load i64, ptr %p
  ret i64 %v
}
define i64 @stackargs(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i, i64 %j) {
  %s = add i64 %i, %j
  ret i64 %s
}
define double @mixed(double %x, float %y, i8 signext %z) {
  %c = fpext float %y to double
  %s = sext i8 %z to i32
  %d = sitofp i32 %s to double
  %r = fadd double %x, %c
  %t = fadd double %r, %d
  ret double %t
}
define i32 @main() {
  ret i32 0
}
)";

struct Big {
  long A, B, C, D, E;
};

TEST(Pipeline, ThunksForwardAbiShapesOnTheHost) {
  auto S = test::simulateAot(AbiIR);
  EXPECT_EQ(S.sym<Big(long)>("sretf")(42).E, 42);
  EXPECT_EQ(
      (S.sym<long(long, long, long, long, long, long, long, long, long, long)>(
          "stackargs")(0, 0, 0, 0, 0, 0, 0, 0, 3, 4)),
      7);
  EXPECT_EQ(S.sym<double(double, float, signed char)>("mixed")(1.5, 2.0f, -1),
            2.5);
  // byval parameters are not thunked: the AArch64 backend cannot forward them.
  EXPECT_EQ(S.Image->NumDispatch, 4u); // sretf, stackargs, mixed, main
  // LLVM's byval is not the C ABI of a struct argument on AArch64, so byvalf is
  // only compiled, not called.
  EXPECT_NE(S.sym<void>("byvalf"), nullptr);
}

constexpr const char *NoIpaIR = R"(
source_filename = "noipa.c"
define internal i32 @ignores(i32 %x, i32 %y) noinline {
  %r = add i32 %x, 1
  ret i32 %r
}
define i32 @caller(i32 %y) {
  %r = call i32 @ignores(i32 7, i32 %y)
  ret i32 %r
}
define i32 @main() {
  ret i32 0
}
)";

TEST(Pipeline, OptimizationDoesNotRewriteThunkedSignatures) {
  Built B(NoIpaIR);
  // Without noipa, DeadArgElim would drop %y and IPSCCP would fold %x to 7:
  // the body JIT-compiled from the snapshot could then not replace this one.
  Function *Body = B.M().getFunction("ignores.orc_pgo.aot");
  ASSERT_NE(Body, nullptr);
  EXPECT_EQ(Body->arg_size(), 2u);
  Function *Caller = B.M().getFunction("caller.orc_pgo.aot");
  ASSERT_NE(Caller, nullptr);
  CallInst *Call = nullptr;
  for (Instruction &I : instructions(*Caller))
    if (auto *C = dyn_cast<CallInst>(&I);
        C && C->getCalledFunction() == B.M().getFunction("ignores"))
      Call = C;
  ASSERT_NE(Call, nullptr);
  EXPECT_EQ(Call->getArgOperand(0),
            ConstantInt::get(Call->getArgOperand(0)->getType(), 7));
  EXPECT_EQ(Call->getArgOperand(1), Caller->getArg(0));
  // IPSCCP did not fold %x into the body either.
  EXPECT_TRUE(isa<Argument>(
      cast<BinaryOperator>(*std::find_if(Body->getEntryBlock().begin(),
                                         Body->getEntryBlock().end(),
                                         [](Instruction &I) {
                                           return isa<BinaryOperator>(I) &&
                                                  I.getType()->isIntegerTy(32);
                                         }))
          .getOperand(0)));
}

TEST(Pipeline, ThunkedFunctionsKeepTheirInferredAttributes) {
  Built B(NoIpaIR);
  // noipa stops FunctionAttrs inferring from the bodies after thunkify, so
  // these were inferred before. The thunk reads its slot with an acquire load
  // and its callee writes counters: no memory or nosync claims.
  for (const char *Name : {"ignores", "caller", "ignores.orc_pgo.aot"}) {
    Function *F = B.M().getFunction(Name);
    ASSERT_NE(F, nullptr) << Name;
    EXPECT_TRUE(F->doesNotThrow()) << Name;
    EXPECT_TRUE(F->willReturn()) << Name;
    EXPECT_TRUE(F->doesNotRecurse()) << Name;
    EXPECT_FALSE(F->hasFnAttribute(Attribute::Memory)) << Name;
    EXPECT_FALSE(F->hasFnAttribute(Attribute::NoSync)) << Name;
  }
  EXPECT_TRUE(B.M().getFunction("ignores")->hasFnAttribute(Attribute::NoIPA));
}

// @die is inferred noreturn and cold; see InstrumentTest.
constexpr const char *ColdCallIR = R"(
source_filename = "cold.c"
declare void @abort() noreturn nounwind cold
define void @die() {
  call void @abort()
  unreachable
}
define i32 @f(i32 %x) {
entry:
  switch i32 %x, label %def [ i32 1, label %c1
                              i32 2, label %c2 ]
c1:
  call void @die()
  br label %m
c2:
  br label %m
def:
  br label %m
m:
  ret i32 %x
}
define i32 @g(i32 %x) {
  %c = icmp eq i32 %x, 1000
  br i1 %c, label %bad, label %ok
bad:
  call void @die()
  br label %ok
ok:
  ret i32 %x
}
)";

TEST(Pipeline, TierUpAnnotatesTheCountersGenPlacedDespiteInferredColdCalls) {
  auto S = test::simulateAot(ColdCallIR);
  auto *F = S.sym<int(int)>("f");
  // Never 1: that case calls @die.
  for (int X : {2, 2, 3, 5})
    EXPECT_EQ(F(X), X);
  EXPECT_EQ(S.sym<int(int)>("g")(7), 7);

  auto Profile =
      buildIndexedProfile(ArrayRef(S.Image->Functions, S.Image->NumFunctions));
  ASSERT_EQ(test::errorText(Profile), "");
  LLVMContext Ctx;
  auto M =
      parseSnapshot(StringRef(S.Image->Snapshot, S.Image->SnapshotSize), Ctx);
  ASSERT_EQ(test::errorText(M), "");
  // The snapshot carries the attributes Gen saw, and Use reports no mismatch.
  EXPECT_TRUE((*M)->getFunction("die")->doesNotReturn());
  EXPECT_EQ(test::errorText(annotateWithProfile(**M, **Profile, {"f", "g"})),
            "");
  // Switch weights: default, then each case.
  auto *Switch =
      cast<SwitchInst>(&(*M)->getFunction("f")->getEntryBlock().back());
  SmallVector<uint32_t> W;
  ASSERT_TRUE(extractBranchWeights(*Switch, W));
  EXPECT_EQ(std::vector<uint32_t>(W.begin(), W.end()),
            (std::vector<uint32_t>{2, 0, 2}));
}

// Loop rotation duplicates a loop header only if TargetTransformInfo costs it
// at 16 instructions or fewer. Without a target every instruction costs 1, so
// this header (2 phis, 3 adds and 3 muls, extractelement, trunc, icmp, br)
// rotates. On a baseline CPU (x86-64's SSE2, AArch64's NEON) each <64 x i64>
// operation legalizes into 32 128-bit ones or more, so it doesn't. The adds and
// muls alternate so that O2 can't fold them into one operation, which would
// make the header cheap. The test builds for the generic CPU: a host CPU could
// be wide enough (e.g. AVX-512) to rotate the loop.
constexpr const char *WideHeaderIR = R"(
source_filename = "w.c"
define i32 @w(i32 %n, ptr %p) {
entry:
  %v = load <64 x i64>, ptr %p
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %acc = phi <64 x i64> [ %v, %entry ], [ %a5, %body ]
  %a0 = add <64 x i64> %acc, %v
  %a1 = mul <64 x i64> %a0, %v
  %a2 = add <64 x i64> %a1, %v
  %a3 = mul <64 x i64> %a2, %v
  %a4 = add <64 x i64> %a3, %v
  %a5 = mul <64 x i64> %a4, %v
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

/// Whether F's entry block ends in a conditional branch, i.e. loop rotation
/// put a guard there.
bool entryIsGuarded(const Function &F) {
  return isa<CondBrInst>(F.getEntryBlock().getTerminator());
}

TEST(Pipeline, LoweringRotatesLoopsWithTheBuildsTarget) {
  // Lowered without a target, the loop is rotated.
  LLVMContext Ctx;
  auto Generic = test::parseHostIR(Ctx, WideHeaderIR);
  ASSERT_EQ(test::errorText(runPgoGen(*Generic)), "");
  ASSERT_EQ(test::errorText(lowerPgo(*Generic).takeError()), "");
  EXPECT_TRUE(entryIsGuarded(*Generic->getFunction("w")));

  // The build lowers with its TargetMachine, as clang does, so neither
  // lowering nor O2 rotates the loop.
  Built B(WideHeaderIR, test::makeTargetMachine(Triple(sys::getProcessTriple()),
                                                Reloc::PIC_));
  Function *Body = B.M().getFunction(("w" + AotSuffix).str());
  ASSERT_NE(Body, nullptr);
  EXPECT_FALSE(entryIsGuarded(*Body));
}

// pow(2.0, x) becomes exp2(x) only if the module has no other idea of what
// exp2 is: SimplifyLibCalls will not call a function whose declared type is
// not the library function's.
constexpr const char *Exp2IR = R"(
source_filename = "exp2.c"
declare double @pow(double, double)
define double @p(double %x) {
  %r = call double @pow(double 2.0, double %x)
  ret double %r
}
define i32 @main() {
  ret i32 0
}
)";

/// The callee names of F's calls, in order.
std::vector<std::string> callees(const Function &F) {
  std::vector<std::string> Names;
  for (const Instruction &I : instructions(F))
    if (const auto *Call = dyn_cast<CallInst>(&I))
      if (const Function *Callee = Call->getCalledFunction())
        Names.push_back(Callee->getName().str());
  return Names;
}

TEST(Pipeline, O2StillFormsLibraryCallsTheModuleDoesNotDeclare) {
  Built B(Exp2IR);
  Function *Body = B.M().getFunction("p.orc_pgo.aot");
  ASSERT_NE(Body, nullptr);
  std::vector<std::string> Callees = callees(*Body);
  EXPECT_EQ(std::count(Callees.begin(), Callees.end(), "exp2"), 1);
  EXPECT_EQ(std::count(Callees.begin(), Callees.end(), "pow"), 0);
  // The table lists the declaration O2 made, not a stub.
  Function *Exp2 = B.M().getFunction("exp2");
  ASSERT_NE(Exp2, nullptr);
  EXPECT_EQ(Exp2->getFunctionType(),
            FunctionType::get(Type::getDoubleTy(B.M().getContext()),
                              {Type::getDoubleTy(B.M().getContext())}, false));
  std::vector<std::string> Names = tableNames(B.M());
  EXPECT_EQ(std::count(Names.begin(), Names.end(), "exp2"), 1);
  EXPECT_EQ(std::count(Names.begin(), Names.end(), "sin"), 1);
}

// O2 turns this pattern store into llvm.experimental.memset.pattern, which code
// generation lowers to a memset_pattern16 call on Darwin, if the module's
// memset_pattern16 has the library's prototype. Built for a Darwin target on
// any host, since the lowering pass runs without generating code.
constexpr const char *PatternIR = R"(
source_filename = "pattern.c"
define void @fill(ptr noalias %p, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %q = getelementptr i32, ptr %p, i64 %i
  store i32 305419896, ptr %q
  %i1 = add i64 %i, 1
  %c = icmp ult i64 %i1, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
define i32 @main() {
  ret i32 0
}
)";

TEST(Pipeline, CodegenFormsMemsetPattern16OnDarwin) {
  LLVMContext Ctx;
  auto Input = test::parseIR(Ctx, PatternIR);
  Triple Darwin(Twine(Triple::getArchTypeName(
                    Triple(sys::getProcessTriple()).getArch())) +
                "-apple-macosx11.0");
  auto TM = test::makeTargetMachine(Darwin, Reloc::PIC_);
  Expected<AotModule> Aot = buildAotModule(*Input, *TM);
  ASSERT_EQ(test::errorText(Aot), "");
  Function *Body = Aot->M->getFunction("fill.orc_pgo.aot");
  ASSERT_NE(Body, nullptr);
  std::vector<std::string> Before = callees(*Body);
  EXPECT_EQ(std::count(Before.begin(), Before.end(),
                       "llvm.experimental.memset.pattern.p0.i32.i64"),
            1);

  PgoAnalysisManagers AM(*Aot->M, TM.get());
  ModulePassManager MPM;
  MPM.addPass(PreISelIntrinsicLoweringPass(TM.get()));
  MPM.run(*Aot->M, AM.MAM);
  ASSERT_FALSE(verifyModule(*Aot->M, &errs()));
  std::vector<std::string> After = callees(*Body);
  EXPECT_EQ(std::count(After.begin(), After.end(), "memset_pattern16"), 1);
  Function *Pattern = Aot->M->getFunction("memset_pattern16");
  ASSERT_NE(Pattern, nullptr);
  EXPECT_EQ(Pattern->arg_size(), 3u);
  std::vector<std::string> Names = tableNames(*Aot->M);
  EXPECT_EQ(std::count(Names.begin(), Names.end(), "memset_pattern16"), 1);
  EXPECT_EQ(std::count(Names.begin(), Names.end(), "bzero"), 1);
}

// Two independent halves of a vector add: only the SLP vectorizer, which
// clang runs at -O2 and PassBuilder does not by default, combines them.
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
define i32 @main() {
  ret i32 0
}
)";

TEST(Pipeline, O2RunsTheSlpVectorizer) {
  Built B(SlpIR);
  Function *Body = B.M().getFunction("slp.orc_pgo.aot");
  ASSERT_NE(Body, nullptr);
  unsigned VectorAdds = 0;
  for (const Instruction &I : instructions(*Body))
    VectorAdds +=
        I.getOpcode() == Instruction::Add && I.getType()->isVectorTy();
  EXPECT_GE(VectorAdds, 1u);
}

TEST(Pipeline, TuningOptionsMirrorClangs) {
  for (OptimizationLevel Level :
       {OptimizationLevel::O2, OptimizationLevel::O3}) {
    PipelineTuningOptions PTO = clangPipelineTuningOptions(Level);
    EXPECT_TRUE(PTO.LoopUnrolling);
    EXPECT_TRUE(PTO.LoopInterleaving);
    EXPECT_TRUE(PTO.LoopVectorization);
    EXPECT_TRUE(PTO.SLPVectorization);
    EXPECT_TRUE(PTO.LoopInterchange);
    EXPECT_FALSE(PTO.LoopFusion);
    EXPECT_FALSE(PTO.MergeFunctions);
    EXPECT_TRUE(PTO.CallGraphProfile);
  }
  // clang neither unrolls nor vectorizes at -O1.
  PipelineTuningOptions O1 = clangPipelineTuningOptions(OptimizationLevel::O1);
  EXPECT_FALSE(O1.LoopUnrolling);
  EXPECT_FALSE(O1.LoopInterleaving);
  EXPECT_FALSE(O1.LoopVectorization);
  EXPECT_FALSE(O1.SLPVectorization);
}

TEST(Pipeline, EmitsANativeObject) {
  LLVMContext Ctx;
  auto Input = test::parseIR(Ctx, ProgramIR);
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  Expected<SmallVector<char, 0>> Object = buildAotObject(*Input, *TM);
  ASSERT_EQ(test::errorText(Object), "");
  auto Obj = cantFail(object::ObjectFile::createObjectFile(
      MemoryBufferRef(StringRef(Object->data(), Object->size()), "aot.o")));
  std::set<std::string> Defined, Undefined;
  for (const object::SymbolRef &Sym : Obj->symbols()) {
    StringRef Name = cantFail(Sym.getName());
    if (Obj->isMachO())
      Name.consume_front("_");
    if (cantFail(Sym.getFlags()) & object::SymbolRef::SF_Undefined)
      Undefined.insert(Name.str());
    else
      Defined.insert(Name.str());
  }
  EXPECT_TRUE(Defined.count("add1"));
  EXPECT_TRUE(Defined.count("main"));
  EXPECT_TRUE(Undefined.count(RegisterImageName.str()));
  EXPECT_TRUE(Undefined.count(StartName.str()));
}

TEST(Pipeline, TheSymbolTableHasWhatCodegenCallsWithoutADeclaration) {
  LLVMContext Ctx;
  auto Input = test::parseIR(Ctx, ProgramIR);
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  Expected<AotModule> Aot = buildAotModule(*Input, *TM, {"__udivti3"});
  ASSERT_EQ(test::errorText(Aot), "");
  ASSERT_NE(Aot->M->getNamedGlobal(SymbolTableName), nullptr);
  std::vector<std::string> Listed = tableNames(*Aot->M);
  std::set<std::string> Names(Listed.begin(), Listed.end());
  // From the program, from the caller, and from the platform list.
  for (const char *Name :
       {"add1", "g", "__udivti3", "_Unwind_Resume", "memcpy", "memset", "sin",
        "sinf", "sinl", "pow", "fmod", "log2f"})
    EXPECT_TRUE(Names.count(Name)) << Name;
  for (const std::string &Name : expectedPlatformSymbols(hostIsDarwin()))
    EXPECT_TRUE(Names.count(Name)) << Name;
  // Declared where the program has no declaration, so tier-up code can link.
  Function *Resume = Aot->M->getFunction("_Unwind_Resume");
  ASSERT_NE(Resume, nullptr);
  EXPECT_TRUE(Resume->isDeclaration());
}

TEST(Pipeline, TheObjectReferencesThePlatformSymbols) {
  LLVMContext Ctx;
  auto Input = test::parseIR(Ctx, ProgramIR);
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  Expected<SmallVector<char, 0>> Object = buildAotObject(*Input, *TM);
  ASSERT_EQ(test::errorText(Object), "");
  auto Obj = cantFail(object::ObjectFile::createObjectFile(
      MemoryBufferRef(StringRef(Object->data(), Object->size()), "aot.o")));
  std::set<std::string> Undefined;
  for (const object::SymbolRef &Sym : Obj->symbols()) {
    StringRef Name = cantFail(Sym.getName());
    if (Obj->isMachO())
      Name.consume_front("_");
    if (cantFail(Sym.getFlags()) & object::SymbolRef::SF_Undefined)
      Undefined.insert(Name.str());
  }
  // The names the linker must resolve for tier-up code, the unwinder's with
  // the leading underscore of the C name.
  for (const char *Name : {"_Unwind_Resume", "memcpy", "memmove", "memset",
                           "sin", "cosf", "powl", "fmod", "log10"})
    EXPECT_TRUE(Undefined.count(Name)) << Name;
}

TEST(Pipeline, AnOptionSetupFailureFailsTheBuild) {
  test::RequiredPositionalOption Positional;
  LLVMContext Ctx;
  auto Input = test::parseIR(Ctx, ProgramIR);
  auto TM = test::makeHostTargetMachine(Reloc::PIC_);
  Expected<SmallVector<char, 0>> Object = buildAotObject(*Input, *TM);
  EXPECT_EQ(test::errorText(Object), Positional.ConfigureError);
}

} // namespace
