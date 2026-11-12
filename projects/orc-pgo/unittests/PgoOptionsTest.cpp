// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/PgoOptions.h"
#include "TestSupport.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LibraryOptionsContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace orc_pgo;

namespace {

// One function per option parsePgoOptions sets.
constexpr const char *IR = R"(
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
define i32 @sel(i1 %c) {
  %s = select i1 %c, i32 1, i32 2
  ret i32 %s
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

/// What PGOInstrumentationGen, under the current options, put into IR.
struct Gen {
  bool EntryCounted = false;  // pgo-instrument-entry
  bool SelectStepped = false; // pgo-instr-select
  // disable-vp, and pgo-instr-memop for the second.
  unsigned IndirectCallSites = 0;
  unsigned MemOpSites = 0;
};

Gen runGen(LLVMContext &Ctx) {
  std::unique_ptr<Module> M = test::parseIR(Ctx, IR);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB(Ctx);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  PGOInstrumentationGen().run(*M, MAM);
  EXPECT_FALSE(verifyModule(*M, &errs()));

  Gen G;
  for (Instruction &I : M->getFunction("diamond")->getEntryBlock())
    G.EntryCounted |= isa<InstrProfIncrementInst>(I);
  for (Instruction &I : instructions(*M->getFunction("sel")))
    G.SelectStepped |= isa<InstrProfIncrementInstStep>(I);
  for (Function &F : *M) {
    for (Instruction &I : instructions(F)) {
      if (auto *VP = dyn_cast<InstrProfValueProfileInst>(&I)) {
        uint64_t Kind = VP->getValueKind()->getZExtValue();
        G.IndirectCallSites += Kind == IPVK_IndirectCallTarget;
        G.MemOpSites += Kind == IPVK_MemOPSize;
      }
    }
  }
  return G;
}

void parse(LLVMContext &Ctx, std::vector<const char *> Args) {
  Args.insert(Args.begin(), "test");
  std::string Message;
  raw_string_ostream OS(Message);
  ASSERT_TRUE(
      cl::parseCommandLineOptions(Ctx.getLibraryOptions(), Args, "", &OS))
      << Message;
}

void expectConfigured(const Gen &G) {
  EXPECT_TRUE(G.EntryCounted);
  EXPECT_TRUE(G.SelectStepped);
  EXPECT_EQ(G.IndirectCallSites, 1u);
  EXPECT_EQ(G.MemOpSites, 1u);
}

TEST(PgoOptions, OverridesEarlierValues) {
  LLVMContext Ctx;
  parse(Ctx, {"-pgo-instrument-entry=false", "-pgo-instr-select=false",
              "-disable-vp=true", "-pgo-instr-memop=false"});
  Gen Before = runGen(Ctx);
  EXPECT_FALSE(Before.EntryCounted);
  EXPECT_FALSE(Before.SelectStepped);
  EXPECT_EQ(Before.IndirectCallSites, 0u);
  EXPECT_EQ(Before.MemOpSites, 0u);

  ASSERT_EQ(test::errorText(parsePgoOptions(Ctx.getLibraryOptions())), "");
  expectConfigured(runGen(Ctx));
}

TEST(PgoOptions, TheEntryIsNotInstrumentedByDefault) {
  // Why orc-pgo sets pgo-instrument-entry: the diamond's spanning tree would
  // otherwise leave its entry block without a counter.
  LLVMContext Ctx;
  Gen G = runGen(Ctx);
  EXPECT_FALSE(G.EntryCounted);
  EXPECT_TRUE(G.SelectStepped);
}

TEST(PgoOptions, OnlyTheContextsGivenThemRunUnderThem) {
  cl::LibraryOptionsContext Parsed;
  ASSERT_EQ(test::errorText(parsePgoOptions(Parsed)), "");
  LLVMContext Assigned;
  Assigned.getLibraryOptions().assign(Parsed);
  expectConfigured(runGen(Assigned));
  LLVMContext Other;
  EXPECT_FALSE(runGen(Other).EntryCounted);
}

TEST(PgoOptions, AFailedParseIsReported) {
  {
    test::RequiredPositionalOption Positional;
    LLVMContext Ctx;
    EXPECT_EQ(test::errorText(parsePgoOptions(Ctx.getLibraryOptions())),
              Positional.ParseError);
  }
  LLVMContext Ctx;
  ASSERT_EQ(test::errorText(parsePgoOptions(Ctx.getLibraryOptions())), "");
  expectConfigured(runGen(Ctx));
}

TEST(PgoOptions, CheckWantsThemParsedLast) {
  LLVMContext Ctx;
  EXPECT_EQ(test::errorText(checkPgoOptions(Ctx)),
            test::MissingPgoOptionsError);
  parse(Ctx, {"-pgo-function-size-threshold=4", "-disable-vp=true",
              "-pgo-instr-memop=false", "-pgo-instr-select=false"});
  EXPECT_EQ(test::errorText(checkPgoOptions(Ctx)),
            test::MissingPgoOptionsError);
  ASSERT_EQ(test::errorText(parsePgoOptions(Ctx.getLibraryOptions())), "");
  EXPECT_EQ(test::errorText(checkPgoOptions(Ctx)), "");
  parse(Ctx, {"-pgo-instr-select=false"});
  EXPECT_EQ(test::errorText(checkPgoOptions(Ctx)),
            test::MissingPgoOptionsError);

  LLVMContext Assigned;
  Assigned.getLibraryOptions().assign(Ctx.getLibraryOptions());
  EXPECT_EQ(test::errorText(checkPgoOptions(Assigned)),
            test::MissingPgoOptionsError);
  ASSERT_EQ(test::errorText(parsePgoOptions(Ctx.getLibraryOptions())), "");
  Assigned.getLibraryOptions().assign(Ctx.getLibraryOptions());
  EXPECT_EQ(test::errorText(checkPgoOptions(Assigned)), "");
}

std::string conflicts(std::vector<std::string> Args) {
  return test::errorText(checkPgoOptionConflicts(Args));
}

TEST(PgoOptionConflicts, MatchingAndUnrelatedValuesAreAccepted) {
  EXPECT_EQ(conflicts({}), "");
  EXPECT_EQ(conflicts({"-pgo-instrument-entry=true", "-pgo-instr-select",
                       "-disable-vp=0", "--pgo-instr-memop=1",
                       "-pgo-function-size-threshold=4", "positional",
                       "-pgo-instrument-entryx=false"}),
            "");
}

TEST(PgoOptionConflicts, AConflictingValueIsRejected) {
  EXPECT_EQ(conflicts({"-pgo-instrument-entry=false"}),
            "LLVM option -pgo-instrument-entry was set to false, but orc-pgo "
            "requires true");
  EXPECT_EQ(conflicts({"--pgo-instr-select=0"}),
            "LLVM option -pgo-instr-select was set to false, but orc-pgo "
            "requires true");
  EXPECT_EQ(conflicts({"-pgo-instr-memop=false"}),
            "LLVM option -pgo-instr-memop was set to false, but orc-pgo "
            "requires true");
}

TEST(PgoOptionConflicts, ABareBoolOptionIsTrue) {
  EXPECT_EQ(
      conflicts({"-disable-vp"}),
      "LLVM option -disable-vp was set to true, but orc-pgo requires false");
  EXPECT_EQ(
      conflicts({"-disable-vp=1"}),
      "LLVM option -disable-vp was set to true, but orc-pgo requires false");
}

TEST(PgoOptionConflicts, TheLastOccurrenceWins) {
  EXPECT_EQ(conflicts({"-disable-vp", "-disable-vp=false"}), "");
  EXPECT_EQ(
      conflicts({"-disable-vp=false", "-disable-vp"}),
      "LLVM option -disable-vp was set to true, but orc-pgo requires false");
}

TEST(PgoOptionsFingerprint, HashesTheParsedArguments) {
  EXPECT_EQ(pgoOptionsFingerprint(),
            xxh3_64bits("-pgo-instrument-entry=true;-pgo-instr-select=true;"
                        "-disable-vp=false;-pgo-instr-memop=true;"));
}

} // namespace
