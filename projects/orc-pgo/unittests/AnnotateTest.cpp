// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Annotate.h"
#include "TestSupport.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/Instrument.h"
#include "orc-pgo/ProfileBuilder.h"
#include "orc-pgo/Snapshot.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ProfileData/InstrProf.h"
#include "gtest/gtest.h"

#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *ProgramIR = R"(
source_filename = "p.c"
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
entry:
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
define i32 @never(i32 %x) {
entry:
  ret i32 %x
}
define void @ta() {
entry:
  ret void
}
define void @tb() {
entry:
  ret void
}
define void @dispatch(ptr %fp) {
entry:
  call void %fp()
  ret void
}
)";

/// A program built the way orc-pgo builds it, ready to run and to annotate.
struct Profiled {
  std::string Snapshot;
  std::unique_ptr<orc::LLJIT> J;
  ArrayRef<OrcPgoFunctionRecord> Table;

  explicit Profiled(StringRef IR) {
    LLVMContext Original;
    Snapshot = writeSnapshot(*test::parseHostIR(Original, IR));
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = cantFail(parseSnapshot(Snapshot, *Ctx));
    EXPECT_EQ(test::errorText(runPgoGen(*M)), "");
    Expected<LoweredPgo> L = lowerPgo(*M);
    EXPECT_EQ(test::errorText(L), "");
    J = test::jitModule(std::move(M), std::move(Ctx));
    auto *Pool = test::lookupSym<void *>(*J, VNodePoolName);
    setValueNodePool(Pool[0], Pool[1]);
    Table =
        ArrayRef(test::lookupSym<OrcPgoFunctionRecord>(*J, FunctionTableName),
                 *test::lookupSym<uint32_t>(*J, NumFunctionsName));
  }

  template <typename Fn>
  Fn *fn(StringRef Name) {
    return test::lookupSym<Fn>(*J, Name);
  }

  /// A fresh parse of the snapshot, annotated with the current counters.
  std::unique_ptr<Module>
  annotate(LLVMContext &Ctx, ArrayRef<StringRef> MustBeAnnotated = {},
           ArrayRef<OrcPgoFunctionRecord> Records = {}) {
    auto Profile = buildIndexedProfile(Records.empty() ? Table : Records);
    EXPECT_EQ(test::errorText(Profile), "");
    auto M = cantFail(parseSnapshot(Snapshot, Ctx));
    LastError =
        test::errorText(annotateWithProfile(*M, **Profile, MustBeAnnotated));
    EXPECT_FALSE(verifyModule(*M, &errs()));
    return M;
  }
  std::string LastError;
};

std::vector<uint64_t> weights(const Instruction &I) {
  SmallVector<uint32_t> W;
  if (!extractBranchWeights(I, W))
    return {};
  return std::vector<uint64_t>(W.begin(), W.end());
}

const Instruction &terminatorOf(Module &M, StringRef Fn, StringRef Block) {
  for (BasicBlock &BB : *M.getFunction(Fn))
    if (BB.getName() == Block)
      return *BB.getTerminator();
  ADD_FAILURE() << "no block " << Block.str() << " in " << Fn.str();
  return M.getFunction(Fn)->getEntryBlock().back();
}

const Instruction &firstOf(Module &M, StringRef Fn, unsigned Opcode) {
  for (Instruction &I : instructions(*M.getFunction(Fn)))
    if (I.getOpcode() == Opcode)
      return I;
  ADD_FAILURE() << "no such instruction in " << Fn.str();
  return M.getFunction(Fn)->getEntryBlock().front();
}

TEST(Annotate, BranchSwitchSelectAndLoopWeightsAreTheEdgeCounts) {
  Profiled P(ProgramIR);
  for (int X : {5, 5, 5, -1})
    P.fn<int(int)>("diamond")(X);
  for (int X : {7, 7, 7, 0, 1, 1})
    P.fn<int(int)>("sw")(X);
  for (int X : {5, -1, -1})
    P.fn<int(int)>("sel")(X);
  P.fn<int(int)>("loop")(3);
  P.fn<int(int)>("loop")(1);

  LLVMContext Ctx;
  auto M = P.annotate(Ctx);
  ASSERT_EQ(P.LastError, "");
  EXPECT_EQ(weights(terminatorOf(*M, "diamond", "entry")),
            (std::vector<uint64_t>{3, 1}));
  // Switch weights: default first, then each case.
  EXPECT_EQ(weights(terminatorOf(*M, "sw", "entry")),
            (std::vector<uint64_t>{3, 1, 2}));
  EXPECT_EQ(weights(firstOf(*M, "sel", Instruction::Select)),
            (std::vector<uint64_t>{1, 2}));
  // loop(3) takes the back edge twice and exits once; loop(1) just exits.
  EXPECT_EQ(weights(terminatorOf(*M, "loop", "body")),
            (std::vector<uint64_t>{2, 2}));
}

TEST(Annotate, EntryCountsAndColdFunctions) {
  Profiled P(ProgramIR);
  for (int X : {5, 5, 5, -1})
    P.fn<int(int)>("diamond")(X);
  LLVMContext Ctx;
  auto M = P.annotate(Ctx);
  ASSERT_EQ(P.LastError, "");
  EXPECT_EQ(M->getFunction("diamond")->getEntryCount(),
            std::optional<uint64_t>(4));
  EXPECT_EQ(M->getFunction("never")->getEntryCount(),
            std::optional<uint64_t>(0));
  EXPECT_TRUE(M->getFunction("never")->hasFnAttribute(Attribute::Cold));
  EXPECT_NE(M->getProfileSummary(/*IsCS=*/false), nullptr);
}

TEST(Annotate, IndirectCallTargetsBecomeValueProfileMetadata) {
  Profiled P(ProgramIR);
  auto *Dispatch = P.fn<void(void (*)())>("dispatch");
  auto *A = P.fn<void()>("ta");
  auto *B = P.fn<void()>("tb");
  for (void (*T)() : {A, B, B, B})
    Dispatch(T);
  LLVMContext Ctx;
  auto M = P.annotate(Ctx);
  ASSERT_EQ(P.LastError, "");
  const Instruction &Call = firstOf(*M, "dispatch", Instruction::Call);
  uint64_t Total = 0;
  SmallVector<InstrProfValueData> VD = getValueProfDataFromInst(
      Call, IPVK_IndirectCallTarget, /*MaxNumValueData=*/4, Total);
  EXPECT_EQ(Total, 4u);
  ASSERT_EQ(VD.size(), 2u);
  // Sorted by count, hottest first, as LLVM writes VP metadata.
  EXPECT_EQ(VD[0].Value, IndexedInstrProf::ComputeHash("tb"));
  EXPECT_EQ(VD[0].Count, 3u);
  EXPECT_EQ(VD[1].Value, IndexedInstrProf::ComputeHash("ta"));
  EXPECT_EQ(VD[1].Count, 1u);
}

/// A random decision tree over the bits of x, emitted as one IR function.
/// Inner nodes are a br on one bit or a 4-way switch on two bits; leaves
/// return a constant, or select between two constants on one bit.
struct Tree {
  struct Node {
    enum Kind { Br, Switch, Ret, Select } K;
    unsigned Bit = 0;
    std::vector<int>
        Children; // Br: {true, false}; Switch: {default=3, 0, 1, 2}
    std::vector<uint64_t> EdgeCounts;
    uint64_t SelectTrue = 0, SelectFalse = 0;
  };
  std::vector<Node> Nodes;

  int build(std::mt19937 &Rng, unsigned Depth) {
    int Id = Nodes.size();
    Nodes.emplace_back();
    unsigned Pick = Depth == 0 ? 2 + Rng() % 2 : Rng() % 4;
    Nodes[Id].K = static_cast<Node::Kind>(Pick);
    Nodes[Id].Bit = Rng() % 5;
    if (Pick == Node::Br || Pick == Node::Switch) {
      unsigned N = Pick == Node::Br ? 2 : 4;
      Nodes[Id].EdgeCounts.assign(N, 0);
      for (unsigned I = 0; I < N; ++I) {
        int Child = build(Rng, Depth - 1);
        Nodes[Id].Children.push_back(Child);
      }
    }
    return Id;
  }

  /// Walks input X through the tree, counting edges; returns the result.
  int run(int X) {
    int Id = 0;
    for (;;) {
      Node &N = Nodes[Id];
      unsigned One = (X >> N.Bit) & 1, Two = (X >> N.Bit) & 3;
      switch (N.K) {
      case Node::Br:
        ++N.EdgeCounts[One ? 0 : 1];
        Id = N.Children[One ? 0 : 1];
        break;
      case Node::Switch: {
        unsigned Slot = Two == 3 ? 0 : Two + 1; // successor 0 is the default
        ++N.EdgeCounts[Slot];
        Id = N.Children[Slot];
        break;
      }
      case Node::Ret:
        return Id;
      case Node::Select:
        ++(One ? N.SelectTrue : N.SelectFalse);
        return One ? Id : -Id;
      }
    }
  }

  std::string ir(const std::string &Fn) const {
    std::string S = "define i32 @" + Fn + "(i32 %x) {\n";
    for (size_t Id = 0; Id < Nodes.size(); ++Id) {
      const Node &N = Nodes[Id];
      std::string I = std::to_string(Id), B = std::to_string(N.Bit);
      S += "n" + I + ":\n";
      S += "  %s" + I + " = lshr i32 %x, " + B + "\n";
      switch (N.K) {
      case Node::Br:
        S += "  %b" + I + " = trunc i32 %s" + I + " to i1\n";
        S += "  br i1 %b" + I + ", label %n" + std::to_string(N.Children[0]) +
             ", label %n" + std::to_string(N.Children[1]) + "\n";
        break;
      case Node::Switch:
        S += "  %v" + I + " = and i32 %s" + I + ", 3\n";
        S += "  switch i32 %v" + I + ", label %n" +
             std::to_string(N.Children[0]) + " [ i32 0, label %n" +
             std::to_string(N.Children[1]) + " i32 1, label %n" +
             std::to_string(N.Children[2]) + " i32 2, label %n" +
             std::to_string(N.Children[3]) + " ]\n";
        break;
      case Node::Ret:
        S += "  ret i32 " + I + "\n";
        break;
      case Node::Select:
        S += "  %b" + I + " = trunc i32 %s" + I + " to i1\n";
        S += "  %r" + I + " = select i1 %b" + I + ", i32 " + I + ", i32 -" + I +
             "\n";
        S += "  ret i32 %r" + I + "\n";
        break;
      }
    }
    return S + "}\n";
  }
};

TEST(Annotate, RandomDecisionTreesMatchTheOracle) {
  std::mt19937 Rng(20261010);
  std::vector<Tree> Trees(50);
  std::string IR = "source_filename = \"rand.c\"\n";
  for (size_t T = 0; T < Trees.size(); ++T) {
    Trees[T].build(Rng, 4);
    IR += Trees[T].ir("t" + std::to_string(T));
  }
  Profiled P(IR);
  for (size_t T = 0; T < Trees.size(); ++T) {
    auto *Fn = P.fn<int(int)>("t" + std::to_string(T));
    for (int X = 0; X < 64; ++X) {
      if (Rng() % 3 == 0)
        continue; // uneven input sets give asymmetric counts
      int Expected = Trees[T].run(X);
      ASSERT_EQ(Fn(X), Expected) << "tree " << T << " input " << X;
    }
  }
  LLVMContext Ctx;
  auto M = P.annotate(Ctx);
  ASSERT_EQ(P.LastError, "");
  for (size_t T = 0; T < Trees.size(); ++T) {
    std::string Fn = "t" + std::to_string(T);
    for (size_t Id = 0; Id < Trees[T].Nodes.size(); ++Id) {
      const Tree::Node &N = Trees[T].Nodes[Id];
      std::string Block = "n" + std::to_string(Id);
      if (N.K == Tree::Node::Br || N.K == Tree::Node::Switch) {
        uint64_t Sum = 0;
        for (uint64_t C : N.EdgeCounts)
          Sum += C;
        // LLVM attaches no weights to a terminator that never ran.
        std::vector<uint64_t> Expected =
            Sum ? N.EdgeCounts : std::vector<uint64_t>{};
        EXPECT_EQ(weights(terminatorOf(*M, Fn, Block)), Expected)
            << Fn << " " << Block;
      } else if (N.K == Tree::Node::Select) {
        std::vector<uint64_t> Expected;
        if (N.SelectTrue + N.SelectFalse)
          Expected = {N.SelectTrue, N.SelectFalse};
        const Instruction *Sel = nullptr;
        for (BasicBlock &BB : *M->getFunction(Fn))
          if (BB.getName() == Block)
            for (Instruction &I : BB)
              if (isa<SelectInst>(I))
                Sel = &I;
        ASSERT_NE(Sel, nullptr);
        EXPECT_EQ(weights(*Sel), Expected) << Fn << " " << Block;
      }
    }
  }
}

TEST(Annotate, AHashMismatchIsAnError) {
  Profiled P(ProgramIR);
  P.fn<int(int)>("diamond")(1);
  std::vector<OrcPgoFunctionRecord> Records(P.Table.begin(), P.Table.end());
  for (OrcPgoFunctionRecord &R : Records)
    if (StringRef(R.Name) == "diamond")
      R.CfgHash ^= 1;
  LLVMContext Ctx;
  P.annotate(Ctx, {}, Records);
  EXPECT_NE(P.LastError.find("hash mismatch"), std::string::npos)
      << P.LastError;
}

TEST(Annotate, TheFirstMismatchIsTheOneReported) {
  Profiled P(ProgramIR);
  P.fn<int(int)>("diamond")(1);
  P.fn<int(int)>("sw")(1);
  std::vector<OrcPgoFunctionRecord> Records(P.Table.begin(), P.Table.end());
  for (OrcPgoFunctionRecord &R : Records)
    R.CfgHash ^= 1;
  LLVMContext Ctx;
  P.annotate(Ctx, {}, Records);
  EXPECT_NE(P.LastError.find("(hash mismatch) diamond "), std::string::npos)
      << P.LastError;
  EXPECT_EQ(P.LastError.find(" sw "), std::string::npos) << P.LastError;
}

TEST(Annotate, ACounterCountMismatchIsAnError) {
  Profiled P(ProgramIR);
  P.fn<int(int)>("diamond")(1);
  std::vector<OrcPgoFunctionRecord> Records(P.Table.begin(), P.Table.end());
  for (OrcPgoFunctionRecord &R : Records)
    if (StringRef(R.Name) == "diamond")
      R.NumCounters -= 1;
  LLVMContext Ctx;
  P.annotate(Ctx, {}, Records);
  EXPECT_NE(P.LastError.find("Inconsistent number of counts"),
            std::string::npos)
      << P.LastError;
}

TEST(Annotate, MustBeAnnotatedFunctionsNeedAProfile) {
  Profiled P(ProgramIR);
  std::vector<OrcPgoFunctionRecord> WithoutSw;
  for (const OrcPgoFunctionRecord &R : P.Table)
    if (StringRef(R.Name) != "sw")
      WithoutSw.push_back(R);
  LLVMContext C1, C2, C3;
  P.annotate(C1, {"diamond"}, WithoutSw);
  EXPECT_EQ(P.LastError, "");
  P.annotate(C2, {"sw"}, WithoutSw);
  EXPECT_EQ(P.LastError, "no profile applied to sw");
  P.annotate(C3, {"missing"}, WithoutSw);
  EXPECT_EQ(P.LastError, "no profile applied to missing");
}

TEST(Annotate, AnInvalidProfileIsAnError) {
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, ProgramIR);
  auto Garbage = MemoryBuffer::getMemBuffer("not a profile", "garbage", false);
  EXPECT_NE(test::errorText(annotateWithProfile(*M, *Garbage)), "");
}

TEST(Annotate, OptionSetupFailuresAreAnError) {
  test::RequiredPositionalOption Positional;
  LLVMContext Ctx;
  auto M = test::parseHostIR(Ctx, ProgramIR);
  auto Empty = MemoryBuffer::getMemBuffer("", "empty", false);
  EXPECT_EQ(test::errorText(annotateWithProfile(*M, *Empty)),
            Positional.ConfigureError);
}

TEST(Annotate, RestoresTheContextsDiagnosticHandler) {
  Profiled P(ProgramIR);
  LLVMContext Ctx;
  bool Called = false;
  Ctx.setDiagnosticHandlerCallBack(
      [](const DiagnosticInfo *, void *Flag) {
        *static_cast<bool *>(Flag) = true;
      },
      &Called);
  P.annotate(Ctx);
  Ctx.diagnose(DiagnosticInfoPGOProfile("x", "after annotate", DS_Warning));
  EXPECT_TRUE(Called);
}

constexpr const char *ExtraIR = R"(
source_filename = "x.c"
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
; while loop whose header is the exiting block, with an if in the body
define i32 @wh(i32 %n) {
entry:
  br label %hdr
hdr:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  %c = icmp slt i32 %i, %n
  br i1 %c, label %body, label %exit
body:
  %odd = and i32 %i, 1
  %o = icmp ne i32 %odd, 0
  br i1 %o, label %latch, label %even
even:
  br label %latch
latch:
  %i1 = add i32 %i, 1
  br label %hdr
exit:
  ret i32 %i
}
; nested loops with an early exit from the inner loop
define i32 @nest(i32 %n) {
entry:
  br label %oh
oh:
  %i = phi i32 [ 0, %entry ], [ %i1, %ol ]
  %c = icmp slt i32 %i, %n
  br i1 %c, label %ih0, label %exit
ih0:
  br label %ih
ih:
  %j = phi i32 [ 0, %ih0 ], [ %j1, %ib ]
  %cj = icmp slt i32 %j, %i
  br i1 %cj, label %ib, label %ol
ib:
  %j1 = add i32 %j, 1
  %stop = icmp eq i32 %j1, 3
  br i1 %stop, label %exit, label %ih
ol:
  %i1 = add i32 %i, 1
  br label %oh
exit:
  ret i32 %i
}
define internal i32 @loc(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %a, label %b
a:
  ret i32 1
b:
  ret i32 2
}
define i32 @useloc(i32 %x) {
entry:
  %r = call i32 @loc(i32 %x)
  ret i32 %r
}
define linkonce_odr i32 @lo(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %a, label %b
a:
  ret i32 1
b:
  ret i32 2
}
define weak i32 @wk(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %a, label %b
a:
  ret i32 1
b:
  ret i32 2
}
define void @cp(ptr %d, ptr %s, i64 %n) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 %n, i1 false)
  ret void
}
)";

TEST(Annotate, RotatedAndNestedLoopsMatchHandCounts) {
  Profiled P(ExtraIR);
  for (int N : {5, 0, 3})
    P.fn<int(int)>("wh")(N);
  EXPECT_EQ(P.fn<int(int)>("nest")(4), 3);
  EXPECT_EQ(P.fn<int(int)>("nest")(1), 1);
  LLVMContext Ctx;
  auto M = P.annotate(Ctx);
  ASSERT_EQ(P.LastError, "");
  EXPECT_EQ(weights(terminatorOf(*M, "wh", "hdr")),
            (std::vector<uint64_t>{8, 3}));
  EXPECT_EQ(weights(terminatorOf(*M, "wh", "body")),
            (std::vector<uint64_t>{3, 5}));
  EXPECT_EQ(M->getFunction("wh")->getEntryCount(), std::optional<uint64_t>(3));
  EXPECT_EQ(weights(terminatorOf(*M, "nest", "oh")),
            (std::vector<uint64_t>{5, 1}));
  EXPECT_EQ(weights(terminatorOf(*M, "nest", "ih")),
            (std::vector<uint64_t>{6, 4}));
  EXPECT_EQ(weights(terminatorOf(*M, "nest", "ib")),
            (std::vector<uint64_t>{1, 5}));
}

TEST(Annotate, InternalFunctionsAreFoundByTheirFilePrefixedName) {
  Profiled P(ExtraIR);
  for (int X : {1, 1, -1})
    P.fn<int(int)>("useloc")(X);
  LLVMContext Ctx;
  auto M = P.annotate(Ctx, {"loc"});
  ASSERT_EQ(P.LastError, "");
  EXPECT_EQ(weights(terminatorOf(*M, "loc", "entry")),
            (std::vector<uint64_t>{2, 1}));
}

TEST(Annotate, AWeakFunctionsHashMismatchIsOnlyCaughtByMustBeAnnotated) {
  Profiled P(ExtraIR);
  std::vector<OrcPgoFunctionRecord> Records(P.Table.begin(), P.Table.end());
  for (OrcPgoFunctionRecord &R : Records)
    if (StringRef(R.Name) == "wk")
      R.CfgHash ^= 1;
  LLVMContext C1, C2;
  auto M = P.annotate(C1, {}, Records);
  EXPECT_EQ(P.LastError, "");
  EXPECT_FALSE(M->getFunction("wk")->getEntryCount());
  P.annotate(C2, {"wk"}, Records);
  EXPECT_EQ(P.LastError, "no profile applied to wk");
}

// linkonce_odr without a comdat is not exempt from the mismatch warning.
TEST(Annotate, ALinkonceFunctionWithoutComdatHashMismatchIsDiagnosed) {
  Profiled P(ExtraIR);
  std::vector<OrcPgoFunctionRecord> Records(P.Table.begin(), P.Table.end());
  for (OrcPgoFunctionRecord &R : Records)
    if (StringRef(R.Name) == "lo")
      R.CfgHash ^= 1;
  LLVMContext Ctx;
  P.annotate(Ctx, {}, Records);
  EXPECT_NE(P.LastError.find("hash mismatch"), std::string::npos)
      << P.LastError;
}

// Warnings that only cost a function some of its profile do not fail the
// annotation. Neither branch of this function ever returns.
constexpr const char *NoReturnIR = R"(
source_filename = "noreturn.c"
define i32 @diverge(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %a, label %b
a:
  br label %a
b:
  br label %b
}
)";

TEST(Annotate, AProfilePartiallyIgnoredForLackOfAReturnPathIsNotAFailure) {
  Profiled P(NoReturnIR);
  // The function cannot return, so counts for it are made up: an entry that
  // the edge counters do not account for.
  for (const OrcPgoFunctionRecord &R : P.Table) {
    ASSERT_EQ(R.NumCounters, 3u);
    R.Counters[0] = 5;
    R.Counters[1] = 0;
    R.Counters[2] = 0;
  }
  LLVMContext Ctx;
  testing::internal::CaptureStderr();
  auto M = P.annotate(Ctx, {"diverge"});
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
  EXPECT_EQ(P.LastError, "");
  EXPECT_TRUE(M->getFunction("diverge")->getEntryCount().has_value());
}

TEST(Annotate, MemopSizesBecomeValueProfileMetadata) {
  Profiled P(ExtraIR);
  char A[200], B[200];
  for (uint64_t N : {8, 8, 100})
    P.fn<void(char *, char *, uint64_t)>("cp")(A, B, N);
  LLVMContext Ctx;
  auto M = P.annotate(Ctx);
  ASSERT_EQ(P.LastError, "");
  const Instruction &Call = firstOf(*M, "cp", Instruction::Call);
  uint64_t Total = 0;
  SmallVector<InstrProfValueData> VD =
      getValueProfDataFromInst(Call, IPVK_MemOPSize, 4, Total);
  EXPECT_EQ(Total, 3u);
  ASSERT_EQ(VD.size(), 2u);
  EXPECT_EQ(VD[0].Value, 8u);
  EXPECT_EQ(VD[0].Count, 2u);
  EXPECT_EQ(VD[1].Value, 65u);
  EXPECT_EQ(VD[1].Count, 1u);
}

// Counters read while calls are in flight need not satisfy flow conservation;
// LLVM annotates them anyway, and the annotation does not fail.
TEST(Annotate, TornCountersAreAnnotatedWithoutFailing) {
  Profiled P(ExtraIR);
  for (int N : {5, 0, 3})
    P.fn<int(int)>("wh")(N);
  for (const OrcPgoFunctionRecord &R : P.Table)
    if (StringRef(R.Name) == "wh")
      R.Counters[2] += 100;
  LLVMContext Ctx;
  auto M = P.annotate(Ctx, {"wh"});
  EXPECT_EQ(P.LastError, "");
}

} // namespace
