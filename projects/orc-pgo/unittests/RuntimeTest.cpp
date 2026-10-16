// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Runtime.h"
#include "SimulatedAot.h"
#include "orc-pgo/PgoOptions.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

extern "C" void orc_pgo_test_throw(int X) {
  if (X >= 0)
    throw X;
}

extern "C" int orc_pgo_test_sentinel(int) { return 777; }

extern "C" void orc_pgo_test_nop() {}

namespace {

constexpr const char *RuntimeIR = R"(
source_filename = "runtime.c"

@orc_pgo_runtime_test_global = global i32 5

define i32 @hot(i32 %x) {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %pos, label %neg
pos:
  %r = add i32 %x, 1
  ret i32 %r
neg:
  ret i32 0
}
define internal i32 @helper(i32 %x) noinline {
  %r = mul i32 %x, 3
  ret i32 %r
}
define i32 @useshelper(i32 %x) {
  %r = call i32 @helper(i32 %x)
  ret i32 %r
}
define i32 @cold(i32 %x) {
  %r = sub i32 %x, 1
  ret i32 %r
}
define i32 @readsg() {
  %v = load i32, ptr @orc_pgo_runtime_test_global
  ret i32 %v
}
define i32 @callit(ptr %f, i32 %x) {
  %r = call i32 %f(i32 %x)
  ret i32 %r
}
)";

constexpr const char *EhIR = R"(
source_filename = "eh.c"

@_ZTIi = external constant ptr

declare void @orc_pgo_test_throw(i32)
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

define i32 @passes(i32 %x) uwtable {
  call void @orc_pgo_test_throw(i32 %x)
  ret i32 1
}
define i32 @catches(i32 %x) uwtable personality ptr @__gxx_personality_v0 {
entry:
  invoke void @orc_pgo_test_throw(i32 %x)
          to label %ok unwind label %lpad
ok:
  ret i32 -1
lpad:
  %lp = landingpad { ptr, i32 } catch ptr @_ZTIi
  %exn = extractvalue { ptr, i32 } %lp, 0
  %p = call ptr @__cxa_begin_catch(ptr %exn)
  %v = load i32, ptr %p
  call void @__cxa_end_catch()
  ret i32 %v
}
)";

constexpr const char *DivIR = R"(
source_filename = "div.c"
define i128 @div(i128 %a, i128 %b) {
  %q = sdiv i128 %a, %b
  ret i128 %q
}
)";

RuntimeOptions options(uint64_t Threshold, unsigned InlineDepth = 2) {
  RuntimeOptions O;
  O.Threshold = Threshold;
  O.PollMs = 0;
  O.InlineDepth = InlineDepth;
  return O;
}

std::unique_ptr<Runtime> runtimeFor(const OrcPgoImage &Image,
                                    RuntimeOptions O) {
  Expected<std::unique_ptr<Runtime>> R = Runtime::create(Image, O);
  if (!R)
    report_fatal_error(R.takeError());
  return std::move(*R);
}

std::vector<OrcPgoFunctionRecord> recordsOf(const OrcPgoImage &Image) {
  return {Image.Functions, Image.Functions + Image.NumFunctions};
}

const char *const EnvVars[] = {"ORC_PGO_THRESHOLD", "ORC_PGO_POLL_MS",
                               "ORC_PGO_INLINE_DEPTH", "ORC_PGO_VERBOSE",
                               "ORC_PGO_DISABLE"};

extern "C" int orc_pgo_runtime_test_sink(const char *S) {
  return static_cast<int>(std::strlen(S));
}

// Everything a tier-up body can mention in the program:
// an alias, an internal function (inlined and not), an internal global, a
// private string, a constant table, a function with an address-taken block, a
// global defined elsewhere in the test binary.
constexpr const char *LinkIR = R"(
source_filename = "link.c"
@.str = private unnamed_addr constant [6 x i8] c"hello\00"
@table = constant [2 x i32] [i32 3, i32 5]
@cnt = internal global i32 0
@valias = alias i32, ptr @cnt
@falias = alias i32 (i32), ptr @far
declare i32 @orc_pgo_runtime_test_sink(ptr)
define internal i32 @far(i32 %x) noinline {
  %r = add i32 %x, 1
  ret i32 %r
}
define internal i32 @small(i32 %x) {
  %r = mul i32 %x, 2
  ret i32 %r
}
define i32 @jumpy(i32 %x) {
entry:
  indirectbr ptr blockaddress(@jumpy, %target), [label %target]
target:
  ret i32 %x
}
define i32 @top(i32 %x) {
entry:
  %n = call i32 @orc_pgo_runtime_test_sink(ptr @.str)
  %p = getelementptr [2 x i32], ptr @table, i64 0, i64 1
  %t = load i32, ptr %p
  %c = load i32, ptr @valias
  %c2 = add i32 %c, 1
  store i32 %c2, ptr @cnt
  %a = call i32 @falias(i32 %x)
  %b = call i32 @small(i32 %a)
  %j = call i32 @jumpy(i32 %b)
  %s1 = add i32 %j, %n
  %s2 = add i32 %s1, %t
  %s3 = add i32 %s2, %c2
  ret i32 %s3
}
)";

TEST(Runtime, TierUpBodiesLinkAgainstTheProgram) {
  auto S = test::simulateAot(
      LinkIR, {},
      {{"orc_pgo_runtime_test_sink",
        reinterpret_cast<const void *>(&orc_pgo_runtime_test_sink)}});
  auto *Top = S.sym<int(int)>("top");
  // top(x) = ((x + 1) * 2) + 5 (strlen) + 5 (table[1]) + (call count so far)
  for (int I = 1; I <= 10; ++I)
    ASSERT_EQ(Top(1), 4 + 5 + 5 + I);
  auto R = runtimeFor(*S.Image, options(10));
  EXPECT_GE(R->pollOnce(), 1u);
  EXPECT_EQ(R->state(*R->slotOf("top")), SlotState::TieredUp);
  // The tier-up body shares the counter global with the AOT body.
  EXPECT_EQ(Top(1), 4 + 5 + 5 + 11);
  EXPECT_EQ(Top(2), 6 + 5 + 5 + 12);
}

TEST(RuntimeOptions, DefaultsAndTheEnvironment) {
  for (const char *V : EnvVars)
    unsetenv(V);
  RuntimeOptions D = RuntimeOptions::fromEnv();
  EXPECT_EQ(D.Threshold, 10000u);
  EXPECT_EQ(D.PollMs, 10u);
  EXPECT_EQ(D.InlineDepth, 2u);
  EXPECT_FALSE(D.Verbose);
  EXPECT_FALSE(D.Disable);

  setenv("ORC_PGO_THRESHOLD", "7", 1);
  setenv("ORC_PGO_POLL_MS", "3", 1);
  setenv("ORC_PGO_INLINE_DEPTH", "deep", 1);
  setenv("ORC_PGO_VERBOSE", "1", 1);
  setenv("ORC_PGO_DISABLE", "1", 1);
  RuntimeOptions O = RuntimeOptions::fromEnv();
  EXPECT_EQ(O.Threshold, 7u);
  EXPECT_EQ(O.PollMs, 3u);
  EXPECT_EQ(O.InlineDepth, 2u); // malformed values are ignored
  EXPECT_TRUE(O.Verbose);
  EXPECT_TRUE(O.Disable);
  for (const char *V : EnvVars)
    unsetenv(V);
}

TEST(ValidateImage, AcceptsWhatThePipelineBuilt) {
  auto S = test::simulateAot(RuntimeIR);
  EXPECT_EQ(test::errorText(validateImage(*S.Image)), "");
}

TEST(ValidateImage, RejectsEachFingerprintMismatch) {
  auto S = test::simulateAot(RuntimeIR);
  OrcPgoImage Bad = *S.Image;
  Bad.Version = 2;
  EXPECT_EQ(test::errorText(validateImage(Bad)),
            "image version 2, runtime expects 1");
  Expected<std::unique_ptr<Runtime>> R = Runtime::create(Bad, options(1));
  EXPECT_EQ(test::errorText(R), "image version 2, runtime expects 1");

  Bad = *S.Image;
  Bad.Fingerprint.LLVMVersion = "0.0.0";
  EXPECT_EQ(
      test::errorText(validateImage(Bad)),
      "image built with LLVM 0.0.0, runtime is LLVM " LLVM_VERSION_STRING);

  Bad = *S.Image;
  Bad.Fingerprint.RawProfileVersion = 11;
  EXPECT_EQ(test::errorText(validateImage(Bad)),
            "image raw profile version 0xb, runtime expects 0x50000000000000b");

  Bad = *S.Image;
  Bad.Fingerprint.OptionsHash ^= 1;
  EXPECT_EQ(test::errorText(validateImage(Bad)),
            "image PGO options differ from the runtime's");
}

TEST(Runtime, TiersUpAHotFunction) {
  auto S = test::simulateAot(RuntimeIR);
  auto R = runtimeFor(*S.Image, options(100));
  auto *Hot = S.sym<int(int)>("hot");
  unsigned Slot = *R->slotOf("hot");
  EXPECT_EQ(Slot, S.slot("hot"));
  EXPECT_FALSE(R->slotOf("nope").has_value());

  for (int I = 0; I < 99; ++I)
    Hot(1);
  EXPECT_EQ(R->pollOnce(), 0u);
  EXPECT_EQ(R->stats().Batches, 0u);
  Hot(1);
  void *Aot = S.Image->Dispatch[Slot];
  EXPECT_EQ(R->pollOnce(), 1u);
  EXPECT_EQ(R->state(Slot), SlotState::TieredUp);
  EXPECT_NE(S.Image->Dispatch[Slot], Aot);
  EXPECT_EQ(Hot(41), 42);
  EXPECT_EQ(Hot(-1), 0);
  // The tier-up body has no counters, so the entry count stops at 100.
  EXPECT_EQ(S.record("hot").Counters[0], 100u);
  EXPECT_EQ(R->state(*R->slotOf("cold")), SlotState::Aot);

  EXPECT_EQ(R->pollOnce(), 0u);
  RuntimeStats St = R->stats();
  EXPECT_EQ(St.Batches, 1u);
  EXPECT_EQ(St.TieredUp, 1u);
  EXPECT_EQ(St.Failed, 0u);
  std::string Out;
  raw_string_ostream OS(Out);
  R->printStats(OS);
  EXPECT_EQ(
      Out,
      "orc-pgo: 1 batches, 1 tiered up, 0 failed, 0 indirect calls promoted\n");
}

TEST(Runtime, OneBatchTiersUpEveryHotFunction) {
  auto S = test::simulateAot(RuntimeIR);
  auto R = runtimeFor(*S.Image, options(10));
  for (int I = 0; I < 10; ++I) {
    S.sym<int(int)>("hot")(1);
    S.sym<int(int)>("useshelper")(2);
  }
  EXPECT_EQ(R->pollOnce(), 3u); // hot, helper, useshelper
  EXPECT_EQ(R->stats().Batches, 1u);
  for (StringRef Name : {"hot", "runtime.c;helper", "useshelper"})
    EXPECT_EQ(R->state(*R->slotOf(Name)), SlotState::TieredUp) << Name.str();
  EXPECT_EQ(S.sym<int(int)>("useshelper")(2), 6);
  EXPECT_EQ(S.sym<int(int)>("hot")(1), 2);
}

TEST(Runtime, CountsPromotedIndirectCalls) {
  auto S = test::simulateAot(RuntimeIR);
  auto R = runtimeFor(*S.Image, options(1000, /*InlineDepth=*/0));
  auto *CallIt = S.sym<int(int (*)(int), int)>("callit");
  auto *Cold = S.sym<int(int)>("cold");
  for (int I = 0; I < 2000; ++I)
    ASSERT_EQ(CallIt(Cold, I), I - 1);
  EXPECT_EQ(R->pollOnce(), 2u); // callit and cold
  EXPECT_EQ(R->stats().PromotedIndirectCalls, 1u);
  EXPECT_EQ(CallIt(Cold, 5), 4);
}

TEST(Runtime, APromotedCallBypassesTheThunkOfATargetTieredUpInTheSameBatch) {
  // callit and cold tier up together. The program passes cold's thunk
  // around, so the promoted compare has to be against that address; when it
  // is, the guarded path calls cold's tier-up body directly. Sending the
  // thunk's slot elsewhere tells the two paths apart.
  auto S = test::simulateAot(RuntimeIR);
  auto R = runtimeFor(*S.Image, options(1000, /*InlineDepth=*/0));
  auto *CallIt = S.sym<int(int (*)(int), int)>("callit");
  auto *Cold = S.sym<int(int)>("cold");
  for (int I = 0; I < 2000; ++I)
    ASSERT_EQ(CallIt(Cold, I), I - 1);
  ASSERT_EQ(R->pollOnce(), 2u);
  ASSERT_EQ(R->stats().PromotedIndirectCalls, 1u);

  S.Image->Dispatch[*R->slotOf("cold")] =
      reinterpret_cast<void *>(&orc_pgo_test_sentinel);
  EXPECT_EQ(Cold(5), 777);
  EXPECT_EQ(CallIt(Cold, 5), 4);
}

TEST(Runtime, ABadRecordFailsOnlyItsOwnRootWhenTheBatchFails) {
  // hot and cold are one batch; the other's record must not fail the retry of
  // either, whichever comes first.
  for (StringRef Bad : {"hot", "cold"}) {
    SCOPED_TRACE(Bad.str());
    StringRef Good = Bad == "hot" ? "cold" : "hot";
    auto S = test::simulateAot(RuntimeIR);
    std::vector<OrcPgoFunctionRecord> Records = recordsOf(*S.Image);
    for (OrcPgoFunctionRecord &Rec : Records)
      if (StringRef(Rec.Name) == Bad)
        Rec.CfgHash ^= 1;
    OrcPgoImage Tampered = *S.Image;
    Tampered.Functions = Records.data();
    auto R = runtimeFor(Tampered, options(10));
    for (int I = 0; I < 10; ++I) {
      S.sym<int(int)>("hot")(1);
      S.sym<int(int)>("cold")(5);
    }
    void *BadAot = S.Image->Dispatch[*R->slotOf(Bad)];
    void *GoodAot = S.Image->Dispatch[*R->slotOf(Good)];
    EXPECT_EQ(R->pollOnce(), 1u);
    EXPECT_EQ(R->state(*R->slotOf(Bad)), SlotState::Failed);
    EXPECT_EQ(R->state(*R->slotOf(Good)), SlotState::TieredUp);
    EXPECT_EQ(S.Image->Dispatch[*R->slotOf(Bad)], BadAot);
    EXPECT_NE(S.Image->Dispatch[*R->slotOf(Good)], GoodAot);
    RuntimeStats St = R->stats();
    EXPECT_EQ(St.Batches, 1u);
    EXPECT_EQ(St.TieredUp, 1u);
    EXPECT_EQ(St.Failed, 1u);
    EXPECT_EQ(S.sym<int(int)>("hot")(41), 42);
    EXPECT_EQ(S.sym<int(int)>("cold")(5), 4);
    EXPECT_EQ(R->pollOnce(), 0u);
  }
}

TEST(Runtime, ARetryReportsTheBatchAndEachFailure) {
  auto S = test::simulateAot(RuntimeIR);
  std::vector<OrcPgoFunctionRecord> Records = recordsOf(*S.Image);
  for (OrcPgoFunctionRecord &Rec : Records)
    if (StringRef(Rec.Name) == "cold")
      Rec.CfgHash ^= 1;
  OrcPgoImage Tampered = *S.Image;
  Tampered.Functions = Records.data();
  RuntimeOptions O = options(10);
  O.Verbose = true;
  auto R = runtimeFor(Tampered, O);
  for (int I = 0; I < 10; ++I) {
    S.sym<int(int)>("hot")(1);
    S.sym<int(int)>("cold")(5);
  }
  testing::internal::CaptureStderr();
  EXPECT_EQ(R->pollOnce(), 1u);
  std::string Err = testing::internal::GetCapturedStderr();
  EXPECT_NE(Err.find("orc-pgo: tier-up of hot, cold failed: "),
            std::string::npos)
      << Err;
  EXPECT_NE(Err.find("orc-pgo: tier-up of cold failed: "), std::string::npos)
      << Err;
  EXPECT_EQ(Err.find("orc-pgo: tier-up of hot failed: "), std::string::npos)
      << Err;
}

TEST(Runtime, ACorruptedHashLeavesTheFunctionOnAotForGood) {
  auto S = test::simulateAot(RuntimeIR);
  std::vector<OrcPgoFunctionRecord> Records = recordsOf(*S.Image);
  for (OrcPgoFunctionRecord &Rec : Records)
    if (StringRef(Rec.Name) == "hot")
      Rec.CfgHash ^= 1;
  OrcPgoImage Tampered = *S.Image;
  Tampered.Functions = Records.data();
  auto R = runtimeFor(Tampered, options(10));
  auto *Hot = S.sym<int(int)>("hot");
  unsigned Slot = *R->slotOf("hot");
  void *Aot = S.Image->Dispatch[Slot];
  for (int I = 0; I < 10; ++I)
    Hot(1);
  EXPECT_EQ(R->pollOnce(), 0u);
  EXPECT_EQ(R->state(Slot), SlotState::Failed);
  EXPECT_EQ(S.Image->Dispatch[Slot], Aot);
  EXPECT_EQ(Hot(41), 42);
  EXPECT_EQ(R->stats().Failed, 1u);

  // hot is not retried, and its record is left out of later profiles, so the
  // next batch is not poisoned by it.
  for (int I = 0; I < 10; ++I)
    S.sym<int(int)>("cold")(5);
  EXPECT_EQ(R->pollOnce(), 1u);
  EXPECT_EQ(R->state(*R->slotOf("cold")), SlotState::TieredUp);
  EXPECT_EQ(R->state(Slot), SlotState::Failed);
  EXPECT_EQ(R->stats().Batches, 2u);
  EXPECT_EQ(S.sym<int(int)>("cold")(5), 4);
}

TEST(Runtime, AProfileWriterErrorFailsTheBatch) {
  auto S = test::simulateAot(RuntimeIR);
  std::vector<OrcPgoFunctionRecord> Records = recordsOf(*S.Image);
  // Same name and hash, different counter count: InstrProfWriter refuses to
  // merge them (count_mismatch).
  OrcPgoFunctionRecord Dup = S.record("hot");
  Dup.NumCounters -= 1;
  Records.push_back(Dup);
  OrcPgoImage Tampered = *S.Image;
  Tampered.Functions = Records.data();
  Tampered.NumFunctions = Records.size();
  auto R = runtimeFor(Tampered, options(10));
  for (int I = 0; I < 10; ++I)
    S.sym<int(int)>("hot")(1);
  EXPECT_EQ(R->pollOnce(), 0u);
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::Failed);
  EXPECT_FALSE(R->imageDisabled());
}

TEST(Runtime, AnUnparsableSnapshotDisablesTheImage) {
  auto S = test::simulateAot(RuntimeIR);
  static const char Garbage[] = "not bitcode";
  OrcPgoImage Tampered = *S.Image;
  Tampered.Snapshot = Garbage;
  Tampered.SnapshotSize = sizeof(Garbage) - 1;
  auto R = runtimeFor(Tampered, options(10));
  for (int I = 0; I < 10; ++I)
    S.sym<int(int)>("hot")(1);
  EXPECT_EQ(R->pollOnce(), 0u);
  EXPECT_TRUE(R->imageDisabled());
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::Failed);
  for (int I = 0; I < 10; ++I)
    S.sym<int(int)>("cold")(1);
  EXPECT_EQ(R->pollOnce(), 0u);
  EXPECT_EQ(R->state(*R->slotOf("cold")), SlotState::Aot);
  EXPECT_EQ(R->stats().Batches, 1u);
}

/// Tier-up of readsg with an empty symbol table: the JIT can't resolve the
/// global it reads.
std::string tierUpWithoutSymbols(bool Verbose, test::SimulatedAot &S,
                                 SlotState &State) {
  OrcPgoImage Tampered = *S.Image;
  Tampered.NumSymbols = 0;
  RuntimeOptions O = options(10);
  O.Verbose = Verbose;
  auto R = runtimeFor(Tampered, O);
  for (int I = 0; I < 10; ++I)
    EXPECT_EQ(S.sym<int()>("readsg")(), 5);
  testing::internal::CaptureStderr();
  EXPECT_EQ(R->pollOnce(), 0u);
  std::string Err = testing::internal::GetCapturedStderr();
  State = R->state(*R->slotOf("readsg"));
  EXPECT_EQ(S.sym<int()>("readsg")(), 5);
  return Err;
}

TEST(Runtime, AnUnresolvedSymbolFailsAtLinkTime) {
  auto S = test::simulateAot(RuntimeIR);
  SlotState State;
  std::string Err = tierUpWithoutSymbols(/*Verbose=*/true, S, State);
  EXPECT_EQ(State, SlotState::Failed);
  EXPECT_NE(Err.find("orc-pgo: JIT error: "), std::string::npos) << Err;
  EXPECT_NE(Err.find("orc-pgo: tier-up of readsg failed: "), std::string::npos)
      << Err;
}

TEST(Runtime, FailuresAreSilentUnlessVerbose) {
  auto S = test::simulateAot(RuntimeIR);
  SlotState State;
  EXPECT_EQ(tierUpWithoutSymbols(/*Verbose=*/false, S, State), "");
  EXPECT_EQ(State, SlotState::Failed);
}

int Clash;

TEST(Runtime, ADuplicateDefinitionFailsTheBatch) {
  // A program symbol that happens to be called hot.orc_pgo.tier.
  auto S = test::simulateAot(RuntimeIR, {"hot.orc_pgo.tier"},
                             {{"hot.orc_pgo.tier", &Clash}});
  RuntimeOptions O = options(10);
  O.Verbose = true;
  auto R = runtimeFor(*S.Image, O);
  for (int I = 0; I < 10; ++I)
    S.sym<int(int)>("hot")(1);
  testing::internal::CaptureStderr();
  EXPECT_EQ(R->pollOnce(), 0u);
  std::string Err = testing::internal::GetCapturedStderr();
  EXPECT_NE(Err.find("orc-pgo: tier-up of hot failed: "), std::string::npos)
      << Err;
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::Failed);
}

TEST(Runtime, PollsInTheBackground) {
  auto S = test::simulateAot(RuntimeIR);
  RuntimeOptions O = options(10);
  O.PollMs = 1;
  auto R = runtimeFor(*S.Image, O);
  auto *Hot = S.sym<int(int)>("hot");
  for (int I = 0; I < 10; ++I)
    Hot(1);
  R->startPolling();
  R->startPolling(); // already polling
  unsigned Slot = *R->slotOf("hot");
  for (int I = 0; I < 10000 && R->state(Slot) != SlotState::TieredUp; ++I)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(R->state(Slot), SlotState::TieredUp);
  R->stopPolling();
  R->stopPolling();
  EXPECT_EQ(Hot(41), 42);
}

TEST(Runtime, NoPollThreadWithoutAPeriod) {
  auto S = test::simulateAot(RuntimeIR);
  auto R = runtimeFor(*S.Image, options(10));
  for (int I = 0; I < 10; ++I)
    S.sym<int(int)>("hot")(1);
  R->startPolling();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::Aot);
  EXPECT_EQ(R->pollOnce(), 1u);
}

TEST(Runtime, CallsDuringTheSwapStayCorrect) {
  auto S = test::simulateAot(RuntimeIR);
  auto R = runtimeFor(*S.Image, options(1000));
  auto *Hot = S.sym<int(int)>("hot");
  std::atomic<bool> Stop{false};
  std::atomic<unsigned> Wrong{0};
  std::vector<std::thread> Callers;
  for (int T = 0; T < 4; ++T) {
    Callers.emplace_back([&] {
      for (int I = 0; !Stop; ++I) {
        int X = I % 1000 + 1;
        if (Hot(X) != X + 1)
          ++Wrong;
      }
    });
  }
  // Non-atomic increments may lose updates, so wait for the count, not the
  // number of calls.
  while (__atomic_load_n(&S.record("hot").Counters[0], __ATOMIC_RELAXED) < 1000)
    std::this_thread::yield();
  EXPECT_EQ(R->pollOnce(), 1u);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  Stop = true;
  for (std::thread &T : Callers)
    T.join();
  EXPECT_EQ(Wrong, 0u);
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::TieredUp);
}

TEST(Runtime, ExceptionsUnwindThroughTieredUpCode) {
  auto S = test::simulateAot(
      EhIR, {},
      {{"orc_pgo_test_throw",
        reinterpret_cast<const void *>(&orc_pgo_test_throw)}});
  auto *Passes = S.sym<int(int)>("passes");
  auto *Catches = S.sym<int(int)>("catches");
  EXPECT_EQ(Passes(-1), 1);
  EXPECT_EQ(Catches(3), 3);
  auto R = runtimeFor(*S.Image, options(1));
  EXPECT_EQ(R->pollOnce(), 2u);
  int Caught = -1;
  try {
    Passes(5);
  } catch (int V) {
    Caught = V;
  }
  EXPECT_EQ(Caught, 5);
  EXPECT_EQ(Catches(7), 7);
  EXPECT_EQ(Catches(-1), -1);
}

extern "C" __int128 orc_pgo_test_divti3_sentinel(__int128, __int128) {
  return 42;
}

TEST(Runtime, LibcallsResolveThroughTheSymbolTable) {
  // The table's __divti3 is a sentinel, not the real one the process also
  // exports (macOS: libcompiler_rt), so tiered-up code that called the
  // process's would return -14.
  const void *Sentinel =
      reinterpret_cast<const void *>(&orc_pgo_test_divti3_sentinel);
  auto S = test::simulateAot(DivIR, {"__divti3"}, {{"__divti3", Sentinel}});
  ASSERT_NE(S.symbol("__divti3"), nullptr);
  EXPECT_EQ(S.symbol("__divti3")->Address, Sentinel);
  auto *Div = S.sym<__int128(__int128, __int128)>("div");
  EXPECT_TRUE(Div(-100, 7) == 42);
  auto R = runtimeFor(*S.Image, options(1));
  EXPECT_EQ(R->pollOnce(), 1u);
  EXPECT_EQ(R->state(*R->slotOf("div")), SlotState::TieredUp);
  EXPECT_TRUE(Div(-100, 7) == 42);
}

TEST(RuntimeOptions, FastStartPollsOftenForTheFirstHundredMilliseconds) {
  using std::chrono::milliseconds;
  EXPECT_EQ(pollPeriodMs(10, milliseconds(0)), 1u);
  EXPECT_EQ(pollPeriodMs(10, milliseconds(99)), 1u);
  EXPECT_EQ(pollPeriodMs(10, milliseconds(100)), 10u);
  EXPECT_EQ(pollPeriodMs(10, milliseconds(5000)), 10u);
  // Never slower than the configured period.
  EXPECT_EQ(pollPeriodMs(1, milliseconds(0)), 1u);
}

constexpr const char *PerfIR = R"(
source_filename = "perf.c"

define i32 @callee(i32 %x) noinline {
  %r = sub i32 %x, 1
  ret i32 %r
}
define i32 @caller(i32 %x) noinline {
  %r = call i32 @callee(i32 %x)
  %s = add i32 %r, 100
  ret i32 %s
}
)";

/// The result of caller(5) after callee and caller are tiered up, in one
/// batch or in two (callee first); callee's slot is then sent to a function
/// returning 777. Through the slot, callee(5) is 777; callee's tier-up body
/// returns 4, so a caller that goes through the slot returns 877 and one that
/// is bound to callee's tier-up body 104.
int callerAfterTierUps(bool SameBatch, int *CalleeThroughSlot) {
  auto S = test::simulateAot(PerfIR);
  auto R = runtimeFor(*S.Image, options(10, /*InlineDepth=*/0));
  auto *Callee = S.sym<int(int)>("callee");
  auto *Caller = S.sym<int(int)>("caller");
  for (int I = 0; I < 10; ++I) {
    EXPECT_EQ(Callee(5), 4);
    if (SameBatch)
      EXPECT_EQ(Caller(5), 104);
  }
  if (!SameBatch) {
    EXPECT_EQ(R->pollOnce(), 1u); // callee, alone
    for (int I = 0; I < 10; ++I)
      EXPECT_EQ(Caller(5), 104);
  }
  EXPECT_EQ(R->pollOnce(), SameBatch ? 2u : 1u);
  EXPECT_EQ(R->state(*R->slotOf("callee")), SlotState::TieredUp);
  EXPECT_EQ(R->state(*R->slotOf("caller")), SlotState::TieredUp);
  S.Image->Dispatch[*R->slotOf("callee")] =
      reinterpret_cast<void *>(&orc_pgo_test_sentinel);
  *CalleeThroughSlot = Callee(5);
  return Caller(5);
}

TEST(Runtime, ACallToAnEarlierTieredFunctionSkipsItsThunk) {
  int Through;
  EXPECT_EQ(callerAfterTierUps(/*SameBatch=*/false, &Through), 104);
  EXPECT_EQ(Through, 777);
}

TEST(Runtime, ACallToARootOfTheSameBatchSkipsItsThunk) {
  int Through;
  EXPECT_EQ(callerAfterTierUps(/*SameBatch=*/true, &Through), 104);
  EXPECT_EQ(Through, 777);
}

TEST(Runtime, FastStartTiersUpBeforeTheFirstPollPeriodHasPassed) {
  auto S = test::simulateAot(PerfIR);
  for (int I = 0; I < 10; ++I)
    S.sym<int(int)>("callee")(5);
  RuntimeOptions O = options(10);
  // The next regular poll is 5 s away; the first batch takes a few ms.
  O.PollMs = 5000;
  auto R = runtimeFor(*S.Image, O);
  R->startPolling();
  unsigned Slot = *R->slotOf("callee");
  auto Deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2500);
  while (R->state(Slot) != SlotState::TieredUp &&
         std::chrono::steady_clock::now() < Deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(R->state(Slot), SlotState::TieredUp);
}

TEST(Runtime, VerboseReportsEachBatchWithItsRoots) {
  auto S = test::simulateAot(PerfIR);
  RuntimeOptions O = options(10, /*InlineDepth=*/0);
  O.Verbose = true;
  auto R = runtimeFor(*S.Image, O);
  for (int I = 0; I < 10; ++I) {
    S.sym<int(int)>("callee")(5);
    S.sym<int(int)>("caller")(5);
  }
  testing::internal::CaptureStderr();
  EXPECT_EQ(R->pollOnce(), 2u);
  std::string Err = testing::internal::GetCapturedStderr();
  EXPECT_NE(Err.find("orc-pgo: batch of 2 roots took "), std::string::npos)
      << Err;
  EXPECT_NE(Err.find(" ms, 2 tiered up: callee, caller\n"), std::string::npos)
      << Err;
}

// Code that registers exit handlers: the table has atexit, __cxa_atexit and
// __dso_handle, which the program's JIT defines itself.
constexpr const char *AtexitIR = R"(
source_filename = "atexit.c"
@exits = global i32 0
@__dso_handle = external hidden global i8
declare i32 @__cxa_atexit(ptr, ptr, ptr)
declare i32 @atexit(ptr)

define void @bump(ptr %arg) {
  %v = load i32, ptr @exits
  %n = add i32 %v, 1
  store i32 %n, ptr @exits
  ret void
}
define void @bump2() {
  %v = load i32, ptr @exits
  %n = add i32 %v, 10
  store i32 %n, ptr @exits
  ret void
}
define i32 @hot(i32 %x) {
  %a = call i32 @__cxa_atexit(ptr @bump, ptr null, ptr @__dso_handle)
  %b = call i32 @atexit(ptr @bump2)
  ret i32 %x
}
)";

TEST(Runtime, TierUpCodeRegistersExitHandlersWithTheProgram) {
  auto S = test::simulateAot(AtexitIR);
  ASSERT_NE(S.symbol("atexit"), nullptr);
  ASSERT_NE(S.symbol("__cxa_atexit"), nullptr);
  ASSERT_NE(S.symbol("__dso_handle"), nullptr);
  auto *Hot = S.sym<int(int)>("hot");
  for (int I = 0; I < 10; ++I)
    ASSERT_EQ(Hot(I), I);
  auto R = runtimeFor(*S.Image, options(10));
  EXPECT_EQ(R->pollOnce(), 1u);
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::TieredUp);
  EXPECT_EQ(Hot(0), 0); // tiered-up: registers one more of each
  int *Exits = S.sym<int>("exits");
  EXPECT_EQ(*Exits, 0);
  // The program's exit handlers run when the program ends: 11 calls, each
  // registering bump (1) and bump2 (10).
  cantFail(S.J->deinitialize(S.J->getMainJITDylib()));
  EXPECT_EQ(*Exits, 11 * (1 + 10));
}

TEST(Runtime, ALinkFailureOfOneRootFailsOnlyThatRoot) {
  auto S = test::simulateAot(RuntimeIR);
  // A table without the global that readsg reads: its tier-up body can't be
  // linked, in a batch with two roots that can.
  std::vector<OrcPgoSymbol> Symbols;
  for (const OrcPgoSymbol &Sym :
       ArrayRef(S.Image->Symbols, S.Image->NumSymbols))
    if (StringRef(Sym.Name) != "orc_pgo_runtime_test_global")
      Symbols.push_back(Sym);
  ASSERT_EQ(Symbols.size() + 1, S.Image->NumSymbols);
  OrcPgoImage Tampered = *S.Image;
  Tampered.Symbols = Symbols.data();
  Tampered.NumSymbols = Symbols.size();
  RuntimeOptions O = options(10);
  O.Verbose = true;
  auto R = runtimeFor(Tampered, O);
  for (int I = 0; I < 10; ++I) {
    S.sym<int(int)>("hot")(1);
    S.sym<int(int)>("cold")(5);
    S.sym<int()>("readsg")();
  }
  testing::internal::CaptureStderr();
  EXPECT_EQ(R->pollOnce(), 2u);
  std::string Err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::TieredUp);
  EXPECT_EQ(R->state(*R->slotOf("cold")), SlotState::TieredUp);
  EXPECT_EQ(R->state(*R->slotOf("readsg")), SlotState::Failed);
  RuntimeStats St = R->stats();
  EXPECT_EQ(St.Batches, 1u);
  EXPECT_EQ(St.TieredUp, 2u);
  EXPECT_EQ(St.Failed, 1u);
  EXPECT_NE(Err.find("orc-pgo: tier-up of hot, cold, readsg failed: "),
            std::string::npos)
      << Err;
  EXPECT_NE(Err.find("orc-pgo: tier-up of readsg failed: "), std::string::npos)
      << Err;
  EXPECT_EQ(Err.find("orc-pgo: tier-up of hot failed: "), std::string::npos)
      << Err;
  EXPECT_EQ(S.sym<int(int)>("hot")(41), 42);
  EXPECT_EQ(S.sym<int(int)>("cold")(5), 4);
  EXPECT_EQ(S.sym<int()>("readsg")(), 5);
}

/// An image whose snapshot has cold start with a call to orc_pgo_forbidden, a
/// declaration with the function attribute Attr, and whose symbol table can
/// resolve it (to a function that does nothing).
struct ForbiddenCallImage {
  ForbiddenCallImage(const OrcPgoImage &Image, StringRef Attr)
      : Bitcode(test::editedSnapshot(Image, [&](Module &M) {
          LLVMContext &Ctx = M.getContext();
          Function *Forbidden = Function::Create(
              FunctionType::get(Type::getVoidTy(Ctx), false),
              GlobalValue::ExternalLinkage, "orc_pgo_forbidden", M);
          Forbidden->addFnAttr(Attr, "a forbidden call");
          IRBuilder<> B(
              &*M.getFunction("cold")->getEntryBlock().getFirstInsertionPt());
          B.CreateCall(Forbidden);
        })) {
    Symbols.assign(Image.Symbols, Image.Symbols + Image.NumSymbols);
    Symbols.push_back({"orc_pgo_forbidden",
                       reinterpret_cast<const void *>(&orc_pgo_test_nop)});
    Tampered = test::withSnapshot(Image, Bitcode);
    Tampered.Symbols = Symbols.data();
    Tampered.NumSymbols = Symbols.size();
  }
  std::string Bitcode;
  std::vector<OrcPgoSymbol> Symbols;
  OrcPgoImage Tampered;
};

TEST(Runtime, AnErrorDiagnosedWhileCompilingFailsTheBatch) {
  // The default diagnostic handler would exit the program.
  auto S = test::simulateAot(RuntimeIR);
  ForbiddenCallImage F(*S.Image, "dontcall-error");
  RuntimeOptions O = options(10);
  O.Verbose = true;
  auto R = runtimeFor(F.Tampered, O);
  for (int I = 0; I < 10; ++I) {
    S.sym<int(int)>("hot")(1);
    S.sym<int(int)>("cold")(5);
  }
  testing::internal::CaptureStderr();
  EXPECT_EQ(R->pollOnce(), 1u);
  std::string Err = testing::internal::GetCapturedStderr();
  EXPECT_EQ(R->state(*R->slotOf("hot")), SlotState::TieredUp);
  EXPECT_EQ(R->state(*R->slotOf("cold")), SlotState::Failed);
  EXPECT_NE(Err.find("orc-pgo: tier-up of cold failed: "), std::string::npos)
      << Err;
  EXPECT_NE(Err.find("a forbidden call"), std::string::npos) << Err;
  EXPECT_EQ(S.sym<int(int)>("cold")(5), 4);
}

TEST(Runtime, WarningsAreReportedOnlyWhenVerbose) {
  for (bool Verbose : {false, true}) {
    SCOPED_TRACE(Verbose);
    auto S = test::simulateAot(RuntimeIR);
    ForbiddenCallImage F(*S.Image, "dontcall-warn");
    RuntimeOptions O = options(10);
    O.Verbose = Verbose;
    auto R = runtimeFor(F.Tampered, O);
    for (int I = 0; I < 10; ++I)
      S.sym<int(int)>("cold")(5);
    testing::internal::CaptureStderr();
    EXPECT_EQ(R->pollOnce(), 1u);
    std::string Err = testing::internal::GetCapturedStderr();
    EXPECT_EQ(R->state(*R->slotOf("cold")), SlotState::TieredUp);
    if (Verbose) {
      EXPECT_NE(Err.find("orc-pgo: warning: "), std::string::npos) << Err;
      EXPECT_NE(Err.find("a forbidden call"), std::string::npos) << Err;
    } else {
      EXPECT_EQ(Err, "");
    }
  }
}

} // namespace
