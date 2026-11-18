// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SimulatedAot.h"
#include "TestSupport.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/ProfData.h"
#include "orc-pgo/Runtime.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace llvm;
using namespace orc_pgo;

namespace {

constexpr const char *IR = R"(
source_filename = "entry.c"
define i32 @hot(i32 %x) {
  %r = add i32 %x, 1
  ret i32 %r
}
define i32 @callit(ptr %f, i32 %x) {
  %r = call i32 %f(i32 %x)
  ret i32 %r
}
)";

class RuntimeEntryDeathTest : public testing::Test {
protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(RuntimeEntryDeathTest, PollTiersUpAndStatsPrintAtExit) {
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_THRESHOLD", "10", 1);
        setenv("ORC_PGO_POLL_MS", "0", 1);
        setenv("ORC_PGO_VERBOSE", "1", 1);
        auto S = test::simulateAot(IR);
        __orc_pgo_register_image(S.Image);
        auto *Hot = S.sym<int(int)>("hot");
        for (int I = 0; I < 10; ++I)
          Hot(I);
        void *Aot = S.Image->Dispatch[S.slot("hot")];
        // A second start must not create a second runtime.
        __orc_pgo_start();
        __orc_pgo_start();
        __orc_pgo_poll();
        bool Swapped = S.Image->Dispatch[S.slot("hot")] != Aot && Hot(41) == 42;
        RuntimeStats St = processRuntimeStats();
        bool Counted = St.TieredUp == 1 && St.Failed == 0 && St.Batches == 1;
        std::exit(Swapped && Counted ? 0 : 1);
      },
      testing::ExitedWithCode(0),
      "orc-pgo: 1 batches, 1 tiered up, 0 failed, 0 indirect calls promoted");
}

TEST_F(RuntimeEntryDeathTest, StartIsIdempotentAndInstallsTheValueNodePool) {
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_POLL_MS", "0", 1);
        auto S = test::simulateAot(IR);
        __orc_pgo_register_image(S.Image);
        setValueNodePool(nullptr, nullptr); // undo the harness's setup
        __orc_pgo_start();
        __orc_pgo_start();
        auto *CallIt = S.sym<int(int (*)(int), int)>("callit");
        CallIt(S.sym<int(int)>("hot"), 1);
        bool Recorded =
            readValueSite(S.record("callit"), IPVK_IndirectCallTarget, 0)
                .size() == 1;
        // The runtime pool would serve the value even with no image pool, so
        // check that the node is the image pool's.
        auto *D = static_cast<const prof::Data *>(S.record("callit").Data);
        auto Node = reinterpret_cast<uintptr_t>(
            static_cast<prof::ValueProfNode *const *>(D->Values)[0]);
        bool InImagePool =
            Node >= reinterpret_cast<uintptr_t>(S.Image->VNodePool[0]) &&
            Node < reinterpret_cast<uintptr_t>(S.Image->VNodePool[1]);
        std::exit(Recorded && InImagePool ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

TEST_F(RuntimeEntryDeathTest, DisabledByTheEnvironment) {
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_DISABLE", "1", 1);
        setenv("ORC_PGO_VERBOSE", "1", 1);
        setenv("ORC_PGO_THRESHOLD", "1", 1);
        auto S = test::simulateAot(IR);
        __orc_pgo_register_image(S.Image);
        S.sym<int(int)>("hot")(1);
        void *Aot = S.Image->Dispatch[S.slot("hot")];
        // However often it is called, start() runs its body once.
        testing::internal::CaptureStderr();
        __orc_pgo_poll();
        __orc_pgo_poll();
        std::string Err = testing::internal::GetCapturedStderr();
        std::fputs(Err.c_str(), stderr);
        size_t First = Err.find("disabled by ORC_PGO_DISABLE");
        bool Once = First != std::string::npos &&
                    Err.find("disabled by ORC_PGO_DISABLE", First + 1) ==
                        std::string::npos;
        bool Untouched = S.Image->Dispatch[S.slot("hot")] == Aot;
        std::exit(Once && Untouched && processRuntimeStats().TieredUp == 0 ? 0
                                                                           : 1);
      },
      testing::ExitedWithCode(0), "orc-pgo: disabled by ORC_PGO_DISABLE");
}

/// Registers S's image with a wrong LLVM version, polls, and exits 0 if the
/// function stayed on its AOT body.
void runMismatchedImage() {
  setenv("ORC_PGO_THRESHOLD", "1", 1);
  setenv("ORC_PGO_POLL_MS", "0", 1);
  auto S = test::simulateAot(IR);
  OrcPgoImage Bad = *S.Image;
  Bad.Fingerprint.LLVMVersion = "0.0.0";
  __orc_pgo_register_image(&Bad);
  S.sym<int(int)>("hot")(1);
  void *Aot = S.Image->Dispatch[S.slot("hot")];
  __orc_pgo_poll();
  std::exit(S.Image->Dispatch[S.slot("hot")] == Aot ? 0 : 1);
}

TEST_F(RuntimeEntryDeathTest, AFingerprintMismatchDisablesTheImage) {
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_VERBOSE", "1", 1);
        runMismatchedImage();
      },
      testing::ExitedWithCode(0), "orc-pgo: image built with LLVM 0.0.0");
}

TEST_F(RuntimeEntryDeathTest, AFingerprintMismatchIsSilentUnlessVerbose) {
  EXPECT_EXIT(runMismatchedImage(), testing::ExitedWithCode(0), "^$");
}

TEST_F(RuntimeEntryDeathTest, AnOptionSetupFailureDisablesTheRuntime) {
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_VERBOSE", "1", 1);
        auto S = test::simulateAot(IR);
        __orc_pgo_register_image(S.Image);
        test::RequiredPositionalOption Positional;
        __orc_pgo_start();
        std::exit(0);
      },
      testing::ExitedWithCode(0),
      "orc-pgo: cannot set orc-pgo's LLVM options: orc-pgo: Not enough "
      "positional");
}

TEST_F(RuntimeEntryDeathTest, ImagesBeyondTheLimitAreIgnored) {
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_VERBOSE", "1", 1);
        setenv("ORC_PGO_POLL_MS", "0", 1);
        auto S = test::simulateAot(IR);
        for (unsigned I = 0; I < MaxImages + 1; ++I)
          __orc_pgo_register_image(S.Image);
        __orc_pgo_start();
        std::exit(0);
      },
      testing::ExitedWithCode(0),
      "orc-pgo: 1 images beyond the first 16 ignored");
}

TEST_F(RuntimeEntryDeathTest, ExitStopsThePollThreadCleanly) {
  // Whether or not a batch is in flight at exit (usually it is: it takes
  // longer than the sleep), the atexit handler waits for it while LLVM is
  // still alive, so the statistics never show a batch that did not finish.
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_THRESHOLD", "1", 1);
        setenv("ORC_PGO_POLL_MS", "1", 1);
        setenv("ORC_PGO_VERBOSE", "1", 1);
        auto S = test::simulateAot(IR);
        __orc_pgo_register_image(S.Image);
        __orc_pgo_start();
        S.sym<int(int)>("hot")(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        std::exit(0);
      },
      testing::ExitedWithCode(0),
      "orc-pgo: (0 batches, 0 tiered up|1 batches, 1 tiered up), 0 failed");
}

TEST_F(RuntimeEntryDeathTest, AForkedChildCanExit) {
  // The child has no poll thread to join. Exiting it must neither abort nor
  // print statistics, and the functions already tiered up keep working.
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_THRESHOLD", "1", 1);
        setenv("ORC_PGO_POLL_MS", "1", 1);
        setenv("ORC_PGO_VERBOSE", "1", 1);
        auto S = test::simulateAot(IR);
        __orc_pgo_register_image(S.Image);
        __orc_pgo_start();
        auto *Hot = S.sym<int(int)>("hot");
        Hot(1);
        for (int I = 0; I < 10000 && processRuntimeStats().TieredUp != 1; ++I)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (processRuntimeStats().TieredUp != 1)
          std::_Exit(2);
        pid_t Pid = fork();
        if (Pid == 0) {
          bool Works = Hot(41) == 42 && processRuntimeStats().TieredUp == 0;
          std::exit(Works ? 0 : 3);
        }
        int Status = 0;
        waitpid(Pid, &Status, 0);
        std::exit(WIFEXITED(Status) ? WEXITSTATUS(Status) : 4);
      },
      testing::ExitedWithCode(0), "orc-pgo: 1 batches, 1 tiered up, 0 failed");
}

TEST_F(RuntimeEntryDeathTest, AnImageRegisteredAfterTheStartIsIgnored) {
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_VERBOSE", "1", 1);
        setenv("ORC_PGO_THRESHOLD", "1", 1);
        setenv("ORC_PGO_POLL_MS", "0", 1);
        auto Early = test::simulateAot(IR);
        __orc_pgo_register_image(Early.Image);
        __orc_pgo_start();
        // As a dlopen'd library's constructor would.
        auto Late = test::simulateAot(IR);
        __orc_pgo_register_image(Late.Image);
        Late.sym<int(int)>("hot")(1);
        Early.sym<int(int)>("hot")(1);
        void *Aot = Late.Image->Dispatch[Late.slot("hot")];
        __orc_pgo_poll();
        bool Ignored = Late.Image->Dispatch[Late.slot("hot")] == Aot;
        bool EarlyServed = processRuntimeStats().TieredUp == 1;
        std::exit(Ignored && EarlyServed ? 0 : 1);
      },
      testing::ExitedWithCode(0),
      "orc-pgo: image registered after the runtime started is ignored");
}

TEST_F(RuntimeEntryDeathTest, ExitingOnThePollThreadDoesNotJoinItself) {
  // A fatal LLVM error on the poll thread whose handler exits (as a host's
  // may): the atexit handler runs on the poll thread, which must not wait for
  // itself, and must still be able to print the statistics.
  EXPECT_EXIT(
      {
        setenv("ORC_PGO_THRESHOLD", "1", 1);
        setenv("ORC_PGO_POLL_MS", "1", 1);
        setenv("ORC_PGO_VERBOSE", "1", 1);
        auto S = test::simulateAot(IR);
        // An intrinsic of the other architecture: no code generator can
        // select it, which is a fatal error.
        std::string Bitcode = test::editedSnapshot(*S.Image, [](Module &M) {
          LLVMContext &Ctx = M.getContext();
          Type *Void = Type::getVoidTy(Ctx);
          Type *I32 = Type::getInt32Ty(Ctx);
          IRBuilder<> B(
              &*M.getFunction("hot")->getEntryBlock().getFirstInsertionPt());
          if (Triple(sys::getProcessTriple()).isX86()) {
            B.CreateCall(Function::Create(FunctionType::get(Void, {I32}, false),
                                          GlobalValue::ExternalLinkage,
                                          "llvm.aarch64.isb", M),
                         {B.getInt32(15)});
          } else {
            B.CreateCall(Function::Create(FunctionType::get(Void, {}, false),
                                          GlobalValue::ExternalLinkage,
                                          "llvm.x86.sse.sfence", M));
          }
        });
        OrcPgoImage Tampered = test::withSnapshot(*S.Image, Bitcode);
        install_fatal_error_handler(
            [](void *, const char *Reason, bool) {
              std::fprintf(stderr, "fatal: %s\n", Reason);
              std::exit(0);
            },
            nullptr);
        __orc_pgo_register_image(&Tampered);
        __orc_pgo_start();
        S.sym<int(int)>("hot")(1);
        std::this_thread::sleep_for(std::chrono::seconds(10));
        std::exit(1);
      },
      testing::ExitedWithCode(0),
      "fatal: .*\n.*orc-pgo: 1 batches, 0 tiered up");
}

} // namespace
