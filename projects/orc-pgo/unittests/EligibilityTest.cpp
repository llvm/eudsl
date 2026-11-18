// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Eligibility.h"
#include "TestSupport.h"

#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace orc_pgo;

namespace {

// @pair names @g twice and @twoloads loads it twice, so the memoized walk
// meets an already-visited constant.
constexpr const char *IR = R"(
@g = global i32 0
@tls = thread_local global i32 0
@ifn = ifunc void (), ptr @resolver
@tlsptr = global ptr @tls
@labels = global [1 x ptr] [ptr blockaddress(@target, %bb)]
@pair = global { ptr, ptr } { ptr @g, ptr getelementptr (i8, ptr @g, i64 4) }

define ptr @resolver() {
  ret ptr @plain
}
define void @plain() {
  ret void
}
define i32 @twoloads() {
  %a = load i32, ptr @g
  %b = load i32, ptr @g
  %s = add i32 %a, %b
  ret i32 %s
}
define ptr @nested() {
  ret ptr getelementptr (i8, ptr getelementptr (i8, ptr @g, i64 1), i64 1)
}
define i32 @usestls() {
  %v = load i32, ptr @tls
  ret i32 %v
}
define ptr @tlsexpr() {
  ret ptr getelementptr (i8, ptr @tls, i64 4)
}
define void @callsifunc() {
  call void @ifn()
  ret void
}
define void @target() {
entry:
  br label %bb
bb:
  ret void
}
define ptr @refstarget() {
  ret ptr blockaddress(@target, %bb)
}
declare void @decl()
)";

TEST(Eligibility, BodiesUsingOrdinaryGlobalsAreRecreatable) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  for (StringRef Name : {"resolver", "plain", "twoloads", "nested", "decl"})
    EXPECT_TRUE(hasRecreatableBody(*M->getFunction(Name))) << Name.str();
}

TEST(Eligibility, ThreadLocalsIfuncsAndBlockAddressesAreNot) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  for (StringRef Name : {"usestls", "tlsexpr", "callsifunc", "refstarget"})
    EXPECT_FALSE(hasRecreatableBody(*M->getFunction(Name))) << Name.str();
}

TEST(Eligibility, FunctionsWithAddressTakenBlocksAreNot) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  EXPECT_FALSE(hasRecreatableBody(*M->getFunction("target")));
}

TEST(Eligibility, ConstantsAreCheckedThroughExpressionsAndAggregates) {
  LLVMContext Ctx;
  auto M = test::parseIR(Ctx, IR);
  auto Init = [&](StringRef Name) -> const Constant & {
    return *M->getNamedGlobal(Name)->getInitializer();
  };
  EXPECT_TRUE(isRecreatable(Init("g")));
  EXPECT_TRUE(isRecreatable(Init("pair")));
  EXPECT_FALSE(isRecreatable(Init("tlsptr")));
  EXPECT_FALSE(isRecreatable(Init("labels")));
}

} // namespace
