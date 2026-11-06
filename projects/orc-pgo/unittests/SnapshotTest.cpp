// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Snapshot.h"
#include "TestSupport.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

// %x has three users. A plain parse pushes each use at the head of the use
// list, giving c, b, a; the uselistorder directive makes the list differ from
// that.
constexpr const char *IRHead = R"(
source_filename = "snap.c"
define i32 @f(i32 %x) {
  %a = add i32 %x, 1
  %b = mul i32 %x, 2
  %c = sub i32 %x, 3
  %s = add i32 %a, %b
  %t = add i32 %s, %c
  ret i32 %t
)";
constexpr const char *UseListOrder = "  uselistorder i32 %x, { 1, 2, 0 }\n";
const std::string IR = std::string(IRHead) + UseListOrder + "}\n";
const std::string PlainIR = std::string(IRHead) + "}\n";

std::string printed(const Module &M) {
  std::string S;
  raw_string_ostream OS(S);
  M.print(OS, nullptr);
  return S;
}

/// Names of the instructions using f's first argument, in use-list order.
std::vector<std::string> argUsers(const Module &M) {
  std::vector<std::string> Names;
  for (const User *U : M.getFunction("f")->getArg(0)->users())
    Names.push_back(U->getName().str());
  return Names;
}

TEST(Snapshot, RoundTripPreservesTheModule) {
  LLVMContext C1, C2;
  auto M = test::parseIR(C1, IR);
  auto Back = parseSnapshot(writeSnapshot(*M), C2);
  ASSERT_EQ(test::errorText(Back), "");
  EXPECT_FALSE(verifyModule(**Back, &errs()));
  // PGO names of local functions are derived from source_filename, which must
  // survive; the module identifier is irrelevant and is not stored in bitcode.
  EXPECT_EQ((*Back)->getSourceFileName(), "snap.c");
  (*Back)->setModuleIdentifier(M->getModuleIdentifier());
  EXPECT_EQ(printed(**Back), printed(*M));
}

TEST(Snapshot, RoundTripPreservesUseListOrder) {
  LLVMContext C1, C2, C3;
  auto M = test::parseIR(C1, IR);
  std::vector<std::string> Original = argUsers(*M);
  LLVMContext PlainCtx;
  ASSERT_NE(Original, argUsers(*test::parseIR(PlainCtx, PlainIR)))
      << "fixture must have a non-default use-list order";

  auto Back = parseSnapshot(writeSnapshot(*M), C2);
  ASSERT_EQ(test::errorText(Back), "");
  EXPECT_FALSE(verifyModule(**Back, &errs()));
  EXPECT_EQ(argUsers(**Back), Original);

  // Without use-list preservation the order is lost, so the check above
  // really depends on writeSnapshot preserving it.
  std::string Plain;
  raw_string_ostream OS(Plain);
  WriteBitcodeToFile(*M, OS, /*ShouldPreserveUseListOrder=*/false);
  auto Lost = parseSnapshot(Plain, C3);
  ASSERT_EQ(test::errorText(Lost), "");
  EXPECT_FALSE(verifyModule(**Lost, &errs()));
  EXPECT_NE(argUsers(**Lost), Original);
}

TEST(Snapshot, RejectsInvalidBytes) {
  LLVMContext Ctx;
  auto Bad = parseSnapshot("not bitcode", Ctx);
  EXPECT_NE(test::errorText(Bad), "");
}

} // namespace
