// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Stats.h"

#include "gtest/gtest.h"

#include <cstring>

namespace {

TEST(Stats, AllZeroWhenNoRuntimeIsRunning) {
  // No image is registered in this process and __orc_pgo_start never ran.
  OrcPgoStats S;
  std::memset(&S, 0xff, sizeof(S));
  __orc_pgo_get_stats(&S);
  EXPECT_EQ(S.TieredUp, 0u);
  EXPECT_EQ(S.Failed, 0u);
  EXPECT_EQ(S.PromotedIndirectCalls, 0u);
}

} // namespace
