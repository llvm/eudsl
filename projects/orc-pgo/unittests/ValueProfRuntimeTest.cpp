// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/ValueProfRuntime.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/ProfData.h"

#include "gtest/gtest.h"

#include <thread>
#include <utility>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

using Values = std::vector<std::pair<uint64_t, uint64_t>>;

/// A function with value sites {IndirectCall, MemOP, VTable} = Sites.
struct Fixture {
  explicit Fixture(std::initializer_list<uint16_t> Sites) {
    std::copy(Sites.begin(), Sites.end(), NumSites);
    Rec = {"f",
           0,
           nullptr,
           &D,
           nullptr,
           0,
           {NumSites[0], NumSites[1], NumSites[2]}};
  }
  uint16_t NumSites[NumValueKinds] = {};
  prof::ValueProfNode *Heads[64] = {};
  prof::Data D{
      /*NameRef=*/0,          /*FuncHash=*/0,
      /*CounterPtr=*/nullptr, /*UniformCounterPtr=*/nullptr,
      /*BitmapPtr=*/nullptr,  /*FunctionPointer=*/nullptr,
      /*Values=*/Heads,       /*NumCounters=*/0,
      /*NumValueSites=*/{},   /*OffloadDeviceWaveSize=*/0,
      /*NumBitmapBytes=*/0};
  OrcPgoFunctionRecord Rec;

  Values site(uint32_t Kind, uint32_t Site) const {
    Values Out;
    for (const InstrProfValueData &V : readValueSite(Rec, Kind, Site))
      Out.push_back({V.Value, V.Count});
    return Out;
  }
};

prof::ValueProfNode Pool[64];

void usePool(unsigned NumNodes) {
  std::fill(std::begin(Pool), std::end(Pool), prof::ValueProfNode{});
  setValueNodePool(Pool, Pool + NumNodes);
}

TEST(ValueProfRuntime, RecordsValuesWithCountsInInsertionOrder) {
  usePool(8);
  Fixture F({1, 0, 0});
  for (uint64_t V : {7, 7, 9, 7})
    __orc_pgo_instrument_target(V, &F.D, 0);
  EXPECT_EQ(F.site(IPVK_IndirectCallTarget, 0), (Values{{7, 3}, {9, 1}}));
  EXPECT_FALSE(valueNodePoolExhausted());
}

TEST(ValueProfRuntime, SitesAreNumberedAcrossKinds) {
  usePool(8);
  Fixture F({2, 1, 0}); // indirect sites 0,1 then memop site 0 at index 2
  __orc_pgo_instrument_target(11, &F.D, 1);
  __orc_pgo_instrument_memop(5, &F.D, 2);
  EXPECT_EQ(F.site(IPVK_IndirectCallTarget, 0), Values{});
  EXPECT_EQ(F.site(IPVK_IndirectCallTarget, 1), (Values{{11, 1}}));
  EXPECT_EQ(F.site(IPVK_MemOPSize, 0), (Values{{5, 1}}));
}

TEST(ValueProfRuntime, MemopSizesUseLlvmRanges) {
  usePool(16);
  Fixture F({0, 1, 0});
  // InstrProfGetRangeRepValue: <=8 exact, powers of two exact, otherwise the
  // previous power of two + 1, and >=513 collapses to 513.
  for (uint64_t Size : {3, 8, 9, 16, 100, 513, 4096})
    __orc_pgo_instrument_memop(Size, &F.D, 0);
  EXPECT_EQ(F.site(IPVK_MemOPSize, 0),
            (Values{{3, 1}, {8, 1}, {9, 1}, {16, 1}, {65, 1}, {513, 2}}));
}

TEST(ValueProfRuntime, EvictsTheColdestValueLikeCompilerRt) {
  usePool(32);
  Fixture F({1, 0, 0});
  for (uint64_t V = 100; V < 124; ++V) { // 24 values, the per-site maximum
    __orc_pgo_instrument_target(V, &F.D, 0);
    __orc_pgo_instrument_target(V, &F.D, 0);
  }
  // A new value first decays the coldest entry (the first one found with
  // the minimum count) ...
  __orc_pgo_instrument_target(500, &F.D, 0);
  Values After1 = F.site(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(After1.size(), 24u);
  EXPECT_EQ(After1[0], (std::pair<uint64_t, uint64_t>{100, 1}));
  // ... and replaces it once that count has dropped to the new value's.
  __orc_pgo_instrument_target(500, &F.D, 0);
  Values After2 = F.site(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(After2.size(), 24u);
  EXPECT_EQ(After2[0], (std::pair<uint64_t, uint64_t>{500, 1}));
  EXPECT_EQ(After2[1], (std::pair<uint64_t, uint64_t>{101, 2}));
}

TEST(ValueProfRuntime, TheRuntimePoolServesValuesOnceTheImagePoolIsUsedUp) {
  usePool(2);
  Fixture F({1, 0, 0});
  for (uint64_t V : {1, 2, 3, 4, 1})
    __orc_pgo_instrument_target(V, &F.D, 0);
  EXPECT_EQ(F.site(IPVK_IndirectCallTarget, 0),
            (Values{{1, 2}, {2, 1}, {3, 1}, {4, 1}}));
  EXPECT_FALSE(valueNodePoolExhausted());
}

TEST(ValueProfRuntime, DropsAndFlagsNewValuesWhenBothPoolsRunOut) {
  // Capacity is the image's pool plus compiler-rt's 1024 runtime nodes:
  // 2 + 1024 = 1026 = 42 sites * 24 values + 18 values. Each site keeps at
  // most 24 values, so the values are spread over 43 sites.
  usePool(2);
  Fixture F({64, 0, 0});
  for (uint32_t Site = 0; Site < 43; ++Site)
    for (uint64_t V = 0; V < (Site < 42 ? 24u : 18u); ++V)
      __orc_pgo_instrument_target(V + 1, &F.D, Site);
  EXPECT_FALSE(valueNodePoolExhausted());
  for (uint32_t Site = 0; Site < 43; ++Site)
    EXPECT_EQ(F.site(IPVK_IndirectCallTarget, Site).size(),
              Site < 42 ? 24u : 18u);

  __orc_pgo_instrument_target(99, &F.D, 43);
  EXPECT_TRUE(valueNodePoolExhausted());
  EXPECT_EQ(F.site(IPVK_IndirectCallTarget, 43), Values{});
  // Values already recorded still count.
  __orc_pgo_instrument_target(1, &F.D, 0);
  EXPECT_EQ(F.site(IPVK_IndirectCallTarget, 0)[0],
            (std::pair<uint64_t, uint64_t>{1, 2}));

  usePool(2);
  EXPECT_FALSE(valueNodePoolExhausted());
}

TEST(ValueProfRuntime, SettingThePoolAgainRestartsTheRuntimePool) {
  usePool(0);
  Fixture A({64, 0, 0});
  // 42 * 24 + 16 = 1024: every runtime node.
  for (uint32_t Site = 0; Site < 43; ++Site)
    for (uint64_t V = 1; V <= (Site < 42 ? 24u : 16u); ++V)
      __orc_pgo_instrument_target(V, &A.D, Site);
  usePool(0);
  Fixture B({1, 0, 0});
  __orc_pgo_instrument_target(5, &B.D, 0);
  EXPECT_EQ(B.site(IPVK_IndirectCallTarget, 0), (Values{{5, 1}}));
  EXPECT_FALSE(valueNodePoolExhausted());
}

TEST(ValueProfRuntime, ValuesAreRecordedWithoutAnImagePool) {
  setValueNodePool(nullptr, nullptr);
  Fixture F({1, 0, 0});
  __orc_pgo_instrument_target(1, &F.D, 0);
  EXPECT_EQ(F.site(IPVK_IndirectCallTarget, 0), (Values{{1, 1}}));
  EXPECT_FALSE(valueNodePoolExhausted());
}

TEST(ValueProfRuntime, IgnoresMissingDataOrValueArrays) {
  usePool(8);
  __orc_pgo_instrument_target(1, nullptr, 0);
  Fixture F({1, 0, 0});
  prof::Data NoValues{
      0, 0,  nullptr, nullptr, nullptr, nullptr, /*Values=*/nullptr,
      0, {}, 0,       0};
  __orc_pgo_instrument_target(1, &NoValues, 0);
  OrcPgoFunctionRecord R = F.Rec;
  R.Data = &NoValues;
  EXPECT_TRUE(readValueSite(R, IPVK_IndirectCallTarget, 0).empty());
  EXPECT_FALSE(valueNodePoolExhausted());
}

TEST(ValueProfRuntime, ConcurrentCallsShareOneNode) {
  usePool(8);
  Fixture F({1, 0, 0});
  std::vector<std::thread> Threads;
  for (int T = 0; T < 4; ++T)
    Threads.emplace_back([&] {
      for (int I = 0; I < 10000; ++I)
        __orc_pgo_instrument_target(5, &F.D, 0);
    });
  for (std::thread &T : Threads)
    T.join();
  Values V = F.site(IPVK_IndirectCallTarget, 0);
  ASSERT_EQ(V.size(), 1u);
  EXPECT_EQ(V[0].first, 5u);
  // compiler-rt updates counts non-atomically, so some may be lost.
  EXPECT_GE(V[0].second, 1u);
  EXPECT_LE(V[0].second, 40000u);
}

} // namespace
