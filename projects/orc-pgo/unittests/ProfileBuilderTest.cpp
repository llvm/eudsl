// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/ProfileBuilder.h"
#include "TestSupport.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/ProfData.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

#include <map>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

struct FakeFunction {
  std::vector<uint64_t> Counters;
  prof::ValueProfNode *Heads[4] = {};
  prof::Data D{0, 0, nullptr, nullptr, nullptr, nullptr, Heads, 0, {}, 0, 0};
};

OrcPgoFunctionRecord record(const char *Name, uint64_t Hash, FakeFunction &F,
                            const void *Address = nullptr,
                            uint16_t IndirectSites = 0,
                            uint16_t MemopSites = 0) {
  return {Name,
          Hash,
          F.Counters.data(),
          &F.D,
          Address,
          static_cast<uint32_t>(F.Counters.size()),
          {IndirectSites, MemopSites, 0}};
}

std::unique_ptr<IndexedInstrProfReader> readBack(const MemoryBuffer &Profile) {
  auto Reader = IndexedInstrProfReader::create(
      MemoryBuffer::getMemBufferCopy(Profile.getBuffer(), "profile"));
  EXPECT_EQ(test::errorText(Reader), "");
  return std::move(*Reader);
}

std::map<uint64_t, uint64_t> asMap(ArrayRef<InstrProfValueData> V) {
  std::map<uint64_t, uint64_t> M;
  for (const InstrProfValueData &D : V)
    M[D.Value] = D.Count;
  return M;
}

TEST(ProfileBuilder, WritesAnIrLevelEntryFirstProfile) {
  FakeFunction F{{4, 3}};
  OrcPgoFunctionRecord Records[] = {record("f", 0x1234, F)};
  auto Profile = buildIndexedProfile(Records);
  ASSERT_EQ(test::errorText(Profile), "");
  auto Reader = readBack(**Profile);
  EXPECT_TRUE(Reader->isIRLevelProfile());
  EXPECT_TRUE(Reader->instrEntryBBEnabled());
  EXPECT_FALSE(Reader->hasCSIRLevelProfile());
}

TEST(ProfileBuilder, RecordsTheLiveCountersOfEveryFunction) {
  FakeFunction F{{4, 3, 1}}, Cold{{0}};
  OrcPgoFunctionRecord Records[] = {record("f", 0x1234, F),
                                    record("p.c;cold", 0x55, Cold)};
  auto Profile = buildIndexedProfile(Records);
  ASSERT_EQ(test::errorText(Profile), "");
  auto Reader = readBack(**Profile);
  auto R = Reader->getInstrProfRecord("f", 0x1234);
  ASSERT_EQ(test::errorText(R), "");
  EXPECT_EQ(R->Counts, (std::vector<uint64_t>{4, 3, 1}));
  // Functions that never ran are kept: their zero counts make them cold.
  auto C = Reader->getInstrProfRecord("p.c;cold", 0x55);
  ASSERT_EQ(test::errorText(C), "");
  EXPECT_EQ(C->Counts, (std::vector<uint64_t>{0}));
  // The summary's max function count is the largest entry count (counter 0).
  EXPECT_EQ(Reader->getSummary(/*UseCS=*/false).getMaxFunctionCount(), 4u);
}

TEST(ProfileBuilder, MapsIndirectCallTargetsToNameHashes) {
  static prof::ValueProfNode Pool[8];
  resetValueNodePools();
  setValueNodePool(Pool, Pool + 8);
  int TargetA, TargetB; // stand-ins for two function addresses
  FakeFunction Caller{{6}}, A{{1}}, B{{1}};
  __orc_pgo_instrument_target(reinterpret_cast<uint64_t>(&TargetA), &Caller.D,
                              0);
  for (int I = 0; I < 4; ++I)
    __orc_pgo_instrument_target(reinterpret_cast<uint64_t>(&TargetB), &Caller.D,
                                0);
  OrcPgoFunctionRecord Records[] = {
      record("caller", 1, Caller, nullptr, /*IndirectSites=*/1),
      record("a", 2, A, &TargetA), record("b", 3, B, &TargetB)};
  auto Profile = buildIndexedProfile(Records);
  ASSERT_EQ(test::errorText(Profile), "");
  auto Reader = readBack(**Profile);
  auto R = Reader->getInstrProfRecord("caller", 1);
  ASSERT_EQ(test::errorText(R), "");
  ASSERT_EQ(R->getNumValueSites(IPVK_IndirectCallTarget), 1u);
  EXPECT_EQ(
      asMap(R->getValueArrayForSite(IPVK_IndirectCallTarget, 0)),
      (std::map<uint64_t, uint64_t>{{IndexedInstrProf::ComputeHash("a"), 1},
                                    {IndexedInstrProf::ComputeHash("b"), 4}}));
}

TEST(ProfileBuilder, KeepsMemopSizesAsRecorded) {
  static prof::ValueProfNode Pool[8];
  resetValueNodePools();
  setValueNodePool(Pool, Pool + 8);
  FakeFunction Copy{{3}};
  for (uint64_t N : {8, 8, 100})
    __orc_pgo_instrument_memop(N, &Copy.D, 0);
  OrcPgoFunctionRecord Records[] = {
      record("copy", 7, Copy, nullptr, 0, /*MemopSites=*/1)};
  auto Profile = buildIndexedProfile(Records);
  ASSERT_EQ(test::errorText(Profile), "");
  auto R = readBack(**Profile)->getInstrProfRecord("copy", 7);
  ASSERT_EQ(test::errorText(R), "");
  EXPECT_EQ(asMap(R->getValueArrayForSite(IPVK_MemOPSize, 0)),
            (std::map<uint64_t, uint64_t>{{8, 2}, {65, 1}}));
}

TEST(ProfileBuilder, RejectsConflictingRecordsForOneName) {
  FakeFunction F{{1}}, G{{1, 2}};
  OrcPgoFunctionRecord Records[] = {record("f", 1, F), record("f", 1, G)};
  auto Profile = buildIndexedProfile(Records);
  EXPECT_NE(test::errorText(Profile), "");
}

} // namespace
