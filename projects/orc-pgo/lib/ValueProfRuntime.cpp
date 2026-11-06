// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A port of compiler-rt's value profiler (lib/profile/InstrProfilingValue.c)
// for statically allocated value sites, which is what lowering emits on Darwin
// and Linux (-vp-static-alloc). Nodes come from the image's static pool, then
// from a runtime pool of the same size as compiler-rt's.

#include "orc-pgo/ValueProfRuntime.h"
#include "orc-pgo/ProfData.h"

// InstrProfGetRangeRepValue, included the way compiler-rt's
// InstrProfilingValue.c includes it.
#define INSTR_PROF_VALUE_PROF_MEMOP_API
#include "llvm/ProfileData/InstrProfData.inc"

#include <algorithm>
#include <atomic>
#include <cstdint>

using namespace llvm;

namespace orc_pgo {

namespace {

// compiler-rt's INSTR_PROF_DEFAULT_NUM_VAL_PER_SITE: the values tracked per
// site when nodes come from the static pool.
constexpr uint32_t MaxValuesPerSite = 24;

// A bump-allocated range of nodes, compiler-rt's CurrentVNode/EndVNode.
struct NodePool {
  std::atomic<prof::ValueProfNode *> Current{nullptr};
  std::atomic<prof::ValueProfNode *> End{nullptr};

  prof::ValueProfNode *allocate() {
    prof::ValueProfNode *Cur = Current.load(std::memory_order_relaxed);
    prof::ValueProfNode *Last = End.load(std::memory_order_relaxed);
    // Early check, so a full pool's cursor never wraps.
    if (!Cur || Cur + 1 > Last)
      return nullptr;
    prof::ValueProfNode *Node = Current.fetch_add(1);
    // LCOV_EXCL_START only reachable when threads race past the early check
    if (Node + 1 > Last)
      return nullptr;
    // LCOV_EXCL_STOP
    return Node;
  }
};

// compiler-rt's lprofValueProfNodes: a shared pool in addition to the nodes
// the compiler allocated statically. compiler-rt places it in the same
// section as those nodes so that [begin, end) covers both; here the image's
// range is the first pool and this array the second.
prof::ValueProfNode RuntimeNodes[RuntimeValueNodePoolSize];

NodePool ImagePool;
NodePool RuntimePool{{RuntimeNodes}, {RuntimeNodes + RuntimeValueNodePoolSize}};
std::atomic<bool> Exhausted{false};

prof::ValueProfNode *allocateOneNode() {
  if (prof::ValueProfNode *Node = ImagePool.allocate())
    return Node;
  if (prof::ValueProfNode *Node = RuntimePool.allocate())
    return Node;
  Exhausted.store(true, std::memory_order_relaxed);
  return nullptr;
}

void instrumentValue(uint64_t TargetValue, void *DataPtr,
                     uint32_t CounterIndex) {
  auto *D = static_cast<prof::Data *>(DataPtr);
  if (!D || !D->Values)
    return;
  auto **Heads = static_cast<prof::ValueProfNode **>(D->Values);
  prof::ValueProfNode *Prev = nullptr;
  prof::ValueProfNode *MinNode = nullptr;
  prof::ValueProfNode *Cur = Heads[CounterIndex];
  uint64_t MinCount = UINT64_MAX;
  uint32_t NumValues = 0;
  while (Cur) {
    if (Cur->Value == TargetValue) {
      ++Cur->Count;
      return;
    }
    if (Cur->Count < MinCount) {
      MinCount = Cur->Count;
      MinNode = Cur;
    }
    Prev = Cur;
    Cur = Cur->Next;
    ++NumValues;
  }

  if (NumValues >= MaxValuesPerSite) {
    // compiler-rt's eviction: decay the coldest value, and replace it once its
    // count has dropped to the new value's, so hot targets are sticky.
    if (MinNode->Count <= 1) {
      MinNode->Value = TargetValue;
      MinNode->Count = 1;
    } else {
      --MinNode->Count;
    }
    return;
  }

  Cur = allocateOneNode();
  if (!Cur)
    return;
  Cur->Value = TargetValue;
  Cur->Count = 1;
  // A node that loses the race stays unused in the pool, as in compiler-rt.
  if (!Heads[CounterIndex])
    __sync_bool_compare_and_swap(&Heads[CounterIndex], nullptr, Cur);
  else if (Prev && !Prev->Next)
    __sync_bool_compare_and_swap(&Prev->Next, nullptr, Cur);
}

} // namespace

void setValueNodePool(void *Begin, void *End) {
  // Nodes already handed out from the runtime pool are not reused except by a
  // second call to this function, which starts both pools afresh.
  prof::ValueProfNode *Used =
      std::min(RuntimePool.Current.load(), RuntimePool.End.load());
  std::fill(RuntimeNodes, Used, prof::ValueProfNode{});
  RuntimePool.Current.store(RuntimeNodes);
  ImagePool.End.store(static_cast<prof::ValueProfNode *>(End));
  ImagePool.Current.store(static_cast<prof::ValueProfNode *>(Begin));
  Exhausted.store(false);
}

bool valueNodePoolExhausted() { return Exhausted.load(); }

std::vector<InstrProfValueData> readValueSite(const OrcPgoFunctionRecord &R,
                                              uint32_t Kind, uint32_t Site) {
  uint32_t Index = Site;
  for (uint32_t K = 0; K < Kind; ++K)
    Index += R.NumValueSites[K];
  std::vector<InstrProfValueData> Out;
  auto *D = static_cast<const prof::Data *>(R.Data);
  auto *const *Heads = static_cast<prof::ValueProfNode *const *>(D->Values);
  if (!Heads)
    return Out;
  for (const prof::ValueProfNode *N = Heads[Index]; N; N = N->Next)
    Out.push_back({N->Value, N->Count});
  return Out;
}

} // namespace orc_pgo

extern "C" {
void __orc_pgo_instrument_target(uint64_t Value, void *Data,
                                 uint32_t CounterIndex) {
  orc_pgo::instrumentValue(Value, Data, CounterIndex);
}

void __orc_pgo_instrument_memop(uint64_t Value, void *Data,
                                uint32_t CounterIndex) {
  orc_pgo::instrumentValue(InstrProfGetRangeRepValue(Value), Data,
                           CounterIndex);
}
}
