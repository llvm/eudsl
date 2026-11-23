// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "stats.h"

#include <cstdio>
#include <stdexcept>

#ifdef MULTI_TU_ORC_PGO
#include "orc-pgo/Stats.h"
extern "C" void __orc_pgo_poll(void);
#endif

namespace {

struct Negative final : Shape {
  long area() const override { return -1; }
  int kind() const override { return 3; }
};

long sumMulDiv() {
  long Sum = 0;
  for (long I = 1; I <= 20000; ++I)
    Sum += mulDiv(I, 3 * I, 7);
  return Sum;
}

} // namespace

int main() {
  std::vector<std::unique_ptr<Shape>> Shapes = makeShapes(1000);
  long Warm = accumulate(Shapes, 20);
  long WarmMulDiv = sumMulDiv();
#ifdef MULTI_TU_ORC_PGO
  // scaled and mulDiv have each been entered 20000 times.
  __orc_pgo_poll();
#endif
  long Hot = accumulate(Shapes, 20);
  long HotMulDiv = sumMulDiv();
  try {
    Negative N;
    scaled(N);
    std::puts("no exception");
  } catch (const std::runtime_error &E) {
    std::printf("caught: %s\n", E.what());
  }
  std::printf("warm=%ld hot=%ld muldiv=%ld/%ld created=%d\n", Warm, Hot,
              WarmMulDiv, HotMulDiv, createdCount());
#ifdef MULTI_TU_ORC_PGO
  OrcPgoStats S;
  __orc_pgo_get_stats(&S);
  std::fprintf(stderr,
               "orc-pgo-stats: tiered_up=%llu failed=%llu promoted=%llu\n",
               static_cast<unsigned long long>(S.TieredUp),
               static_cast<unsigned long long>(S.Failed),
               static_cast<unsigned long long>(S.PromotedIndirectCalls));
#endif
  return Warm == Hot && WarmMulDiv == HotMulDiv ? 0 : 1;
}
