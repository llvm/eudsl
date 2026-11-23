// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C ABI for programs built by orc-pgo to read the runtime's statistics.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OrcPgoStats {
  /// Functions whose dispatch slot now points at JIT'd code.
  uint64_t TieredUp;
  /// Tier-ups that failed; those functions keep their AOT body.
  uint64_t Failed;
  /// Indirect calls that PGOIndirectCallPromotion promoted in tier-up code.
  uint64_t PromotedIndirectCalls;
} OrcPgoStats;

/// Totals over every image in the process. All zero before __orc_pgo_start,
/// and when the runtime is disabled.
void __orc_pgo_get_stats(OrcPgoStats *Out);

#ifdef __cplusplus
}
#endif
