// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Stats.h"
#include "orc-pgo/Runtime.h"

extern "C" void __orc_pgo_get_stats(OrcPgoStats *Out) {
  orc_pgo::RuntimeStats S = orc_pgo::processRuntimeStats();
  Out->TieredUp = S.TieredUp;
  Out->Failed = S.Failed;
  Out->PromotedIndirectCalls = S.PromotedIndirectCalls;
}
