// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/InstrProf.h"

#include <cstdint>

namespace orc_pgo {

inline constexpr llvm::StringLiteral FunctionTableName = "__orc_pgo_functions";
inline constexpr llvm::StringLiteral NumFunctionsName =
    "__orc_pgo_num_functions";
/// [2 x ptr] {begin, end} of the value-node pool lowering emits, or nulls.
inline constexpr llvm::StringLiteral VNodePoolName = "__orc_pgo_vnodes";
inline constexpr unsigned NumValueKinds = llvm::IPVK_Last + 1;

/// One instrumented function, as recorded by lowerPgo.
struct OrcPgoFunctionRecord {
  /// IR PGO name (getIRPGOObjectName), NUL-terminated.
  const char *Name;
  uint64_t CfgHash;
  /// Counters[0] is the entry count (-pgo-instrument-entry).
  uint64_t *Counters;
  /// The function's __profd record, as passed to the value-profiling callbacks.
  void *Data;
  /// The address indirect calls to the function observe, or null.
  const void *Address;
  uint32_t NumCounters;
  uint16_t NumValueSites[NumValueKinds];
};

static_assert(sizeof(OrcPgoFunctionRecord) == 56, "layout emitted by lowerPgo");

} // namespace orc_pgo
