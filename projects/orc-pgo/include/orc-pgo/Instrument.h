// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/StringRef.h"

namespace llvm {
class GlobalVariable;
class Module;
} // namespace llvm

namespace orc_pgo {
class CounterLayout;

inline constexpr llvm::StringLiteral CountersName = "__orc_pgo_counters";

/// Adds `__orc_pgo_counters` ([N x i64], hidden) and increments for every
/// counter in L. L must have been computed on M before calling this.
///
/// Each function's entry counter is bumped at the top of its entry block
/// (after leading allocas); each site's counter is bumped immediately before
/// the site instruction, with the index chosen by the same condition the
/// instruction tests. No edges are split and no blocks are added. Increments
/// are a monotonic atomic load, add and monotonic atomic store, not an atomic
/// read-modify-write, so concurrent increments may be lost.
llvm::GlobalVariable *instrumentModule(llvm::Module &M, const CounterLayout &L);
} // namespace orc_pgo
