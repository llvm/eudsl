// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <vector>

namespace llvm {
class Function;
class Instruction;
class Module;
} // namespace llvm

namespace orc_pgo {

inline constexpr llvm::StringLiteral InternalPrefix = "__orc_pgo";

bool isInternalName(llvm::StringRef Name);

/// Number of counters for a profiled instruction (one per successor/arm), or
/// 0 if the instruction is not profiled:
///   - conditional br: 2;
///   - switch with more than one successor: one per successor slot (default
///     first, then each case), even when several slots share a destination
///     block, since switch branch_weights carry one weight per slot;
///   - select with a scalar i1 condition: 2.
/// Everything else is 0, including vector-condition selects and the other
/// multi-successor terminators (invoke, indirectbr, callbr). Profiled sites are
/// counted just before they execute, from the condition that picks the
/// successor; these have no such condition (an invoke unwinds only if the
/// callee throws, indirectbr/callbr pick their target at run time), so they
/// would need counters in their successors instead. Little is lost: unwind
/// edges are cold by default and indirectbr/callbr are rare.
unsigned numSiteCounters(const llvm::Instruction &I);

struct CounterSite {
  llvm::Instruction *Inst;
  uint64_t FirstCounter;
  unsigned NumCounters;
};

struct FunctionCounters {
  llvm::Function *F;
  /// Position in CounterLayout::functions().
  unsigned FunctionIndex;
  uint64_t EntryCounter;
  /// In instruction order; counters follow EntryCounter contiguously.
  std::vector<CounterSite> Sites;
};

/// Positional counter assignment. Functions are visited in module order,
/// skipping declarations, available_externally definitions and InternalPrefix
/// names; each gets an entry counter followed by its sites' counters.
///
/// Indices depend only on function order, instruction order and these skip
/// rules, so computing the layout on two copies of the same IR yields identical
/// indices. That is what lets two copies (e.g. one instrumented and one kept
/// pristine) agree on counter positions without hashes or name tables. The
/// flip side: any IR difference between the copies silently misaligns every
/// later counter.
///
/// The module must be fully materialized (an unmaterialized body would count
/// as a definition with no sites).
class CounterLayout {
public:
  static CounterLayout compute(llvm::Module &M);

  llvm::ArrayRef<FunctionCounters> functions() const { return Functions; }
  /// nullptr for functions the layout skips (or from another module).
  const FunctionCounters *lookup(const llvm::Function &F) const;
  uint64_t numCounters() const { return NumCounters; }

private:
  std::vector<FunctionCounters> Functions;
  llvm::DenseMap<const llvm::Function *, unsigned> Index;
  uint64_t NumCounters = 0;
};

} // namespace orc_pgo
