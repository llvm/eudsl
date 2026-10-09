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

/// Prefix of every symbol orc-pgo itself emits or provides.
inline constexpr llvm::StringLiteral InternalPrefix = "__orc_pgo";

bool isInternalName(llvm::StringRef Name);

/// Number of counters for a profiled instruction (one per successor/arm), or
/// 0 if the instruction is not profiled.
unsigned numSiteCounters(const llvm::Instruction &I);

struct CounterSite {
  llvm::Instruction *Inst;
  uint64_t FirstCounter;
  unsigned NumCounters;
};

struct FunctionCounters {
  llvm::Function *F;
  unsigned FunctionIndex;
  uint64_t EntryCounter;
  std::vector<CounterSite> Sites;
};

/// Positional counter assignment. Computing it on two copies of the same IR
/// yields identical indices, which is what lets the runtime map the AOT
/// counters back onto the embedded snapshot without hashes or name tables.
class CounterLayout {
public:
  static CounterLayout compute(llvm::Module &M);

  llvm::ArrayRef<FunctionCounters> functions() const { return Functions; }
  const FunctionCounters *lookup(const llvm::Function &F) const;
  uint64_t numCounters() const { return NumCounters; }

private:
  std::vector<FunctionCounters> Functions;
  llvm::DenseMap<const llvm::Function *, unsigned> Index;
  uint64_t NumCounters = 0;
};

} // namespace orc_pgo
