// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace llvm {
class MemoryBuffer;
class Module;
class TargetMachine;
} // namespace llvm

namespace orc_pgo {

inline constexpr llvm::StringLiteral TierUpSuffix = ".orc_pgo.tier";

struct TierUpBatch {
  /// The tier-up symbol of each root, in order.
  std::vector<std::string> Names;
  /// Indirect call sites PGOIndirectCallPromotion promoted in the batch's
  /// definitions (roots and callee copies): promotions, not executions.
  unsigned PromotedIndirectCalls = 0;
};

/// Turns M, a fresh parse of the snapshot, into the module for one tier-up
/// batch:
/// 1. annotates all of M with Profile; every root must get an entry count;
/// 2. runs indirect-call promotion over all of M;
/// 3. reduces M to the roots and their callees:
///    - each root stays an external definition, renamed <name>.orc_pgo.tier.
///      Calls to it go to that definition; every other use (a promoted
///      indirect call's compare, an address in a table) keeps the original
///      name, a declaration resolved to the program's thunk;
///    - direct callees up to InlineDepth calls away become available_externally
///      copies when their bodies are recreatable and not interposable (not
///      weak);
///    - constant, non-thread-local, non-interposable variables with
///      recreatable initializers become available_externally copies;
///    - every other global becomes an external declaration, resolved against
///      the program at JIT-link time;
///    - module inline asm is dropped: the program's object already has it.
/// Roots are IR PGO names (OrcPgoFunctionRecord::Name).
llvm::Expected<TierUpBatch>
prepareTierUpModule(llvm::Module &M, const llvm::MemoryBuffer &Profile,
                    llvm::ArrayRef<llvm::StringRef> Roots,
                    unsigned InlineDepth);

/// The default O3 pipeline, with clang's pipeline tuning options
/// (clangPipelineTuningOptions) and without PGOOptions (the profile is already
/// in M's metadata), with PGOMemOPSizeOpt added to its function simplification
/// pipeline, which is where clang's -fprofile-use pipeline has it.
void optimizeTierUp(llvm::Module &M, llvm::TargetMachine &TM);

/// Direct binding. Tiered holds the tier-up symbols (<name>.orc_pgo.tier) that
/// earlier batches defined in the JIT. A call in M to a function whose tier-up
/// symbol is in Tiered is redirected to a declaration of that symbol, so it no
/// longer goes through the function's thunk and dispatch slot. Only uses as a
/// call's callee are redirected: an address (a compare, a table entry, an
/// argument) stays the original function's, as for roots in
/// prepareTierUpModule. Run it after optimizeTierUp, so that the callees can
/// still be inlined. Returns the number of calls redirected.
unsigned bindTieredCalls(llvm::Module &M, const llvm::StringSet<> &Tiered);

} // namespace orc_pgo
