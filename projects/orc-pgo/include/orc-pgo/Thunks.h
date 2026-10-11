// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/ModRef.h"

#include <string>
#include <vector>

namespace llvm {
class Function;
class GlobalVariable;
class Module;
} // namespace llvm

namespace orc_pgo {

inline constexpr llvm::StringLiteral AotSuffix = ".orc_pgo.aot";
inline constexpr llvm::StringLiteral DispatchTableName = "__orc_pgo_dispatch";

/// What the source declared, by function name, of the two kinds of attribute
/// that dropInvalidatedAttributes undoes.
struct DeclaredAttributes {
  llvm::StringMap<llvm::MemoryEffects> Memory;
  llvm::StringSet<> NoSync;
};

/// Infers function attributes (nounwind, willreturn, norecurse, nofree,
/// noreturn, cold, ...) as the default pipeline does. FunctionAttrs infers
/// nothing for a noipa function, which is what prepareThunkCandidates makes of
/// every candidate, so this must run first. It must also run before the
/// snapshot is written: Gen and tier-up's Use build their spanning trees from
/// branch probabilities, which depend on noreturn and cold, so both must see
/// the same attributes. The attributes hold for either body behind a slot, the
/// AOT one and the tier-up one, which are compilations of the same source.
/// Returns what M declared before inference.
DeclaredAttributes inferFunctionAttributes(llvm::Module &M);

/// For the AOT module after runPgoGen: removes the inferred `memory` effects
/// and `nosync`, which Gen and thunkify make false (counters are written, and
/// a thunk's slot load is an acquire), restoring what Declared says. Tier-up
/// code has no counters and keeps them.
void dropInvalidatedAttributes(llvm::Module &M,
                               const DeclaredAttributes &Declared);

/// Whether F, in a module runPgoGen has instrumented and nothing has optimized
/// yet, can get a dispatch slot. F must be a definition Gen gave counters to,
/// with a recreatable body. It must not be available_externally, varargs or
/// naked, carry prefix or prologue data, or take inalloca, preallocated, byval
/// or inreg arguments (the AArch64 backend cannot forward the last two through
/// a musttail call, and the rule does not depend on the target).
bool isThunkEligible(const llvm::Function &F);

struct ThunkCandidate {
  /// Candidates are found before optimization and thunked after it. The handle
  /// does not keep F alive: it is cleared if optimization deletes F, and the
  /// candidate is then dropped by thunkify. (The AOT pipeline does keep every
  /// function, through the symbol table's use in llvm.compiler.used, but
  /// nothing here relies on that.)
  llvm::WeakVH Handle;
  /// The candidate, or null if it was deleted.
  llvm::Function *function() const {
    return llvm::cast_or_null<llvm::Function>(Handle);
  }
  /// IR PGO name of F's counters, from Gen's name variable.
  std::string PgoName;
};

/// The eligible functions of M, in module order. Must run after runPgoGen and
/// before lowerPgo or any optimization. Marks each candidate `noipa`, which
/// stops interprocedural optimization from using its body (see
/// inferFunctionAttributes for the attributes inferred beforehand).
std::vector<ThunkCandidate> prepareThunkCandidates(llvm::Module &M);

struct Thunk {
  /// The original function, now `load acquire dispatch[i]; musttail call`.
  llvm::Function *Entry;
  /// The original body, internal, named <F>.orc_pgo.aot.
  llvm::Function *Body;
  std::string PgoName;
};

struct ThunkTable {
  /// [N x ptr] named DispatchTableName; slot i initially holds Thunks[i].Body.
  llvm::GlobalVariable *Dispatch;
  std::vector<Thunk> Thunks;
};

/// Moves each candidate's body to <F>.orc_pgo.aot and turns F into a thunk
/// through its slot. Candidates that no longer exist get no slot.
ThunkTable thunkify(llvm::Module &M, llvm::ArrayRef<ThunkCandidate> Candidates);

} // namespace orc_pgo
