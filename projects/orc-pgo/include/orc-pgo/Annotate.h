// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class MemoryBuffer;
class Module;
} // namespace llvm

namespace orc_pgo {

/// Runs LLVM's PGOInstrumentationUse on M with Profile (an indexed profile)
/// served from memory (LLVM sets the profile summary itself). Fails on a
/// profile error and on a diagnostic that the profile does not match the IR (a
/// CFG hash or counter count mismatch), and if a function named in
/// MustBeAnnotated ends up without an entry count: hash mismatches of
/// functions in a comdat or with weak/available_externally linkage are not
/// diagnosed by LLVM. Other profile warnings, like a profile partially ignored
/// for lack of a return path, are not failures. The context's diagnostic
/// handler is restored afterwards, but its respect-filters setting, which
/// LLVMContext cannot report, is reset to false.
llvm::Error
annotateWithProfile(llvm::Module &M, const llvm::MemoryBuffer &Profile,
                    llvm::ArrayRef<llvm::StringRef> MustBeAnnotated = {});

} // namespace orc_pgo
