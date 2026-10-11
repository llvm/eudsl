// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace orc_pgo {

/// The hidden LLVM option configurePgoOptions sets to remember it has run. Not
/// for users: checkPgoOptionConflicts rejects it.
inline constexpr llvm::StringLiteral ReservedOptionName = "orc-pgo-configured";

/// Sets the LLVM PGO options orc-pgo relies on (entry-block instrumentation,
/// select and value profiling) by passing them to cl::ParseCommandLineOptions,
/// after any options given before, which they override. Both the build tool
/// and the runtime call this, so instrumentation and annotation agree. Later
/// calls do nothing until cl::ResetAllOptionOccurrences, which restores the
/// defaults. Fails if the parse does, e.g. in a program with a required
/// positional cl::opt.
llvm::Error configurePgoOptions();

/// Fails if LLVMArgs, arguments for cl::ParseCommandLineOptions that parsed
/// (such as the -mllvm values), give one of the options configurePgoOptions
/// sets a different value, or use ReservedOptionName.
llvm::Error checkPgoOptionConflicts(llvm::ArrayRef<std::string> LLVMArgs);

/// Hash of the arguments configurePgoOptions parses; recorded in the image so
/// the runtime can check it was built with the same settings.
uint64_t pgoOptionsFingerprint();

} // namespace orc_pgo
