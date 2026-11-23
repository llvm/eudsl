// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace llvm {
class LLVMContext;
namespace cl {
class LibraryOptionsContext;
} // namespace cl
} // namespace llvm

namespace orc_pgo {

/// Parses the LLVM PGO options orc-pgo relies on (entry-block instrumentation,
/// select and value profiling) into Opts, after any options parsed into it
/// before, which they override. They are library options, which passes read
/// from their LLVMContext, so Opts is a context's getLibraryOptions(), or what
/// contexts then assign, as clang's CodeGenAction does with the -mllvm values
/// CompilerInstance parsed. Both the build tool and the runtime parse them, so
/// instrumentation and annotation agree. Fails if the parse does, e.g. in a
/// program with a required positional cl::opt.
llvm::Error parsePgoOptions(llvm::cl::LibraryOptionsContext &Opts);

/// Fails unless the options parsePgoOptions parses are the last ones parsed
/// into Ctx's library options, so that passes in Ctx run under them.
llvm::Error checkPgoOptions(const llvm::LLVMContext &Ctx);

/// Fails if LLVMArgs, arguments for cl::ParseCommandLineOptions that parsed
/// (such as the -mllvm values), give one of the options parsePgoOptions sets a
/// different value.
llvm::Error checkPgoOptionConflicts(llvm::ArrayRef<std::string> LLVMArgs);

/// Hash of the arguments parsePgoOptions parses; recorded in the image so
/// the runtime can check it was built with the same settings.
uint64_t pgoOptionsFingerprint();

} // namespace orc_pgo
