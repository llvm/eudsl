// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace orc_pgo {

/// Bitcode for M with use-list order preserved, so a module parsed from it
/// iterates uses (and so computes analyses) exactly like every other parse.
std::string writeSnapshot(const llvm::Module &M);

/// Parses (and fully materializes) a snapshot written by writeSnapshot.
llvm::Expected<std::unique_ptr<llvm::Module>>
parseSnapshot(llvm::StringRef Bytes, llvm::LLVMContext &Ctx);

} // namespace orc_pgo
