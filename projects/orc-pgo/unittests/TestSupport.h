// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <memory>

namespace orc_pgo::test {

/// Aborts the test binary with the diagnostic if Text is invalid IR.
std::unique_ptr<llvm::Module> parseIR(llvm::LLVMContext &Ctx, llvm::StringRef Text);

} // namespace orc_pgo::test
