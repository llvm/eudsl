// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <memory>

namespace orc_pgo::test {

/// Aborts the test binary with the diagnostic if Text is invalid IR.
std::unique_ptr<llvm::Module> parseIR(llvm::LLVMContext &Ctx, llvm::StringRef Text);

std::unique_ptr<llvm::orc::LLJIT> makeJIT();

/// Parses IR into a fresh context, applies Transform, and adds it to a new JIT.
std::unique_ptr<llvm::orc::LLJIT>
jitIR(llvm::StringRef IR, llvm::function_ref<void(llvm::Module &)> Transform = {});

template <typename Fn> Fn *lookupSym(llvm::orc::LLJIT &J, llvm::StringRef Name) {
  return llvm::cantFail(J.lookup(Name)).toPtr<Fn *>();
}

} // namespace orc_pgo::test
