// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>

namespace orc_pgo::test {

/// Initializes the native target, asm printer and asm parser once.
void initLLVM();

/// Parses textual IR; aborts the test binary with the diagnostic on error.
std::unique_ptr<llvm::Module> parseIR(llvm::LLVMContext &Ctx, llvm::StringRef Text);

/// TargetMachine for the host, matching LLJIT's default data layout.
std::unique_ptr<llvm::TargetMachine> hostTargetMachine();

std::unique_ptr<llvm::orc::LLJIT> makeJIT();

/// Parses IR into a fresh context, applies Transform, and adds it to a new JIT.
std::unique_ptr<llvm::orc::LLJIT>
jitIR(llvm::StringRef IR, llvm::function_ref<void(llvm::Module &)> Transform = {});

template <typename Fn> Fn *lookupFn(llvm::orc::LLJIT &J, llvm::StringRef Name) {
  return llvm::cantFail(J.lookup(Name)).toPtr<Fn *>();
}

} // namespace orc_pgo::test
