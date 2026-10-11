// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace orc_pgo {

/// The instrumented program and the context that owns it.
struct AotModule {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> M;
};

/// Build steps 2-9 on M, the merged program. M itself only gets TM's triple
/// and data layout, names for its unnamed globals, and inferred function
/// attributes. The result is a
/// re-parse of M's snapshot in a new context, then:
/// - instrumented by Gen and lowered for TM, as clang -fprofile-generate does;
/// - optimized at O2;
/// - thunked;
/// - given the image, the symbol table, the registration constructor and the
///   __orc_pgo_start call in main.
/// ExtraSymbols are added to the symbol table: the global symbols of the
/// compiler-rt builtins archive, which codegen may call with no IR declaration.
llvm::Expected<AotModule>
buildAotModule(llvm::Module &M, llvm::TargetMachine &TM,
               llvm::ArrayRef<std::string> ExtraSymbols = {});

/// Native object code for M (build step 10).
llvm::Expected<llvm::SmallVector<char, 0>> emitObject(llvm::Module &M,
                                                      llvm::TargetMachine &TM);

/// buildAotModule, then emitObject.
llvm::Expected<llvm::SmallVector<char, 0>>
buildAotObject(llvm::Module &M, llvm::TargetMachine &TM,
               llvm::ArrayRef<std::string> ExtraSymbols = {});

} // namespace orc_pgo
