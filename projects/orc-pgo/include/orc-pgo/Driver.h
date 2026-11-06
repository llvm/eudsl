// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Linker;
class Module;
} // namespace llvm

namespace orc_pgo {

/// The orc-pgo executable: clang's driver with in-process cc1. Link
/// invocations build the instrumented program (see Driver.cpp). Argv[0] is the
/// program path.
int driverMain(llvm::ArrayRef<const char *> Argv);

/// Argv with orc-pgo's defaults inserted right after Argv[0], so later user
/// arguments override them:
/// - --driver-mode=g++, unless a --driver-mode= is given;
/// - -flto=full;
/// - -isysroot DefaultSysroot, if DefaultSysroot is non-empty and there is no
///   -isysroot, --sysroot or SDKROOT.
llvm::SmallVector<const char *, 0>
augmentArgs(llvm::ArrayRef<const char *> Argv, const char *DefaultSysroot);

/// Links Src into L's module. llvm::Linker's errors become the returned Error;
/// its warnings are printed. The context's diagnostic handler is restored.
llvm::Error linkModule(llvm::Linker &L, std::unique_ptr<llvm::Module> Src);

/// Merges the bitcode files at Paths into one module, then names its anonymous
/// globals.
llvm::Expected<std::unique_ptr<llvm::Module>>
linkBitcodeFiles(llvm::ArrayRef<std::string> Paths, llvm::LLVMContext &Ctx);

/// The global symbols of a static archive's symbol table, each once, with one
/// leading GlobalPrefix (if non-NUL) removed, i.e. as IR names. A universal
/// Mach-O file is read through its ArchName slice.
llvm::Expected<std::vector<std::string>>
archiveGlobalSymbols(llvm::MemoryBufferRef Archive, llvm::StringRef ArchName,
                     char GlobalPrefix);

} // namespace orc_pgo
