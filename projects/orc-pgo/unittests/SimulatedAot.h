// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TestSupport.h"
#include "orc-pgo/Image.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

#include <memory>
#include <string>

namespace orc_pgo::test {

/// A host function or variable the simulated program links against.
struct HostSymbol {
  const char *Name;
  const void *Address;
};

/// A program built by buildAotModule and loaded into a JIT: its static
/// constructors have run, main has not.
struct SimulatedAot {
  std::unique_ptr<llvm::orc::LLJIT> J;
  /// What the registration constructor passed to __orc_pgo_register_image.
  const OrcPgoImage *Image = nullptr;

  template <typename T>
  T *sym(llvm::StringRef Name) {
    return lookupSym<T>(*J, Name);
  }
  /// The record with this IR PGO name; fails the test if there is none.
  const OrcPgoFunctionRecord &record(llvm::StringRef PgoName) const;
  /// The dispatch slot of the function with this IR PGO name; fails the test
  /// if there is none.
  unsigned slot(llvm::StringRef PgoName) const;
  /// The symbol table entry with this name, or null.
  const OrcPgoSymbol *symbol(llvm::StringRef Name) const;
};

/// Runs buildAotModule on IR for the host, then JITs and initializes the
/// result as described above. Installs the image's value-node pool with
/// setValueNodePool (after resetValueNodePools), as __orc_pgo_start does.
SimulatedAot simulateAot(llvm::StringRef IR,
                         llvm::ArrayRef<std::string> ExtraSymbols = {},
                         llvm::ArrayRef<HostSymbol> HostSymbols = {});

/// The bitcode of Image's snapshot after Edit has changed the parsed module:
/// what a runtime that tiers up from it sees, while the program's AOT code
/// stays as built.
std::string editedSnapshot(const OrcPgoImage &Image,
                           llvm::function_ref<void(llvm::Module &)> Edit);

/// Image with Bitcode, which must outlive it, as its snapshot.
OrcPgoImage withSnapshot(const OrcPgoImage &Image, const std::string &Bitcode);

/// How many times simulated programs have called __orc_pgo_start.
unsigned simulatedStartCalls();

} // namespace orc_pgo::test
