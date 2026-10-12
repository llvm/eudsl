// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <optional>

namespace llvm {
class Module;
} // namespace llvm

namespace orc_pgo {

enum class ArchiveContents { Native, Bitcode };

/// Whether a static archive's members are native objects or bitcode. An
/// empty archive is Native; one that mixes both is an error.
llvm::Expected<ArchiveContents> classifyArchive(llvm::MemoryBufferRef Archive);

/// classifyArchive for a library that the linker finds through -l: one that
/// orc-pgo can't read and that has no bitcode members (a member that is not an
/// object file, a damaged archive) may still be fine for the system linker,
/// so it is not an error but std::nullopt. Archives with bitcode members are
/// orc-pgo's to read, and their errors stay errors.
llvm::Expected<std::optional<ArchiveContents>>
classifyLibrary(llvm::MemoryBufferRef Library);

/// `orc-pgo --orc-pgo-ar {create|append} <archive> [<member>...]`: CMake's
/// archive rules. create writes a new archive; append adds the members to an
/// existing one. Archives with bitcode members get no symbol table (only
/// orc-pgo reads them). Returns the exit code.
int runArchiver(llvm::ArrayRef<const char *> Args);

/// An archive to read members from.
struct ArchiveInput {
  llvm::MemoryBufferRef Buffer;
  /// Load every member, as -force_load and --whole-archive do, not only the
  /// ones that define something undefined.
  bool Whole;
  ArchiveContents Contents;

  ArchiveInput(llvm::MemoryBufferRef Buffer, bool Whole = false,
               ArchiveContents Contents = ArchiveContents::Bitcode)
      : Buffer(Buffer), Whole(Whole), Contents(Contents) {}
};

/// Links into Dest the bitcode archive members a static linker would load:
/// scanning Archives in the order given, the first member that defines a
/// symbol still undefined, until none does. A symbol is undefined if Dest, a
/// loaded native object or a loaded native archive member references it
/// strongly and nothing defines it; main, which the C runtime references,
/// counts as referenced. Weak references load nothing. The members of native
/// archives (Contents) take part like those of bitcode ones, but are only
/// scanned for symbols, not linked. Whole archives are loaded first. Returns
/// the number of bitcode members linked.
llvm::Expected<unsigned>
linkArchiveMembers(llvm::Module &Dest, llvm::ArrayRef<ArchiveInput> Archives,
                   llvm::ArrayRef<llvm::MemoryBufferRef> NativeObjects);

} // namespace orc_pgo
