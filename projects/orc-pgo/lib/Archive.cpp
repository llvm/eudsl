// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Archive.h"
#include "orc-pgo/Driver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace orc_pgo {

namespace {

/// An object file's global symbols, as IR names. Weak references are not
/// included: they don't make a static linker load anything.
struct NativeSymbols {
  std::vector<std::string> Defined;
  std::vector<std::string> Undefined;
};

struct Member {
  /// "<archive>(<member>)"
  std::string Name;
  /// A bitcode member's module; null for a native member, and once linked.
  std::unique_ptr<Module> M;
  /// A native member's symbols.
  NativeSymbols Symbols;
  std::vector<std::string> Defines;
  bool Whole = false;
  bool Loaded = false;
};

} // namespace

static Error forEachMember(const object::Archive &Archive,
                           function_ref<Error(MemoryBufferRef)> F) {
  Error Err = Error::success();
  for (const object::Archive::Child &C : Archive.children(Err)) {
    Expected<MemoryBufferRef> Buf = C.getMemoryBufferRef();
    if (!Buf)
      return Buf.takeError();
    if (Error E = F(*Buf))
      return E;
  }
  return Err;
}

Expected<ArchiveContents> classifyArchive(MemoryBufferRef Buf) {
  StringRef Id = Buf.getBufferIdentifier();
  Expected<std::unique_ptr<object::Archive>> Archive =
      object::Archive::create(Buf);
  if (!Archive)
    return createFileError(Id, Archive.takeError());
  unsigned NumBitcode = 0, NumNative = 0;
  Error E = forEachMember(**Archive, [&](MemoryBufferRef M) {
    ++(identify_magic(M.getBuffer()) == file_magic::bitcode ? NumBitcode
                                                            : NumNative);
    return Error::success();
  });
  if (E)
    return createFileError(Id, std::move(E));
  if (NumBitcode && NumNative)
    return createFileError(
        Id, createStringError("archive mixes native and bitcode members"));
  return NumBitcode ? ArchiveContents::Bitcode : ArchiveContents::Native;
}

int runArchiver(ArrayRef<const char *> Args) {
  auto Fail = [](Error E) {
    errs() << "orc-pgo: error: " << toString(std::move(E)) << '\n';
    return 1;
  };
  StringRef Mode = Args.empty() ? "" : Args[0];
  if (Args.size() < 2 || (Mode != "create" && Mode != "append")) {
    return Fail(createStringError(
        "usage: orc-pgo --orc-pgo-ar {create|append} <archive> [<member>...]"));
  }
  StringRef ArchivePath = Args[1];

  std::vector<NewArchiveMember> Members;
  std::unique_ptr<MemoryBuffer> OldBuf;
  std::unique_ptr<object::Archive> Old;
  if (Mode == "append") {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
        MemoryBuffer::getFile(ArchivePath);
    if (!Buf)
      return Fail(createFileError(ArchivePath, Buf.getError()));
    OldBuf = std::move(*Buf);
    Expected<std::unique_ptr<object::Archive>> A =
        object::Archive::create(OldBuf->getMemBufferRef());
    if (!A)
      return Fail(createFileError(ArchivePath, A.takeError()));
    Old = std::move(*A);
    Error Err = Error::success();
    for (const object::Archive::Child &C : Old->children(Err)) {
      Expected<NewArchiveMember> M =
          NewArchiveMember::getOldMember(C, /*Deterministic=*/true);
      if (!M)
        return Fail(createFileError(ArchivePath, M.takeError()));
      Members.push_back(std::move(*M));
    }
    if (Err)
      return Fail(createFileError(ArchivePath, std::move(Err)));
  }
  for (const char *Path : Args.drop_front(2)) {
    Expected<NewArchiveMember> M =
        NewArchiveMember::getFile(Path, /*Deterministic=*/true);
    if (!M)
      return Fail(createFileError(Path, M.takeError()));
    // As llvm-ar: members are named by file name, not by the path given.
    M->MemberName = sys::path::filename(M->MemberName);
    Members.push_back(std::move(*M));
  }

  bool AnyBitcode = any_of(Members, [](const NewArchiveMember &M) {
    return identify_magic(M.Buf->getBuffer()) == file_magic::bitcode;
  });
  // OldBuf backs the old members; writeArchive keeps it alive while writing.
  if (Error E = writeArchive(ArchivePath, Members,
                             AnyBitcode ? SymtabWritingMode::NoSymtab
                                        : SymtabWritingMode::NormalSymtab,
                             object::Archive::getDefaultKind(),
                             /*Deterministic=*/true,
                             /*Thin=*/false, std::move(OldBuf))) {
    return Fail(std::move(E));
  }
  return 0;
}

static Expected<NativeSymbols> nativeSymbols(MemoryBufferRef Buf) {
  Expected<std::unique_ptr<object::ObjectFile>> Obj =
      object::ObjectFile::createObjectFile(Buf);
  if (!Obj)
    return createFileError(Buf.getBufferIdentifier(), Obj.takeError());
  NativeSymbols Out;
  // Mach-O prefixes C symbols with '_'. Other names are \1-escaped in IR.
  StringRef Prefix = (*Obj)->isMachO() ? "_" : "";
  for (const object::SymbolRef &Sym : (*Obj)->symbols()) {
    Expected<uint32_t> Flags = Sym.getFlags();
    Expected<StringRef> Name = Sym.getName();
    if (Error E = joinErrors(Flags.takeError(), Name.takeError()))
      return createFileError(Buf.getBufferIdentifier(), std::move(E));
    if (!(*Flags & object::SymbolRef::SF_Global))
      continue;
    std::string IRName = Name->starts_with(Prefix)
                             ? Name->drop_front(Prefix.size()).str()
                             : ("\1" + *Name).str();
    if (!(*Flags & object::SymbolRef::SF_Undefined))
      Out.Defined.push_back(std::move(IRName));
    else if (!(*Flags & object::SymbolRef::SF_Weak))
      Out.Undefined.push_back(std::move(IRName));
  }
  return std::move(Out);
}

static bool hasBitcodeMembers(MemoryBufferRef Buf) {
  Expected<std::unique_ptr<object::Archive>> Archive =
      object::Archive::create(Buf);
  if (!Archive) {
    consumeError(Archive.takeError());
    return false;
  }
  bool Found = false;
  consumeError(forEachMember(**Archive, [&](MemoryBufferRef M) {
    Found |= identify_magic(M.getBuffer()) == file_magic::bitcode;
    return Error::success();
  }));
  return Found;
}

Expected<std::optional<ArchiveContents>> classifyLibrary(MemoryBufferRef Buf) {
  Expected<ArchiveContents> Kind = classifyArchive(Buf);
  if (!Kind) {
    if (hasBitcodeMembers(Buf))
      return Kind.takeError();
    consumeError(Kind.takeError());
    return std::nullopt;
  }
  if (*Kind == ArchiveContents::Native) {
    // classifyArchive has read the archive, so it parses.
    std::unique_ptr<object::Archive> Archive =
        cantFail(object::Archive::create(Buf));
    Error E = forEachMember(*Archive, [](MemoryBufferRef M) {
      return nativeSymbols(M).takeError();
    });
    if (E) {
      consumeError(std::move(E));
      return std::nullopt;
    }
  }
  return *Kind;
}

/// The symbols that the objects and archive members loaded so far define and
/// reference, other than the ones in Dest.
struct NativeLoaded {
  StringSet<> Undefined;
  StringSet<> Defined;

  void add(const NativeSymbols &Symbols) {
    for (const std::string &Name : Symbols.Defined)
      Defined.insert(Name);
    for (const std::string &Name : Symbols.Undefined)
      Undefined.insert(Name);
  }
};

static StringSet<> undefinedNames(const Module &Dest,
                                  const NativeLoaded &Native) {
  auto IsDefined = [&](StringRef Name) {
    const GlobalValue *GV = Dest.getNamedValue(Name);
    return Native.Defined.contains(Name) ||
           (GV && !GV->isDeclarationForLinker());
  };
  StringSet<> Out;
  for (const auto &Entry : Native.Undefined)
    if (!IsDefined(Entry.getKey()))
      Out.insert(Entry.getKey());
  if (!IsDefined("main"))
    Out.insert("main");
  for (const GlobalValue &GV : Dest.global_values()) {
    if (GV.isDeclarationForLinker() && !GV.hasLocalLinkage() &&
        !GV.hasExternalWeakLinkage() && !Native.Defined.contains(GV.getName()))
      Out.insert(GV.getName());
  }
  return Out;
}

Expected<unsigned> linkArchiveMembers(Module &Dest,
                                      ArrayRef<ArchiveInput> Archives,
                                      ArrayRef<MemoryBufferRef> NativeObjects) {
  NativeLoaded Native;
  for (MemoryBufferRef Obj : NativeObjects) {
    Expected<NativeSymbols> Symbols = nativeSymbols(Obj);
    if (!Symbols)
      return Symbols.takeError();
    Native.add(*Symbols);
  }

  // Every member is parsed up front: bitcode archives have no symbol table.
  std::vector<Member> Members;
  auto AddArchive = [&](const ArchiveInput &In) -> Error {
    const bool IsBitcode = In.Contents == ArchiveContents::Bitcode;
    StringRef Id = In.Buffer.getBufferIdentifier();
    Expected<std::unique_ptr<object::Archive>> Archive =
        object::Archive::create(In.Buffer);
    if (!Archive)
      return createFileError(Id, Archive.takeError());
    Error E = forEachMember(**Archive, [&](MemoryBufferRef MB) -> Error {
      Member P;
      P.Name = (Id + "(" + MB.getBufferIdentifier() + ")").str();
      P.Whole = In.Whole;
      if (IsBitcode) {
        Expected<std::unique_ptr<Module>> M =
            parseBitcodeFile(MB, Dest.getContext());
        if (!M)
          return createFileError(MB.getBufferIdentifier(), M.takeError());
        P.M = std::move(*M);
        for (const GlobalValue &GV : P.M->global_values())
          if (!GV.isDeclarationForLinker() && !GV.hasLocalLinkage())
            P.Defines.push_back(GV.getName().str());
      } else {
        Expected<NativeSymbols> Symbols = nativeSymbols(MB);
        if (!Symbols)
          return Symbols.takeError();
        P.Symbols = std::move(*Symbols);
        P.Defines = P.Symbols.Defined;
      }
      Members.push_back(std::move(P));
      return Error::success();
    });
    return E ? createFileError(Id, std::move(E)) : Error::success();
  };
  for (const ArchiveInput &In : Archives)
    if (Error E = AddArchive(In))
      return std::move(E);

  Linker L(Dest);
  unsigned Linked = 0;
  // Loads a member: bitcode is linked into Dest, and a native member's
  // references and definitions join the native ones.
  auto Load = [&](Member &P) -> Error {
    P.Loaded = true;
    if (!P.M) {
      Native.add(P.Symbols);
      return Error::success();
    }
    if (Error E = linkModule(L, std::move(P.M)))
      return createFileError(P.Name, std::move(E));
    ++Linked;
    return Error::success();
  };

  for (Member &P : Members)
    if (P.Whole)
      if (Error E = Load(P))
        return std::move(E);
  for (;;) {
    StringSet<> Undefined = undefinedNames(Dest, Native);
    auto Next = find_if(Members, [&](const Member &P) {
      return !P.Loaded && any_of(P.Defines, [&](const std::string &Name) {
        return Undefined.contains(Name);
      });
    });
    if (Next == Members.end())
      return Linked;
    if (Error E = Load(*Next))
      return std::move(E);
  }
}

} // namespace orc_pgo
