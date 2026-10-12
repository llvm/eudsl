// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The orc-pgo driver: clang's driver, with cc1 run in-process. A link
// invocation compiles its sources, merges every bitcode input, builds the
// instrumented AOT object with buildAotObject, and then links natively. The
// native link is re-planned by a second clang Driver over native inputs, plus
// the orc-pgo runtime, static LLVM and compiler-rt builtins.

#include "orc-pgo/Driver.h"
#include "orc-pgo/Archive.h"
#include "orc-pgo/Config.h"
#include "orc-pgo/PgoOptions.h"
#include "orc-pgo/Pipeline.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/Stack.h"
#include "clang/CodeGen/ObjectFilePCHContainerWriter.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Options/Options.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "clang/Serialization/PCHContainerOperations.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BuryPointer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/IOSandbox.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

using namespace llvm;
using clang::driver::Command;
using clang::driver::Compilation;
using clang::driver::Driver;
using clang::driver::InputInfo;
using clang::driver::JobList;

namespace orc_pgo {

namespace {

// Added to the native link of an instrumented program, around the runtime
// archives. On Linux, --no-whole-archive ends any --whole-archive of the
// user's, which must not apply to the runtime. compiler-rt builtins are
// already clang's only choice on Darwin. On Linux, -rtlib=compiler-rt alone
// would also drop the unwinder, so the dynamic libgcc_s is requested
// explicitly: libstdc++ needs it, and the JIT registers its frames with it.
#ifdef __APPLE__
constexpr const char *RuntimeLinkPrefix[] = {"-rtlib=compiler-rt"};
constexpr const char *RuntimeLinkSuffix[] = {"-Wl,-dead_strip"};
#else
constexpr const char *RuntimeLinkPrefix[] = {
    "-rtlib=compiler-rt", "-unwindlib=libgcc", "-Wl,--gc-sections",
    "-Wl,--no-whole-archive", "-Wl,--start-group"};
constexpr const char *RuntimeLinkSuffix[] = {"-Wl,--end-group"};
#endif

constexpr const char *RuntimeArchives[] = {
    ORC_PGO_RUNTIME_LIB, ORC_PGO_COMMON_LIB, ORC_PGO_LLVM_LINK_ARGS};

/// Collects llvm::Linker's errors and prints its warnings.
class LinkDiagnostics : public DiagnosticHandler {
public:
  explicit LinkDiagnostics(std::string &Errors) : Errors(Errors) {}

  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    std::string Text;
    raw_string_ostream OS(Text);
    DiagnosticPrinterRawOStream Printer(OS);
    DI.print(Printer);
    if (DI.getSeverity() == DS_Error)
      Errors += Text;
    else
      errs() << "orc-pgo: warning: " << Text << '\n';
    return true;
  }

private:
  std::string &Errors;
};

/// A static archive input.
struct LinkArchive {
  std::unique_ptr<MemoryBuffer> Buffer;
  ArchiveContents Contents;
  /// Linked in full: -force_load, -all_load, or between --whole-archive and
  /// --no-whole-archive.
  bool Whole;
};

/// An option and the index of one of its values.
using OptionValue = std::pair<const llvm::opt::Arg *, unsigned>;

/// The link job's inputs, by how the link step handles them.
struct LinkInputs {
  /// Merged into the AOT module.
  std::vector<std::string> Bitcode;
  /// In command-line order. The members of the bitcode ones are linked into
  /// the AOT module as needed; the references of the native ones decide which
  /// of those members are needed.
  std::vector<LinkArchive> Archives;
  /// Their references decide which archive members are needed.
  std::vector<std::unique_ptr<MemoryBuffer>> NativeObjects;
  /// Inputs that the AOT object replaces on the native link line.
  StringSet<> Replaced;
  /// Option values that name bitcode archives (-force_load), which the native
  /// link must not see.
  DenseSet<OptionValue> Dropped;
  /// -all_load is in effect. ld64 applies it to every archive, runtime
  /// archives included, so with an AOT object it is replaced by a -force_load
  /// of each native archive in Archives instead.
  bool AllLoad = false;

  bool hasBitcode() const {
    return !Bitcode.empty() || any_of(Archives, [](const LinkArchive &A) {
      return A.Contents == ArchiveContents::Bitcode;
    });
  }
};

} // namespace

static int reportError(Error E) {
  errs() << "orc-pgo: error: " << toString(std::move(E)) << '\n';
  return 1;
}

static void initializeNativeTarget() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
}

/// clang's cc1_main for the native target. Argv starts with "-cc1".
static int cc1Main(ArrayRef<const char *> Argv, const char *Argv0) {
  auto PCHOps = std::make_shared<clang::PCHContainerOperations>();
  PCHOps->registerWriter(
      std::make_unique<clang::ObjectFilePCHContainerWriter>());
  PCHOps->registerReader(
      std::make_unique<clang::ObjectFilePCHContainerReader>());
  initializeNativeTarget();

  // Argument-parsing diagnostics are buffered until the real engine exists.
  clang::DiagnosticOptions DiagOpts;
  auto *DiagsBuffer = new clang::TextDiagnosticBuffer;
  clang::DiagnosticsEngine Diags(clang::DiagnosticIDs::create(), DiagOpts,
                                 DiagsBuffer);
  auto Invocation = std::make_shared<clang::CompilerInvocation>();
  bool Success = clang::CompilerInvocation::CreateFromArgs(*Invocation, Argv,
                                                           Diags, Argv0);
  auto Clang = std::make_unique<clang::CompilerInstance>(std::move(Invocation),
                                                         std::move(PCHOps));
  clang::HeaderSearchOptions &HSOpts = Clang->getHeaderSearchOpts();
  if (HSOpts.UseBuiltinIncludes && HSOpts.ResourceDir.empty())
    HSOpts.ResourceDir = ORC_PGO_CLANG_RESOURCE_DIR;
  // As in cc1_main: the real file system is not itself sandboxed.
  auto VFS = [] {
    auto BypassSandbox = sys::sandbox::scopedDisable();
    return vfs::getRealFileSystem();
  }();
  Clang->createVirtualFileSystem(std::move(VFS), DiagsBuffer);
  Clang->createDiagnostics();
  DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
  if (!Success)
    return 1;

  Success = clang::ExecuteCompilerInvocation(Clang.get());
  if (Clang->getFrontendOpts().DisableFree) {
    // As in cc1_main: the consumer must be destroyed; the rest is leaked.
    Clang->getDiagnosticClient().~DiagnosticConsumer();
    BuryPointer(std::move(Clang));
  }
  return !Success;
}

/// Driver::CC1Main, and direct `orc-pgo -cc1...`: clang's ExecuteCC1Tool.
static int cc1Tool(SmallVectorImpl<const char *> &ArgV) {
  // cl::opts are process-global, and orc-pgo's build pipeline sets several.
  cl::ResetAllOptionOccurrences();
  StringRef Tool = ArgV[1];
  if (Tool != "-cc1") {
    errs() << "orc-pgo: error: " << Tool
           << " is not supported (orc-pgo compiles C and C++ sources only)\n";
    return 1;
  }
  return cc1Main(ArrayRef(ArgV).slice(1), ArgV[0]);
}

static Expected<std::unique_ptr<TargetMachine>>
createTargetMachine(const Triple &TT) {
  std::string Message;
  const llvm::Target *T = TargetRegistry::lookupTarget(TT, Message);
  if (!T)
    return createStringError("cannot target %s: %s", TT.str().c_str(),
                             Message.c_str());
  llvm::TargetOptions Options;
  // clang's default (-fuse-init-array); ignored for Mach-O.
  Options.UseInitArray = true;
  return std::unique_ptr<TargetMachine>(
      T->createTargetMachine(TT, "", "", Options, Reloc::PIC_));
}

Error linkModule(Linker &L, std::unique_ptr<llvm::Module> Src) {
  LLVMContext &Ctx = Src->getContext();
  std::string Errors;
  std::unique_ptr<DiagnosticHandler> Saved = Ctx.getDiagnosticHandler();
  Ctx.setDiagnosticHandler(std::make_unique<LinkDiagnostics>(Errors));
  bool Failed = L.linkInModule(std::move(Src));
  Ctx.setDiagnosticHandler(std::move(Saved));
  return Failed ? createStringError(Errors) : Error::success();
}

Expected<std::unique_ptr<llvm::Module>>
linkBitcodeFiles(ArrayRef<std::string> Paths, LLVMContext &Ctx) {
  // Like llvm-link: link everything into an empty module, which takes the
  // first input's triple and data layout.
  auto Merged = std::make_unique<llvm::Module>("orc-pgo", Ctx);
  Linker L(*Merged);
  for (const std::string &Path : Paths) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(Path);
    if (!Buf)
      return createFileError(Path, Buf.getError());
    Expected<std::unique_ptr<llvm::Module>> M =
        parseBitcodeFile((*Buf)->getMemBufferRef(), Ctx);
    if (!M)
      return createFileError(Path, M.takeError());
    if (Error E = linkModule(L, std::move(*M)))
      return createFileError(Path, std::move(E));
  }
  nameUnamedGlobals(*Merged);
  return std::move(Merged);
}

Expected<std::vector<std::string>> archiveGlobalSymbols(MemoryBufferRef Buf,
                                                        StringRef ArchName,
                                                        char GlobalPrefix) {
  std::unique_ptr<object::Archive> Archive;
  if (identify_magic(Buf.getBuffer()) == file_magic::macho_universal_binary) {
    Expected<std::unique_ptr<object::MachOUniversalBinary>> Fat =
        object::MachOUniversalBinary::create(Buf);
    if (!Fat)
      return Fat.takeError();
    Expected<std::unique_ptr<object::Archive>> Slice =
        (*Fat)->getArchiveForArch(ArchName);
    if (!Slice)
      return Slice.takeError();
    Archive = std::move(*Slice);
  } else {
    Expected<std::unique_ptr<object::Archive>> Plain =
        object::Archive::create(Buf);
    if (!Plain)
      return Plain.takeError();
    Archive = std::move(*Plain);
  }

  std::vector<std::string> Names;
  StringSet<> Seen;
  for (const object::Archive::Symbol &Sym : Archive->symbols()) {
    StringRef Name = Sym.getName();
    if (GlobalPrefix != '\0' && Name.starts_with(StringRef(&GlobalPrefix, 1)))
      Name = Name.drop_front();
    if (Seen.insert(Name).second)
      Names.push_back(Name.str());
  }
  return Names;
}

/// Applies the -mllvm options to orc-pgo's own pipeline, like clang's LTO.
/// The pipeline's configurePgoOptions then overrides the PGO options orc-pgo
/// needs, so -mllvm must not give them other values.
static Error applyLinkTimeLLVMOptions(const Compilation &C) {
  // Start from defaults: an earlier cc1 or build may have set options.
  cl::ResetAllOptionOccurrences();
  std::vector<std::string> Values =
      C.getArgs().getAllArgValues(clang::options::OPT_mllvm);
  SmallVector<const char *, 8> Argv{"orc-pgo"};
  for (const std::string &V : Values)
    Argv.push_back(V.c_str());
  std::string Message;
  raw_string_ostream OS(Message);
  if (!cl::ParseCommandLineOptions(Argv.size(), Argv.data(), "", &OS))
    return createStringError("invalid -mllvm option: %s", Message.c_str());
  return checkPgoOptionConflicts(Values);
}

// The suffixes of the shared libraries that the linker prefers to lib<name>.a
// in the same directory.
#ifdef __APPLE__
constexpr const char *SharedLibrarySuffixes[] = {".tbd", ".dylib"};
#else
constexpr const char *SharedLibrarySuffixes[] = {".so"};
#endif

/// The static archive that the linker takes for -l<Name>, or null if it takes
/// something else. In each -L directory in order, lib<Name>.a comes after the
/// shared library lib<Name>.so (.tbd, .dylib on Darwin), unless -static is
/// given, and the first match wins. The file may not be an archive orc-pgo can
/// read (a universal one, say): classifyLibrary decides. Not modeled: the
/// linker's default search directories, -l:<file>, -Bstatic and -Bdynamic
/// (positional), -search_dylibs_first.
static std::unique_ptr<MemoryBuffer> findLibrary(const Compilation &C,
                                                 StringRef Name) {
  const bool Static = C.getArgs().hasArg(clang::options::OPT_static);
  for (const std::string &Dir :
       C.getArgs().getAllArgValues(clang::options::OPT_L)) {
    SmallString<256> Path(Dir);
    sys::path::append(Path, "lib" + Name);
    SmallString<256> Base = Path;
    if (!Static) {
      for (const char *Suffix : SharedLibrarySuffixes) {
        if (sys::fs::exists(Base + Suffix))
          return nullptr;
      }
    }
    Path += ".a";
    if (ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
            MemoryBuffer::getFile(Path))
      return std::move(*Buf);
  }
  return nullptr;
}

/// Adds an archive once: when it's given again, it keeps its place and is
/// Whole if either is. Whole archives are linked in full, and so are all
/// archives with -all_load.
static void addArchive(LinkInputs &Inputs, std::unique_ptr<MemoryBuffer> Buffer,
                       ArchiveContents Contents, bool Whole) {
  Whole |= Inputs.AllLoad;
  for (LinkArchive &A : Inputs.Archives) {
    if (A.Buffer->getBufferIdentifier() == Buffer->getBufferIdentifier()) {
      A.Whole |= Whole;
      return;
    }
  }
  Inputs.Archives.push_back({std::move(Buffer), Contents, Whole});
}

/// Classifies the static archive in Buffer and adds it.
static Expected<ArchiveContents>
addArchiveInput(LinkInputs &Inputs, std::unique_ptr<MemoryBuffer> Buffer,
                bool Whole) {
  Expected<ArchiveContents> Kind = classifyArchive(Buffer->getMemBufferRef());
  if (!Kind)
    return Kind.takeError();
  addArchive(Inputs, std::move(Buffer), *Kind, Whole);
  return *Kind;
}

static bool isLinkerOption(const llvm::opt::Arg &A) {
  return A.getOption().matches(clang::options::OPT_Wl_COMMA) ||
         A.getOption().matches(clang::options::OPT_Xlinker);
}

/// An archive that the linker reads from an option value (-force_load's path,
/// or a file name in -Wl,): a bitcode one is linked here instead, so the
/// native link must not see the values Keys. Anything else is for the system
/// linker to read or report.
static Error addLinkerOptionArchive(LinkInputs &Inputs, StringRef Path,
                                    bool Whole, ArrayRef<OptionValue> Keys) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(Path);
  if (!Buf || identify_magic((*Buf)->getBuffer()) != file_magic::archive)
    return Error::success();
  Expected<ArchiveContents> Kind =
      addArchiveInput(Inputs, std::move(*Buf), Whole);
  if (!Kind)
    return Kind.takeError();
  if (*Kind == ArchiveContents::Bitcode)
    Inputs.Dropped.insert(Keys.begin(), Keys.end());
  return Error::success();
}

/// Reads the link job's inputs. The -l libraries are only looked at if
/// Libraries is set.
static Expected<LinkInputs>
readLinkInputs(const Compilation &C, const Command &Link, bool Libraries) {
  namespace options = clang::options;
  LinkInputs Inputs;
#ifdef __APPLE__
  for (const llvm::opt::Arg *A : C.getInputArgs()) {
    if (A->getOption().matches(options::OPT_all__load) ||
        (isLinkerOption(*A) && any_of(A->getValues(), [](StringRef V) {
           return V == "-all_load";
         }))) {
      Inputs.AllLoad = true;
    }
  }
  OptionValue ForceLoadOption{nullptr, 0};
  // Direct -force_load options are not among the link action's inputs. Each
  // is added when the walk reaches its place on the command line.
  SmallVector<const llvm::opt::Arg *, 2> DirectForceLoads;
  for (const llvm::opt::Arg *A :
       C.getInputArgs().filtered(options::OPT_force__load))
    DirectForceLoads.push_back(A);
  size_t NextForceLoad = 0;
  auto AddForceLoadsBefore = [&](unsigned Index) -> Error {
    while (NextForceLoad != DirectForceLoads.size() &&
           DirectForceLoads[NextForceLoad]->getIndex() < Index) {
      const llvm::opt::Arg *A = DirectForceLoads[NextForceLoad++];
      if (Error E = addLinkerOptionArchive(Inputs, A->getValue(),
                                           /*Whole=*/true, {{A, 0}}))
        return E;
    }
    return Error::success();
  };
#endif

  // Command::getInputInfos has only the file inputs; the linker-input
  // arguments (-l, -Wl, -Xlinker, ...) are found among the link action's
  // inputs. Their order is the command line's.
  bool WholeArchive = false;
  SmallVector<bool, 2> SavedWholeArchive;
  size_t NextFile = 0;
  for (const clang::driver::Action *A : Link.getSource().getInputs()) {
    const auto *IA = dyn_cast<clang::driver::InputAction>(A);
#ifdef __APPLE__
    // Compiled sources are not archives: only the other inputs flush the
    // -force_load options that precede them.
    if (IA)
      if (Error E = AddForceLoadsBefore(IA->getInputArg().getIndex()))
        return std::move(E);
#endif
    if (IA && !IA->getInputArg().getOption().matches(options::OPT_INPUT)) {
      const llvm::opt::Arg &Arg = IA->getInputArg();
      if (Arg.getOption().matches(options::OPT_l)) {
        std::unique_ptr<MemoryBuffer> Buf =
            Libraries ? findLibrary(C, Arg.getValue()) : nullptr;
        if (!Buf)
          continue;
        std::string Path = Buf->getBufferIdentifier().str();
        Expected<std::optional<ArchiveContents>> Kind =
            classifyLibrary(Buf->getMemBufferRef());
        if (!Kind)
          return Kind.takeError();
        if (!*Kind)
          continue;
        // The static linker can't read a bitcode archive found in a -L
        // directory, so that one must be passed by path.
        if (**Kind == ArchiveContents::Bitcode)
          return createStringError(
              "-l%s resolves to the bitcode archive %s; pass bitcode archives "
              "by path",
              Arg.getValue(), Path.c_str());
        addArchive(Inputs, std::move(Buf), **Kind, WholeArchive);
      } else if (isLinkerOption(Arg)) {
        for (unsigned I = 0; I != Arg.getNumValues(); ++I) {
          StringRef V = Arg.getValue(I);
#ifdef __APPLE__
          if (ForceLoadOption.first) {
            if (Error E = addLinkerOptionArchive(Inputs, V, /*Whole=*/true,
                                                 {ForceLoadOption, {&Arg, I}}))
              return std::move(E);
            ForceLoadOption = {nullptr, 0};
            continue;
          }
          if (V == "-force_load")
            ForceLoadOption = {&Arg, I};
#endif
          if (V == "--whole-archive") {
            WholeArchive = true;
          } else if (V == "--no-whole-archive") {
            WholeArchive = false;
          } else if (V == "--push-state") {
            SavedWholeArchive.push_back(WholeArchive);
          } else if (V == "--pop-state") {
            if (!SavedWholeArchive.empty())
              WholeArchive = SavedWholeArchive.pop_back_val();
          } else if (!V.starts_with("-")) {
            // CMake passes a whole archive as
            // -Wl,--push-state,--whole-archive,<path>,--pop-state.
            if (Error E = addLinkerOptionArchive(Inputs, V, WholeArchive,
                                                 {{&Arg, I}}))
              return std::move(E);
          }
        }
      }
      continue;
    }

    assert(NextFile < Link.getInputInfos().size());
    StringRef Path = Link.getInputInfos()[NextFile++].getFilename();
    // Unreadable inputs are left for the system linker to report.
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(Path);
    file_magic Magic = Buf ? identify_magic((*Buf)->getBuffer())
                           : file_magic(file_magic::unknown);
    if (Magic == file_magic::bitcode) {
      Inputs.Bitcode.push_back(Path.str());
      Inputs.Replaced.insert(Path);
    } else if (Magic == file_magic::archive) {
      Expected<ArchiveContents> Kind =
          addArchiveInput(Inputs, std::move(*Buf), WholeArchive);
      if (!Kind)
        return Kind.takeError();
      if (*Kind == ArchiveContents::Bitcode)
        Inputs.Replaced.insert(Path);
    } else if (Magic == file_magic::elf_relocatable ||
               Magic == file_magic::macho_object) {
      Inputs.NativeObjects.push_back(std::move(*Buf));
    }
  }
#ifdef __APPLE__
  if (Error E = AddForceLoadsBefore(std::numeric_limits<unsigned>::max()))
    return std::move(E);
#endif
  return std::move(Inputs);
}

/// The libraries that -l names are of no interest to a link without bitcode
/// inputs, which is the system linker's alone.
static Expected<LinkInputs> classifyLinkInputs(const Compilation &C,
                                               const Command &Link) {
  Expected<LinkInputs> Inputs = readLinkInputs(C, Link, /*Libraries=*/false);
  if (!Inputs || !Inputs->hasBitcode())
    return Inputs;
  return readLinkInputs(C, Link, /*Libraries=*/true);
}

/// Merges the bitcode inputs and the archive members they need, then writes
/// the instrumented AOT object to a temporary file that C removes. Returns
/// the file's path, or null if no bitcode was linked (the only bitcode is in
/// archives, and no member of them is needed).
static Expected<const char *> buildAot(Compilation &C,
                                       const LinkInputs &Inputs) {
  // The runtime starts from main, and there is one image per executable.
  if (const llvm::opt::Arg *A = C.getArgs().getLastArg(
          clang::options::OPT_shared, clang::options::OPT_dynamiclib,
          clang::options::OPT_bundle, clang::options::OPT_r)) {
    return createStringError("%s is not supported (orc-pgo builds "
                             "executables only)",
                             A->getAsString(C.getArgs()).c_str());
  }
  LLVMContext Ctx;
  Expected<std::unique_ptr<llvm::Module>> M =
      linkBitcodeFiles(Inputs.Bitcode, Ctx);
  if (!M)
    return M.takeError();
  SmallVector<ArchiveInput, 4> Archives;
  SmallVector<MemoryBufferRef, 4> Natives;
  for (const LinkArchive &A : Inputs.Archives)
    Archives.emplace_back(A.Buffer->getMemBufferRef(), A.Whole, A.Contents);
  for (const std::unique_ptr<MemoryBuffer> &B : Inputs.NativeObjects)
    Natives.push_back(B->getMemBufferRef());
  Expected<unsigned> Members = linkArchiveMembers(**M, Archives, Natives);
  if (!Members)
    return Members.takeError();
  if (Inputs.Bitcode.empty() && *Members == 0)
    return static_cast<const char *>(nullptr);
  Expected<std::unique_ptr<TargetMachine>> TM =
      createTargetMachine((*M)->getTargetTriple());
  if (!TM)
    return TM.takeError();
  (*M)->setDataLayout((*TM)->createDataLayout());

  // The JIT can call any builtin, so the image's symbol table takes the
  // address of all of them.
  std::string Builtins =
      C.getDefaultToolChain().getCompilerRT(C.getArgs(), "builtins");
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuiltinsBuf =
      MemoryBuffer::getFile(Builtins);
  if (!BuiltinsBuf)
    return createStringError("cannot read the compiler-rt builtins archive "
                             "%s: %s",
                             Builtins.c_str(),
                             BuiltinsBuf.getError().message().c_str());
  Expected<std::vector<std::string>> ExtraSymbols = archiveGlobalSymbols(
      (*BuiltinsBuf)->getMemBufferRef(), (*M)->getTargetTriple().getArchName(),
      (*M)->getDataLayout().getGlobalPrefix());
  if (!ExtraSymbols)
    return ExtraSymbols.takeError();

  if (Error E = applyLinkTimeLLVMOptions(C))
    return std::move(E);
  Expected<SmallVector<char, 0>> Object =
      buildAotObject(**M, **TM, *ExtraSymbols);
  if (!Object)
    return Object.takeError();

  SmallString<128> Path;
  int FD;
  if (std::error_code EC =
          sys::fs::createTemporaryFile("orc-pgo-aot", "o", FD, Path)) {
    return createStringError(EC, "cannot create a temporary file: %s",
                             EC.message().c_str());
  }
  // Registered before anything is written, so that C removes it on any error.
  const char *Temp = C.addTempFile(C.getArgs().MakeArgString(Path));
  raw_fd_ostream OS(FD, /*shouldClose=*/true);
  OS.write(Object->data(), Object->size());
  OS.close();
  if (OS.has_error()) {
    std::error_code EC = OS.error();
    // An unread error would be fatal when OS is destroyed.
    OS.clear_error();
    return createFileError(Path, EC);
  }
  return Temp;
}

/// buildAot, with the crash recovery that clang gives each cc1 job: a fatal
/// LLVM error (report_fatal_error aborts) or a crash in the merge or the code
/// generation becomes an error instead of killing the driver.
static Expected<const char *> buildAotSafely(Compilation &C,
                                             const LinkInputs &Inputs) {
  std::optional<Expected<const char *>> Result;
  CrashRecoveryContext CRC;
  CRC.DumpStackAndCleanupOnFailure = true;
  // As in CC1Command::Execute: unwinding skipped the pretty stack entries'
  // destructors.
  const void *PrettyState = SavePrettyStackState();
  if (!CRC.RunSafely([&] { Result.emplace(buildAot(C, Inputs)); })) {
    RestorePrettyStackState(PrettyState);
    return createStringError("building the AOT object failed with a fatal "
                             "error (see above)");
  }
  return std::move(*Result);
}

/// Renders a -Wl, or -Xlinker option, without the values that are not for the
/// native link.
static void renderLinkerOption(const llvm::opt::Arg &A,
                               const llvm::opt::ArgList &Args,
                               const LinkInputs &Inputs, bool DropAllLoad,
                               llvm::opt::ArgStringList &Out) {
  SmallVector<StringRef, 4> Kept;
  for (unsigned I = 0; I != A.getNumValues(); ++I) {
    StringRef V = A.getValue(I);
    if (!Inputs.Dropped.contains({&A, I}) && !(DropAllLoad && V == "-all_load"))
      Kept.push_back(V);
  }
  if (Kept.size() == A.getNumValues())
    A.render(Args, Out);
  else if (!Kept.empty())
    Out.push_back(Args.MakeArgString("-Wl," + join(Kept, ",")));
}

/// The command line for the native link:
/// - the original options, minus inputs, linker inputs, LTO, -x, -mllvm, --
///   and -force_load of bitcode archives;
/// - the link job's inputs in order, with the first replaced input swapped
///   for AotObject and the other replaced inputs dropped;
/// - then, if there is an AOT object, the runtime.
static llvm::opt::ArgStringList relinkArgs(const Compilation &C,
                                           const Command &Link,
                                           const LinkInputs &Inputs,
                                           const char *AotObject) {
  namespace options = clang::options;
  const llvm::opt::InputArgList &Args = C.getInputArgs();
  llvm::opt::ArgStringList Out{C.getDriver().getDriverProgramPath()};
  const bool DropAllLoad = AotObject && Inputs.AllLoad;
  for (const llvm::opt::Arg *A : Args) {
    const llvm::opt::Option &O = A->getOption();
    if (Inputs.Dropped.contains({A, 0}) ||
        (DropAllLoad && O.matches(options::OPT_all__load)) ||
        O.getKind() == llvm::opt::Option::InputClass ||
        O.hasFlag(options::LinkerInput) || O.matches(options::OPT_flto_EQ) ||
        O.matches(options::OPT_fno_lto) || O.matches(options::OPT_x) ||
        O.matches(options::OPT_mllvm) || O.matches(options::OPT__DASH_DASH)) {
      continue;
    }
    A->render(Args, Out);
  }

  // Command::getInputInfos has only the file inputs; the linker-input
  // arguments (-l, -Wl, -Xlinker, ...) are found among the link action's
  // inputs.
  bool AotPlaced = false;
  size_t NextFile = 0;
  for (const clang::driver::Action *A : Link.getSource().getInputs()) {
    if (const auto *IA = dyn_cast<clang::driver::InputAction>(A)) {
      const llvm::opt::Arg &Arg = IA->getInputArg();
      if (!Arg.getOption().matches(options::OPT_INPUT)) {
        if (isLinkerOption(Arg))
          renderLinkerOption(Arg, C.getArgs(), Inputs, DropAllLoad, Out);
        else
          Arg.render(C.getArgs(), Out);
        continue;
      }
    }
    assert(NextFile < Link.getInputInfos().size());
    const InputInfo &II = Link.getInputInfos()[NextFile++];
    if (!Inputs.Replaced.contains(II.getFilename())) {
      Out.push_back(II.getFilename());
      continue;
    }
    if (AotObject && !AotPlaced)
      Out.push_back(AotObject);
    AotPlaced = true;
  }

  // The original compile options are still present; their objects aren't.
  Out.push_back("-Qunused-arguments");
  if (AotObject) {
    // The runtime is C++ whatever the program's language: link as clang++
    // does, even if this was a C link (--driver-mode=gcc). The last mode wins.
    Out.push_back("--driver-mode=g++");
#ifdef __APPLE__
    // In place of -all_load, which would also load the runtime archives.
    if (Inputs.AllLoad) {
      for (const LinkArchive &A : Inputs.Archives) {
        if (A.Contents == ArchiveContents::Native)
          Out.push_back(C.getArgs().MakeArgString(
              "-Wl,-force_load," + A.Buffer->getBufferIdentifier()));
      }
    }
#endif
    Out.append(std::begin(RuntimeLinkPrefix), std::end(RuntimeLinkPrefix));
    Out.append(std::begin(RuntimeArchives), std::end(RuntimeArchives));
    Out.append(std::begin(RuntimeLinkSuffix), std::end(RuntimeLinkSuffix));
    // Coverage builds only: the instrumented runtime archives need it.
    if (*ORC_PGO_EXTRA_LINK_INPUT != '\0')
      Out.push_back(ORC_PGO_EXTRA_LINK_INPUT);
  }
  return Out;
}

#ifdef __APPLE__
/// Darwin's linker job always passes -lto_library <InstalledDir>/../lib/
/// libLTO.dylib, and ld64 warns when that file doesn't exist (orc-pgo isn't
/// installed next to LLVM's lib directory). The native link has no bitcode
/// input, so ld64 never uses the library.
static void dropLtoLibrary(Compilation &C) {
  for (Command &Job : C.getJobs()) {
    if (Job.getCreator().isLinkJob()) {
      llvm::opt::ArgStringList Args;
      ArrayRef<const char *> Old = Job.getArguments();
      for (size_t I = 0; I != Old.size(); ++I) {
        if (StringRef(Old[I]) == "-lto_library")
          ++I; // and its value
        else
          Args.push_back(Old[I]);
      }
      Job.replaceArguments(std::move(Args));
    }
  }
}
#endif

/// clang passes -disable-free to cc1 when the process is about to exit, and
/// cc1 then leaks everything it allocated. With several jobs, upstream would
/// run each cc1 in its own process; here they share one, so every translation
/// unit would leak its compiler instance.
static void dropDisableFree(Compilation &C) {
  for (Command &Job : C.getJobs()) {
    ArrayRef<const char *> Old = Job.getArguments();
    if (Old.empty() || StringRef(Old.front()) != "-cc1")
      continue;
    llvm::opt::ArgStringList Args;
    for (const char *A : Old) {
      if (StringRef(A) != "-disable-free")
        Args.push_back(A);
    }
    Job.replaceArguments(std::move(Args));
  }
}

/// Driver::ExecuteCompilation returns 0 when a job failed with status 1 (the
/// "normal" compiler failure); clang_main takes the first failing job's status.
static int executeCompilation(Driver &D, Compilation &C) {
  SmallVector<std::pair<int, const Command *>, 4> Failing;
  int Res = D.ExecuteCompilation(C, Failing);
  if (Res == 0 && !Failing.empty())
    Res = Failing.front().first;
  return Res;
}

static int runDriver(ArrayRef<const char *> Args) {
  initializeNativeTarget();
  std::unique_ptr<clang::DiagnosticOptions> DiagOpts =
      clang::CreateAndPopulateDiagOpts(Args);
  DiagOpts->DiagnosticSuppressionMappingsFile.clear();
  auto *DiagClient = new clang::TextDiagnosticPrinter(errs(), *DiagOpts);
  DiagClient->setPrefix(sys::path::stem(Args[0]).str());
  clang::DiagnosticsEngine Diags(clang::DiagnosticIDs::create(), *DiagOpts,
                                 DiagClient);
  clang::ProcessWarningOptions(Diags, *DiagOpts, *vfs::getRealFileSystem(),
                               /*ReportDiags=*/false);

  std::string Exe =
      sys::fs::getMainExecutable(Args[0], (void *)(intptr_t)&driverMain);
  Driver D(Exe, sys::getDefaultTargetTriple(), Diags, "orc-pgo");
  D.ResourceDir = ORC_PGO_CLANG_RESOURCE_DIR;
  D.CC1Main = cc1Tool;
  CrashRecoveryContext::Enable(/*NeedsPOSIXUtilitySignalHandling=*/true);

  if (count_if(Args, [](const char *A) { return StringRef(A) == "-arch"; }) > 1)
    return reportError(
        createStringError("more than one -arch is not supported"));

  std::unique_ptr<Compilation> C(D.BuildCompilation(Args));
  if (C->containsError() || Diags.hasErrorOccurred())
    return 1;

  // clang::Driver::BuildCompilation turns integrated cc1 off whenever a
  // compilation has more than one job (e.g. compile + link): cc1 would be
  // spawned as `<driver executable> -cc1 ...`. Keep it in-process, so cc1 never
  // depends on what the driver executable is (the unit tests' is gtest_main).
  for (Command &Job : C->getJobs())
    Job.InProcess = true;

  if (C->getJobs().size() > 1)
    dropDisableFree(*C);

  const JobList &Jobs = C->getJobs();
  auto LinkJob = find_if(
      Jobs, [](const Command &Job) { return Job.getCreator().isLinkJob(); });
  if (LinkJob == Jobs.end() ||
      C->getArgs().hasArg(clang::options::OPT__HASH_HASH_HASH,
                          clang::options::OPT_fdriver_only)) {
    return executeCompilation(D, *C);
  }
  const Command &Link = *LinkJob;

  // Jobs after the link (Darwin's dsymutil) would need its output.
  for (const Command &Job : Jobs) {
    if (&Job == &Link)
      break;
    const Command *FailingJob = nullptr;
    if (int Res = C->ExecuteCommand(Job, FailingJob))
      return Res;
  }

  Expected<LinkInputs> Inputs = classifyLinkInputs(*C, Link);
  if (!Inputs)
    return reportError(Inputs.takeError());
  const char *AotObject = nullptr;
  if (Inputs->hasBitcode()) {
    Expected<const char *> Object = buildAotSafely(*C, *Inputs);
    if (!Object)
      return reportError(Object.takeError());
    AotObject = *Object;
  }

  llvm::opt::ArgStringList NativeArgs =
      relinkArgs(*C, Link, *Inputs, AotObject);
  Driver Native(Exe, sys::getDefaultTargetTriple(), Diags, "orc-pgo");
  Native.ResourceDir = ORC_PGO_CLANG_RESOURCE_DIR;
  std::unique_ptr<Compilation> NativeLink(Native.BuildCompilation(NativeArgs));
#ifdef __APPLE__
  dropLtoLibrary(*NativeLink);
#endif
  return executeCompilation(Native, *NativeLink);
}

SmallVector<const char *, 0> augmentArgs(ArrayRef<const char *> Argv,
                                         const char *DefaultSysroot) {
  bool HasMode = false;
  bool HasSysroot = std::getenv("SDKROOT") != nullptr;
  for (StringRef A : Argv.drop_front()) {
    HasMode |= A.starts_with("--driver-mode=");
    HasSysroot |= A.starts_with("-isysroot") || A.starts_with("--sysroot");
  }
  SmallVector<const char *, 0> Out{Argv.front()};
  if (!HasMode)
    Out.push_back("--driver-mode=g++");
  Out.push_back("-flto=full");
  if (*DefaultSysroot != '\0' && !HasSysroot) {
    Out.push_back("-isysroot");
    Out.push_back(DefaultSysroot);
  }
  Out.append(Argv.begin() + 1, Argv.end());
  return Out;
}

int driverMain(ArrayRef<const char *> Argv) {
  clang::noteBottomOfStack();
  BumpPtrAllocator Alloc;
  SmallVector<const char *, 64> Args(Argv.begin(), Argv.end());
  cl::ExpansionContext ECtx(Alloc, cl::TokenizeGNUCommandLine);
  if (Error E = ECtx.expandResponseFiles(Args))
    return reportError(std::move(E));
  if (Args.size() >= 2 && StringRef(Args[1]).starts_with("-cc1"))
    return cc1Tool(Args);
  if (Args.size() >= 2 && StringRef(Args[1]) == "--orc-pgo-ar")
    return runArchiver(ArrayRef(Args).drop_front(2));
  return runDriver(augmentArgs(Args, ORC_PGO_DEFAULT_SYSROOT));
}

} // namespace orc_pgo
