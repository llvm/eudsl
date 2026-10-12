// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Driver.h"
#include "orc-pgo/Archive.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/MachOUniversalWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <sys/resource.h>

using namespace llvm;
using namespace orc_pgo;

extern char **environ;

namespace {

std::string errorText(Error E) {
  return E ? toString(std::move(E)) : std::string();
}

template <typename T>
std::string errorText(Expected<T> &E) {
  return E ? std::string() : errorText(E.takeError());
}

bool contains(const std::string &Haystack, StringRef Needle) {
  return StringRef(Haystack).contains(Needle);
}

/// Points TMPDIR at Dir for its lifetime, creating Dir unless asked not to.
class TempDirOverride {
public:
  explicit TempDirOverride(StringRef Dir, bool Create = true) {
    if (const char *Old = std::getenv("TMPDIR"))
      Saved = Old;
    if (Create)
      EXPECT_FALSE(sys::fs::create_directories(Dir));
    setenv("TMPDIR", Dir.str().c_str(), 1);
  }
  ~TempDirOverride() {
    if (Saved)
      setenv("TMPDIR", Saved->c_str(), 1);
    else
      unsetenv("TMPDIR");
  }

private:
  std::optional<std::string> Saved;
};

/// The names of the entries of directory Dir.
std::vector<std::string> entriesOf(StringRef Dir) {
  std::vector<std::string> Names;
  std::error_code EC;
  for (sys::fs::directory_iterator I(Dir, EC), E; !EC && I != E;
       I.increment(EC))
    Names.push_back(sys::path::filename(I->path()).str());
  EXPECT_FALSE(EC) << Dir.str();
  return Names;
}

std::vector<std::string> strings(ArrayRef<const char *> Args) {
  return std::vector<std::string>(Args.begin(), Args.end());
}

constexpr const char *LibSource =
    "int twice(int x) { return x > 0 ? 2 * x : 0; }\n";
constexpr const char *NativeSource =
    "extern \"C\" int addOne(int x) { return x + 1; }\n";
constexpr const char *MainSource = R"(#include <cstdio>
int twice(int);
extern "C" int addOne(int);
int main() {
  int S = 0;
  for (int I = 0; I < 1000; ++I)
    S += twice(I % 7);
  std::printf("%d %d\n", S, addOne(41));
  return 0;
}
)";

class DriverTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("orc-pgo-driver", Dir));
  }
  void TearDown() override { sys::fs::remove_directories(Dir); }

  std::string path(StringRef Name) const {
    SmallString<128> P(Dir);
    sys::path::append(P, Name);
    return std::string(P);
  }

  std::string write(StringRef Name, StringRef Text) const {
    std::string P = path(Name);
    std::error_code EC;
    raw_fd_ostream OS(P, EC);
    EXPECT_FALSE(EC) << P;
    OS << Text;
    return P;
  }

  std::string writeBitcode(StringRef Name, StringRef IR) const {
    LLVMContext Ctx;
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
    std::string P = path(Name);
    if (!M) {
      ADD_FAILURE() << Err.getMessage().str();
      return P;
    }
    std::error_code EC;
    raw_fd_ostream OS(P, EC);
    WriteBitcodeToFile(*M, OS);
    return P;
  }

  static int drive(const std::vector<std::string> &Args) {
    std::vector<const char *> Argv{"orc-pgo"};
    for (const std::string &A : Args)
      Argv.push_back(A.c_str());
    return driverMain(Argv);
  }

  /// drive(), returning everything written to stderr, the linker's included.
  static std::string driveCapturingStderr(const std::vector<std::string> &Args,
                                          int &Exit) {
    ::testing::internal::CaptureStderr();
    Exit = drive(Args);
    return ::testing::internal::GetCapturedStderr();
  }

  /// Runs Exe and returns its exit code and stdout. In coverage builds the
  /// program links the coverage profile runtime, which would write into the
  /// gate's profraw directory (LLVM_PROFILE_FILE): give it its own file.
  std::pair<int, std::string> run(StringRef Exe) const {
    std::string Out = path("stdout.txt");
    sys::fs::remove(Out);
    std::optional<StringRef> Redirects[] = {std::nullopt, StringRef(Out),
                                            std::nullopt};
    std::vector<std::string> Env;
    for (char **E = environ; *E; ++E)
      if (!StringRef(*E).starts_with("LLVM_PROFILE_FILE="))
        Env.push_back(*E);
    Env.push_back("LLVM_PROFILE_FILE=" + path("app.profraw"));
    std::vector<StringRef> EnvRefs(Env.begin(), Env.end());
    int Exit = sys::ExecuteAndWait(Exe, {Exe}, ArrayRef<StringRef>(EnvRefs),
                                   Redirects);
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(Out);
    return {Exit, Buf ? (*Buf)->getBuffer().str() : std::string()};
  }

  static file_magic magicOf(StringRef P) {
    file_magic M = file_magic::unknown;
    EXPECT_FALSE(identify_magic(P, M)) << P.str();
    return M;
  }

  /// Whether Exe's symbol table has Name (Mach-O: with its '_' prefix).
  static bool hasSymbol(StringRef Exe, StringRef Name) {
    Expected<object::OwningBinary<object::ObjectFile>> Obj =
        object::ObjectFile::createObjectFile(Exe);
    if (!Obj) {
      ADD_FAILURE() << errorText(Obj.takeError());
      return false;
    }
    std::string Prefixed = ("_" + Name).str();
    for (const object::SymbolRef &Sym : Obj->getBinary()->symbols()) {
      Expected<StringRef> SymName = Sym.getName();
      if (!SymName) {
        consumeError(SymName.takeError());
        continue;
      }
      if (*SymName == Name || *SymName == Prefixed)
        return true;
    }
    return false;
  }

  /// main.cpp and lib.cpp compiled to bitcode objects. Linking them alone
  /// fails at the native link (addOne is undefined), so tests that expect an
  /// earlier failure use them.
  std::pair<std::string, std::string> bitcodeObjects() {
    std::string Main = path("main.o"), Lib = path("lib.o");
    EXPECT_EQ(drive({"-c", write("main.cpp", MainSource), "-o", Main}), 0);
    EXPECT_EQ(drive({"-c", write("lib.cpp", LibSource), "-o", Lib}), 0);
    return {Main, Lib};
  }

  /// A native object defining _bar, _data and baz (exact symbol names).
  void nativeObjectA(std::string &Obj) {
    Obj = path("a.o");
    ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                     write("a.c", "void f(void) __asm__(\"_bar\");\n"
                                  "void f(void) {}\n"
                                  "int d __asm__(\"_data\") = 1;\n"
                                  "static int local(void) { return 0; }\n"
                                  "int g(void) __asm__(\"baz\");\n"
                                  "int g(void) { return local(); }\n"),
                     "-o", Obj}),
              0);
  }

  /// A native object with a weak definition of baz.
  void nativeObjectB(std::string &Obj) {
    Obj = path("b.o");
    ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                     write("b.c", "int h(void) __asm__(\"baz\");\n"
                                  "__attribute__((weak)) int h(void) "
                                  "{ return 2; }\n"),
                     "-o", Obj}),
              0);
  }

  /// The archive of Members, with a symbol table. Any failure fails the test
  /// and leaves Ar null.
  static void archiveOf(ArrayRef<std::string> Members,
                        std::unique_ptr<MemoryBuffer> &Ar) {
    std::vector<NewArchiveMember> NM;
    for (const std::string &P : Members) {
      Expected<NewArchiveMember> Member = NewArchiveMember::getFile(P, true);
      if (!Member) {
        ADD_FAILURE() << errorText(Member);
        return;
      }
      NM.push_back(std::move(*Member));
    }
    Expected<std::unique_ptr<MemoryBuffer>> Buf = writeArchiveToBuffer(
        NM, SymtabWritingMode::NormalSymtab, object::Archive::getDefaultKind(),
        /*Deterministic=*/true, /*Thin=*/false);
    if (!Buf) {
      ADD_FAILURE() << errorText(Buf);
      return;
    }
    Ar = std::move(*Buf);
  }

  /// A -B directory whose `ld` drops the GNU --whole-archive family of options
  /// and runs the system linker, so that one command line works on every
  /// platform (ld64 has no such options). Returns the -B option.
  std::string linkerShim() {
    std::string Shim = path("bin");
    EXPECT_FALSE(sys::fs::create_directory(Shim));
    ErrorOr<std::string> RealLd = sys::findProgramByName("ld");
    EXPECT_TRUE(RealLd);
    std::string Script = "#!/bin/sh\n"
                         "for a in \"$@\"; do\n"
                         "  shift\n"
                         "  case \"$a\" in\n"
                         "  --whole-archive|--no-whole-archive|--push-state|"
                         "--pop-state) ;;\n"
                         "  *) set -- \"$@\" \"$a\" ;;\n"
                         "  esac\n"
                         "done\n"
                         "exec \"" +
                         (RealLd ? *RealLd : std::string()) + "\" \"$@\"\n";
    std::string Ld = write("bin/ld", Script);
    EXPECT_FALSE(sys::fs::setPermissions(Ld, sys::fs::perms::all_all));
    return "-B" + Shim;
  }

  /// An archive of one bitcode member that defines Symbol and prints Message
  /// from a static initializer, so linking the member is visible at run time.
  /// Nothing references the symbol.
  std::string noisyArchive(StringRef Name, StringRef Message) {
    std::string Object = path((Name + ".o").str());
    std::string Source = ("#include <cstdio>\n"
                          "static int Linked = (std::puts(\"" +
                          Message +
                          "\"), 0);\n"
                          "int " +
                          Name + "Symbol() { return Linked; }\n")
                             .str();
    EXPECT_EQ(drive({"-c", write((Name + ".cpp").str(), Source), "-o", Object}),
              0);
    std::string Archive = path(("lib" + Name + ".a").str());
    EXPECT_EQ(runArchiver({"create", Archive.c_str(), Object.c_str()}), 0);
    return Archive;
  }

  /// A bitcode archive of lib.cpp and native.cpp (twice and addOne), and the
  /// bitcode main.o that uses them: "5994 42\n".
  std::pair<std::string, std::string> bitcodeProgram() {
    std::string Main = path("main.o"), Lib = path("lib.o"), Add = path("add.o");
    EXPECT_EQ(drive({"-c", write("main.cpp", MainSource), "-o", Main}), 0);
    EXPECT_EQ(drive({"-c", write("lib.cpp", LibSource), "-o", Lib}), 0);
    EXPECT_EQ(drive({"-c", write("add.cpp", NativeSource), "-o", Add}), 0);
    std::string Archive = path("libprog.a");
    EXPECT_EQ(
        runArchiver({"create", Archive.c_str(), Lib.c_str(), Add.c_str()}), 0);
    return {Main, Archive};
  }

  SmallString<128> Dir;
};

std::vector<std::string> sorted(std::vector<std::string> V) {
  std::sort(V.begin(), V.end());
  return V;
}

// ---- augmentArgs ----

TEST(AugmentArgs, AddsDefaultsBeforeTheUserArguments) {
  unsetenv("SDKROOT");
  EXPECT_EQ(strings(augmentArgs({"orc-pgo", "-c", "x.cpp"}, "")),
            (std::vector<std::string>{"orc-pgo", "--driver-mode=g++",
                                      "-flto=full", "-c", "x.cpp"}));
  EXPECT_EQ(
      strings(augmentArgs({"orc-pgo", "x.cpp"}, "/sdk")),
      (std::vector<std::string>{"orc-pgo", "--driver-mode=g++", "-flto=full",
                                "-isysroot", "/sdk", "x.cpp"}));
}

TEST(AugmentArgs, KeepsAnExplicitDriverMode) {
  unsetenv("SDKROOT");
  EXPECT_EQ(strings(augmentArgs({"orc-pgo", "--driver-mode=gcc", "x.c"}, "")),
            (std::vector<std::string>{"orc-pgo", "-flto=full",
                                      "--driver-mode=gcc", "x.c"}));
}

TEST(AugmentArgs, AnExplicitSysrootWins) {
  unsetenv("SDKROOT");
  auto Count = [](ArrayRef<const char *> A) {
    return std::count_if(A.begin(), A.end(), [](const char *S) {
      return StringRef(S).starts_with("-isysroot");
    });
  };
  EXPECT_EQ(Count(augmentArgs({"orc-pgo", "-isysroot", "/x", "a.c"}, "/sdk")),
            1);
  EXPECT_EQ(Count(augmentArgs({"orc-pgo", "-isysroot/x", "a.c"}, "/sdk")), 1);
  EXPECT_EQ(Count(augmentArgs({"orc-pgo", "--sysroot=/y", "a.c"}, "/sdk")), 0);
  EXPECT_EQ(Count(augmentArgs({"orc-pgo", "--sysroot", "/y", "a.c"}, "/sdk")),
            0);
  setenv("SDKROOT", "/z", 1);
  EXPECT_EQ(Count(augmentArgs({"orc-pgo", "a.c"}, "/sdk")), 0);
  unsetenv("SDKROOT");
}

// ---- linkBitcodeFiles ----

TEST_F(DriverTest, LinkBitcodeFilesMergesAndNamesAnonymousGlobals) {
  std::string A = writeBitcode("a.bc", "@0 = private global i32 1\n"
                                       "define ptr @getA() {\n"
                                       "  ret ptr @0\n"
                                       "}\n");
  std::string B = writeBitcode("b.bc", "define i32 @b() {\n"
                                       "  ret i32 2\n"
                                       "}\n");
  LLVMContext Ctx;
  Expected<std::unique_ptr<Module>> M = linkBitcodeFiles({A, B}, Ctx);
  ASSERT_EQ(errorText(M), "");
  ASSERT_NE((*M)->getFunction("b"), nullptr);
  Function *GetA = (*M)->getFunction("getA");
  ASSERT_NE(GetA, nullptr);
  auto *Ret = cast<ReturnInst>(GetA->getEntryBlock().getTerminator());
  StringRef Name = Ret->getReturnValue()->getName();
  EXPECT_TRUE(Name.starts_with("anon.")) << Name.str();
  EXPECT_TRUE(Name.ends_with(".0")) << Name.str();
}

TEST_F(DriverTest, LinkBitcodeFilesReportsUnreadableAndCorruptFiles) {
  LLVMContext Ctx;
  Expected<std::unique_ptr<Module>> Missing =
      linkBitcodeFiles({path("missing.bc")}, Ctx);
  EXPECT_TRUE(contains(errorText(Missing), "missing.bc"));
  std::string Bad = write("bad.bc", StringRef("BC\xC0\xDE junk", 9));
  Expected<std::unique_ptr<Module>> Corrupt = linkBitcodeFiles({Bad}, Ctx);
  EXPECT_TRUE(contains(errorText(Corrupt), "bad.bc"));
}

TEST_F(DriverTest, LinkBitcodeFilesReportsDuplicateDefinitions) {
  constexpr const char *Dup = "define i32 @dup() {\n  ret i32 1\n}\n";
  std::string A = writeBitcode("a.bc", Dup), B = writeBitcode("b.bc", Dup);
  LLVMContext Ctx;
  Expected<std::unique_ptr<Module>> M = linkBitcodeFiles({A, B}, Ctx);
  std::string Err = errorText(M);
  EXPECT_TRUE(contains(Err, "b.bc")) << Err;
  EXPECT_TRUE(contains(Err, "Linking globals named 'dup': symbol multiply "
                            "defined!"))
      << Err;
}

TEST_F(DriverTest, LinkBitcodeFilesPrintsLinkerWarnings) {
  std::string A =
      writeBitcode("a.bc", "target triple = \"x86_64-unknown-linux-gnu\"\n");
  std::string B =
      writeBitcode("b.bc", "target triple = \"aarch64-unknown-linux-gnu\"\n");
  LLVMContext Ctx;
  ::testing::internal::CaptureStderr();
  Expected<std::unique_ptr<Module>> M = linkBitcodeFiles({A, B}, Ctx);
  std::string Err = ::testing::internal::GetCapturedStderr();
  ASSERT_EQ(errorText(M), "");
  EXPECT_TRUE(contains(Err, "orc-pgo: warning: Linking two modules of "
                            "different target triples"))
      << Err;
}

TEST_F(DriverTest, LinkModuleRestoresTheDiagnosticHandler) {
  LLVMContext Ctx;
  Ctx.setDiagnosticHandler(std::make_unique<DiagnosticHandler>());
  const DiagnosticHandler *Before = Ctx.getDiagHandlerPtr();
  Module Dest("dest", Ctx);
  Linker L(Dest);
  auto Src = std::make_unique<Module>("src", Ctx);
  ASSERT_EQ(errorText(linkModule(L, std::move(Src))), "");
  EXPECT_EQ(Ctx.getDiagHandlerPtr(), Before);
}

// ---- archiveGlobalSymbols ----

TEST_F(DriverTest, ArchiveGlobalSymbolsReadsThePlainSymbolTable) {
  std::string A, B;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(A));
  ASSERT_NO_FATAL_FAILURE(nativeObjectB(B));
  std::unique_ptr<MemoryBuffer> Ar;
  ASSERT_NO_FATAL_FAILURE(archiveOf({A, B}, Ar));
  ASSERT_NE(Ar, nullptr);
  Expected<std::vector<std::string>> Names =
      archiveGlobalSymbols(Ar->getMemBufferRef(), "unused", '\0');
  ASSERT_EQ(errorText(Names), "");
  // baz is defined by both members; it is listed once.
  EXPECT_EQ(sorted(*Names), (std::vector<std::string>{"_bar", "_data", "baz"}));
}

TEST_F(DriverTest, ArchiveGlobalSymbolsStripsTheGlobalPrefix) {
  std::string A, B;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(A));
  ASSERT_NO_FATAL_FAILURE(nativeObjectB(B));
  std::unique_ptr<MemoryBuffer> Ar;
  ASSERT_NO_FATAL_FAILURE(archiveOf({A, B}, Ar));
  ASSERT_NE(Ar, nullptr);
  Expected<std::vector<std::string>> Names =
      archiveGlobalSymbols(Ar->getMemBufferRef(), "unused", '_');
  ASSERT_EQ(errorText(Names), "");
  EXPECT_EQ(sorted(*Names), (std::vector<std::string>{"bar", "baz", "data"}));
}

TEST_F(DriverTest, ArchiveGlobalSymbolsReadsASliceOfAUniversalFile) {
  std::string Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  std::unique_ptr<MemoryBuffer> Ar;
  ASSERT_NO_FATAL_FAILURE(archiveOf({Obj}, Ar));
  ASSERT_NE(Ar, nullptr);
  std::unique_ptr<object::Archive> A =
      cantFail(object::Archive::create(Ar->getMemBufferRef()));
  object::Slice S(*A, MachO::CPU_TYPE_ARM64, MachO::CPU_SUBTYPE_ARM64_ALL,
                  "arm64", 12);
  SmallString<0> Fat;
  raw_svector_ostream OS(Fat);
  ASSERT_EQ(errorText(object::writeUniversalBinaryToStream({S}, OS)), "");
  MemoryBufferRef FatRef(Fat, "fat.a");

  Expected<std::vector<std::string>> Names =
      archiveGlobalSymbols(FatRef, "arm64", '_');
  ASSERT_EQ(errorText(Names), "");
  EXPECT_EQ(sorted(*Names), (std::vector<std::string>{"bar", "baz", "data"}));

  Expected<std::vector<std::string>> Missing =
      archiveGlobalSymbols(FatRef, "x86_64", '_');
  EXPECT_TRUE(contains(errorText(Missing), "fat file does not contain x86_64"));
}

TEST_F(DriverTest, ArchiveGlobalSymbolsRejectsMalformedInput) {
  Expected<std::vector<std::string>> NotArchive = archiveGlobalSymbols(
      MemoryBufferRef("not an archive at all", "junk.a"), "arm64", '_');
  EXPECT_NE(errorText(NotArchive), "");
  // A fat header announcing one slice, and nothing after it.
  StringRef Truncated("\xCA\xFE\xBA\xBE\x00\x00\x00\x01", 8);
  Expected<std::vector<std::string>> BadFat =
      archiveGlobalSymbols(MemoryBufferRef(Truncated, "fat.a"), "arm64", '_');
  EXPECT_TRUE(contains(errorText(BadFat), "truncated or malformed fat file"));
}

// ---- compile-only and immediate modes ----

TEST_F(DriverTest, CompileOnlyEmitsBitcode) {
  std::string Obj = path("lib.o");
  ASSERT_EQ(drive({"-c", write("lib.cpp", LibSource), "-o", Obj}), 0);
  EXPECT_EQ(magicOf(Obj), file_magic::bitcode);
}

TEST_F(DriverTest, NoLtoEmitsANativeObject) {
  std::string Obj = path("lib.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", write("lib.cpp", LibSource), "-o", Obj}),
            0);
  file_magic M = magicOf(Obj);
  EXPECT_TRUE(M == file_magic::elf_relocatable ||
              M == file_magic::macho_object);
}

TEST_F(DriverTest, FindsBuiltinAndSystemHeaders) {
  EXPECT_EQ(drive({"-fsyntax-only",
                   write("h.cpp", "#include <stddef.h>\n#include <cstdio>\n"
                                  "#include <vector>\n")}),
            0);
}

TEST_F(DriverTest, FrontendErrorsFail) {
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-fsyntax-only", write("bad.cpp", "int x = ;\n")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "expected expression")) << Err;
}

TEST_F(DriverTest, UnknownDriverOptionsFail) {
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-no-such-driver-flag", "-c", write("lib.cpp", LibSource)}, Exit);
  EXPECT_EQ(Exit, 1);
  // clang may append a "did you mean" suggestion, so match the pieces.
  EXPECT_TRUE(contains(Err, "unknown argument")) << Err;
  EXPECT_TRUE(contains(Err, "-no-such-driver-flag")) << Err;
}

TEST_F(DriverTest, VersionSucceeds) { EXPECT_EQ(drive({"--version"}), 0); }

TEST_F(DriverTest, RunsCC1InProcess) {
  // No -resource-dir: cc1 falls back to orc-pgo's, which has stddef.h. cc1
  // only searches <resource-dir>/include itself for targets whose header
  // search the driver does not own (not Darwin, not Linux), so use a bare
  // OS-less triple to observe the fallback on every host.
  EXPECT_EQ(drive({"-cc1", "-triple", "aarch64-unknown-unknown",
                   "-nostdsysteminc", "-fsyntax-only",
                   write("ok.c", "#include <stddef.h>\nsize_t n;\n")}),
            0);
  int Exit;
  std::string Err = driveCapturingStderr({"-cc1", "-no-such-cc1-flag"}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "unknown argument")) << Err;
  EXPECT_TRUE(contains(Err, "-no-such-cc1-flag")) << Err;
}

TEST_F(DriverTest, OtherIntegratedToolsAreRejected) {
  int Exit;
  std::string Err = driveCapturingStderr({"-cc1as"}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "orc-pgo: error: -cc1as is not supported")) << Err;
}

TEST_F(DriverTest, ExpandsResponseFiles) {
  std::string Rsp =
      write("args.rsp", "-fsyntax-only " + write("ok.cpp", "int x;\n") + "\n");
  EXPECT_EQ(drive({"@" + Rsp}), 0);
  std::string Loop = path("loop.rsp");
  write("loop.rsp", "@" + Loop + "\n");
  int Exit;
  std::string Err = driveCapturingStderr({"@" + Loop}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "recursive expansion of")) << Err;
}

TEST_F(DriverTest, RejectsMoreThanOneArch) {
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-arch", "arm64", "-arch", "x86_64", "-c", write("lib.cpp", LibSource)},
      Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "more than one -arch is not supported")) << Err;
}

TEST_F(DriverTest, PrintOnlyModesDoNotLink) {
  std::string Exe = path("app");
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-###", write("main.cpp", MainSource), "-o", Exe}, Exit);
  EXPECT_EQ(Exit, 0);
  EXPECT_TRUE(contains(Err, "\"-cc1\"")) << Err;
  EXPECT_EQ(drive({"-fdriver-only", path("main.cpp"), "-o", Exe}), 0);
  EXPECT_FALSE(sys::fs::exists(Exe));
}

// ---- link mode ----

TEST_F(DriverTest, LinksBitcodeSourcesAndNativeObjects) {
  std::string Lib = path("lib.o"), Native = path("native.o"), Exe = path("app");
  ASSERT_EQ(drive({"-c", "-O2", write("lib.cpp", LibSource), "-o", Lib}), 0);
  ASSERT_EQ(drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o",
                   Native}),
            0);
  // main.cpp is compiled to bitcode in-process; -lm is a linker-input
  // argument, which keeps its place on the native link line.
  ASSERT_EQ(drive({"-O2", write("main.cpp", MainSource), Lib, Native, "-lm",
                   "-o", Exe}),
            0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "5994 42\n");
  EXPECT_TRUE(hasSymbol(Exe, "__orc_pgo_start"));
}

TEST_F(DriverTest, NativeOnlyLinksSkipTheRuntime) {
  std::string Exe = path("app");
  ASSERT_EQ(drive({"-fno-lto", "-O2", write("main.cpp", MainSource),
                   write("lib.cpp", LibSource),
                   write("native.cpp", NativeSource), "-o", Exe}),
            0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "5994 42\n");
  EXPECT_FALSE(hasSymbol(Exe, "__orc_pgo_start"));
}

TEST_F(DriverTest, NativeLinkErrorsPropagate) {
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-fno-lto", write("main.cpp", MainSource), "-o", path("app")}, Exit);
  EXPECT_NE(Exit, 0);
  EXPECT_TRUE(contains(Err, "linker command failed")) << Err;
}

TEST_F(DriverTest, CompileErrorsInLinkModeFail) {
  std::string Exe = path("app");
  EXPECT_EQ(drive({write("bad.cpp", "int x = ;\n"), write("lib.cpp", LibSource),
                   "-o", Exe}),
            1);
  EXPECT_FALSE(sys::fs::exists(Exe));
}

TEST_F(DriverTest, UnknownTargetsAreReported) {
  std::string Gpu =
      writeBitcode("gpu.o", "target triple = \"nvptx64-nvidia-cuda\"\n"
                            "define void @k() {\n  ret void\n}\n");
  int Exit;
  std::string Err = driveCapturingStderr({Gpu, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(
      contains(Err, "orc-pgo: error: cannot target nvptx64-nvidia-cuda"))
      << Err;
}

TEST_F(DriverTest, MissingBuiltinsAreReported) {
  auto [Main, Lib] = bitcodeObjects();
  std::string Empty = path("empty-resource-dir");
  ASSERT_FALSE(sys::fs::create_directory(Empty));
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-resource-dir", Empty, Main, Lib, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(
      contains(Err, "cannot read the compiler-rt builtins archive " + Empty))
      << Err;
}

TEST_F(DriverTest, LinkerInputArgumentsReachTheNativeLink) {
  // Command::getInputInfos has only file inputs; -l must still be passed on.
  auto [Main, Lib] = bitcodeObjects();
  std::string Native = path("native.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o",
                   Native}),
            0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, "-lorc_pgo_no_such_library", Lib, Native, "-o", path("app")},
      Exit);
  EXPECT_NE(Exit, 0);
  EXPECT_TRUE(contains(Err, "orc_pgo_no_such_library")) << Err;
}

TEST_F(DriverTest, LinkerInputsKeepTheirOrderOnTheNativeLink) {
  // The first bitcode input becomes the AOT object, the others are dropped;
  // everything else stays where the command line had it, before the runtime.
  auto [Main, Lib] = bitcodeObjects();
  std::string Native = path("native.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o",
                   Native}),
            0);
  int Exit;
  std::string Err =
      driveCapturingStderr({"-v", Main, "-lorc_pgo_first", Lib,
                            "-Wl,--orc_pgo_wl", "-lorc_pgo_second", "-Xlinker",
                            "--orc_pgo_xl", Native, "-o", path("app")},
                           Exit);
  EXPECT_NE(Exit, 0);
  // The ld command is the printed line that names the AOT object. (The test
  // directory's name has no characters that clang would quote.)
  size_t Begin = Err.find("orc-pgo-aot");
  ASSERT_NE(Begin, std::string::npos) << Err;
  Begin = Err.rfind('\n', Begin) + 1;
  StringRef Line = StringRef(Err).drop_front(Begin).take_until(
      [](char C) { return C == '\n'; });
  size_t Last = 0;
  for (StringRef Token : {"orc-pgo-aot", " -lorc_pgo_first ", " --orc_pgo_wl ",
                          " -lorc_pgo_second ", " --orc_pgo_xl ", "/native.o ",
                          "libOrcPgoRuntime.a"}) {
    size_t At = Line.find(Token, Last);
    ASSERT_NE(At, StringRef::npos)
        << Token.str() << " after " << Last << " in " << Line.str();
    Last = At;
  }
  // The dropped input is not on the line at all.
  EXPECT_FALSE(Line.contains("lib.o")) << Line.str();
}

TEST_F(DriverTest, TheNativeLinkDoesNotNameLibLTO) {
  // Darwin's linker job passes -lto_library <InstalledDir>/../lib/libLTO.dylib
  // unconditionally, and ld64 warns when that file is missing.
  auto [Main, Archive] = bitcodeProgram();
  int Exit;
  std::string Err =
      driveCapturingStderr({"-v", Main, Archive, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 0) << Err;
  EXPECT_FALSE(contains(Err, "lto_library")) << Err;
  EXPECT_TRUE(contains(Err, "-o " + path("app"))) << Err;
}

TEST_F(DriverTest, CLinksUseTheCXXRuntime) {
  // --driver-mode=gcc is how CMake's C compiler is spelled; the runtime
  // archives still need the C++ standard library.
  std::string Exe = path("app");
  ASSERT_EQ(drive({"--driver-mode=gcc",
                   write("main.c", "int puts(const char *);\n"
                                   "int main(void) {\n"
                                   "  int new = 4;\n"
                                   "  puts(\"c program\");\n"
                                   "  return new - 4;\n"
                                   "}\n"),
                   "-o", Exe}),
            0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "c program\n");
  EXPECT_TRUE(hasSymbol(Exe, "__orc_pgo_start"));
}

TEST_F(DriverTest, DuplicateBitcodeDefinitionsFailTheLink) {
  std::string A = path("a.o"), B = path("b.o");
  ASSERT_EQ(drive({"-c", write("a.cpp", "int dup() { return 1; }\n"), "-o", A}),
            0);
  ASSERT_EQ(drive({"-c", write("b.cpp", "int dup() { return 2; }\n"), "-o", B}),
            0);
  int Exit;
  std::string Err = driveCapturingStderr({A, B, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "symbol multiply defined")) << Err;
}

TEST_F(DriverTest, UnreadableBuiltinsArchivesAreReported) {
  auto [Main, Lib] = bitcodeObjects();
  std::string Dir = path("junk-resource-dir");
  ASSERT_FALSE(sys::fs::create_directory(Dir));
  // Learn where the toolchain looks for the builtins from the missing-file
  // error.
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-resource-dir", Dir, Main, Lib, "-o", path("app")}, Exit);
  size_t Begin = Err.find("builtins archive ");
  size_t End = Err.find(": ", Begin);
  ASSERT_NE(Begin, std::string::npos) << Err;
  ASSERT_NE(End, std::string::npos) << Err;
  Begin += StringRef("builtins archive ").size();
  std::string Archive = Err.substr(Begin, End - Begin);
  ASSERT_FALSE(sys::fs::create_directories(sys::path::parent_path(Archive)));
  write("junk", "this is not an archive");
  ASSERT_FALSE(sys::fs::copy_file(path("junk"), Archive));
  Err = driveCapturingStderr(
      {"-resource-dir", Dir, Main, Lib, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "orc-pgo: error: ")) << Err;
  EXPECT_FALSE(contains(Err, "cannot read the compiler-rt builtins")) << Err;
}

TEST_F(DriverTest, InvalidMllvmOptionsAreReported) {
  auto [Main, Lib] = bitcodeObjects();
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-mllvm", "-no-such-option", Main, Lib, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "Unknown command line argument '-no-such-option'"))
      << Err;
}

TEST_F(DriverTest, ConflictingPgoOptionsFailTheBuild) {
  auto [Main, Lib] = bitcodeObjects();
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-mllvm", "-pgo-instrument-entry=false", Main, Lib, "-o", path("app")},
      Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err,
                       "LLVM option -pgo-instrument-entry was set to false, "
                       "but orc-pgo requires true"))
      << Err;
}

TEST_F(DriverTest, ConflictingPgoOptionsAreFoundInEveryMllvmSpelling) {
  auto [Main, Lib] = bitcodeObjects();
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-mllvm=--disable-vp", Main, Lib, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "LLVM option -disable-vp was set to true, but "
                            "orc-pgo requires false"))
      << Err;
}

TEST_F(DriverTest, MatchingPgoOptionsAreAccepted) {
  std::string Exe = path("app");
  ASSERT_EQ(
      drive({"-mllvm", "-pgo-instrument-entry", "-mllvm=-disable-vp=false",
             write("main.cpp", MainSource), write("lib.cpp", LibSource),
             write("native.cpp", NativeSource), "-o", Exe}),
      0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "5994 42\n");
}

TEST_F(DriverTest, PipelineErrorsFailTheBuild) {
  auto [Main, Lib] = bitcodeObjects();
  // A required positional cl::opt, as an embedder of the driver might have:
  // the -mllvm value satisfies it, the pipeline's configurePgoOptions doesn't.
  cl::opt<std::string> Positional(cl::Positional, cl::Required);
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-mllvm", "input", Main, Lib, "-o", path("app")}, Exit);
  Positional.removeArgument();
  cl::ResetAllOptionOccurrences();
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "cannot set orc-pgo's LLVM options: orc-pgo: Not "
                            "enough positional command line arguments"))
      << Err;
}

TEST_F(DriverTest, UnwritableTempDirFailsTheBuild) {
  auto [Main, Lib] = bitcodeObjects();
  // Only .o inputs: clang plans no temporary files for this link, so the
  // AOT object is the first one created.
  TempDirOverride Tmp(path("missing"), /*Create=*/false);
  int Exit;
  std::string Err = driveCapturingStderr({Main, Lib, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "cannot create a temporary file")) << Err;
}

TEST_F(DriverTest, AotObjectWriteErrorsFailTheBuild) {
  auto [Main, Lib] = bitcodeObjects();
  TempDirOverride Tmp(path("tmp"));
  // The AOT object is far larger than this; the error message is far smaller
  // (gtest's stderr capture is a file, too).
  struct rlimit Saved;
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &Saved), 0);
  struct rlimit Small = Saved;
  Small.rlim_cur = 8192;
  void (*SavedHandler)(int) = std::signal(SIGXFSZ, SIG_IGN);
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &Small), 0);
  int Exit;
  std::string Err = driveCapturingStderr({Main, Lib, "-o", path("app")}, Exit);
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &Saved), 0);
  std::signal(SIGXFSZ, SavedHandler);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "orc-pgo: error: ")) << Err;
  EXPECT_TRUE(contains(Err, path("tmp") + "/orc-pgo-aot")) << Err;
  EXPECT_TRUE(contains(Err, "File too large")) << Err;
  EXPECT_EQ(entriesOf(path("tmp")), std::vector<std::string>());
}

TEST_F(DriverTest, FatalErrorsInTheAotBuildAreReported) {
  // main calls an intrinsic of the other architecture: code generation cannot
  // select it and aborts. The driver recovers, reports it and cleans up the
  // temporary object that compiling helper.cpp left.
#if defined(__x86_64__)
  constexpr const char *Foreign = "llvm.aarch64.isb";
  constexpr const char *Call = "call void @llvm.aarch64.isb(i32 15)";
  constexpr const char *Decl = "declare void @llvm.aarch64.isb(i32)";
#else
  constexpr const char *Foreign = "llvm.x86.sse2.pause";
  constexpr const char *Call = "call void @llvm.x86.sse2.pause()";
  constexpr const char *Decl = "declare void @llvm.x86.sse2.pause()";
#endif
  std::string Bad = writeBitcode(
      "bad.o", ("target triple = \"" + sys::getDefaultTargetTriple() +
                "\"\ndefine i32 @main() {\n  " + Call + "\n  ret i32 0\n}\n" +
                Decl + "\n"));
  std::string Source = write("helper.cpp", "int helper() { return 1; }\n");
  TempDirOverride Tmp(path("tmp"));
  // An entry that outlives the recovered crash: destroying it asserts if the
  // driver left the pretty stack trace's list in the crashed state.
  PrettyStackTraceString Outer(
      "DriverTest.FatalErrorsInTheAotBuildAreReported");
  int Exit;
  std::string Err =
      driveCapturingStderr({"-v", Source, Bad, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  // -v names helper.cpp's temporary object: it was created in TMPDIR.
  EXPECT_TRUE(contains(Err, path("tmp") + "/helper-")) << Err;
  EXPECT_TRUE(contains(Err, "LLVM ERROR: Cannot select")) << Err;
  EXPECT_TRUE(contains(Err, Foreign)) << Err;
  EXPECT_TRUE(contains(Err, "orc-pgo: error: ")) << Err;
  EXPECT_FALSE(sys::fs::exists(path("app")));
  EXPECT_EQ(entriesOf(path("tmp")), std::vector<std::string>());
  // The process is still usable.
  EXPECT_EQ(drive({"-fsyntax-only", write("ok.cpp", "int x;\n")}), 0);
}

TEST_F(DriverTest, SharedAndRelocatableLinksOfBitcodeAreRejected) {
  auto [Main, Lib] = bitcodeObjects();
  for (const char *Flag : {"-shared", "-dynamiclib", "-bundle", "-r"}) {
    int Exit;
    std::string Err =
        driveCapturingStderr({Flag, Main, Lib, "-o", path("out")}, Exit);
    EXPECT_EQ(Exit, 1) << Flag;
    EXPECT_TRUE(contains(Err, std::string("orc-pgo: error: ") + Flag +
                                  " is not supported"))
        << Err;
    EXPECT_FALSE(sys::fs::exists(path("out"))) << Flag;
  }
}

TEST_F(DriverTest, SharedLinksOfNativeObjectsAreNotRejected) {
  std::string Native = path("native.o"), Out = path("native.so");
  ASSERT_EQ(drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o",
                   Native}),
            0);
  EXPECT_EQ(drive({"-shared", Native, "-o", Out}), 0);
  EXPECT_TRUE(sys::fs::exists(Out));
}

TEST_F(DriverTest, MultiJobCompilationsDoNotLeakEachTranslationUnit) {
  // With more than one job, cc1 runs in-process for every one of them, so
  // -disable-free (clang's "the process is about to exit") must not be passed.
  int Exit;
  std::string Err =
      driveCapturingStderr({"-###", write("main.cpp", MainSource),
                            write("lib.cpp", LibSource), "-o", path("app")},
                           Exit);
  EXPECT_EQ(Exit, 0);
  EXPECT_TRUE(contains(Err, "\"-cc1\"")) << Err;
  EXPECT_FALSE(contains(Err, "-disable-free")) << Err;
}

TEST_F(DriverTest, SingleJobCompilationsKeepClangsDefaults) {
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-###", "-c", write("lib.cpp", LibSource), "-o", path("lib.o")}, Exit);
  EXPECT_EQ(Exit, 0);
  EXPECT_TRUE(contains(Err, "\"-disable-free\"")) << Err;
}

// ---- Archives ----

constexpr const char *DeadSource =
    "#include <cstdio>\n"
    "static int Linked = (std::puts(\"dead member linked\"), 0);\n"
    "int deadSymbol() { return Linked; }\n";

/// A GNU ar member header: name, mtime, uid, gid, mode, size, terminator.
std::string gnuHeader(StringRef Name, StringRef Size) {
  auto Pad = [](StringRef S, size_t Width) {
    return (S + std::string(Width - S.size(), ' ')).str();
  };
  return Pad(Name, 16) + Pad("0", 12) + Pad("0", 6) + Pad("0", 6) +
         Pad("644", 8) + Pad(Size, 10) + "`\n";
}

/// One good member, then a header whose size field isn't a number.
std::string corruptArchive() {
  return "!<arch>\n" + gnuHeader("a.txt/", "4") + "abcd" +
         gnuHeader("b.txt/", "zz");
}

std::unique_ptr<MemoryBuffer> readFile(StringRef Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf = MemoryBuffer::getFile(Path);
  EXPECT_TRUE(Buf) << Path.str();
  return Buf ? std::move(*Buf) : MemoryBuffer::getMemBuffer("");
}

std::unique_ptr<Module> parseInto(LLVMContext &Ctx, StringRef IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
  EXPECT_TRUE(M) << Err.getMessage().str();
  return M ? std::move(M) : std::make_unique<Module>("invalid", Ctx);
}

std::vector<std::string> memberNames(MemoryBufferRef Buf) {
  std::vector<std::string> Names;
  Expected<std::unique_ptr<object::Archive>> A = object::Archive::create(Buf);
  if (!A) {
    ADD_FAILURE() << errorText(A);
    return Names;
  }
  Error Err = Error::success();
  for (const object::Archive::Child &C : (*A)->children(Err)) {
    Expected<StringRef> Name = C.getName();
    if (!Name) {
      ADD_FAILURE() << errorText(Name);
      break;
    }
    Names.push_back(Name->str());
  }
  EXPECT_EQ(errorText(std::move(Err)), "");
  return Names;
}

/// Whether the archive in Buf has a symbol table.
bool hasSymbolTable(MemoryBufferRef Buf) {
  Expected<std::unique_ptr<object::Archive>> A = object::Archive::create(Buf);
  if (!A) {
    ADD_FAILURE() << errorText(A);
    return false;
  }
  return (*A)->hasSymbolTable();
}

/// Writes a thin archive at ArchivePath that refers to the file MemberPath.
bool writeThinArchive(StringRef ArchivePath, StringRef MemberPath) {
  Expected<NewArchiveMember> Member =
      NewArchiveMember::getFile(MemberPath, /*Deterministic=*/true);
  if (!Member) {
    ADD_FAILURE() << errorText(Member);
    return false;
  }
  std::vector<NewArchiveMember> NM;
  NM.push_back(std::move(*Member));
  std::string Err = errorText(writeArchive(
      ArchivePath, NM, SymtabWritingMode::NoSymtab, object::Archive::K_GNU,
      /*Deterministic=*/true, /*Thin=*/true));
  EXPECT_EQ(Err, "");
  return Err.empty();
}

constexpr const char *MemberA = "declare i32 @b()\n"
                                "define i32 @a() {\n"
                                "  %r = call i32 @b()\n"
                                "  ret i32 %r\n"
                                "}\n";
constexpr const char *MemberB = "define i32 @b() {\n  ret i32 1\n}\n";
constexpr const char *MemberDead =
    "@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] "
    "[{ i32, ptr, ptr } { i32 65535, ptr @init, ptr null }]\n"
    "define internal void @init() {\n  ret void\n}\n"
    "define i32 @dead() {\n  ret i32 0\n}\n";

TEST_F(DriverTest, ArchiverCreatesAndAppends) {
  std::string Ar = path("libab.a"), ObjA, ObjB;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(ObjA));
  ASSERT_NO_FATAL_FAILURE(nativeObjectB(ObjB));
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), ObjA.c_str()}), 0);
  ASSERT_EQ(runArchiver({"append", Ar.c_str(), ObjB.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  EXPECT_EQ(memberNames(Buf->getMemBufferRef()),
            (std::vector<std::string>{"a.o", "b.o"}));
  EXPECT_TRUE(hasSymbolTable(Buf->getMemBufferRef()));
}

TEST_F(DriverTest, ArchiverWritesNoSymbolTableForBitcode) {
  std::string Ar = path("libbc.a");
  std::string M = writeBitcode("b.o", MemberB);
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), M.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  EXPECT_FALSE(hasSymbolTable(Buf->getMemBufferRef()));
}

TEST_F(DriverTest, ArchiverRejectsBadUsage) {
  EXPECT_EQ(runArchiver({}), 1);
  EXPECT_EQ(runArchiver({"create"}), 1);
  ::testing::internal::CaptureStderr();
  int Exit = runArchiver({"frob", "x.a"});
  std::string Err = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "usage: orc-pgo --orc-pgo-ar {create|append}"))
      << Err;
}

TEST_F(DriverTest, ArchiverReportsMissingFiles) {
  std::string Ar = path("x.a"), Missing = path("missing.o");
  EXPECT_EQ(runArchiver({"create", Ar.c_str(), Missing.c_str()}), 1);
  EXPECT_EQ(runArchiver({"append", Ar.c_str()}), 1);
  std::string NotArchive = write("not.a", "plain text\n");
  EXPECT_EQ(runArchiver({"append", NotArchive.c_str()}), 1);
}

TEST_F(DriverTest, ArchiverReportsUnreadableOldMembers) {
  // A thin archive refers to its members by path; remove one.
  std::string Member = writeBitcode("b.o", MemberB), Ar = path("thin.a");
  ASSERT_TRUE(writeThinArchive(Ar, Member));
  ASSERT_FALSE(sys::fs::remove(Member));
  EXPECT_EQ(runArchiver({"append", Ar.c_str()}), 1);
}

TEST_F(DriverTest, ArchiverReportsCorruptArchives) {
  std::string Ar = write("corrupt.a", corruptArchive());
  EXPECT_EQ(runArchiver({"append", Ar.c_str()}), 1);
}

TEST_F(DriverTest, ArchiverReportsWriteErrors) {
  std::string Ar = path("no-such-dir/x.a"), Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  EXPECT_EQ(runArchiver({"create", Ar.c_str(), Obj.c_str()}), 1);
}

TEST_F(DriverTest, ClassifiesArchives) {
  std::string Native = path("libn.a"), Bitcode = path("libb.a"),
              Empty = path("libe.a"), Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  ASSERT_EQ(runArchiver({"create", Native.c_str(), Obj.c_str()}), 0);
  std::string M = writeBitcode("b.o", MemberB);
  ASSERT_EQ(runArchiver({"create", Bitcode.c_str(), M.c_str()}), 0);
  ASSERT_EQ(runArchiver({"create", Empty.c_str()}), 0);
  auto Classify = [](StringRef P) {
    std::unique_ptr<MemoryBuffer> Buf = readFile(P);
    Expected<ArchiveContents> Kind = classifyArchive(Buf->getMemBufferRef());
    EXPECT_EQ(errorText(Kind), "");
    // The expectations below then fail for a Native archive too.
    return Kind ? *Kind : ArchiveContents::Bitcode;
  };
  EXPECT_EQ(Classify(Native), ArchiveContents::Native);
  EXPECT_EQ(Classify(Bitcode), ArchiveContents::Bitcode);
  EXPECT_EQ(Classify(Empty), ArchiveContents::Native);
}

TEST_F(DriverTest, RejectsMixedArchives) {
  std::string Mixed = path("libmixed.a"), Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  std::string M = writeBitcode("b.o", MemberB);
  ASSERT_EQ(runArchiver({"create", Mixed.c_str(), Obj.c_str(), M.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Mixed);
  Expected<ArchiveContents> Kind = classifyArchive(Buf->getMemBufferRef());
  std::string Err = errorText(Kind);
  EXPECT_TRUE(contains(Err, "libmixed.a")) << Err;
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
}

TEST_F(DriverTest, ClassifyReportsMalformedArchives) {
  Expected<ArchiveContents> Garbage =
      classifyArchive(MemoryBufferRef("plain text, not an archive", "g.a"));
  EXPECT_TRUE(contains(errorText(Garbage), "g.a"));
  std::string Corrupt = corruptArchive();
  Expected<ArchiveContents> BadHeader =
      classifyArchive(MemoryBufferRef(Corrupt, "corrupt.a"));
  EXPECT_TRUE(contains(errorText(BadHeader), "corrupt.a"));

  std::string Member = writeBitcode("b.o", MemberB), Ar = path("thin.a");
  ASSERT_TRUE(writeThinArchive(Ar, Member));
  ASSERT_FALSE(sys::fs::remove(Member));
  std::unique_ptr<MemoryBuffer> Thin = readFile(Ar);
  Expected<ArchiveContents> MissingMember =
      classifyArchive(Thin->getMemBufferRef());
  EXPECT_TRUE(contains(errorText(MissingMember), "thin.a"));
}

TEST_F(DriverTest, LinksOnlyTheMembersAStaticLinkerWould) {
  std::string Ar = path("lib.a");
  std::string A = writeBitcode("m_a.o", MemberA),
              B = writeBitcode("m_b.o", MemberB),
              Dead = writeBitcode("m_dead.o", MemberDead);
  ASSERT_EQ(
      runArchiver({"create", Ar.c_str(), A.c_str(), B.c_str(), Dead.c_str()}),
      0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  LLVMContext Ctx;
  std::unique_ptr<Module> Dest = parseInto(Ctx, "declare i32 @a()\n"
                                                "define i32 @main() {\n"
                                                "  %r = call i32 @a()\n"
                                                "  ret i32 %r\n"
                                                "}\n");
  Expected<unsigned> Linked =
      linkArchiveMembers(*Dest, {Buf->getMemBufferRef()}, {});
  ASSERT_EQ(errorText(Linked), "");
  // m_a defines a and needs b, so m_b follows; nothing needs m_dead.
  EXPECT_EQ(*Linked, 2u);
  Function *BFn = Dest->getFunction("b");
  ASSERT_NE(BFn, nullptr);
  EXPECT_FALSE(BFn->isDeclaration());
  EXPECT_EQ(Dest->getFunction("dead"), nullptr);
  EXPECT_EQ(Dest->getNamedGlobal("llvm.global_ctors"), nullptr);
}

TEST_F(DriverTest, NativeObjectsPullMembers) {
  std::string Native = path("use.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("use.c", "extern int c(void);\n"
                                  "int use(void) { return c(); }\n"
                                  "int other(void) { return 4; }\n"),
                   "-o", Native}),
            0);
  std::string Ar = path("lib.a");
  std::string C = writeBitcode("m_c.o", "define i32 @c() {\n  ret i32 3\n}\n");
  std::string Other =
      writeBitcode("m_other.o", "define i32 @other() {\n  ret i32 5\n}\n");
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), C.c_str(), Other.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> ArBuf = readFile(Ar), ObjBuf = readFile(Native);
  LLVMContext Ctx;
  // Dest's reference to other is resolved by the native object, so m_other
  // stays out; the native object's reference to c pulls m_c in.
  std::unique_ptr<Module> Dest = parseInto(Ctx, "declare i32 @other()\n"
                                                "define i32 @main() {\n"
                                                "  %r = call i32 @other()\n"
                                                "  ret i32 %r\n"
                                                "}\n");
  Expected<unsigned> Linked = linkArchiveMembers(
      *Dest, {ArBuf->getMemBufferRef()}, {ObjBuf->getMemBufferRef()});
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 1u);
  Function *CFn = Dest->getFunction("c");
  ASSERT_NE(CFn, nullptr);
  EXPECT_FALSE(CFn->isDeclaration());
  EXPECT_TRUE(Dest->getFunction("other")->isDeclaration());
}

TEST_F(DriverTest, NativeDefinitionsSatisfyNativeReferences) {
  // use.o needs c, which c.o defines, so the archive's c stays out.
  std::string Use = path("use.o"), C = path("c.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("use.c", "extern int c(void);\n"
                                  "int use(void) { return c(); }\n"),
                   "-o", Use}),
            0);
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("c.c", "int c(void) { return 3; }\n"), "-o", C}),
            0);
  std::string Ar = path("lib.a");
  std::string M = writeBitcode("m_c.o", "define i32 @c() {\n  ret i32 4\n}\n");
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), M.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> ArBuf = readFile(Ar), UseBuf = readFile(Use),
                                CBuf = readFile(C);
  LLVMContext Ctx;
  Module Dest("dest", Ctx);
  Expected<unsigned> Linked =
      linkArchiveMembers(Dest, {ArBuf->getMemBufferRef()},
                         {UseBuf->getMemBufferRef(), CBuf->getMemBufferRef()});
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 0u);
}

TEST_F(DriverTest, MainIsResolvedFromArchives) {
  std::string Ar = path("lib.a");
  std::string Main =
      writeBitcode("m_main.o", "define i32 @main() {\n  ret i32 0\n}\n");
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), Main.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  LLVMContext Ctx;
  Module Dest("dest", Ctx);
  Expected<unsigned> Linked =
      linkArchiveMembers(Dest, {Buf->getMemBufferRef()}, {});
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 1u);
}

TEST_F(DriverTest, WeakReferencesDoNotPullMembers) {
  std::string Ar = path("lib.a");
  std::string W = writeBitcode("m_w.o", "define i32 @w() {\n  ret i32 0\n}\n");
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), W.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  LLVMContext Ctx;
  std::unique_ptr<Module> Dest =
      parseInto(Ctx, "declare extern_weak i32 @w()\n@p = global ptr @w\n");
  Expected<unsigned> Linked =
      linkArchiveMembers(*Dest, {Buf->getMemBufferRef()}, {});
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 0u);
}

TEST_F(DriverTest, TheFirstDefiningMemberWins) {
  std::string Ar = path("lib.a");
  std::string X1 = writeBitcode("x1.o", "define i32 @x() {\n  ret i32 1\n}\n");
  std::string X2 = writeBitcode("x2.o", "define i32 @x() {\n  ret i32 2\n}\n");
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), X1.c_str(), X2.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  LLVMContext Ctx;
  std::unique_ptr<Module> Dest = parseInto(Ctx, "declare i32 @x()\n"
                                                "define i32 @main() {\n"
                                                "  %r = call i32 @x()\n"
                                                "  ret i32 %r\n"
                                                "}\n");
  Expected<unsigned> Linked =
      linkArchiveMembers(*Dest, {Buf->getMemBufferRef()}, {});
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 1u);
  Function *X = Dest->getFunction("x");
  ASSERT_NE(X, nullptr);
  auto *Ret = cast<ReturnInst>(X->getEntryBlock().getTerminator());
  EXPECT_EQ(cast<ConstantInt>(Ret->getReturnValue())->getZExtValue(), 1u);
}

TEST_F(DriverTest, ArchiveMemberErrorsAreReported) {
  LLVMContext Ctx;
  // A needed member that also redefines one of Dest's functions.
  std::string Ar = path("libdup.a");
  std::string XY = writeBitcode("xy.o", "define i32 @x() {\n  ret i32 1\n}\n"
                                        "define i32 @y() {\n  ret i32 2\n}\n");
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), XY.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Dup = readFile(Ar);
  std::unique_ptr<Module> Dest =
      parseInto(Ctx, "define i32 @x() {\n  ret i32 0\n}\n"
                     "declare i32 @y()\n"
                     "define i32 @main() {\n"
                     "  %r = call i32 @y()\n"
                     "  ret i32 %r\n"
                     "}\n");
  Expected<unsigned> DupLinked =
      linkArchiveMembers(*Dest, {Dup->getMemBufferRef()}, {});
  std::string Err = errorText(DupLinked);
  EXPECT_TRUE(contains(Err, "libdup.a(xy.o)")) << Err;
  EXPECT_TRUE(contains(Err, "symbol multiply defined")) << Err;

  // A member with bitcode magic and junk after it.
  std::string BadAr = path("libbad.a");
  std::string Bad = write("bad.o", StringRef("BC\xC0\xDE junk", 9));
  ASSERT_EQ(runArchiver({"create", BadAr.c_str(), Bad.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> BadBuf = readFile(BadAr);
  Module Empty("empty", Ctx);
  Expected<unsigned> BadLinked =
      linkArchiveMembers(Empty, {BadBuf->getMemBufferRef()}, {});
  Err = errorText(BadLinked);
  EXPECT_TRUE(contains(Err, "libbad.a")) << Err;
  EXPECT_TRUE(contains(Err, "bad.o")) << Err;

  Expected<unsigned> NotArchive = linkArchiveMembers(
      Empty, {MemoryBufferRef("plain text, not an archive", "g.a")}, {});
  EXPECT_TRUE(contains(errorText(NotArchive), "g.a"));
}

TEST_F(DriverTest, BadNativeObjectsAreReported) {
  LLVMContext Ctx;
  Module Dest("dest", Ctx);
  Expected<unsigned> Junk =
      linkArchiveMembers(Dest, {}, {MemoryBufferRef("junk", "junk.o")});
  EXPECT_TRUE(contains(errorText(Junk), "junk.o"));

  // An ELF object whose only symbol names a string past its string table.
  SmallString<0> Elf;
  raw_svector_ostream OS(Elf);
  yaml::Input YIn("--- !ELF\n"
                  "FileHeader:\n"
                  "  Class:   ELFCLASS64\n"
                  "  Data:    ELFDATA2LSB\n"
                  "  Type:    ET_REL\n"
                  "  Machine: EM_X86_64\n"
                  "Symbols:\n"
                  "  - Name:    bad\n"
                  "    StName:  0x1000\n"
                  "    Binding: STB_GLOBAL\n");
  ASSERT_TRUE(yaml::convertYAML(
      YIn, OS, [](const Twine &Msg) { ADD_FAILURE() << Msg.str(); }));
  Expected<unsigned> BadName =
      linkArchiveMembers(Dest, {}, {MemoryBufferRef(Elf, "bad-name.o")});
  EXPECT_TRUE(contains(errorText(BadName), "bad-name.o"));
}

TEST_F(DriverTest, LinksBitcodeAndNativeArchives) {
  auto [Main, Lib] = bitcodeObjects();
  std::string Dead = path("dead.o"), LibBc = path("libbc.a");
  ASSERT_EQ(drive({"-c", write("dead.cpp", DeadSource), "-o", Dead}), 0);
  ASSERT_EQ(drive({"--orc-pgo-ar", "create", LibBc, Lib, Dead}), 0);
  std::string Native = path("native.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o",
                   Native}),
            0);
  ASSERT_EQ(drive({"--orc-pgo-ar", "create", path("libnat.a"), Native}), 0);
  std::string Empty = path("empty");
  ASSERT_FALSE(sys::fs::create_directory(Empty));
  std::string Exe = path("app");
  // -lnat is found in the second -L directory and is native, so the system
  // linker gets it; -lm is in neither directory.
  ASSERT_EQ(drive({Main, LibBc, "-L" + Empty, "-L" + std::string(Dir), "-lnat",
                   "-lm", "-o", Exe}),
            0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "5994 42\n");
  EXPECT_TRUE(hasSymbol(Exe, "__orc_pgo_start"));
}

TEST_F(DriverTest, UnusedBitcodeArchivesAreDropped) {
  std::string Main = path("main.o"), Lib = path("lib.o"),
              Native = path("native.o");
  ASSERT_EQ(
      drive({"-c", "-fno-lto", write("main.cpp", MainSource), "-o", Main}), 0);
  ASSERT_EQ(drive({"-c", "-fno-lto", write("lib.cpp", LibSource), "-o", Lib}),
            0);
  ASSERT_EQ(drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o",
                   Native}),
            0);
  std::string Dead = path("dead.o"), LibDead = path("libdead.a");
  ASSERT_EQ(drive({"-c", write("dead.cpp", DeadSource), "-o", Dead}), 0);
  ASSERT_EQ(runArchiver({"create", LibDead.c_str(), Dead.c_str()}), 0);
  std::string Exe = path("app");
  ASSERT_EQ(drive({Main, Lib, Native, LibDead, "-o", Exe}), 0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "5994 42\n");
  EXPECT_FALSE(hasSymbol(Exe, "__orc_pgo_start"));
}

TEST_F(DriverTest, MixedArchivesAreRejected) {
  auto [Main, Lib] = bitcodeObjects();
  std::string Mixed = path("libmixed.a"), Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  ASSERT_EQ(runArchiver({"create", Mixed.c_str(), Obj.c_str(), Lib.c_str()}),
            0);
  int Exit;
  std::string Err =
      driveCapturingStderr({Main, Mixed, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
}

TEST_F(DriverTest, BitcodeLibrariesMustBePassedByPath) {
  auto [Main, Lib] = bitcodeObjects();
  ASSERT_EQ(runArchiver({"create", path("libbc.a").c_str(), Lib.c_str()}), 0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, "-L" + std::string(Dir), "-lbc", "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "-lbc resolves to the bitcode archive")) << Err;
  EXPECT_TRUE(contains(Err, "pass bitcode archives by path")) << Err;
}

TEST_F(DriverTest, UnreadableLibrariesAreReported) {
  auto [Main, Lib] = bitcodeObjects();
  write("libjunk.a", "plain text, not an archive");
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, Lib, "-L" + std::string(Dir), "-ljunk", "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "libjunk.a")) << Err;
}

TEST_F(DriverTest, CorruptArchiveMembersAreReported) {
  auto [Main, Lib] = bitcodeObjects();
  std::string Bad = write("bad.o", StringRef("BC\xC0\xDE junk", 9));
  std::string LibBad = path("libbad.a");
  ASSERT_EQ(runArchiver({"create", LibBad.c_str(), Lib.c_str(), Bad.c_str()}),
            0);
  int Exit;
  std::string Err =
      driveCapturingStderr({Main, LibBad, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "bad.o")) << Err;
}

/// A good bitcode member, then a header whose size field isn't a number.
std::string corruptBitcodeArchive(StringRef BitcodeFile) {
  std::string Bytes = readFile(BitcodeFile)->getBuffer().str();
  return "!<arch>\n" + gnuHeader("m.o/", std::to_string(Bytes.size())) + Bytes +
         (Bytes.size() % 2 ? "\n" : "") + gnuHeader("b.txt/", "zz");
}

TEST_F(DriverTest, ArchivesAreScannedInTheOrderGiven) {
  std::string Obj = path("nat.o"), Nat = path("libnat.a"), Bc = path("libbc.a");
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("nat.c", "int f(void) { return 1; }\n"), "-o", Obj}),
            0);
  ASSERT_EQ(runArchiver({"create", Nat.c_str(), Obj.c_str()}), 0);
  std::string M = writeBitcode("m_f.o", "define i32 @f() {\n  ret i32 2\n}\n");
  ASSERT_EQ(runArchiver({"create", Bc.c_str(), M.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> NatBuf = readFile(Nat), BcBuf = readFile(Bc);
  ArchiveInput N(NatBuf->getMemBufferRef(), /*Whole=*/false,
                 ArchiveContents::Native),
      B(BcBuf->getMemBufferRef());
  LLVMContext Ctx;
  constexpr const char *Program = "declare i32 @f()\n"
                                  "define i32 @main() {\n"
                                  "  %r = call i32 @f()\n"
                                  "  ret i32 %r\n"
                                  "}\n";
  // Both archives define f. The one given first wins, as with a static
  // linker: here the native one, so no bitcode member is linked.
  std::unique_ptr<Module> NativeFirst = parseInto(Ctx, Program);
  Expected<unsigned> Linked = linkArchiveMembers(*NativeFirst, {N, B}, {});
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 0u);
  EXPECT_TRUE(NativeFirst->getFunction("f")->isDeclaration());

  std::unique_ptr<Module> BitcodeFirst = parseInto(Ctx, Program);
  Linked = linkArchiveMembers(*BitcodeFirst, {B, N}, {});
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 1u);
  EXPECT_FALSE(BitcodeFirst->getFunction("f")->isDeclaration());
}

TEST_F(DriverTest, ClassifyLibraryLeavesWhatTheLinkerMayReadToIt) {
  std::string Obj, Junk = write("junk.o", "junk");
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  std::unique_ptr<MemoryBuffer> NativeAr, JunkAr;
  ASSERT_NO_FATAL_FAILURE(archiveOf({Obj}, NativeAr));
  ASSERT_NO_FATAL_FAILURE(archiveOf({Obj, Junk}, JunkAr));
  ASSERT_NE(NativeAr, nullptr);
  ASSERT_NE(JunkAr, nullptr);
  std::string M = writeBitcode("b.o", MemberB), Bc = path("libbc.a"),
              Mixed = path("libmixed.a");
  ASSERT_EQ(runArchiver({"create", Bc.c_str(), M.c_str()}), 0);
  ASSERT_EQ(runArchiver({"create", Mixed.c_str(), Obj.c_str(), M.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> BcBuf = readFile(Bc),
                                MixedBuf = readFile(Mixed);
  std::string Corrupt = corruptArchive(), CorruptBc = corruptBitcodeArchive(M);

  auto Classify = [](MemoryBufferRef B) -> std::string {
    Expected<std::optional<ArchiveContents>> Kind = classifyLibrary(B);
    if (!Kind)
      return "error: " + errorText(Kind);
    if (!*Kind)
      return "left to the linker";
    return **Kind == ArchiveContents::Native ? "native" : "bitcode";
  };
  EXPECT_EQ(Classify(NativeAr->getMemBufferRef()), "native");
  EXPECT_EQ(Classify(BcBuf->getMemBufferRef()), "bitcode");
  // The linker may be able to read what orc-pgo can't, if there is nothing
  // bitcode in it: a member that isn't an object file, a broken archive.
  EXPECT_EQ(Classify(JunkAr->getMemBufferRef()), "left to the linker");
  EXPECT_EQ(Classify(MemoryBufferRef("plain text", "g.a")),
            "left to the linker");
  EXPECT_EQ(Classify(MemoryBufferRef(Corrupt, "corrupt.a")),
            "left to the linker");
  // Archives with bitcode members are orc-pgo's to read.
  std::string Err = Classify(MixedBuf->getMemBufferRef());
  EXPECT_TRUE(contains(Err, "error: ")) << Err;
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
  Err = Classify(MemoryBufferRef(CorruptBc, "corrupt-bc.a"));
  EXPECT_TRUE(contains(Err, "error: ")) << Err;
  EXPECT_TRUE(contains(Err, "corrupt-bc.a")) << Err;
}

// ---- Whole archives and native archives ----

TEST_F(DriverTest, WholeArchivesLinkEveryMember) {
  std::string Ar = path("lib.a");
  std::string A = writeBitcode("m_a.o", MemberA),
              B = writeBitcode("m_b.o", MemberB),
              Dead = writeBitcode("m_dead.o", MemberDead);
  ASSERT_EQ(
      runArchiver({"create", Ar.c_str(), A.c_str(), B.c_str(), Dead.c_str()}),
      0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  LLVMContext Ctx;
  Module Dest("dest", Ctx);
  Expected<unsigned> Linked = linkArchiveMembers(
      Dest, {ArchiveInput(Buf->getMemBufferRef(), /*Whole=*/true)}, {});
  ASSERT_EQ(errorText(Linked), "");
  // Nothing references a, b or dead; a whole archive links them anyway,
  // static initializer included.
  EXPECT_EQ(*Linked, 3u);
  EXPECT_NE(Dest.getFunction("dead"), nullptr);
  EXPECT_NE(Dest.getNamedGlobal("llvm.global_ctors"), nullptr);
}

TEST_F(DriverTest, WholeArchiveMemberErrorsAreReported) {
  std::string Ar = path("libdup.a");
  std::string X = writeBitcode("x.o", "define i32 @x() {\n  ret i32 1\n}\n");
  ASSERT_EQ(runArchiver({"create", Ar.c_str(), X.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> Buf = readFile(Ar);
  LLVMContext Ctx;
  std::unique_ptr<Module> Dest =
      parseInto(Ctx, "define i32 @x() {\n  ret i32 0\n}\n");
  Expected<unsigned> Linked = linkArchiveMembers(
      *Dest, {ArchiveInput(Buf->getMemBufferRef(), /*Whole=*/true)}, {});
  std::string Err = errorText(Linked);
  EXPECT_TRUE(contains(Err, "libdup.a(x.o)")) << Err;
  EXPECT_TRUE(contains(Err, "symbol multiply defined")) << Err;
}

TEST_F(DriverTest, NativeArchiveMembersPullBitcodeMembersOnDemand) {
  // u1.o references c1 and u2.o references c2; only bitcode members define
  // them.
  std::string U1 = path("u1.o"), U2 = path("u2.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("u1.c", "extern int c1(void);\n"
                                 "int u1(void) { return c1(); }\n"),
                   "-o", U1}),
            0);
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("u2.c", "extern int c2(void);\n"
                                 "int u2(void) { return c2(); }\n"),
                   "-o", U2}),
            0);
  std::string Nat = path("libnat.a"), Bc = path("libbc.a");
  ASSERT_EQ(runArchiver({"create", Nat.c_str(), U1.c_str(), U2.c_str()}), 0);
  std::string C1 = writeBitcode("m_c1.o",
                                "define i32 @c1() {\n  ret i32 1\n}\n"),
              C2 = writeBitcode("m_c2.o",
                                "define i32 @c2() {\n  ret i32 2\n}\n");
  ASSERT_EQ(runArchiver({"create", Bc.c_str(), C1.c_str(), C2.c_str()}), 0);
  std::unique_ptr<MemoryBuffer> NatBuf = readFile(Nat), BcBuf = readFile(Bc);

  auto Link = [&](bool Whole, Module &Dest) {
    return linkArchiveMembers(Dest,
                              {ArchiveInput(BcBuf->getMemBufferRef()),
                               ArchiveInput(NatBuf->getMemBufferRef(), Whole,
                                            ArchiveContents::Native)},
                              {});
  };
  LLVMContext Ctx;
  // Dest needs u1, which is a native member's: its reference to c1 counts,
  // u2's does not.
  std::unique_ptr<Module> Dest = parseInto(Ctx, "declare i32 @u1()\n"
                                                "define i32 @main() {\n"
                                                "  %r = call i32 @u1()\n"
                                                "  ret i32 %r\n"
                                                "}\n");
  Expected<unsigned> Linked = Link(/*Whole=*/false, *Dest);
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 1u);
  EXPECT_NE(Dest->getFunction("c1"), nullptr);
  EXPECT_EQ(Dest->getFunction("c2"), nullptr);

  // A whole native archive loads u2 too, and with it its reference to c2.
  Module Empty("empty", Ctx);
  Linked = Link(/*Whole=*/true, Empty);
  ASSERT_EQ(errorText(Linked), "");
  EXPECT_EQ(*Linked, 2u);
  EXPECT_NE(Empty.getFunction("c2"), nullptr);
}

TEST_F(DriverTest, BadNativeArchiveMembersAreReported) {
  std::string Junk = write("junk.o", "junk");
  Expected<NewArchiveMember> Member =
      NewArchiveMember::getFile(Junk, /*Deterministic=*/true);
  ASSERT_EQ(errorText(Member), "");
  std::vector<NewArchiveMember> NM;
  NM.push_back(std::move(*Member));
  Expected<std::unique_ptr<MemoryBuffer>> Buf = writeArchiveToBuffer(
      NM, SymtabWritingMode::NoSymtab, object::Archive::getDefaultKind(),
      /*Deterministic=*/true, /*Thin=*/false);
  ASSERT_EQ(errorText(Buf), "");
  LLVMContext Ctx;
  Module Dest("dest", Ctx);
  Expected<unsigned> Linked = linkArchiveMembers(
      Dest,
      {ArchiveInput((*Buf)->getMemBufferRef(), /*Whole=*/false,
                    ArchiveContents::Native)},
      {});
  std::string Err = errorText(Linked);
  EXPECT_TRUE(contains(Err, "junk.o")) << Err;

  Expected<unsigned> NotArchive = linkArchiveMembers(
      Dest,
      {ArchiveInput(MemoryBufferRef("plain text, not an archive", "n.a"),
                    /*Whole=*/false, ArchiveContents::Native)},
      {});
  EXPECT_TRUE(contains(errorText(NotArchive), "n.a"));
}

TEST_F(DriverTest, UnreadableNativeLibrariesAreLeftToTheLinker) {
  // libbad.a has a member that is not an object file. It is the linker's to
  // accept (GNU ld never extracts it) or reject (ld64 reads every member).
  auto [Main, Lib] = bitcodeObjects();
  std::string Obj = path("native.o"), Junk = write("junk.o", "junk");
  ASSERT_EQ(
      drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o", Obj}),
      0);
  std::unique_ptr<MemoryBuffer> Ar;
  ASSERT_NO_FATAL_FAILURE(archiveOf({Obj, Junk}, Ar));
  ASSERT_NE(Ar, nullptr);
  write("libbad.a", Ar->getBuffer());
  std::string Exe = path("app");
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, Lib, "-L" + std::string(Dir), "-lbad", "-o", Exe}, Exit);
  EXPECT_FALSE(contains(Err, "orc-pgo: error: '" + path("libbad.a") + "'"))
      << Err;
#ifdef __APPLE__
  EXPECT_TRUE(contains(Err, "not a mach-o file")) << Err;
#else
  ASSERT_EQ(Exit, 0) << Err;
  EXPECT_EQ(run(Exe).second, "5994 42\n");
#endif
}

TEST_F(DriverTest, MalformedNativeLibrariesAreLeftToTheLinker) {
  auto [Main, Lib] = bitcodeObjects();
  write("libcorrupt.a", corruptArchive());
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, Lib, "-L" + std::string(Dir), "-lcorrupt", "-o", path("app")},
      Exit);
  // It is the linker that fails: addOne is undefined.
  EXPECT_NE(Exit, 0);
  EXPECT_TRUE(contains(Err, "linker command failed")) << Err;
}

TEST_F(DriverTest, MixedLibrariesFoundThroughDashLAreRejected) {
  auto [Main, Lib] = bitcodeObjects();
  std::string Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  ASSERT_EQ(runArchiver({"create", path("libmixed.a").c_str(), Obj.c_str(),
                         Lib.c_str()}),
            0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, "-L" + std::string(Dir), "-lmixed", "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
}

TEST_F(DriverTest, LibrariesAreOnlyReadForLinksWithBitcode) {
  // Without bitcode inputs the link is the system linker's alone, whatever
  // the libraries are.
  std::string Obj = path("native.o"), Lib = path("lib.o"),
              Main = path("main.o");
  ASSERT_EQ(
      drive({"-c", "-fno-lto", write("main.cpp", MainSource), "-o", Main}), 0);
  ASSERT_EQ(drive({"-c", "-fno-lto", write("lib.cpp", LibSource), "-o", Lib}),
            0);
  ASSERT_EQ(
      drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o", Obj}),
      0);
  std::string Bitcode = writeBitcode("b.o", MemberB);
  ASSERT_EQ(runArchiver({"create", path("libmixed.a").c_str(), Obj.c_str(),
                         Bitcode.c_str()}),
            0);
  write("libcorrupt.a", corruptArchive());
  int Exit;
  std::string Err =
      driveCapturingStderr({Main, Lib, Obj, "-L" + std::string(Dir), "-lmixed",
                            "-lcorrupt", "-o", path("app")},
                           Exit);
  EXPECT_FALSE(contains(Err, "archive mixes native and bitcode members"))
      << Err;
}

TEST_F(DriverTest, ArchivesAreLinkedInCommandLineOrder) {
  // libnat.a and libbc.a both define f; the one given first on the command
  // line provides it.
  std::string Obj = path("nat.o"), Nat = path("libnat.a"), Bc = path("libbc.a"),
              Main = path("main.o"), M = path("f.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("nat.c", "int f(void) { return 1; }\n"), "-o", Obj}),
            0);
  ASSERT_EQ(runArchiver({"create", Nat.c_str(), Obj.c_str()}), 0);
  ASSERT_EQ(drive({"-c", write("f.cpp", "extern \"C\" int f() { return 2; }\n"),
                   "-o", M}),
            0);
  ASSERT_EQ(runArchiver({"create", Bc.c_str(), M.c_str()}), 0);
  ASSERT_EQ(
      drive({"-c",
             write("main.cpp", "#include <cstdio>\n"
                               "extern \"C\" int f();\n"
                               "int main() { std::printf(\"%d\\n\", f()); }\n"),
             "-o", Main}),
      0);
  std::string Exe = path("app");
  ASSERT_EQ(drive({Main, Nat, Bc, "-o", Exe}), 0);
  EXPECT_EQ(run(Exe).second, "1\n");
  ASSERT_EQ(drive({Main, Bc, Nat, "-o", Exe}), 0);
  EXPECT_EQ(run(Exe).second, "2\n");
}

/// A directory with a shared libfoo (foo returns 5) next to a bitcode libfoo.a
/// (foo returns 7), and a bitcode main.o that prints foo().
class SharedLibraryTest : public DriverTest {
protected:
  void SetUp() override {
    DriverTest::SetUp();
#ifdef __APPLE__
    constexpr const char *Suffix = ".dylib";
#else
    constexpr const char *Suffix = ".so";
#endif
    ASSERT_EQ(
        drive({"-shared", "-fno-lto",
               write("foo5.cpp", "extern \"C\" int foo() { return 5; }\n"),
               "-o", path(std::string("libfoo") + Suffix)}),
        0);
    std::string Foo = path("foo.o"), Archive = path("libfoo.a");
    ASSERT_EQ(
        drive({"-c",
               write("foo7.cpp", "extern \"C\" int foo() { return 7; }\n"),
               "-o", Foo}),
        0);
    ASSERT_EQ(runArchiver({"create", Archive.c_str(), Foo.c_str()}), 0);
    Main = path("main.o");
    ASSERT_EQ(drive({"-c",
                     write("main.cpp",
                           "#include <cstdio>\n"
                           "extern \"C\" int foo();\n"
                           "int main() { std::printf(\"%d\\n\", foo()); }\n"),
                     "-o", Main}),
              0);
  }

  std::string Main;
};

TEST_F(SharedLibraryTest, SharedLibrariesComeBeforeStaticOnes) {
  // The linker takes libfoo.so (libfoo.dylib), not libfoo.a, from the same
  // directory.
  std::string Exe = path("app");
  int Exit;
  std::string Err =
      driveCapturingStderr({Main, "-L" + std::string(Dir), "-lfoo",
                            "-Wl,-rpath," + std::string(Dir), "-o", Exe},
                           Exit);
  ASSERT_EQ(Exit, 0) << Err;
  EXPECT_EQ(run(Exe).second, "5\n");
}

TEST_F(SharedLibraryTest, StaticLinksIgnoreSharedLibraries) {
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-static", Main, "-L" + std::string(Dir), "-lfoo", "-o", path("app")},
      Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "-lfoo resolves to the bitcode archive")) << Err;
}

/// main.o calls natfun, defined by libnat.a's native member, which calls
/// provided, which only the bitcode libprovided.a defines.
class NativeArchiveLinkTest : public DriverTest {
protected:
  void SetUp() override {
    DriverTest::SetUp();
    Main = path("main.o");
    ASSERT_EQ(
        drive({"-c",
               write("main.cpp",
                     "#include <cstdio>\n"
                     "extern \"C\" int natfun(int);\n"
                     "int main() { std::printf(\"%d\\n\", natfun(20)); }\n"),
               "-o", Main}),
        0);
    Provided = path("libprovided.a");
    std::string P = path("provided.o");
    ASSERT_EQ(
        drive({"-c",
               write("provided.cpp",
                     "extern \"C\" int provided(int x) { return 2 * x; }\n"),
               "-o", P}),
        0);
    ASSERT_EQ(runArchiver({"create", Provided.c_str(), P.c_str()}), 0);
    Native = path("libnat.a");
    std::string N = path("nat.o");
    ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                     write("nat.c",
                           "int provided(int);\n"
                           "int natfun(int x) { return provided(x) + 1; }\n"),
                     "-o", N}),
              0);
    ASSERT_EQ(runArchiver({"create", Native.c_str(), N.c_str()}), 0);
  }

  std::string Main, Provided, Native;
};

TEST_F(NativeArchiveLinkTest, NativeArchiveMembersPullBitcodeArchiveMembers) {
  std::string Exe = path("app");
  ASSERT_EQ(drive({Main, Native, Provided, "-o", Exe}), 0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "41\n");
}

TEST_F(NativeArchiveLinkTest, LibrariesFoundInLibraryDirectoriesAreScanned) {
  std::string Exe = path("app");
  ASSERT_EQ(
      drive({Main, "-L" + std::string(Dir), "-lnat", Provided, "-o", Exe}), 0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "41\n");
}

TEST_F(DriverTest, WholeArchiveOptionsLinkEveryMemberOfBitcodeArchives) {
  auto [Main, Prog] = bitcodeProgram();
  std::string Whole = noisyArchive("whole", "whole member linked"),
              Lazy = noisyArchive("lazy", "lazy member linked");
  std::string Shim = linkerShim(), Exe = path("app");
  int Exit;
  std::string Err =
      driveCapturingStderr({Shim, "-v", Main, Prog, "-Wl,--whole-archive",
                            Whole, "-Wl,--no-whole-archive", Lazy, "-o", Exe},
                           Exit);
  ASSERT_EQ(Exit, 0) << Err;
  // Bitcode archives never reach the system linker.
  EXPECT_FALSE(contains(Err, "libwhole.a")) << Err;
  EXPECT_FALSE(contains(Err, "liblazy.a")) << Err;
  auto [RunExit, Out] = run(Exe);
  EXPECT_EQ(RunExit, 0);
  EXPECT_EQ(Out, "whole member linked\n5994 42\n");
}

TEST_F(DriverTest, WholeArchiveStateIsSavedAndRestored) {
  auto [Main, Prog] = bitcodeProgram();
  std::string Whole = noisyArchive("whole", "whole member linked"),
              Lazy = noisyArchive("lazy", "lazy member linked");
  std::string Shim = linkerShim(), Exe = path("app");
  // CMake's spelling of a whole archive, with the path inside the option.
  ASSERT_EQ(drive({Shim, Main, Prog,
                   "-Wl,--push-state,--whole-archive," + Whole + ",--pop-state",
                   Lazy, "-o", Exe}),
            0);
  EXPECT_EQ(run(Exe).second, "whole member linked\n5994 42\n");

  // The state after --pop-state is the one --push-state saved.
  ASSERT_EQ(
      drive({Shim, Main, Prog, "-Wl,--whole-archive",
             "-Wl,--push-state,--no-whole-archive", Lazy, "-Wl,--pop-state",
             Whole, "-Wl,--no-whole-archive", "-o", Exe}),
      0);
  EXPECT_EQ(run(Exe).second, "whole member linked\n5994 42\n");

  // An unmatched --pop-state changes nothing.
  ASSERT_EQ(drive({Shim, Main, Prog, "-Wl,--pop-state", Lazy, "-o", Exe}), 0);
  EXPECT_EQ(run(Exe).second, "5994 42\n");
}

TEST_F(DriverTest, MixedArchivesInLinkerOptionsAreRejected) {
  auto [Main, Prog] = bitcodeProgram();
  std::string Mixed = path("libmixed.a"), Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  ASSERT_EQ(runArchiver(
                {"create", Mixed.c_str(), Obj.c_str(), path("lib.o").c_str()}),
            0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, Prog, "-Wl,--whole-archive," + Mixed, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
}

TEST_F(DriverTest, RepeatedArchivesAreLinkedOnce) {
  auto [Main, Prog] = bitcodeProgram();
  std::string Lazy = noisyArchive("lazy", "lazy member linked");
  std::string Shim = linkerShim(), Exe = path("app");
  // CMake repeats static libraries to resolve cycles. A later occurrence in
  // a --whole-archive region makes the archive whole.
  ASSERT_EQ(drive({Shim, Main, Prog, Prog, Lazy, "-Wl,--whole-archive", Lazy,
                   Prog, "-Wl,--no-whole-archive", "-o", Exe}),
            0);
  auto [Exit, Out] = run(Exe);
  EXPECT_EQ(Exit, 0);
  EXPECT_EQ(Out, "lazy member linked\n5994 42\n");
}

TEST_F(DriverTest, WholeArchiveOptionsDoNotReachTheRuntimeArchives) {
  // A --whole-archive that is never ended must not apply to the runtime.
  auto [Main, Prog] = bitcodeProgram();
  std::string Exe = path("app");
  std::vector<std::string> Args{"-v", Main, "-Wl,--whole-archive",
                                Prog, "-o", Exe};
#ifdef __APPLE__
  // ld64 has no such option.
  Args.insert(Args.begin(), linkerShim());
#endif
  int Exit;
  std::string Err = driveCapturingStderr(Args, Exit);
  ASSERT_EQ(Exit, 0) << Err;
#ifndef __APPLE__
  // The real GNU ld applied the options: with the runtime archives under a
  // --whole-archive the link would not have worked. They are also ordered on
  // the command line.
  size_t Whole = Err.find("--whole-archive"),
         NoWhole = Err.find("--no-whole-archive");
  ASSERT_NE(Whole, std::string::npos) << Err;
  ASSERT_NE(NoWhole, std::string::npos) << Err;
  EXPECT_LT(Whole, NoWhole) << Err;
#endif
  EXPECT_EQ(run(Exe).second, "5994 42\n");
}

#ifdef __APPLE__
TEST_F(DriverTest, ForceLoadLinksEveryMemberOfBitcodeArchives) {
  auto [Main, Prog] = bitcodeProgram();
  std::string Whole = noisyArchive("whole", "whole member linked"),
              Lazy = noisyArchive("lazy", "lazy member linked");
  const std::vector<std::vector<std::string>> Forms = {
      {"-Wl,-force_load," + Whole},
      {"-force_load", Whole},
      {"-Xlinker", "-force_load", "-Xlinker", Whole},
      {"-Wl,-rpath,/nowhere,-force_load," + Whole}};
  for (const std::vector<std::string> &Form : Forms) {
    std::string Exe = path("app");
    std::vector<std::string> Args{"-v", Main, Prog};
    Args.insert(Args.end(), Form.begin(), Form.end());
    Args.insert(Args.end(), {Lazy, "-o", Exe});
    int Exit;
    std::string Err = driveCapturingStderr(Args, Exit);
    ASSERT_EQ(Exit, 0) << Err;
    EXPECT_FALSE(contains(Err, "force_load")) << Err;
    EXPECT_EQ(run(Exe).second, "whole member linked\n5994 42\n") << Form[0];
    sys::fs::remove(Exe);
  }
}

TEST_F(DriverTest, ForceLoadOfOtherFilesIsLeftToTheLinker) {
  auto [Main, Prog] = bitcodeProgram();
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, Prog, "-Wl,-force_load," + path("missing.a"), "-o", path("app")},
      Exit);
  EXPECT_NE(Exit, 0);
  EXPECT_TRUE(contains(Err, "missing.a")) << Err;
  driveCapturingStderr({Main, Prog, "-force_load",
                        write("junk.a", "not an archive"), "-o", path("app")},
                       Exit);
  EXPECT_NE(Exit, 0);
}

TEST_F(DriverTest, ForceLoadOfMixedArchivesIsRejected) {
  auto [Main, Prog] = bitcodeProgram();
  std::string Mixed = path("libmixed.a"), Obj;
  ASSERT_NO_FATAL_FAILURE(nativeObjectA(Obj));
  ASSERT_EQ(runArchiver(
                {"create", Mixed.c_str(), Obj.c_str(), path("lib.o").c_str()}),
            0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, Prog, "-Wl,-force_load," + Mixed, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
  Err = driveCapturingStderr(
      {Main, Prog, "-force_load", Mixed, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
  // The same when an input follows the option.
  Err = driveCapturingStderr(
      {"-force_load", Mixed, Main, Prog, "-o", path("app")}, Exit);
  EXPECT_EQ(Exit, 1);
  EXPECT_TRUE(contains(Err, "archive mixes native and bitcode members")) << Err;
}

TEST_F(DriverTest, ForceLoadedArchivesFollowTheCommandLineOrder) {
  // Both archives define f, so the second one loaded fails the link, and the
  // error names it. Direct -force_load options are in the order of the
  // command line with the ones in -Wl, and with the source files.
  auto Make = [&](const std::string &Name, const std::string &Value) {
    std::string Obj = path("f" + Name + ".o"),
                Archive = path("lib" + Name + ".a");
    EXPECT_EQ(drive({"-c",
                     write("f" + Name + ".cpp",
                           "extern \"C\" int f() { return " + Value + "; }\n"),
                     "-o", Obj}),
              0);
    EXPECT_EQ(runArchiver({"create", Archive.c_str(), Obj.c_str()}), 0);
    return Archive;
  };
  std::string X = Make("x", "1"), Y = Make("y", "2");
  std::string Src = write("m.cpp", "int main() { return 0; }\n");
  auto Second = [&](const std::vector<std::string> &Args) {
    int Exit;
    std::string Err = driveCapturingStderr(Args, Exit);
    EXPECT_EQ(Exit, 1) << Err;
    return Err;
  };
  std::string Err = Second({"-force_load", X, Src, "-force_load", Y});
  EXPECT_TRUE(contains(Err, "liby.a(fy.o)")) << Err;
  EXPECT_FALSE(contains(Err, "libx.a(fx.o)")) << Err;
  Err = Second({"-force_load", Y, Src, "-force_load", X});
  EXPECT_TRUE(contains(Err, "libx.a(fx.o)")) << Err;
  EXPECT_FALSE(contains(Err, "liby.a(fy.o)")) << Err;
  Err = Second({"-Wl,-force_load," + Y, Src, "-force_load", X});
  EXPECT_TRUE(contains(Err, "libx.a(fx.o)")) << Err;
  EXPECT_FALSE(contains(Err, "liby.a(fy.o)")) << Err;
  Err = Second({"-force_load", X, Src, "-Wl,-force_load," + Y});
  EXPECT_TRUE(contains(Err, "liby.a(fy.o)")) << Err;
  EXPECT_FALSE(contains(Err, "libx.a(fx.o)")) << Err;
}

TEST_F(DriverTest, AllLoadLinksEveryMemberOfBitcodeArchives) {
  auto [Main, Prog] = bitcodeProgram();
  std::string Whole = noisyArchive("whole", "whole member linked");
  for (const char *Option : {"-Wl,-all_load", "-all_load"}) {
    std::string Exe = path("app");
    int Exit;
    std::string Err = driveCapturingStderr(
        {"-v", Main, Prog, Whole, Option, "-o", Exe}, Exit);
    ASSERT_EQ(Exit, 0) << Err;
    // ld64 would load all of the runtime's LLVM archives too.
    EXPECT_FALSE(contains(Err, "all_load")) << Err;
    EXPECT_EQ(run(Exe).second, "whole member linked\n5994 42\n") << Option;
    sys::fs::remove(Exe);
  }
}

TEST_F(NativeArchiveLinkTest, AllLoadAppliesToNativeArchivesOnly) {
  // unused.o, which nothing references, has a constructor (a root for
  // -dead_strip) that needs a symbol only a bitcode member defines. all_load
  // loads it into the program, so that member is needed (or the link fails:
  // dep is undefined).
  std::string Unused = path("unused.o"), Dep = path("dep.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("unused.c",
                         "int dep(void);\n"
                         "__attribute__((constructor)) static void init(void) "
                         "{ dep(); }\n"),
                   "-o", Unused}),
            0);
  ASSERT_EQ(runArchiver({"append", Native.c_str(), Unused.c_str()}), 0);
  ASSERT_EQ(
      drive({"-c", write("dep.cpp", "extern \"C\" int dep() { return 7; }\n"),
             "-o", Dep}),
      0);
  ASSERT_EQ(runArchiver({"append", Provided.c_str(), Dep.c_str()}), 0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-v", Main, Native, Provided, "-Wl,-all_load", "-o", path("app")}, Exit);
  ASSERT_EQ(Exit, 0) << Err;
  // all_load is replaced by -force_load of the native archive.
  EXPECT_FALSE(contains(Err, "all_load")) << Err;
  EXPECT_TRUE(contains(Err, "-force_load")) << Err;
  EXPECT_TRUE(contains(Err, "libnat.a")) << Err;
  EXPECT_EQ(run(path("app")).second, "41\n");

  // The same through -l.
  Err = driveCapturingStderr({Main, "-L" + std::string(Dir), "-lnat", Provided,
                              "-all_load", "-o", path("app2")},
                             Exit);
  EXPECT_EQ(Exit, 0) << Err;
}

TEST_F(SharedLibraryTest, AllLoadSkipsStaticCopiesNextToDylibs) {
  // libfoo.a is a static copy of the dylib that the linker takes: it is not
  // force-loaded.
  ASSERT_EQ(drive({"-c", "-fno-lto",
                   write("foo9.cpp", "extern \"C\" int foo() { return 9; }\n"),
                   "-o", path("foo9.o")}),
            0);
  ASSERT_EQ(
      runArchiver({"create", path("libfoo.a").c_str(), path("foo9.o").c_str()}),
      0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-v", Main, "-L" + std::string(Dir), "-lfoo", "-Wl,-all_load",
       "-Wl,-rpath," + std::string(Dir), "-o", path("app")},
      Exit);
  ASSERT_EQ(Exit, 0) << Err;
  EXPECT_FALSE(contains(Err, "-force_load")) << Err;
  EXPECT_EQ(run(path("app")).second, "5\n");
}

TEST_F(DriverTest, UniversalLibrariesAreLeftToTheLinker) {
  // A universal libuni.a, which only the linker reads.
  auto [Main, Lib] = bitcodeObjects();
  std::string Obj = path("native.o");
  ASSERT_EQ(
      drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o", Obj}),
      0);
  std::unique_ptr<MemoryBuffer> Ar;
  ASSERT_NO_FATAL_FAILURE(archiveOf({Obj}, Ar));
  ASSERT_NE(Ar, nullptr);
  Expected<std::unique_ptr<object::Archive>> A =
      object::Archive::create(Ar->getMemBufferRef());
  ASSERT_EQ(errorText(A), "");
#if defined(__aarch64__)
  object::Slice S(**A, MachO::CPU_TYPE_ARM64, MachO::CPU_SUBTYPE_ARM64_ALL,
                  "arm64", 14);
#else
  object::Slice S(**A, MachO::CPU_TYPE_X86_64, MachO::CPU_SUBTYPE_X86_64_ALL,
                  "x86_64", 12);
#endif
  SmallString<0> Fat;
  raw_svector_ostream OS(Fat);
  ASSERT_EQ(errorText(object::writeUniversalBinaryToStream({S}, OS)), "");
  write("libuni.a", Fat);
  std::string Exe = path("app");
  int Exit;
  std::string Err = driveCapturingStderr(
      {Main, Lib, "-L" + std::string(Dir), "-luni", "-o", Exe}, Exit);
  ASSERT_EQ(Exit, 0) << Err;
  EXPECT_EQ(run(Exe).second, "5994 42\n");
}

TEST_F(NativeArchiveLinkTest, AllLoadForceLoadsEachNativeArchiveOnce) {
  int Exit;
  std::string Err = driveCapturingStderr({"-v", Main, Native, Native, Provided,
                                          "-Wl,-all_load", "-o", path("app")},
                                         Exit);
  ASSERT_EQ(Exit, 0) << Err;
  size_t Count = 0;
  for (size_t At = Err.find("-force_load " + Native); At != std::string::npos;
       At = Err.find("-force_load " + Native, At + 1))
    ++Count;
  EXPECT_EQ(Count, 1u) << Err;
}

TEST_F(NativeArchiveLinkTest, ForceLoadedNativeArchivesAreScannedInFull) {
  // As above: the force-loaded unused.o has a constructor (a root for
  // -dead_strip) that needs dep, which only a bitcode member defines.
  std::string Unused = path("unused.o"), Dep = path("dep.o");
  ASSERT_EQ(drive({"-c", "-fno-lto", "-x", "c",
                   write("unused.c",
                         "int dep(void);\n"
                         "__attribute__((constructor)) static void init(void) "
                         "{ dep(); }\n"),
                   "-o", Unused}),
            0);
  ASSERT_EQ(runArchiver({"append", Native.c_str(), Unused.c_str()}), 0);
  ASSERT_EQ(
      drive({"-c", write("dep.cpp", "extern \"C\" int dep() { return 7; }\n"),
             "-o", Dep}),
      0);
  ASSERT_EQ(runArchiver({"append", Provided.c_str(), Dep.c_str()}), 0);
  std::string Exe = path("app");
  ASSERT_EQ(drive({Main, Provided, "-Wl,-force_load," + Native, "-o", Exe}), 0);
  EXPECT_EQ(run(Exe).second, "41\n");
}
#endif

} // namespace

// Not gtest_main: if a driver job were ever run out of process, it would
// re-exec this binary as `DriverTest -cc1 ...`, which would run the whole suite
// again, recursively. Refuse instead.
int main(int Argc, char **Argv) {
  for (int I = 1; I < Argc; ++I) {
    if (std::strncmp(Argv[I], "-cc1", 4) == 0) {
      std::fprintf(stderr,
                   "DriverTest: refusing to run as %s (a driver job "
                   "was run out of process)\n",
                   Argv[I]);
      return 1;
    }
  }
  // What InitLLVM does for the orc-pgo tool. The signal handlers must be
  // registered before the driver enables crash recovery, or they would replace
  // its handlers (a signal would then kill the test instead of failing a job).
  sys::PrintStackTraceOnErrorSignal(Argv[0]);
  ::testing::InitGoogleTest(&Argc, Argv);
  return RUN_ALL_TESTS();
}
