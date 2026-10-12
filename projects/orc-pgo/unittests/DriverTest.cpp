// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Driver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
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
  std::string Lib = path("lib.o"), Native = path("native.o");
  ASSERT_EQ(drive({"-c", write("lib.cpp", LibSource), "-o", Lib}), 0);
  ASSERT_EQ(drive({"-c", "-fno-lto", write("native.cpp", NativeSource), "-o",
                   Native}),
            0);
  int Exit;
  std::string Err = driveCapturingStderr(
      {"-v", write("main.cpp", MainSource), Lib, Native, "-o", path("app")},
      Exit);
  EXPECT_EQ(Exit, 0) << Err;
  EXPECT_FALSE(contains(Err, "lto_library")) << Err;
  EXPECT_TRUE(contains(Err, "-o " + path("app"))) << Err;
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
