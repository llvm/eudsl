// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Annotate.h"
#include "orc-pgo/Analyses.h"
#include "orc-pgo/PgoOptions.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"

#include <memory>
#include <string>

using namespace llvm;

namespace orc_pgo {

namespace {

constexpr StringLiteral ProfilePath = "/orc-pgo/live.profdata";

/// Whether a PGO warning says that the profile of a function does not match
/// its IR, so that the counts would be applied to the wrong edges or not at
/// all. The diagnostics carry no code, so this goes by their text
/// (InstrProfError::message() and PGOInstrumentation.cpp); AnnotateTest pins
/// it. Warnings that only lose some of a function's profile are not mismatches
/// in this sense: "malformed" and "no profile" leave a root without an entry
/// count, which MustBeAnnotated catches, and "Inconsistent number of value
/// sites" only skips the function's value profile.
bool isProfileMismatch(StringRef Message) {
  return Message.contains("hash mismatch") ||
         Message.contains("Inconsistent number of counts");
}

/// Keeps the first PGO profile error, and the first warning of a mismatch.
/// Other warnings (e.g. a profile "partially ignored, possibly due to the lack
/// of a return path") leave the function annotated as far as LLVM could, and
/// are dropped.
struct ProfileDiagnostics : DiagnosticHandler {
  std::string First;
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    if (DI.getKind() != DK_PGOProfile || DI.getSeverity() > DS_Warning ||
        !First.empty()) {
      return true;
    }
    std::string Text;
    raw_string_ostream OS(Text);
    DiagnosticPrinterRawOStream DP(OS);
    DI.print(DP);
    if (DI.getSeverity() == DS_Error || isProfileMismatch(Text))
      First = std::move(Text);
    return true;
  }
};

} // namespace

Error annotateWithProfile(Module &M, const MemoryBuffer &Profile,
                          ArrayRef<StringRef> MustBeAnnotated) {
  if (Error E = configurePgoOptions())
    return E;
  auto FS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  FS->addFile(ProfilePath, /*ModificationTime=*/0,
              MemoryBuffer::getMemBuffer(Profile.getMemBufferRef(),
                                         /*RequiresNullTerminator=*/false));

  LLVMContext &Ctx = M.getContext();
  std::unique_ptr<DiagnosticHandler> Previous = Ctx.getDiagnosticHandler();
  auto Diagnostics = std::make_unique<ProfileDiagnostics>();
  ProfileDiagnostics *Collected = Diagnostics.get();
  Ctx.setDiagnosticHandler(std::move(Diagnostics));
  {
    PgoAnalysisManagers AM(M);
    ModulePassManager MPM;
    MPM.addPass(
        PGOInstrumentationUse(ProfilePath.str(), "", /*IsCS=*/false, FS));
    MPM.run(M, AM.MAM);
  }
  std::string Problem = std::move(Collected->First);
  Ctx.setDiagnosticHandler(std::move(Previous));
  if (!Problem.empty())
    return createStringError(Problem);

  for (StringRef Name : MustBeAnnotated) {
    Function *F = M.getFunction(Name);
    if (!F || !F->getEntryCount())
      return createStringError("no profile applied to " + Name);
  }
  return Error::success();
}

} // namespace orc_pgo
