// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestSupport.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>

using namespace llvm;

namespace orc_pgo::test {

std::unique_ptr<Module> parseIR(LLVMContext &Ctx, StringRef Text) {
  SMDiagnostic Err;
  auto M = parseAssemblyString(Text, Err, Ctx);
  if (!M) {
    Err.print("orc-pgo-test", errs());
    report_fatal_error("invalid test IR");
  }
  return M;
}

void initLLVM() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
  });
}

std::unique_ptr<orc::LLJIT> makeJIT() {
  initLLVM();
  return cantFail(orc::LLJITBuilder().create());
}

std::unique_ptr<orc::LLJIT> jitIR(StringRef IR, function_ref<void(Module &)> Transform) {
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = parseIR(*Ctx, IR);
  if (Transform) {
    Transform(*M);
  }
  auto J = makeJIT();
  cantFail(J->addIRModule(orc::ThreadSafeModule(std::move(M), std::move(Ctx))));
  return J;
}

} // namespace orc_pgo::test
