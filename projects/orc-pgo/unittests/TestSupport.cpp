// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestSupport.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

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

} // namespace orc_pgo::test
