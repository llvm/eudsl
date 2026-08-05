// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Errors.h"

#include <llvm/Support/ErrorHandling.h>

#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;

namespace {
// Fatal errors abort the process; convert the message to something visible
// before LLVM calls abort(). The handler cannot return, so this is a
// best-effort last word rather than a recoverable path. Only runs on an LLVM
// fatal error, which the test suite does not (and must not) trigger.
// LCOV_EXCL_START
void fatalHandler(void *, const char *reason, bool) {
  PyErr_WarnEx(PyExc_RuntimeWarning,
               (std::string("LLVM fatal error: ") + reason).c_str(), 1);
}
// LCOV_EXCL_STOP
} // namespace

namespace eudsl {

void registerExceptions(nb::module_ &m) {
  nb::exception<ParseError>(m, "ParseError");
  nb::exception<VerifyError>(m, "VerifyError");
  llvm::install_fatal_error_handler(fatalHandler);
}

} // namespace eudsl
