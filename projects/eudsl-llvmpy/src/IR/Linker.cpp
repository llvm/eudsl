// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"
#include "MIR/Diagnostics.h"

#include <llvm/Linker/Linker.h>

#include <memory>
#include <string>

void populate_linker(nb::module_ &m) {
  m.def(
      "link_into",
      [](eudsl::Module &dest, eudsl::Module &src) {
        // linkModules consumes the source module; take() marks the Python
        // wrapper moved-from so later use raises rather than segfaults.
        std::unique_ptr<llvm::Module> srcOwned = src.take();
        // A link conflict is reported through the context's diagnostic handler,
        // and this LLVM's default handler calls exit(1) on an error-severity
        // diagnostic (newer LLVM just prints and lets linkModules return). Swap
        // in a capturing handler so the failure surfaces as a catchable
        // exception with the linker's message instead of aborting the process.
        std::string diag;
        eudsl::ScopedDiagnosticCapture capture(dest.get().getContext(), diag);
        if (llvm::Linker::linkModules(dest.get(), std::move(srcOwned)))
          throw std::runtime_error(
              eudsl::withDetail("linkModules failed", diag));
      },
      "dest"_a, "src"_a);
}
