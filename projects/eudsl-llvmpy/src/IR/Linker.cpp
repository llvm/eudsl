// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/Linker/Linker.h>

#include <memory>

void populate_linker(nb::module_ &m) {
  m.def(
      "link_into",
      [](eudsl::Module &dest, eudsl::Module &src) {
        // linkModules consumes the source module; take() marks the Python
        // wrapper moved-from so later use raises rather than segfaults.
        std::unique_ptr<llvm::Module> srcOwned = src.take();
        if (llvm::Linker::linkModules(dest.get(), std::move(srcOwned)))
          throw std::runtime_error("linkModules failed");
      },
      "dest"_a, "src"_a);
}
