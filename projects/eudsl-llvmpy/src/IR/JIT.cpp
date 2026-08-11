// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>

#include <memory>
#include <string>

void populate_jit(nb::module_ &m) {
  nb::class_<llvm::orc::LLJIT>(m, "LLJIT")
      .def(nb::new_([]() -> llvm::orc::LLJIT * {
             return eudsl::unwrap(llvm::orc::LLJITBuilder().create()).release();
           }))
      .def(
          "add_module",
          [](llvm::orc::LLJIT &self, eudsl::Module &mod) {
            std::string buf;
            {
              llvm::raw_string_ostream os(buf);
              llvm::WriteBitcodeToFile(mod.get(), os);
            }
            auto ctx = std::make_unique<llvm::LLVMContext>();
            std::unique_ptr<llvm::MemoryBuffer> memBuf =
                llvm::MemoryBuffer::getMemBufferCopy(buf, "<jit>");
            std::unique_ptr<llvm::Module> cloned = eudsl::unwrap(
                llvm::parseBitcodeFile(memBuf->getMemBufferRef(), *ctx));
            llvm::orc::ThreadSafeModule tsm(std::move(cloned), std::move(ctx));
            eudsl::unwrap(self.addIRModule(std::move(tsm)));
            (void)mod.take();
          },
          "module"_a)
      .def(
          "lookup",
          [](llvm::orc::LLJIT &self, const std::string &name) {
            llvm::orc::ExecutorAddr addr = eudsl::unwrap(self.lookup(name));
            return static_cast<uint64_t>(addr.getValue());
          },
          "name"_a);
}
