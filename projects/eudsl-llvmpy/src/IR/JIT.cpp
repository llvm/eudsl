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

namespace {
struct JIT {
  std::unique_ptr<llvm::orc::LLJIT> jit;
};
} // namespace

void populate_jit(nb::module_ &m) {
  nb::class_<JIT>(m, "LLJIT")
      .def("__init__",
           [](JIT *self) {
             auto jit = eudsl::unwrap(llvm::orc::LLJITBuilder().create());
             new (self) JIT{std::move(jit)};
           })
      .def(
          "add_module",
          [](JIT &self, eudsl::Module &mod) {
            // Move the module into the JIT by round-tripping through bitcode
            // into a fresh, JIT-owned LLVMContext. This avoids extracting a
            // unique_ptr<LLVMContext> from the shared_ptr the Context/Module
            // lifetime model uses. The source module is then marked consumed.
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
            eudsl::unwrap(self.jit->addIRModule(std::move(tsm)));
            // Mark the source module consumed so later use raises.
            (void)mod.take();
          },
          "module"_a)
      .def(
          "lookup",
          [](JIT &self, const std::string &name) {
            llvm::orc::ExecutorAddr addr = eudsl::unwrap(self.jit->lookup(name));
            return static_cast<uint64_t>(addr.getValue());
          },
          "name"_a);
}
