// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <cstdint>
#include <memory>
#include <string>

namespace eudsl {

/// Owns an llvm::LLVMContext. Counts live instances so tests can assert that
/// nothing leaked, mirroring mlir/test/python/ir/*.py.
class Context {
public:
  Context();
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  llvm::LLVMContext &get() const;

  /// Destroy the underlying LLVMContext and drop the live count. Idempotent;
  /// called by both __exit__ and the destructor.
  void release();
  bool isReleased() const { return ctx == nullptr; }

  static int64_t liveCount();

private:
  std::unique_ptr<llvm::LLVMContext> ctx;
};

/// Owns an llvm::Module. `get()` throws once the module has been handed to the
/// JIT, so a stale reference is a Python exception rather than a segfault.
class Module {
public:
  Module(const std::string &name, Context &ctx);
  Module(std::unique_ptr<llvm::Module> mod, Context &ctx);

  llvm::Module &get() const;
  /// Relinquish ownership. Every later get() throws.
  std::unique_ptr<llvm::Module> take();
  bool isConsumed() const { return mod == nullptr; }

  Context &context() const { return *owner; }

private:
  std::unique_ptr<llvm::Module> mod;
  Context *owner;
};

} // namespace eudsl
