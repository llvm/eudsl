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

/// Owns an llvm::LLVMContext via shared_ptr. Counts live instances so tests can
/// assert nothing leaked, mirroring mlir/test/python/ir/*.py. The shared_ptr is
/// so a Module (which holds its own copy) can safely outlive the Python context
/// manager's __exit__: release() drops this object's reference and the live
/// count, but the underlying LLVMContext survives until the last Module is gone.
class Context {
public:
  Context();
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  llvm::LLVMContext &get() const;

  /// Shared handle for a Module to keep the LLVMContext alive independently.
  std::shared_ptr<llvm::LLVMContext> shared() const { return ctx; }

  /// Drop this object's reference and the live count. Idempotent; called by
  /// both __exit__ and the destructor.
  void release();
  bool isReleased() const { return ctx == nullptr; }

  static int64_t liveCount();

private:
  std::shared_ptr<llvm::LLVMContext> ctx;
};

/// Owns an llvm::Module plus a shared reference to the LLVMContext it was built
/// in, so destruction order is safe regardless of when the Context manager
/// exits. `get()` throws once the module has been handed to the JIT.
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
  std::shared_ptr<llvm::LLVMContext> ctxKeepAlive;
  std::unique_ptr<llvm::Module> mod;
  Context *owner;
};

} // namespace eudsl
