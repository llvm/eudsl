// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Ownership.h"

#include <stdexcept>
#include <vector>

namespace eudsl {

static int64_t gLiveContexts = 0;
static int64_t gLiveModules = 0;

Context::Context() : ctx(std::make_shared<llvm::LLVMContext>()) {
  ++gLiveContexts;
}

Context::~Context() { release(); }

void Context::release() {
  if (ctx) {
    ctx.reset();
    --gLiveContexts;
  }
}

int64_t Context::liveCount() { return gLiveContexts; }

static thread_local std::vector<Context *> gContextStack;

void Context::pushCurrent() { gContextStack.push_back(this); }

void Context::popCurrent() {
  if (!gContextStack.empty())
    gContextStack.pop_back();
}

Context *Context::current() {
  return gContextStack.empty() ? nullptr : gContextStack.back();
}

llvm::LLVMContext &Context::get() const {
  if (!ctx)
    throw std::runtime_error("context has been released");
  return *ctx;
}

Module::Module(const std::string &name, Context &ctx)
    : ctxKeepAlive(ctx.shared()),
      mod(std::make_unique<llvm::Module>(name, ctx.get())), owner(&ctx) {
  ++gLiveModules;
}

Module::Module(std::unique_ptr<llvm::Module> m, Context &ctx)
    : ctxKeepAlive(ctx.shared()), mod(std::move(m)), owner(&ctx) {
  ++gLiveModules;
}

Module::~Module() { --gLiveModules; }

int64_t Module::liveCount() { return gLiveModules; }

llvm::Module &Module::get() const {
  if (!mod) {
    throw std::runtime_error(
        "module has been consumed (moved into the JIT) and can no longer be "
        "used");
  }
  return *mod;
}

std::unique_ptr<llvm::Module> Module::take() {
  get(); // throws if already consumed
  return std::move(mod);
}

} // namespace eudsl
