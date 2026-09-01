// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Ownership.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace eudsl {

static int64_t gLiveContexts = 0;
static int64_t gLiveModules = 0;

/// Maps an llvm::Module pointer to the eudsl::Module wrapper that was
/// constructed around it, so a view built from a bare llvm::Value can find the
/// wrapper (and pin its Python object) via moduleWrapperFor. The key is the
/// llvm::Module pointer captured at construction; the entry is inserted by the
/// ctors and removed by the dtor, so it tracks *wrapper* lifetime. take() does
/// not update it: after a module is consumed the wrapper stays registered under
/// its (now-dangling) former key until the wrapper is destroyed --
/// moduleWrapperFor only hashes/compares the pointer and never dereferences it,
/// so a stale key is not itself unsafe (see the guarded erase in ~Module).
///
/// Not synchronized: like the gLiveContexts/gLiveModules counters below, this
/// map assumes the GIL and is not safe under free-threading (the module is
/// built Py_MOD_GIL_NOT_USED but thread-safety is not verified).
static std::unordered_map<const llvm::Module *, Module *> &liveModuleMap() {
  static std::unordered_map<const llvm::Module *, Module *> map;
  return map;
}

Module *moduleWrapperFor(const llvm::Module *m) {
  auto &map = liveModuleMap();
  auto it = map.find(m);
  return it == map.end() ? nullptr : it->second;
}

Context::Context() : ctx(std::make_shared<llvm::LLVMContext>()) {
  // This LLVM still supports typed pointers and defaults to them; the bindings
  // only ever build opaque pointers, so put every context in opaque-pointer
  // mode up front (newer LLVM drops this call as opaque is the only mode).
  ctx->setOpaquePointers(true);
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
  keyMod = mod.get();
  liveModuleMap()[keyMod] = this;
  ++gLiveModules;
}

Module::Module(std::unique_ptr<llvm::Module> m, Context &ctx)
    : ctxKeepAlive(ctx.shared()), mod(std::move(m)), owner(&ctx) {
  keyMod = mod.get();
  liveModuleMap()[keyMod] = this;
  ++gLiveModules;
}

Module::~Module() {
  // Guard the erase: only remove the entry if it still maps to *this*. After
  // take() frees the llvm::Module and a new one is allocated at the same
  // address and wrapped by a different Module, the map key collides; an
  // unconditional erase(keyMod) would clobber the newer wrapper's entry (ABA),
  // unpinning its views back into a use-after-free.
  if (keyMod) {
    auto &map = liveModuleMap();
    auto it = map.find(keyMod);
    if (it != map.end() && it->second == this)
      map.erase(it);
  }
  --gLiveModules;
}

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
