// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/CounterLayout.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace orc_pgo {

bool isInternalName(StringRef Name) { return Name.starts_with(InternalPrefix); }

unsigned numSiteCounters(const Instruction &I) {
  if (isa<CondBrInst>(I))
    return 2;
  if (const auto *SI = dyn_cast<SwitchInst>(&I))
    return SI->getNumSuccessors() > 1 ? SI->getNumSuccessors() : 0;
  if (const auto *Sel = dyn_cast<SelectInst>(&I))
    return Sel->getCondition()->getType()->isIntegerTy(1) ? 2 : 0;
  return 0;
}

CounterLayout CounterLayout::compute(Module &M) {
  CounterLayout L;
  for (Function &F : M) {
    if (F.isDeclaration() || F.hasAvailableExternallyLinkage() ||
        isInternalName(F.getName()))
      continue;
    FunctionCounters FC{&F, static_cast<unsigned>(L.Functions.size()),
                        L.NumCounters++, {}};
    for (Instruction &I : instructions(F)) {
      if (unsigned N = numSiteCounters(I)) {
        FC.Sites.push_back({&I, L.NumCounters, N});
        L.NumCounters += N;
      }
    }
    L.Index[&F] = FC.FunctionIndex;
    L.Functions.push_back(std::move(FC));
  }
  return L;
}

const FunctionCounters *CounterLayout::lookup(const Function &F) const {
  auto It = Index.find(&F);
  return It == Index.end() ? nullptr : &Functions[It->second];
}

} // namespace orc_pgo
