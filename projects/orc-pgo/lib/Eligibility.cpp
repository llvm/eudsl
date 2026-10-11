// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Eligibility.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/InstIterator.h"

using namespace llvm;

namespace orc_pgo {

static bool recreatable(const Constant &C,
                        SmallPtrSetImpl<const Constant *> &Seen) {
  if (!Seen.insert(&C).second)
    return true;
  if (isa<BlockAddress>(C) || isa<GlobalIFunc>(C))
    return false;
  if (const auto *GV = dyn_cast<GlobalValue>(&C))
    return !GV->isThreadLocal();
  return all_of(C.operands(), [&](const Use &Op) {
    return recreatable(*cast<Constant>(Op.get()), Seen);
  });
}

bool isRecreatable(const Constant &C) {
  SmallPtrSet<const Constant *, 16> Seen;
  return recreatable(C, Seen);
}

bool hasRecreatableBody(const Function &F) {
  if (any_of(F, [](const BasicBlock &BB) { return BB.hasAddressTaken(); }))
    return false;
  SmallPtrSet<const Constant *, 32> Seen;
  for (const Instruction &I : instructions(F))
    for (const Value *Op : I.operands())
      if (const auto *C = dyn_cast<Constant>(Op); C && !recreatable(*C, Seen))
        return false;
  return true;
}

} // namespace orc_pgo
