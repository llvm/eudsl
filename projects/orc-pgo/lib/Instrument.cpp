// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Instrument.h"
#include "orc-pgo/CounterLayout.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace orc_pgo {

static void emitIncrement(IRBuilder<> &B, GlobalVariable *Counters, Value *Idx) {
  Value *Ptr = B.CreateInBoundsGEP(Counters->getValueType(), Counters,
                                   {B.getInt64(0), Idx});
  LoadInst *Old = B.CreateAlignedLoad(B.getInt64Ty(), Ptr, Align(8));
  Old->setAtomic(AtomicOrdering::Monotonic);
  StoreInst *St =
      B.CreateAlignedStore(B.CreateAdd(Old, B.getInt64(1)), Ptr, Align(8));
  St->setAtomic(AtomicOrdering::Monotonic);
}

/// Counter index taken at a site; successor/arm k maps to FirstCounter + k.
static Value *siteIndex(IRBuilder<> &B, const CounterSite &S) {
  uint64_t First = S.FirstCounter;
  if (auto *SI = dyn_cast<SwitchInst>(S.Inst)) {
    Value *Idx = B.getInt64(First); // successor 0 is the default
    for (auto &Case : SI->cases()) {
      Idx = B.CreateSelect(
          B.CreateICmpEQ(SI->getCondition(), Case.getCaseValue()),
          B.getInt64(First + Case.getSuccessorIndex()), Idx);
    }
    return Idx;
  }
  Value *Cond = isa<SelectInst>(S.Inst)
                    ? cast<SelectInst>(S.Inst)->getCondition()
                    : cast<CondBrInst>(S.Inst)->getCondition();
  return B.CreateSelect(Cond, B.getInt64(First), B.getInt64(First + 1));
}

GlobalVariable *instrumentModule(Module &M, const CounterLayout &L) {
  auto *ArrTy = ArrayType::get(Type::getInt64Ty(M.getContext()), L.numCounters());
  auto *Counters = new GlobalVariable(M, ArrTy, /*isConstant=*/false,
                                      GlobalValue::ExternalLinkage,
                                      Constant::getNullValue(ArrTy), CountersName);
  Counters->setVisibility(GlobalValue::HiddenVisibility);
  Counters->setAlignment(Align(8));

  for (const FunctionCounters &FC : L.functions()) {
    BasicBlock &Entry = FC.F->getEntryBlock();
    IRBuilder<> B(Entry.getFirstNonPHIOrDbgOrAlloca());
    emitIncrement(B, Counters, B.getInt64(FC.EntryCounter));
    for (const CounterSite &S : FC.Sites) {
      IRBuilder<> SB(S.Inst);
      emitIncrement(SB, Counters, siteIndex(SB, S));
    }
  }
  return Counters;
}

} // namespace orc_pgo
