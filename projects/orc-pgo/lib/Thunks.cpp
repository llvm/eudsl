// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Thunks.h"
#include "orc-pgo/Analyses.h"
#include "orc-pgo/Eligibility.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"

using namespace llvm;

namespace orc_pgo {

/// The first counter increment Gen placed in F, or null if Gen skipped F.
static const InstrProfIncrementInst *firstIncrement(const Function &F) {
  for (const Instruction &I : instructions(F))
    if (const auto *Inc = dyn_cast<InstrProfIncrementInst>(&I))
      return Inc;
  return nullptr;
}

bool isThunkEligible(const Function &F) {
  if (F.isDeclaration() || F.hasAvailableExternallyLinkage() || F.isVarArg() ||
      F.hasFnAttribute(Attribute::Naked) || F.hasPrefixData() ||
      F.hasPrologueData()) {
    return false;
  }
  // The AArch64 backend cannot forward byval and inreg arguments through a
  // musttail call and aborts instead. Other targets may manage, but the rule
  // is unconditional so that which functions get a slot does not depend on the
  // target. A function left unthunked only loses tier-up.
  if (any_of(F.args(), [](const Argument &A) {
        return A.hasInAllocaAttr() || A.hasPreallocatedAttr() ||
               A.hasByValAttr() || A.hasAttribute(Attribute::InReg);
      })) {
    return false;
  }
  return firstIncrement(F) && hasRecreatableBody(F);
}

DeclaredAttributes inferFunctionAttributes(Module &M) {
  DeclaredAttributes Declared;
  for (const Function &F : M) {
    Declared.Memory.try_emplace(F.getName(), F.getMemoryEffects());
    if (F.hasFnAttribute(Attribute::NoSync))
      Declared.NoSync.insert(F.getName());
  }
  PgoAnalysisManagers AM(M);
  ModulePassManager MPM;
  MPM.addPass(
      createModuleToPostOrderCGSCCPassAdaptor(PostOrderFunctionAttrsPass()));
  MPM.addPass(ReversePostOrderFunctionAttrsPass());
  MPM.run(M, AM.MAM);
  return Declared;
}

void dropInvalidatedAttributes(Module &M, const DeclaredAttributes &Declared) {
  for (Function &F : M) {
    // Functions Gen declared, such as intrinsics, are not in the map.
    auto It = Declared.Memory.find(F.getName());
    if (It == Declared.Memory.end())
      continue;
    if (It->second == MemoryEffects::unknown())
      F.removeFnAttr(Attribute::Memory);
    else
      F.setMemoryEffects(It->second);
    if (!Declared.NoSync.contains(F.getName()))
      F.removeFnAttr(Attribute::NoSync);
  }
}

std::vector<ThunkCandidate> prepareThunkCandidates(Module &M) {
  std::vector<ThunkCandidate> Candidates;
  for (Function &F : M) {
    if (!isThunkEligible(F))
      continue;
    // The body behind F's slot is replaced at run time, so optimization must
    // not derive interprocedural facts from it (e.g. DeadArgElim passing
    // poison for arguments the AOT body ignores). Inlining is unaffected.
    F.addFnAttr(Attribute::NoIPA);
    Candidates.push_back(
        {&F, getPGOFuncNameVarInitializer(firstIncrement(F)->getName()).str()});
  }
  return Candidates;
}

static Function *moveBody(Function &F) {
  // copyAttributesFrom copies F's visibility, which local linkage forbids, so
  // the body becomes internal only afterwards.
  Function *Body = Function::Create(
      F.getFunctionType(), GlobalValue::ExternalLinkage, F.getAddressSpace(),
      F.getName() + AotSuffix, F.getParent());
  Body->copyAttributesFrom(&F);
  Body->setVisibility(GlobalValue::DefaultVisibility);
  Body->setDLLStorageClass(GlobalValue::DefaultStorageClass);
  Body->setLinkage(GlobalValue::InternalLinkage);
  Body->setDSOLocal(true);
  Body->splice(Body->end(), &F);
  for (auto [Old, New] : zip(F.args(), Body->args())) {
    New.takeName(&Old);
    Old.replaceAllUsesWith(&New);
  }
  Body->setSubprogram(F.getSubprogram());
  F.setSubprogram(nullptr);
  F.setPersonalityFn(nullptr);
  // The thunk reads its slot with an acquire load.
  F.removeFnAttr(Attribute::Memory);
  F.removeFnAttr(Attribute::NoSync);
  return Body;
}

static void emitThunk(Function &F, GlobalVariable &Dispatch, uint64_t Slot) {
  LLVMContext &Ctx = F.getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", &F));
  Value *SlotAddr =
      B.CreateConstInBoundsGEP2_64(Dispatch.getValueType(), &Dispatch, 0, Slot);
  LoadInst *Target = B.CreateAlignedLoad(
      B.getPtrTy(), SlotAddr, DL.getPointerABIAlignment(0), "target");
  Target->setAtomic(AtomicOrdering::Acquire);
  SmallVector<Value *> Args;
  for (Argument &A : F.args())
    Args.push_back(&A);
  CallInst *Call = B.CreateCall(F.getFunctionType(), Target, Args);
  Call->setCallingConv(F.getCallingConv());
  Call->setTailCallKind(CallInst::TCK_MustTail);
  // musttail requires the ABI-affecting parameter and return attributes of
  // the call to match the caller's.
  AttributeList Attrs = F.getAttributes();
  SmallVector<AttributeSet> ParamAttrs;
  for (unsigned I = 0, E = F.arg_size(); I != E; ++I)
    ParamAttrs.push_back(Attrs.getParamAttrs(I));
  Call->setAttributes(
      AttributeList::get(Ctx, AttributeSet(), Attrs.getRetAttrs(), ParamAttrs));
  if (F.getReturnType()->isVoidTy())
    B.CreateRetVoid();
  else
    B.CreateRet(Call);
}

ThunkTable thunkify(Module &M, ArrayRef<ThunkCandidate> Candidates) {
  SmallVector<std::pair<Function *, StringRef>> Live;
  for (const ThunkCandidate &C : Candidates)
    if (Function *F = C.function())
      Live.push_back({F, C.PgoName});
  auto *TableTy =
      ArrayType::get(PointerType::getUnqual(M.getContext()), Live.size());
  auto *Dispatch = new GlobalVariable(
      M, TableTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
      Constant::getNullValue(TableTy), DispatchTableName);
  ThunkTable Table{Dispatch, {}};
  SmallVector<Constant *> Bodies;
  for (auto [F, PgoName] : Live) {
    Function *Body = moveBody(*F);
    emitThunk(*F, *Dispatch, Bodies.size());
    Bodies.push_back(Body);
    Table.Thunks.push_back({F, Body, PgoName.str()});
  }
  Dispatch->setInitializer(ConstantArray::get(TableTy, Bodies));
  return Table;
}

} // namespace orc_pgo
