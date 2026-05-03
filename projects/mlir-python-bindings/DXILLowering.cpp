//===----- DXILLowering.cpp - Rewrite MLIR-lowered IR to DXIL shape -*- C++
//-*-===//
//
// Rewrites an LLVM module produced by MLIR's memref/func lowering into a form
// the DirectX backend can codegen:
//
//   * Buffer parameters (`ptr addrspace(1)` / `ptr addrspace(2)`) are dropped
//     from the entry-point signature and replaced with
//     `target("dx.RawBuffer", T, IsUAV, 0, 0)` handles obtained via
//     `@llvm.dx.resource.handlefrombinding`.
//   * Each GEP + load on such a buffer becomes
//     `@llvm.dx.resource.load.rawbuffer` + `extractvalue 0`.
//   * Each GEP + store becomes `@llvm.dx.resource.store.rawbuffer`.
//   * A `<3 x i32>` parameter is treated as the thread grid id: each
//     `extractelement` on it becomes `@llvm.dx.thread.id(i)`.
//   * The entry function's signature is rewritten to `void ()`.
//
// The buffer/vector parameter recognition is based on types only:
//
//   addrspace 1   -> RWBuffer (UAV=1) if any store, else SRV (UAV=0)
//   addrspace 2   -> read-only RawBuffer (UAV=0)
//
// Buffers are assigned consecutive register indices in space 0 in declaration
// order.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

Type *detectBufferElementType(Value *BufArg) {
  for (User *U : BufArg->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U))
      return GEP->getSourceElementType();
    if (auto *LI = dyn_cast<LoadInst>(U))
      if (LI->getPointerOperand() == BufArg)
        return LI->getType();
    if (auto *SI = dyn_cast<StoreInst>(U))
      if (SI->getPointerOperand() == BufArg)
        return SI->getValueOperand()->getType();
  }
  return nullptr;
}

bool bufferIsWritten(Value *BufArg) {
  for (User *U : BufArg->users()) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      for (User *GU : GEP->users())
        if (isa<StoreInst>(GU))
          return true;
    } else if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() == BufArg)
        return true;
    }
  }
  return false;
}

Value *createHandleFromBinding(IRBuilderBase &B, Module &M, Type *ElemTy,
                               bool IsUAV, unsigned RegisterSpace,
                               unsigned Binding) {
  LLVMContext &Ctx = B.getContext();
  Type *HandleTy = TargetExtType::get(Ctx, "dx.RawBuffer", {ElemTy},
                                      {IsUAV ? 1u : 0u, 0u, 0u});
  Function *Decl = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::dx_resource_handlefrombinding, {HandleTy});
  Value *Args[] = {
      B.getInt32(RegisterSpace),
      B.getInt32(Binding),
      B.getInt32(1),
      B.getInt32(0),
      ConstantPointerNull::get(PointerType::get(Ctx, 0)),
  };
  return B.CreateCall(Decl, Args);
}

void rewriteBufferUses(Value *BufArg, Value *Handle, Type *ElemTy, Module &M) {
  LLVMContext &Ctx = M.getContext();
  Type *I32 = Type::getInt32Ty(Ctx);

  SmallVector<User *, 8> Users(BufArg->users());
  for (User *U : Users) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (GEP->getNumIndices() != 1) {
        errs() << "DXILLowering: unexpected multi-index GEP: " << *GEP << "\n";
        continue;
      }
      Value *Idx = GEP->getOperand(1);
      if (Idx->getType() != I32) {
        IRBuilder<> CB(GEP);
        Idx = CB.CreateIntCast(Idx, I32, /*isSigned*/ true);
      }
      SmallVector<User *, 4> GEPUsers(GEP->users());
      for (User *GU : GEPUsers) {
        if (auto *LI = dyn_cast<LoadInst>(GU)) {
          IRBuilder<> B(LI);
          Function *LDecl = Intrinsic::getOrInsertDeclaration(
              &M, Intrinsic::dx_resource_load_rawbuffer,
              {ElemTy, Handle->getType()});
          Value *Loaded = B.CreateCall(LDecl, {Handle, Idx, B.getInt32(0)});
          Value *Val = B.CreateExtractValue(Loaded, 0);
          LI->replaceAllUsesWith(Val);
          LI->eraseFromParent();
        } else if (auto *SI = dyn_cast<StoreInst>(GU)) {
          IRBuilder<> B(SI);
          Function *SDecl = Intrinsic::getOrInsertDeclaration(
              &M, Intrinsic::dx_resource_store_rawbuffer,
              {Handle->getType(), ElemTy});
          B.CreateCall(SDecl,
                       {Handle, Idx, B.getInt32(0), SI->getValueOperand()});
          SI->eraseFromParent();
        } else {
          errs() << "DXILLowering: unhandled GEP user: " << *GU << "\n";
        }
      }
      if (GEP->use_empty())
        GEP->eraseFromParent();
    } else if (auto *LI = dyn_cast<LoadInst>(U)) {
      IRBuilder<> B(LI);
      Function *LDecl = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_load_rawbuffer,
          {ElemTy, Handle->getType()});
      Value *Loaded =
          B.CreateCall(LDecl, {Handle, B.getInt32(0), B.getInt32(0)});
      Value *Val = B.CreateExtractValue(Loaded, 0);
      LI->replaceAllUsesWith(Val);
      LI->eraseFromParent();
    } else if (auto *SI = dyn_cast<StoreInst>(U)) {
      IRBuilder<> B(SI);
      Function *SDecl = Intrinsic::getOrInsertDeclaration(
          &M, Intrinsic::dx_resource_store_rawbuffer,
          {Handle->getType(), ElemTy});
      B.CreateCall(
          SDecl, {Handle, B.getInt32(0), B.getInt32(0), SI->getValueOperand()});
      SI->eraseFromParent();
    } else {
      errs() << "DXILLowering: unhandled buffer arg user: " << *U << "\n";
    }
  }
}

void rewriteGidUses(Value *GidArg, Module &M) {
  SmallVector<User *, 4> Users(GidArg->users());
  for (User *U : Users) {
    auto *Ext = dyn_cast<ExtractElementInst>(U);
    if (!Ext) {
      errs() << "DXILLowering: unhandled gid user: " << *U << "\n";
      continue;
    }
    auto *IdxC = dyn_cast<ConstantInt>(Ext->getIndexOperand());
    if (!IdxC) {
      errs() << "DXILLowering: non-constant extractelement on gid: " << *Ext
             << "\n";
      continue;
    }
    IRBuilder<> B(Ext);
    Function *Decl =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dx_thread_id);
    Value *Tid = B.CreateCall(Decl, {B.getInt32(IdxC->getZExtValue())});
    Ext->replaceAllUsesWith(Tid);
    Ext->eraseFromParent();
  }
}

Function *makeVoidEntry(Function &F) {
  Module &M = *F.getParent();
  LLVMContext &Ctx = F.getContext();
  FunctionType *NewFT =
      FunctionType::get(Type::getVoidTy(Ctx), /*vararg*/ false);
  Function *NF = Function::Create(NewFT, F.getLinkage(), F.getAddressSpace(),
                                  F.getName() + ".dxil.tmp", &M);
  NF->copyAttributesFrom(&F);
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  F.getAllMetadata(MDs);
  for (auto &P : MDs)
    NF->addMetadata(P.first, *P.second);
  NF->splice(NF->begin(), &F);
  std::string OrigName = F.getName().str();
  F.setName(F.getName() + ".dxil.dead");
  F.eraseFromParent();
  NF->setName(OrigName);
  return NF;
}

void rewriteKernel(Function &F) {
  Module &M = *F.getParent();
  BasicBlock &Entry = F.getEntryBlock();
  IRBuilder<> EntryB(&Entry, Entry.getFirstInsertionPt());

  unsigned Binding = 0;
  for (Argument &Arg : F.args()) {
    Type *ArgTy = Arg.getType();
    if (auto *PT = dyn_cast<PointerType>(ArgTy)) {
      unsigned AS = PT->getAddressSpace();
      if (AS != 1 && AS != 2) {
        errs() << "DXILLowering: unexpected pointer addrspace on " << Arg
               << "\n";
        continue;
      }
      Type *ElemTy = detectBufferElementType(&Arg);
      if (!ElemTy) {
        errs() << "DXILLowering: could not infer element type for " << Arg
               << "\n";
        continue;
      }
      bool IsUAV = (AS == 1) && bufferIsWritten(&Arg);
      Value *Handle = createHandleFromBinding(EntryB, M, ElemTy, IsUAV,
                                              /*RegisterSpace*/ 0, Binding);
      rewriteBufferUses(&Arg, Handle, ElemTy, M);
      ++Binding;
    } else if (auto *VT = dyn_cast<FixedVectorType>(ArgTy)) {
      if (VT->getNumElements() == 3 && VT->getElementType()->isIntegerTy(32)) {
        rewriteGidUses(&Arg, M);
      } else {
        errs() << "DXILLowering: unexpected vector arg: " << Arg << "\n";
      }
    } else {
      errs() << "DXILLowering: unexpected arg type: " << Arg << "\n";
    }
  }

  makeVoidEntry(F);
}

} // end anonymous namespace

namespace llvm {

void lowerMLIRToDXIL(Module &M) {
  SmallVector<Function *, 2> Kernels;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (!F.hasFnAttribute("hlsl.shader"))
      continue;
    Kernels.push_back(&F);
  }
  for (Function *F : Kernels)
    rewriteKernel(*F);
}

} // namespace llvm
