// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/TierUp.h"
#include "orc-pgo/Analyses.h"
#include "orc-pgo/Annotate.h"
#include "orc-pgo/Eligibility.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"

using namespace llvm;

namespace orc_pgo {

/// Callees of Roots up to InlineDepth calls away whose bodies the JIT can
/// recompile and that the linker cannot replace: a weak definition may be
/// overridden by another one in the program, so the call must stay a call.
static SmallPtrSet<Function *, 16> collectCopies(ArrayRef<Function *> Roots,
                                                 unsigned InlineDepth) {
  SmallPtrSet<Function *, 16> Copies;
  SmallVector<Function *> Frontier(Roots.begin(), Roots.end());
  for (unsigned Depth = 0; Depth != InlineDepth && !Frontier.empty(); ++Depth) {
    SmallVector<Function *> Next;
    for (Function *F : Frontier) {
      for (Instruction &I : instructions(*F)) {
        auto *CB = dyn_cast<CallBase>(&I);
        Function *Callee = CB ? CB->getCalledFunction() : nullptr;
        if (!Callee || Callee->isDeclaration() || is_contained(Roots, Callee) ||
            Callee->isInterposable(/*CheckNoIPA=*/false) ||
            !hasRecreatableBody(*Callee) || !Copies.insert(Callee).second) {
          continue;
        }
        Next.push_back(Callee);
      }
    }
    Frontier = std::move(Next);
  }
  return Copies;
}

/// GV is resolved at JIT-link time to wherever the program has it.
static void makeImported(GlobalValue &GV) {
  if (!GV.hasExternalWeakLinkage())
    GV.setLinkage(GlobalValue::ExternalLinkage);
  GV.setVisibility(GlobalValue::DefaultVisibility);
  GV.setDSOLocal(false);
}

/// Replaces an alias or ifunc with a declaration of the same name.
static void replaceWithDeclaration(GlobalValue &GV) {
  Module &M = *GV.getParent();
  GlobalValue *Decl;
  if (auto *FT = dyn_cast<FunctionType>(GV.getValueType()))
    Decl = Function::Create(FT, GlobalValue::ExternalLinkage, "", M);
  else
    Decl = new GlobalVariable(M, GV.getValueType(), /*isConstant=*/false,
                              GlobalValue::ExternalLinkage, nullptr);
  Decl->takeName(&GV);
  GV.replaceAllUsesWith(Decl);
  GV.eraseFromParent();
}

static void reduceToBatch(Module &M, ArrayRef<Function *> Roots,
                          const SmallPtrSetImpl<Function *> &Copies) {
  // The program's object already assembled it, and defining its symbols again
  // would clash.
  M.removeModuleInlineAsm();

  for (GlobalAlias &GA : make_early_inc_range(M.aliases()))
    replaceWithDeclaration(GA);
  for (GlobalIFunc &GI : make_early_inc_range(M.ifuncs()))
    replaceWithDeclaration(GI);

  // Before function bodies go: deleting a block rewrites the block addresses
  // in initializers.
  for (GlobalVariable &GV : make_early_inc_range(M.globals())) {
    if (GV.hasAppendingLinkage()) {
      GV.eraseFromParent();
      continue;
    }
    GV.setComdat(nullptr);
    if (GV.isConstant() && GV.hasInitializer() && !GV.isThreadLocal() &&
        !GV.isInterposable() && isRecreatable(*GV.getInitializer())) {
      GV.setLinkage(GlobalValue::AvailableExternallyLinkage);
      GV.setVisibility(GlobalValue::DefaultVisibility);
      GV.setDSOLocal(false);
    } else {
      GV.setInitializer(nullptr);
      makeImported(GV);
    }
  }

  for (Function &F : M) {
    F.setComdat(nullptr);
    if (is_contained(Roots, &F)) {
      F.setLinkage(GlobalValue::ExternalLinkage);
      F.setVisibility(GlobalValue::DefaultVisibility);
    } else if (Copies.contains(&F)) {
      F.setLinkage(GlobalValue::AvailableExternallyLinkage);
      F.setVisibility(GlobalValue::DefaultVisibility);
      F.setDSOLocal(false);
    } else if (!F.isIntrinsic()) {
      if (!F.isDeclaration())
        F.deleteBody();
      makeImported(F);
    }
  }
}

namespace {
/// Records where PGOIndirectCallPromotion reports a promotion. Swallows every
/// other diagnostic: tier-up must never stop the program.
struct PromotionRemarks : DiagnosticHandler {
  SmallVector<const Function *> PromotedIn;
  bool isAnyRemarkEnabled() const override { return true; }
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    const auto *R = dyn_cast<OptimizationRemark>(&DI);
    if (R && R->getPassName() == "pgo-icall-prom" &&
        R->getRemarkName() == "Promoted")
      PromotedIn.push_back(&R->getFunction());
    return true;
  }
};
} // namespace

/// Runs PGOIndirectCallPromotion over M; returns the functions it promoted
/// calls in, once per promotion.
static SmallVector<const Function *> runIndirectCallPromotion(Module &M) {
  LLVMContext &Ctx = M.getContext();
  std::unique_ptr<DiagnosticHandler> Previous = Ctx.getDiagnosticHandler();
  auto Remarks = std::make_unique<PromotionRemarks>();
  PromotionRemarks *Collected = Remarks.get();
  Ctx.setDiagnosticHandler(std::move(Remarks));
  {
    PgoAnalysisManagers AM(M);
    ModulePassManager MPM;
    MPM.addPass(
        PGOIndirectCallPromotion(/*IsInLTO=*/false, /*SamplePGO=*/false));
    MPM.run(M, AM.MAM);
  }
  SmallVector<const Function *> PromotedIn = std::move(Collected->PromotedIn);
  Ctx.setDiagnosticHandler(std::move(Previous));
  return PromotedIn;
}

Expected<TierUpBatch> prepareTierUpModule(Module &M,
                                          const MemoryBuffer &Profile,
                                          ArrayRef<StringRef> Roots,
                                          unsigned InlineDepth) {
  StringMap<Function *> ByPgoName;
  for (Function &F : M)
    if (!F.isDeclaration())
      ByPgoName[getIRPGOObjectName(F)] = &F;
  SmallVector<Function *> RootFns;
  SmallVector<StringRef> RootIRNames;
  for (StringRef Name : Roots) {
    Function *F = ByPgoName.lookup(Name);
    if (!F)
      return createStringError("the snapshot has no function with PGO name '" +
                               Name + "'");
    RootFns.push_back(F);
    RootIRNames.push_back(F->getName());
  }

  if (Error E = annotateWithProfile(M, Profile, RootIRNames))
    return std::move(E);
  // As in clang's -fprofile-use pipeline, ICP runs right after Use. It also
  // must run before the reduction: it only promotes to targets defined in M
  // (-icp-allow-decls=false), and finds them by their original PGO names.
  // Promoted targets are then direct callees, copied like any other.
  SmallVector<const Function *> PromotedIn = runIndirectCallPromotion(M);
  SmallPtrSet<Function *, 16> Copies = collectCopies(RootFns, InlineDepth);
  TierUpBatch Batch;
  Batch.PromotedIndirectCalls = count_if(PromotedIn, [&](const Function *F) {
    return is_contained(RootFns, F) || Copies.contains(F);
  });
  reduceToBatch(M, RootFns, Copies);

  // The program defines the original names (the thunks), so tier-up
  // definitions need their own. Only calls go to the tier-up definition: any
  // other use of a root (a compare from indirect-call promotion, a vtable or
  // table entry in a copied constant, an escaping address) is of the address
  // the program has for it, the thunk, so that it matches what the program
  // passes around. Were the promoted compare against the tier-up definition,
  // it would never succeed.
  for (Function *F : RootFns) {
    std::string Original = F->getName().str();
    F->setName(Original + TierUpSuffix);
    Batch.Names.push_back(F->getName().str());
    auto IsAddressUse = [](Use &U) {
      auto *CB = dyn_cast<CallBase>(U.getUser());
      return !(CB && CB->isCallee(&U));
    };
    if (any_of(F->uses(), IsAddressUse)) {
      Function *Decl =
          Function::Create(F->getFunctionType(), GlobalValue::ExternalLinkage,
                           F->getAddressSpace(), Original, &M);
      F->replaceUsesWithIf(Decl, IsAddressUse);
    }
  }
  return Batch;
}

/// Whether U is the callee operand of a call.
static bool isCalleeUse(Use &U) {
  auto *CB = dyn_cast<CallBase>(U.getUser());
  return CB && CB->isCallee(&U);
}

unsigned bindTieredCalls(Module &M, const StringSet<> &Tiered) {
  unsigned Bound = 0;
  // Declarations are added to M while going through it.
  SmallVector<Function *> Callees;
  for (Function &F : M)
    if (Tiered.contains((F.getName() + TierUpSuffix).str()))
      Callees.push_back(&F);
  for (Function *F : Callees) {
    SmallVector<Use *> Calls;
    for (Use &U : F->uses())
      if (isCalleeUse(U))
        Calls.push_back(&U);
    if (Calls.empty())
      continue;
    std::string TierName = (F->getName() + TierUpSuffix).str();
    Function *Tier = M.getFunction(TierName);
    if (!Tier) {
      Tier =
          Function::Create(F->getFunctionType(), GlobalValue::ExternalLinkage,
                           F->getAddressSpace(), TierName, &M);
    }
    for (Use *U : Calls)
      U->set(Tier);
    Bound += Calls.size();
  }
  return Bound;
}

void optimizeTierUp(Module &M, TargetMachine &TM) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB(&TM, clangPipelineTuningOptions(OptimizationLevel::O3));
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  // ScalarOptimizerLate is the extension point of the function simplification
  // pipeline that is invoked once, after inlining. clang's IRUse pipeline adds
  // PGOMemOPSizeOpt a few passes earlier in that pipeline; the peephole point
  // next to it is also invoked before inlining and at several later points.
  PB.registerScalarOptimizerLateEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel) {
        FPM.addPass(PGOMemOPSizeOpt());
      });
  PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3).run(M, MAM);
}

} // namespace orc_pgo
