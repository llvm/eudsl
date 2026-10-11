// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Instrument.h"
#include "orc-pgo/Analyses.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/PgoOptions.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <string>
#include <utility>

using namespace llvm;

namespace orc_pgo {

Error runPgoGen(Module &M) {
  if (Error E = configurePgoOptions())
    return E;
  PgoAnalysisManagers AM(M);
  ModulePassManager MPM;
  MPM.addPass(PGOInstrumentationGen(PGOInstrumentationType::FDO));
  MPM.run(M, AM.MAM);
  return Error::success();
}

namespace {

// Operand indices of a __profd initializer (INSTR_PROF_DATA in
// llvm/ProfileData/InstrProfData.inc, raw profile version 11).
enum DataField : unsigned {
  NameRefField = 0,
  FuncHashField = 1,
  CounterPtrField = 2,
  FunctionPointerField = 5,
  NumCountersField = 7,
  NumValueSitesField = 8,
  NumDataFields = 11,
};

/// The global other than Except that C refers to, e.g. the __profc array in a
/// record's relative CounterPtr expression.
GlobalVariable *findGlobal(Constant *C, const GlobalVariable *Except) {
  if (auto *GV = dyn_cast<GlobalVariable>(C))
    return GV == Except ? nullptr : GV;
  for (Use &Op : C->operands())
    if (GlobalVariable *GV = findGlobal(cast<Constant>(Op), Except))
      return GV;
  return nullptr;
}

uint16_t elementOrZero(Constant *Array, unsigned Index) {
  if (auto *CDS = dyn_cast<ConstantDataSequential>(Array))
    return CDS->getElementAsInteger(Index);
  return 0; // zeroinitializer
}

/// The function a record's FunctionPointer names, looking through the private
/// alias lowering may use; null when lowering recorded no address.
Function *recordedFunction(Constant *FunctionPointer) {
  auto *GV = dyn_cast<GlobalValue>(FunctionPointer->stripPointerCasts());
  return GV ? dyn_cast_or_null<Function>(GV->getAliaseeObject()) : nullptr;
}

} // namespace

Expected<LoweredPgo> lowerPgo(Module &M, TargetMachine *TM) {
  // Lowering deletes the name variables, so collect the names first.
  DenseMap<uint64_t, std::string> NameByMD5;
  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      if (auto *Inc = dyn_cast<InstrProfCntrInstBase>(&I)) {
        StringRef Name = getPGOFuncNameVarInitializer(Inc->getName());
        NameByMD5[IndexedInstrProf::ComputeHash(Name)] = Name.str();
      }
    }
  }

  // Lowering references compiler-rt's __llvm_profile_runtime unless the module
  // already has one; orc-pgo programs don't link compiler-rt's profile runtime.
  LLVMContext &Ctx = M.getContext();
  Type *I32Ty = Type::getInt32Ty(Ctx);
  if (!M.getNamedValue(getInstrProfRuntimeHookVarName())) {
    auto *Hook = new GlobalVariable(
        M, I32Ty, /*isConstant=*/false, GlobalValue::LinkOnceODRLinkage,
        ConstantInt::get(I32Ty, 0), getInstrProfRuntimeHookVarName());
    Hook->setVisibility(GlobalValue::HiddenVisibility);
  }

  {
    // PassBuilder::addPGOInstrPasses: post-PGO loop rotation (on by default),
    // then lowering with promotion; -fprofile-update=single is non-atomic.
    InstrProfOptions Options;
    Options.Atomic = false;
    Options.DoCounterPromotion = true;
    PgoAnalysisManagers AM(M, TM);
    ModulePassManager MPM;
    MPM.addPass(
        createModuleToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(
            LoopRotatePass(), /*UseMemorySSA=*/false)));
    MPM.addPass(InstrProfilingLoweringPass(Options));
    MPM.run(M, AM.MAM);
  }

  for (auto [From, To] :
       {std::pair<StringRef, StringRef>{"__llvm_profile_instrument_target",
                                        "__orc_pgo_instrument_target"},
        {"__llvm_profile_instrument_memop", "__orc_pgo_instrument_memop"}})
    if (Function *Callee = M.getFunction(From))
      Callee->setName(To);

  Type *PtrTy = PointerType::getUnqual(Ctx);
  Type *I16Ty = Type::getInt16Ty(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  auto *SitesTy = ArrayType::get(I16Ty, NumValueKinds);
  auto *RecordTy =
      StructType::get(Ctx, {PtrTy, I64Ty, PtrTy, PtrTy, PtrTy, I32Ty, SitesTy});

  std::string DataSection =
      getInstrProfSectionName(IPSK_data, M.getTargetTriple().getObjectFormat());
  SmallVector<GlobalVariable *> DataVars;
  for (GlobalVariable &GV : M.globals())
    if (GV.getSection() == DataSection)
      DataVars.push_back(&GV);

  SmallVector<Constant *> Records;
  for (GlobalVariable *Data : DataVars) {
    auto *Init = dyn_cast<ConstantStruct>(Data->getInitializer());
    if (!Init || Init->getNumOperands() != NumDataFields) {
      return createStringError("unsupported __profd record %s",
                               Data->getName().str().c_str());
    }
    uint64_t NameRef =
        cast<ConstantInt>(Init->getOperand(NameRefField))->getZExtValue();
    auto NameIt = NameByMD5.find(NameRef);
    if (NameIt == NameByMD5.end()) {
      return createStringError("no PGO name for __profd record %s",
                               Data->getName().str().c_str());
    }
    GlobalVariable *Counters =
        findGlobal(Init->getOperand(CounterPtrField), Data);
    if (!Counters) {
      return createStringError("no counters for __profd record %s",
                               Data->getName().str().c_str());
    }
    Constant *Address = ConstantPointerNull::get(cast<PointerType>(PtrTy));
    if (Function *F = recordedFunction(Init->getOperand(FunctionPointerField)))
      Address = F;

    Constant *NameStr = ConstantDataArray::getString(Ctx, NameIt->second);
    auto *NameVar = new GlobalVariable(
        M, NameStr->getType(), /*isConstant=*/true, GlobalValue::PrivateLinkage,
        NameStr, "__orc_pgo_name");
    NameVar->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    Constant *SitesInit = Init->getOperand(NumValueSitesField);
    SmallVector<Constant *, NumValueKinds> Sites;
    for (unsigned K = 0; K < NumValueKinds; ++K)
      Sites.push_back(ConstantInt::get(I16Ty, elementOrZero(SitesInit, K)));

    Records.push_back(ConstantStruct::get(
        RecordTy, {NameVar, Init->getOperand(FuncHashField), Counters, Data,
                   Address, Init->getOperand(NumCountersField),
                   ConstantArray::get(SitesTy, Sites)}));
  }

  auto *TableTy = ArrayType::get(RecordTy, Records.size());
  auto *Table = new GlobalVariable(
      M, TableTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantArray::get(TableTy, Records), FunctionTableName);
  auto *Count = new GlobalVariable(
      M, I32Ty, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantInt::get(I32Ty, Records.size()), NumFunctionsName);

  Constant *PoolBegin = ConstantPointerNull::get(cast<PointerType>(PtrTy));
  Constant *PoolEnd = PoolBegin;
  if (GlobalVariable *Pool = M.getGlobalVariable(getInstrProfVNodesVarName(),
                                                 /*AllowInternal=*/true)) {
    PoolBegin = Pool;
    PoolEnd = ConstantExpr::getPtrAdd(
        Pool,
        ConstantInt::get(
            I64Ty, M.getDataLayout().getTypeAllocSize(Pool->getValueType())),
        GEPNoWrapFlags::inBounds());
  }
  auto *PoolTy = ArrayType::get(PtrTy, 2);
  auto *VNodes = new GlobalVariable(
      M, PoolTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
      ConstantArray::get(PoolTy, {PoolBegin, PoolEnd}), VNodePoolName);

  for (GlobalVariable *GV : {Table, Count, VNodes})
    GV->setVisibility(GlobalValue::HiddenVisibility);
  appendToCompilerUsed(M, {Table, Count, VNodes});
  return LoweredPgo{Table, Count, VNodes};
}

} // namespace orc_pgo
