// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Pipeline.h"
#include "orc-pgo/Analyses.h"
#include "orc-pgo/Image.h"
#include "orc-pgo/Instrument.h"
#include "orc-pgo/PgoOptions.h"
#include "orc-pgo/Snapshot.h"
#include "orc-pgo/SymbolTable.h"
#include "orc-pgo/Thunks.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <cassert>

using namespace llvm;

namespace orc_pgo {

static constexpr StringLiteral SnapshotName = "__orc_pgo_snapshot";
static constexpr StringLiteral DispatchFunctionsName =
    "__orc_pgo_dispatch_functions";
static constexpr StringLiteral RegisterFunctionName = "__orc_pgo_register";

static void optimizeAot(Module &M, TargetMachine &TM) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB(&TM, clangPipelineTuningOptions(OptimizationLevel::O2));
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2).run(M, MAM);
}

/// The value Gen stored in __llvm_profile_raw_version.
static uint64_t rawProfileVersion(const Module &M) {
  const GlobalVariable *GV =
      M.getNamedGlobal(INSTR_PROF_QUOTE(INSTR_PROF_RAW_VERSION_VAR));
  return cast<ConstantInt>(GV->getInitializer())->getZExtValue();
}

static uint64_t numElements(const GlobalVariable &GV) {
  return cast<ArrayType>(GV.getValueType())->getNumElements();
}

static GlobalVariable *privateConstant(Module &M, Constant *Init,
                                       StringRef Name) {
  return new GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                            GlobalValue::PrivateLinkage, Init, Name);
}

/// For each slot, the index of the record carrying its function's PGO name.
static SmallVector<uint32_t> slotRecords(const LoweredPgo &Lowered,
                                         const ThunkTable &Thunks) {
  StringMap<uint32_t> RecordOf;
  Constant *Records = Lowered.Functions->getInitializer();
  for (unsigned I = 0, E = numElements(*Lowered.Functions); I != E; ++I) {
    StringRef Name;
    [[maybe_unused]] bool IsString = getConstantStringInfo(
        Records->getAggregateElement(I)->getAggregateElement(0u), Name);
    assert(IsString && "record names are constant strings");
    RecordOf[Name] = I;
  }
  SmallVector<uint32_t> Slots;
  for (const Thunk &T : Thunks.Thunks) {
    auto It = RecordOf.find(T.PgoName);
    assert(It != RecordOf.end() && "Gen counts every thunk's entry block");
    Slots.push_back(It->second);
  }
  return Slots;
}

static void emitGlue(Module &M, StringRef Snapshot, uint64_t RawVersion,
                     const LoweredPgo &Lowered, const ThunkTable &Thunks,
                     GlobalVariable &Symbols) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *Ptr = PointerType::getUnqual(Ctx);

  GlobalVariable *SnapshotGV = privateConstant(
      M, ConstantDataArray::getString(Ctx, Snapshot, /*AddNull=*/false),
      SnapshotName);
  SnapshotGV->setAlignment(Align(8));
  SmallVector<uint32_t> Slots = slotRecords(Lowered, Thunks);
  GlobalVariable *SlotRecords =
      privateConstant(M, ConstantDataArray::get(Ctx, ArrayRef<uint32_t>(Slots)),
                      DispatchFunctionsName);
  GlobalVariable *Version =
      privateConstant(M, ConstantDataArray::getString(Ctx, LLVM_VERSION_STRING),
                      "__orc_pgo_llvm_version");

  auto *FingerprintTy = StructType::get(Ctx, {Ptr, I64, I64});
  auto *ImageTy = StructType::get(Ctx, {I64, FingerprintTy, Ptr, I64, Ptr, I64,
                                        Ptr, Ptr, Ptr, I64, Ptr, I64});
  Constant *Fields[] = {
      ConstantInt::get(I64, ImageVersion),
      ConstantStruct::get(FingerprintTy,
                          {Version, ConstantInt::get(I64, RawVersion),
                           ConstantInt::get(I64, pgoOptionsFingerprint())}),
      SnapshotGV,
      ConstantInt::get(I64, Snapshot.size()),
      Lowered.Functions,
      ConstantInt::get(I64, numElements(*Lowered.Functions)),
      Lowered.VNodes,
      Thunks.Dispatch,
      SlotRecords,
      ConstantInt::get(I64, Thunks.Thunks.size()),
      &Symbols,
      ConstantInt::get(I64, numElements(Symbols))};
  GlobalVariable *Image =
      privateConstant(M, ConstantStruct::get(ImageTy, Fields), ImageName);

  Function *Register =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::InternalLinkage, RegisterFunctionName, M);
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", Register));
  B.CreateCall(M.getOrInsertFunction(RegisterImageName, VoidTy, Ptr), {Image});
  B.CreateRetVoid();
  appendToGlobalCtors(M, Register, /*Priority=*/65535);

  // Starting the runtime needs LLVM's static initializers, which in a static
  // link may run after the registration constructor. main runs after all of
  // them.
  if (Function *Main = M.getFunction("main"); Main && !Main->isDeclaration()) {
    IRBuilder<> MB(Main->getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
    MB.CreateCall(M.getOrInsertFunction(StartName, VoidTy));
  }
}

Expected<AotModule> buildAotModule(Module &Input, TargetMachine &TM,
                                   ArrayRef<std::string> ExtraSymbols) {
  // 2. Target and names.
  Input.setTargetTriple(TM.getTargetTriple());
  Input.setDataLayout(TM.createDataLayout());
  nameUnamedGlobals(Input);
  // Inferred before the snapshot, so that Gen here and Use in tier-up see the
  // same attributes (noreturn and cold change branch probabilities, and with
  // them where Gen places counters).
  DeclaredAttributes Declared = inferFunctionAttributes(Input);

  // 3. Snapshot, then continue on a re-parse of it.
  std::string Snapshot = writeSnapshot(Input);
  AotModule Aot;
  Aot.Ctx = std::make_unique<LLVMContext>();
  Expected<std::unique_ptr<Module>> Parsed = parseSnapshot(Snapshot, *Aot.Ctx);
  // The bytes writeSnapshot just wrote parse.
  if (!Parsed)
    return Parsed.takeError(); // LCOV_EXCL_LINE
  Aot.M = std::move(*Parsed);
  Module &M = *Aot.M;
  std::vector<GlobalValue *> Symbols = collectSymbols(M);

  // 4 and 5. Configure and instrument. Candidates are found while Gen's
  // intrinsics are still there.
  if (Error E = runPgoGen(M))
    return std::move(E);
  dropInvalidatedAttributes(M, Declared);
  uint64_t RawVersion = rawProfileVersion(M);
  std::vector<ThunkCandidate> Candidates = prepareThunkCandidates(M);
  GlobalVariable *SymbolTable = emitSymbolTable(M, Symbols);

  // 6. Lower, with TM so that loop rotation uses the target's cost model, as
  // under clang. The records name each function itself; thunkify keeps that
  // Function as the thunk, so they name the address indirect calls observe.
  Expected<LoweredPgo> Lowered = lowerPgo(M, &TM);
  if (!Lowered)
    return Lowered.takeError(); // LCOV_EXCL_LINE runPgoGen made every record

  // 7. Optimize.
  optimizeAot(M, TM);

  // 8. Thunkify.
  // Every function is an entry of the symbol table, which is in
  // llvm.compiler.used, so O2 keeps all candidates; thunkify would drop any it
  // did not.
  ThunkTable Thunks = thunkify(M, Candidates);
  assert(Thunks.Thunks.size() == Candidates.size() &&
         "optimization deleted a thunk candidate");

  // 9. Glue. The names the module does not mention are added to the symbol
  // table only now: a stub declaration of the wrong type, present during O2,
  // would keep it from emitting calls to the library function of that name
  // (isLibFuncEmittable), such as memset_pattern16 and exp2.
  std::vector<std::string> Extra(ExtraSymbols.begin(), ExtraSymbols.end());
  for (std::string &Name : platformSymbols(TM.getTargetTriple()))
    Extra.push_back(std::move(Name));
  SymbolTable = extendSymbolTable(M, *SymbolTable, Extra);
  emitGlue(M, Snapshot, RawVersion, *Lowered, Thunks, *SymbolTable);
  return std::move(Aot);
}

Expected<SmallVector<char, 0>> emitObject(Module &M, TargetMachine &TM) {
  SmallVector<char, 0> Object;
  raw_svector_ostream OS(Object);
  legacy::PassManager PM;
  // LCOV_EXCL_START native targets can emit object files
  if (TM.addPassesToEmitFile(PM, OS, nullptr, CodeGenFileType::ObjectFile))
    return createStringError("orc-pgo: the target cannot emit object files");
  // LCOV_EXCL_STOP
  PM.run(M);
  return std::move(Object);
}

Expected<SmallVector<char, 0>>
buildAotObject(Module &M, TargetMachine &TM,
               ArrayRef<std::string> ExtraSymbols) {
  Expected<AotModule> Aot = buildAotModule(M, TM, ExtraSymbols);
  if (!Aot)
    return Aot.takeError();
  return emitObject(*Aot->M, TM);
}

} // namespace orc_pgo
