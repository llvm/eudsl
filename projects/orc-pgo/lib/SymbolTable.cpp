// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/SymbolTable.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace orc_pgo {

static bool isTierUpVisible(const GlobalValue &GV) {
  if (!GV.hasName() || GV.getName().starts_with("llvm.") || GV.isThreadLocal())
    return false;
  if (GV.isDeclaration() || GV.hasAvailableExternallyLinkage())
    return !GV.use_empty();
  return true;
}

std::vector<GlobalValue *> collectSymbols(Module &M) {
  std::vector<GlobalValue *> Symbols;
  for (Function &F : M)
    if (isTierUpVisible(F))
      Symbols.push_back(&F);
  for (GlobalVariable &GV : M.globals())
    if (isTierUpVisible(GV))
      Symbols.push_back(&GV);
  for (GlobalAlias &GA : M.aliases())
    if (isTierUpVisible(GA))
      Symbols.push_back(&GA);
  return Symbols;
}

std::vector<std::string> platformSymbols(const Triple &T) {
  std::vector<std::string> Symbols = {"_Unwind_Resume", "memcpy", "memmove",
                                      "memset", "__stack_chk_fail"};
  if (T.isOSDarwin())
    for (const char *Name : {"bzero", "memset_pattern4", "memset_pattern8",
                             "memset_pattern16", "__stack_chk_guard"})
      Symbols.push_back(Name);
  for (const char *Name :
       {"sin",  "cos",  "tan",   "asin",  "acos", "atan",      "atan2", "sinh",
        "cosh", "tanh", "pow",   "exp",   "exp2", "log",       "log2",  "log10",
        "fmod", "ceil", "floor", "trunc", "rint", "nearbyint", "round", "fma",
        "fmin", "fmax", "ldexp", "frexp", "modf"})
    for (const char *Suffix : {"", "f", "l"})
      Symbols.push_back(std::string(Name) + Suffix);
  return Symbols;
}

static StructType *entryType(LLVMContext &Ctx) {
  auto *PtrTy = PointerType::getUnqual(Ctx);
  return StructType::get(Ctx, {PtrTy, PtrTy});
}

static Constant *makeEntry(Module &M, GlobalValue *GV) {
  LLVMContext &Ctx = M.getContext();
  Constant *NameStr = ConstantDataArray::getString(Ctx, GV->getName());
  auto *Name = new GlobalVariable(M, NameStr->getType(), /*isConstant=*/true,
                                  GlobalValue::PrivateLinkage, NameStr,
                                  "__orc_pgo_symbol_name");
  Name->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return ConstantStruct::get(entryType(Ctx), {Name, GV});
}

static GlobalVariable *makeTable(Module &M, ArrayRef<Constant *> Entries) {
  auto *TableTy = ArrayType::get(entryType(M.getContext()), Entries.size());
  auto *Table = new GlobalVariable(M, TableTy, /*isConstant=*/true,
                                   GlobalValue::InternalLinkage,
                                   ConstantArray::get(TableTy, Entries), "");
  return Table;
}

GlobalVariable *emitSymbolTable(Module &M, ArrayRef<GlobalValue *> Symbols) {
  std::vector<Constant *> Entries;
  for (GlobalValue *GV : Symbols)
    Entries.push_back(makeEntry(M, GV));
  GlobalVariable *Table = makeTable(M, Entries);
  Table->setName(SymbolTableName);
  appendToCompilerUsed(M, {Table});
  return Table;
}

/// The type of the declaration of a name M does not mention. Only the name
/// matters, except for functions that code generation looks up by name and
/// type: it forms a memset_pattern16 call from llvm.experimental.memset.pattern
/// only if M's memset_pattern16 has the library's prototype.
static FunctionType *stubType(Module &M, StringRef Name) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  if (Name == "memset_pattern16") {
    auto *PtrTy = PointerType::getUnqual(Ctx);
    return FunctionType::get(
        VoidTy, {PtrTy, PtrTy, M.getDataLayout().getIntPtrType(Ctx)},
        /*isVarArg=*/false);
  }
  return FunctionType::get(VoidTy, /*isVarArg=*/false);
}

GlobalVariable *extendSymbolTable(Module &M, GlobalVariable &Table,
                                  ArrayRef<std::string> Names) {
  SetVector<GlobalValue *> Seen;
  std::vector<Constant *> Entries;
  Constant *Init = Table.getInitializer();
  for (unsigned I = 0,
                E = cast<ArrayType>(Table.getValueType())->getNumElements();
       I != E; ++I) {
    Constant *Entry = Init->getAggregateElement(I);
    Seen.insert(cast<GlobalValue>(Entry->getAggregateElement(1u)));
    Entries.push_back(Entry);
  }
  for (const std::string &Name : Names) {
    GlobalValue *GV = M.getNamedValue(Name);
    if (!GV)
      GV = Function::Create(stubType(M, Name), GlobalValue::ExternalLinkage,
                            Name, M);
    if (Seen.insert(GV))
      Entries.push_back(makeEntry(M, GV));
  }
  GlobalVariable *Extended = makeTable(M, Entries);
  Table.replaceAllUsesWith(Extended);
  Table.eraseFromParent();
  Extended->setName(SymbolTableName);
  return Extended;
}

} // namespace orc_pgo
