// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SimulatedAot.h"
#include "orc-pgo/Pipeline.h"
#include "orc-pgo/Snapshot.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace orc_pgo::test {

namespace {
const OrcPgoImage *Registered = nullptr;
unsigned StartCalls = 0;

void registerImageStub(const OrcPgoImage *Image) { Registered = Image; }
void startStub() { ++StartCalls; }
} // namespace

const OrcPgoFunctionRecord &SimulatedAot::record(StringRef PgoName) const {
  for (const OrcPgoFunctionRecord &R :
       ArrayRef(Image->Functions, Image->NumFunctions))
    if (PgoName == R.Name)
      return R;
  ADD_FAILURE() << "no record for " << PgoName.str();
  return Image->Functions[0];
}

unsigned SimulatedAot::slot(StringRef PgoName) const {
  for (unsigned I = 0; I != Image->NumDispatch; ++I)
    if (PgoName == Image->Functions[Image->DispatchFunctions[I]].Name)
      return I;
  ADD_FAILURE() << "no dispatch slot for " << PgoName.str();
  return 0;
}

const OrcPgoSymbol *SimulatedAot::symbol(StringRef Name) const {
  for (const OrcPgoSymbol &S : ArrayRef(Image->Symbols, Image->NumSymbols))
    if (Name == S.Name)
      return &S;
  return nullptr;
}

SimulatedAot simulateAot(StringRef IR, ArrayRef<std::string> ExtraSymbols,
                         ArrayRef<HostSymbol> HostSymbols) {
  LLVMContext InputCtx;
  auto Input = parseIR(InputCtx, IR);
  auto TM = makeHostTargetMachine(Reloc::PIC_);
  Expected<AotModule> Aot = buildAotModule(*Input, *TM, ExtraSymbols);
  if (!Aot)
    report_fatal_error(Aot.takeError());
  EXPECT_FALSE(verifyModule(*Aot->M, &errs()));

  SimulatedAot S;
  S.J = makeJIT();
  orc::SymbolMap Symbols;
  auto Define = [&](StringRef Name, const void *Address) {
    Symbols[S.J->mangleAndIntern(Name)] = {orc::ExecutorAddr::fromPtr(Address),
                                           JITSymbolFlags::Exported};
  };
  Define(RegisterImageName, reinterpret_cast<const void *>(&registerImageStub));
  Define(StartName, reinterpret_cast<const void *>(&startStub));
  for (const HostSymbol &H : HostSymbols)
    Define(H.Name, H.Address);
  cantFail(
      S.J->getMainJITDylib().define(orc::absoluteSymbols(std::move(Symbols))));
  // jitModule also provides the value-profiling callbacks an executable links
  // from OrcPgoRuntime, and, on ELF, makes the instrumented module's comdat any
  // globals weak (weakenElfComdatAnyGlobals in TestSupport.cpp), as every JIT
  // of instrumented code must.
  S.J = jitModule(std::move(S.J), std::move(Aot->M), std::move(Aot->Ctx));

  Registered = nullptr;
  cantFail(S.J->initialize(S.J->getMainJITDylib()));
  if (!Registered)
    report_fatal_error("the registration constructor did not run");
  S.Image = Registered;
  resetValueNodePools();
  setValueNodePool(S.Image->VNodePool[0], S.Image->VNodePool[1]);
  return S;
}

std::string editedSnapshot(const OrcPgoImage &Image,
                           function_ref<void(Module &)> Edit) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = cantFail(
      parseSnapshot(StringRef(Image.Snapshot, Image.SnapshotSize), Ctx));
  Edit(*M);
  return writeSnapshot(*M);
}

OrcPgoImage withSnapshot(const OrcPgoImage &Image, const std::string &Bitcode) {
  OrcPgoImage Edited = Image;
  Edited.Snapshot = Bitcode.data();
  Edited.SnapshotSize = Bitcode.size();
  return Edited;
}

unsigned simulatedStartCalls() { return StartCalls; }

} // namespace orc_pgo::test
