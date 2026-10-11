// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TestSupport.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>

using namespace llvm;

namespace orc_pgo::test {

static void initLLVM() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
  });
}

std::unique_ptr<Module> parseIR(LLVMContext &Ctx, StringRef Text) {
  SMDiagnostic Err;
  auto M = parseAssemblyString(Text, Err, Ctx);
  if (!M) {
    Err.print("orc-pgo-test", errs());
    report_fatal_error("invalid test IR");
  }
  return M;
}

std::unique_ptr<Module> parseHostIR(LLVMContext &Ctx, StringRef Text) {
  initLLVM();
  auto M = parseIR(Ctx, Text);
  auto JTMB = cantFail(orc::JITTargetMachineBuilder::detectHost());
  M->setTargetTriple(JTMB.getTargetTriple());
  M->setDataLayout(cantFail(JTMB.getDefaultDataLayoutForTarget()));
  return M;
}

std::string errorText(Error E) {
  return E ? toString(std::move(E)) : std::string();
}

std::unique_ptr<TargetMachine> makeHostTargetMachine() {
  initLLVM();
  auto JTMB = cantFail(orc::JITTargetMachineBuilder::detectHost());
  return cantFail(JTMB.createTargetMachine());
}

std::unique_ptr<orc::LLJIT> makeJIT() {
  initLLVM();
  return cantFail(orc::LLJITBuilder().create());
}

std::unique_ptr<orc::LLJIT> jitModule(std::unique_ptr<Module> M,
                                      std::unique_ptr<LLVMContext> Ctx) {
  auto J = makeJIT();
  orc::SymbolMap Runtime;
  auto Define = [&](StringRef Name, const void *Addr) {
    Runtime[J->mangleAndIntern(Name)] = {orc::ExecutorAddr::fromPtr(Addr),
                                         JITSymbolFlags::Exported};
  };
  Define("__orc_pgo_instrument_target",
         reinterpret_cast<const void *>(&__orc_pgo_instrument_target));
  Define("__orc_pgo_instrument_memop",
         reinterpret_cast<const void *>(&__orc_pgo_instrument_memop));
  cantFail(
      J->getMainJITDylib().define(orc::absoluteSymbols(std::move(Runtime))));
  cantFail(J->addIRModule(orc::ThreadSafeModule(std::move(M), std::move(Ctx))));
  return J;
}

void runO2(Module &M) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2).run(M, MAM);
}

} // namespace orc_pgo::test
