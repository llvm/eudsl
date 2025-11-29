//===----------------- Wasm.cpp - Wasm Interpreter --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements interpreter support for code execution in WebAssembly.
//
//===----------------------------------------------------------------------===//

#include "WasmCompilerLinkerLoaderCAPI.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <string>

namespace lld {
enum Flavor {
  Invalid,
  Gnu,     // -flavor gnu
  MinGW,   // -flavor gnu MinGW
  WinLink, // -flavor link
  Darwin,  // -flavor darwin
  Wasm,    // -flavor wasm
};

using Driver = bool (*)(llvm::ArrayRef<const char *>, llvm::raw_ostream &,
                        llvm::raw_ostream &, bool, bool);

struct DriverDef {
  Flavor f;
  Driver d;
};

struct Result {
  int retCode;
  bool canRunAgain;
};

Result lldMain(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
               llvm::raw_ostream &stderrOS, llvm::ArrayRef<DriverDef> drivers);

namespace wasm {
bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
} // namespace wasm
} // namespace lld

#include <dlfcn.h>

MlirStringRef compileModule(MlirOperation module, MlirStringRef moduleName,
                            int optLevel) {
  static bool initOnce = [] {
    // TODO(max): why doesn't this work? "no targets registered"
    // LLVMInitializeWebAssemblyTarget();
    // LLVMInitializeWebAssemblyAsmParser();
    // LLVMInitializeWebAssemblyAsmPrinter();
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initOnce;

  mlir::Operation *m = unwrap(module);
  std::unique_ptr<llvm::LLVMContext> ctx(new llvm::LLVMContext);
  auto llvmModule = mlir::translateModuleToLLVMIR(m, *ctx);
  if (!llvmModule)
    llvm::report_fatal_error("could not convert to LLVM IR");

  std::string errorString;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      llvmModule->getTargetTriple(), errorString);
  if (!target) {
    auto reason = "Failed to create Wasm Target: " + errorString;
    llvm::report_fatal_error(reason.c_str());
  }

  llvm::TargetOptions to = llvm::TargetOptions();
  llvm::TargetMachine *targetMachine = target->createTargetMachine(
      llvmModule->getTargetTriple(), "", "", to, llvm::Reloc::Model::PIC_);
  assert(optLevel >= 0 && optLevel <= 3 && "expected optLevel between 0 and 3");
  targetMachine->setOptLevel(static_cast<llvm::CodeGenOptLevel>(optLevel));
  llvmModule->setDataLayout(targetMachine->createDataLayout());
  std::string moduleNameStr(unwrap(moduleName));
  std::string objectFileName = moduleNameStr + ".o";

  std::error_code error;
  llvm::raw_fd_ostream objectFileOutput(llvm::StringRef(objectFileName), error);

  llvm::legacy::PassManager pm;
  if (targetMachine->addPassesToEmitFile(pm, objectFileOutput, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    llvm::report_fatal_error("Wasm backend cannot produce object.");
  }

  if (!pm.run(*llvmModule))
    llvm::report_fatal_error("Failed to emit Wasm object.");

  objectFileOutput.close();
  return mlirStringRefCreateFromCString(objectFileName.c_str());
}

void linkLoadModule(MlirStringRef objectFileName,
                    MlirStringRef binaryFileName) {
  std::string objectFileNameStr(unwrap(objectFileName));
  std::string binaryFileNameStr(unwrap(binaryFileName));
  std::vector<const char *> linkerArgs = {"wasm-ld",
                                          "-shared",
                                          "--import-memory",
                                          "--experimental-pic",
                                          "--stack-first",
                                          "--allow-undefined",
                                          objectFileNameStr.c_str(),
                                          "-o",
                                          binaryFileNameStr.c_str()};

  const lld::DriverDef wasmDriver = {lld::Flavor::Wasm, &lld::wasm::link};
  std::vector<lld::DriverDef> wasmDriverArgs;
  wasmDriverArgs.push_back(wasmDriver);
  lld::Result result =
      lld::lldMain(linkerArgs, llvm::outs(), llvm::errs(), wasmDriverArgs);

  if (result.retCode)
    llvm::report_fatal_error("Failed to link incremental module");

  void *loadedLibModule =
      dlopen(binaryFileNameStr.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!loadedLibModule)
    llvm::report_fatal_error("Failed to link incremental module");
}

void *getSymbolAddress(MlirStringRef name) {
  std::string nameStr(unwrap(name));
  void *sym = dlsym(RTLD_DEFAULT, nameStr.c_str());
  if (!sym)
    llvm::report_fatal_error("dlsym failed for symbol: ");
  return sym;
}
