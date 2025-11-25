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
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

#include <string>

namespace nb = nanobind;

NB_MODULE(_mlirWasmExecutionEngine, m) {
  m.def("compile", [](MlirOperation module, const std::string &moduleName) {
    MlirStringRef name =
        compile(module, mlirStringRefCreateFromCString(moduleName.c_str()));
    return std::string(name.data, name.length);
  });
  m.def("link_load", [](const std::string &objectFileName,
                        const std::string &binaryFileName) {
    linkLoad(mlirStringRefCreateFromCString(objectFileName.c_str()),
             mlirStringRefCreateFromCString(binaryFileName.c_str()));
  });
  m.def("get_symbol_address", [](const std::string &name) {
    return reinterpret_cast<uintptr_t>(
        getSymbolAddress(mlirStringRefCreateFromCString(name.c_str())));
  });
}