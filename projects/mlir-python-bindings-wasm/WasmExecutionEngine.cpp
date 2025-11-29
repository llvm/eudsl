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
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/Target/TargetMachine.h"

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_mlirWasmExecutionEngine, m) {
  m.def(
      "compile_module",
      [](MlirOperation module, const std::string &moduleName, int optLevel) {
        MlirStringRef name = compileModule(
            module, mlirStringRefCreateFromCString(moduleName.c_str()),
            optLevel);
        return std::string(name.data, name.length);
      },
      "module"_a, "module_name"_a, "opt_level"_a = 2);
  m.def(
      "link_load_module",
      [](const std::string &objectFileName, const std::string &binaryFileName) {
        linkLoadModule(mlirStringRefCreateFromCString(objectFileName.c_str()),
                       mlirStringRefCreateFromCString(binaryFileName.c_str()));
      },
      "object_filename"_a, "binary_filename"_a);
  m.def(
      "get_symbol_address",
      [](const std::string &name) {
        return reinterpret_cast<uintptr_t>(
            getSymbolAddress(mlirStringRefCreateFromCString(name.c_str())));
      },
      "name"_a);
}
