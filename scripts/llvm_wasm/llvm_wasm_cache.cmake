#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.
#

set(LLVM_ENABLE_PROJECTS "mlir;llvm;lld" CACHE STRING "")

set(LLVM_TARGETS_TO_BUILD "WebAssembly" CACHE STRING "")
set(LLVM_TARGET_ARCH "wasm32" CACHE STRING "")
set(LLVM_DEFAULT_TARGET_TRIPLE "wasm32-unknown-emscripten" CACHE STRING "")
set(LLVM_HOST_TRIPLE "wasm32-unknown-emscripten" CACHE STRING "")
set(LLVM_BUILD_STATIC ON CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_ENABLE_PIC ON CACHE BOOL "")

set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")
set(MLIR_ENABLE_EXECUTION_ENGINE ON CACHE BOOL "")
set(MLIR_ENABLE_SPIRV_CPU_RUNNER ON CACHE BOOL "")

set(LLVM_BUILD_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_ENABLE_THREADS OFF CACHE BOOL "")
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_GO_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_BUILD_TESTS OFF CACHE BOOL "")
set(MLIR_INCLUDE_INTEGRATION_TESTS OFF CACHE BOOL "")
set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "")

### Distributions ###

set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")

set(LLVM_DISTRIBUTIONS MlirDevelopment CACHE STRING "")
set(LLVM_MlirDevelopment_DISTRIBUTION_COMPONENTS
    # triggers ClangConfig.cmake and etc
    cmake-exports
    # triggers LLVMMlirDevelopmentExports.cmake
    mlirdevelopment-cmake-exports

    lldCommon
    lldWasm
    lld-headers
    lld-libraries
    lld-cmake-exports
    lld-mlirdevelopment-cmake-exports

    llvm-config
    llvm-headers
    llvm-libraries

    MLIRPythonModules
    # triggers MLIRMlirDevelopmentTargets.cmake
    mlir-mlirdevelopment-cmake-exports
    # triggers MLIRConfig.cmake and etc
    mlir-cmake-exports
    mlir-headers
    mlir-libraries
    mlir-python-sources
    CACHE STRING "")
