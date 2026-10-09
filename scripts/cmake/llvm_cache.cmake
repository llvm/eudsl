#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.
#

include(CMakePrintHelpers)

set(LLVM_ENABLE_PROJECTS "llvm;mlir;clang" CACHE STRING "")

if (NOT WIN32)
  # compiler-rt builtins only (built by the separate `builtins` sub-build);
  # every compiler-rt runtime is off.
  set(LLVM_ENABLE_RUNTIMES "compiler-rt" CACHE STRING "")
  set(COMPILER_RT_BUILD_BUILTINS ON CACHE BOOL "")
  foreach(_crt SANITIZERS XRAY LIBFUZZER PROFILE MEMPROF CTX_PROFILE ORC GWP_ASAN)
    set(COMPILER_RT_BUILD_${_crt} OFF CACHE BOOL "")
  endforeach()
  set(COMPILER_RT_INCLUDE_TESTS OFF CACHE BOOL "")
  # macOS only; avoids needing iOS/watchOS/tvOS/visionOS SDKs.
  foreach(_os IOS WATCHOS TVOS XROS)
    set(COMPILER_RT_ENABLE_${_os} OFF CACHE BOOL "")
  endforeach()
endif()

# LLVM options

set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_UTILS ON CACHE BOOL "")
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_ENABLE_DUMP ON CACHE BOOL "")

set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
if (WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded CACHE STRING "")
  list(APPEND CMAKE_C_FLAGS "/MT")
  list(APPEND CMAKE_CXX_FLAGS "/MT")
endif()

# useful things
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
if (WIN32)
  set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
else()
  set(LLVM_ENABLE_WARNINGS ON CACHE BOOL "")
endif()
set(LLVM_FORCE_ENABLE_STATS ON CACHE BOOL "")
# because AMD target td files are insane...
set(LLVM_TARGETS_TO_BUILD "host;NVPTX;AMDGPU" CACHE STRING "")
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "DirectX" CACHE STRING "")
set(LLVM_OPTIMIZED_TABLEGEN ON CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_VERSION_SUFFIX "" CACHE STRING "")
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON CACHE BOOL "")

# MLIR options

set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")
set(MLIR_ENABLE_EXECUTION_ENGINE ON CACHE BOOL "")
set(MLIR_ENABLE_SPIRV_CPU_RUNNER ON CACHE BOOL "")

# space savings

set(LLVM_BUILD_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")

set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_ENABLE_LZMA OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_GO_TESTS OFF CACHE BOOL "")

# tests

option(RUN_TESTS "" OFF)
set(LLVM_INCLUDE_TESTS ${RUN_TESTS} CACHE BOOL "")
set(LLVM_BUILD_TESTS ${RUN_TESTS} CACHE BOOL "")
set(MLIR_INCLUDE_INTEGRATION_TESTS ${RUN_TESTS} CACHE BOOL "")
set(MLIR_INCLUDE_TESTS ${RUN_TESTS} CACHE BOOL "")

### Distributions ###

set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")

set(LLVM_DISTRIBUTIONS MlirDevelopment CACHE STRING "")
set(_mlir_dev_components
    clang-libraries
    clang-headers
    clang-resource-headers
    # triggers ClangConfig.cmake and etc
    clang-cmake-exports
    # triggers ClangMlirDevelopmentTargets.cmake
    clang-mlirdevelopment-cmake-exports

    # triggers ClangConfig.cmake and etc
    cmake-exports
    # triggers LLVMMlirDevelopmentExports.cmake
    mlirdevelopment-cmake-exports
    llvm-config
    llvm-cov
    llvm-headers
    llvm-libraries
    llvm-profdata
    llvm-tblgen

    split-file
    count
    FileCheck
    not
    MLIRPythonModules
    # triggers MLIRMlirDevelopmentTargets.cmake
    mlir-mlirdevelopment-cmake-exports
    # triggers MLIRConfig.cmake and etc
    mlir-cmake-exports
    mlir-headers
    mlir-libraries
    mlir-linalg-ods-yaml-gen
    mlir-opt
    mlir-pdll
    mlir-python-sources
    mlir-reduce
    mlir-tblgen
    mlir-translate
)

if (NOT WIN32)
  list(APPEND _mlir_dev_components LLVM MLIR builtins)
endif()

# Only cache entries survive a -C script, so the list must be complete before
# it is cached (appending to the variable afterwards has no effect).
set(LLVM_MlirDevelopment_DISTRIBUTION_COMPONENTS ${_mlir_dev_components} CACHE STRING "")

get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)
cmake_print_variables(${_variableNames})
