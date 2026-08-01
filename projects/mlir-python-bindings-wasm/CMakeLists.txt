# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2025.

cmake_minimum_required(VERSION 3.29)
project(mlir-python-bindings-wasm LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0116 OLD)
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLD REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLDConfig.cmake in: ${LLD_DIR}")

set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir" CACHE STRING "")
set(MLIR_PYTHON_PACKAGE_PREFIX "mlir" CACHE STRING "")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_CURRENT_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)
include(MLIRDetectPythonEnv)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

mlir_configure_python_dev_packages()

add_mlir_public_c_api_library(WasmCompilerLinkerLoaderCAPI
  WasmCompilerLinkerLoaderCAPI.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRToLLVMIRTranslationRegistration
  MLIRTargetLLVMIRExport
  MLIRCAPIIR
  MLIRLLVMToLLVMIRTranslation
  MLIRLLVMIRToLLVMTranslation
  MLIRSupport
  lldWasm
  lldCommon
)

declare_mlir_python_sources(MLIRPythonExtension.WasmExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES
    wasm_execution_engine.py
)

declare_mlir_python_extension(MLIRPythonExtension.WasmExecutionEngine.extension
  MODULE_NAME _mlirWasmExecutionEngine
  ADD_TO_PARENT MLIRPythonExtension.WasmExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    WasmExecutionEngine.cpp
  EMBED_CAPI_LINK_LIBS
    WasmCompilerLinkerLoaderCAPI
)

# ##############################################################################
# SPIR-V -> WGSL, so kernels compiled to SPIR-V in the browser can be handed to
# WebGPU. Requires Tint, so it is opt-in: without MLIR_PYTHON_BINDINGS_WGSL the
# wheel is built exactly as before.
#
# Only Tint's SPIR-V reader and WGSL writer are needed; Dawn's runtime is not.
# The browser already implements WebGPU, so nothing here talks to a device.
# ##############################################################################

option(MLIR_PYTHON_BINDINGS_WGSL "Build the SPIR-V -> WGSL extension (needs Tint)" OFF)

if(MLIR_PYTHON_BINDINGS_WGSL)
  if(NOT TARGET tint_lang_spirv_reader OR NOT TARGET tint_lang_wgsl_writer)
    message(FATAL_ERROR
      "MLIR_PYTHON_BINDINGS_WGSL requires the tint_lang_spirv_reader and "
      "tint_lang_wgsl_writer targets. Add dawn (and SPIRV-Tools before it, see "
      "the note in this file's git history) to the build first.")
  endif()

  add_mlir_public_c_api_library(SPIRVToWGSLCAPI
    SPIRVToWGSL.cpp

    PARTIAL_SOURCES_INTENDED

    LINK_LIBS PUBLIC
    tint_lang_spirv_reader
    tint_lang_wgsl_writer
    tint_lang_core_ir
  )

  declare_mlir_python_sources(MLIRPythonExtension.SPIRVToWGSL
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
    ADD_TO_PARENT MLIRPythonSources
    SOURCES
      wgsl.py
  )

  declare_mlir_python_extension(MLIRPythonExtension.SPIRVToWGSL.extension
    MODULE_NAME _mlirSPIRVToWGSL
    ADD_TO_PARENT MLIRPythonExtension.SPIRVToWGSL
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
      SPIRVToWGSLModule.cpp
    EMBED_CAPI_LINK_LIBS
      SPIRVToWGSLCAPI
  )
endif()

add_mlir_python_common_capi_library(MLIRPythonWasmCAPI
  INSTALL_COMPONENT MLIRPythonWasmModules
  INSTALL_DESTINATION mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
)

# ##############################################################################
# The fully assembled package of modules. This must come last.
# ##############################################################################

add_mlir_python_modules(MLIRPythonWasmModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/mlir"
  INSTALL_PREFIX "mlir"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    MLIRPythonWasmCAPI
)

# Without STANDALONE, add_mlir_library appends `LINK_COMPONENTS Support`, which pulls
# LLVMSupport (and with it CommandLine.cpp) into this side module. Its file-scope
# static cl::opt objects then re-register options that libMLIRPythonWasmCAPI.so already
# owns, and cl::Option::addArgument aborts the moment Pyodide dlopens this library:
#   CommandLine Error: Option 'print-inst-addrs' registered more than once!
# 32bRunnerUtils.cpp only needs the C++ stdlib, so drop the dependency entirely. This
# mirrors upstream's own mlir_runner_utils, which is also declared STANDALONE.
add_mlir_library(32b_mlir_runner_utils
  SHARED
  32bRunnerUtils.cpp

  PARTIAL_SOURCES_INTENDED
  EXCLUDE_FROM_LIBMLIR

  STANDALONE
)

set(_runtimes
  32b_mlir_runner_utils
)

foreach(r ${_runtimes})
  add_custom_command(
    TARGET MLIRPythonWasmModules
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${r}>
            ${CMAKE_CURRENT_BINARY_DIR}/mlir/_mlir_libs)
endforeach()

install(
  IMPORTED_RUNTIME_ARTIFACTS ${_runtimes}
  LIBRARY DESTINATION "mlir/_mlir_libs"
  COMPONENT MLIRPythonWasmModules
)

message(STATUS "CMAKE_GENERATOR=${CMAKE_GENERATOR}")
