# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2025.

include_directories(${LLVM_SOURCE_DIR}/include)
include_directories(${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

add_mlir_upstream_c_api_library(WasmCompilerLinkerLoaderCAPI
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

declare_mlir_python_extension(MLIRPythonExtension.WasmExecutionEngine
  MODULE_NAME _mlirWasmExecutionEngine
  ADD_TO_PARENT MLIRPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    WasmExecutionEngine.cpp
  EMBED_CAPI_LINK_LIBS
    WasmCompilerLinkerLoaderCAPI
)

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

set(_declared_sources MLIRPythonSources MLIRPythonExtension.RegisterEverything)
if(NOT CMAKE_CROSSCOMPILING)
  list(APPEND _declared_sources MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_modules(MLIRPythonWasmModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/mlir"
  INSTALL_PREFIX "mlir"
  DECLARED_SOURCES
    ${_declared_sources}
  COMMON_CAPI_LINK_LIBS
    MLIRPythonWasmCAPI
)

set(_runtimes
  mlir_async_runtime
  mlir_c_runner_utils
  mlir_float16_utils
  mlir_runner_utils
)

if (TARGET mlir_rocm_runtime)
  list(APPEND _runtimes mlir_rocm_runtime)
endif()

if (TARGET mlir_cuda_runtime)
  list(APPEND _runtimes mlir_cuda_runtime)
endif()

if (TARGET omp)
  list(APPEND _runtimes omp)
endif()

if (TARGET vulkan-runtime-wrappers)
  list(APPEND _runtimes vulkan-runtime-wrappers)
endif()

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

# TODO(max): look up CMP0002 policy