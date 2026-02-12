# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2025.

cmake_minimum_required(VERSION 3.29)
project(eudsl-tile LANGUAGES CXX C)

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

if(MSVC OR WIN32)
  add_compile_options(
    $<$<CONFIG:>:/MT> # ---------|
    $<$<CONFIG:Debug>:/MTd> # ---|-- Statically link the runtime libraries
    $<$<CONFIG:Release>:/MT> # --|
  )
endif()

set(CMAKE_C_VISIBILITY_PRESET "hidden")
set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "eudsl_tile" CACHE STRING "")
set(MLIR_PYTHON_PACKAGE_PREFIX "eudsl_tile" CACHE STRING "")

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

# This includes eudsl-python-extras
declare_mlir_python_sources(EUDSLTilePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eudsl_tile"
  SOURCES_GLOB "*.py"
)

set(_ext_sources
  # not MLIRPythonExtension.Core, which doesn't depend on MLIRPythonSupport and etc.
  MLIRPythonSources.Core
  # we register everything even though we don't use all the dialects
  # because currently RegisterEverything isn't factored (and probably won't ever be...)
  MLIRPythonExtension.RegisterEverything
  # other extensions
  MLIRPythonExtension.Dialects.GPU.Nanobind
  MLIRPythonExtension.Dialects.LLVM.Nanobind
  MLIRPythonExtension.Dialects.Linalg.Nanobind
  MLIRPythonExtension.Dialects.Transform.Nanobind
  MLIRPythonExtension.ExecutionEngine
  MLIRPythonExtension.GPUDialectPasses
  MLIRPythonExtension.LinalgPasses
  MLIRPythonExtension.TransformInterpreter
)

add_mlir_python_common_capi_library(EUDSLTilePythonCAPI
  INSTALL_COMPONENT EUDSLTilePythonModules
  INSTALL_DESTINATION eudsl_tile/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/eudsl_tile/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  # Source groups from which to discover dependent EMBED_CAPI_LINK_LIBS
  DECLARED_SOURCES ${_ext_sources}
)

# ##############################################################################
# The fully assembled package of modules. This must come last.
# ##############################################################################

set(_declared_sources
  ${_ext_sources}
  # dialects
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.bufferization
  MLIRPythonSources.Dialects.bufferization_transform
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.gpu
  MLIRPythonSources.Dialects.gpu_transform
  MLIRPythonSources.Dialects.index
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.memref_transform
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.spirv
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.tensor_transform
  MLIRPythonSources.Dialects.transform
  MLIRPythonSources.Dialects.transform.extras
  MLIRPythonSources.Dialects.transform.interpreter
  MLIRPythonSources.Dialects.vector
  MLIRPythonSources.Dialects.vector_transform
  # other sources
  MLIRPythonSources.ExecutionEngine
  EUDSLTilePythonSources
)
if(NOT CMAKE_CROSSCOMPILING)
  list(APPEND _declared_sources MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_modules(EUDSLTilePythonModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/eudsl_tile"
  INSTALL_PREFIX "eudsl_tile"
  MLIR_BINDINGS_PYTHON_NB_DOMAIN ${MLIR_BINDINGS_PYTHON_NB_DOMAIN}
  DECLARED_SOURCES
    ${_declared_sources}
  COMMON_CAPI_LINK_LIBS
    EUDSLTilePythonCAPI
)

set(_runtimes
  mlir_async_runtime
  mlir_c_runner_utils
  mlir_float16_utils
  mlir_runner_utils
)

if (TARGET mlir_apfloat_wrappers)
  list(APPEND _runtimes mlir_apfloat_wrappers)
endif()

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
    TARGET EUDSLTilePythonModules
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${r}>
            ${CMAKE_CURRENT_BINARY_DIR}/eudsl_tile/_mlir_libs)
endforeach()

install(
  IMPORTED_RUNTIME_ARTIFACTS ${_runtimes}
  LIBRARY DESTINATION "eudsl_tile/_mlir_libs"
  COMPONENT EUDSLTilePythonModules
)
