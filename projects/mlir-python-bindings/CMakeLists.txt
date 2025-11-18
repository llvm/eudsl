# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2025.

project(mlir-python-bindings LANGUAGES CXX C)
cmake_minimum_required(VERSION 3.29)

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

declare_mlir_python_sources(MLIRPythonCAPI.TableGenSources
  ADD_TO_PARENT MLIRPythonCAPI.HeaderSources
  ROOT_DIR "${MLIR_INCLUDE_DIRS}"
  SOURCES_GLOB "mlir/*.td"
)

add_mlir_python_common_capi_library(MLIRPythonCAPI
  INSTALL_COMPONENT MLIRPythonModules
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
# Custom targets.
# ##############################################################################

_flatten_mlir_python_targets(mlir_python_sources_deps MLIRPythonSources)
add_custom_target("mlir-python-sources" DEPENDS ${mlir_python_sources_deps})
if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-mlir-python-sources DEPENDS
    mlir-python-sources COMPONENT mlir-python-sources)
endif()

# ##############################################################################
# The fully assembled package of modules. This must come last.
# ##############################################################################

set(_declared_sources MLIRPythonSources MLIRPythonExtension.RegisterEverything)
if(NOT CMAKE_CROSSCOMPILING)
  list(APPEND _declared_sources MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_modules(MLIRPythonModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/mlir"
  INSTALL_PREFIX "mlir"
  DECLARED_SOURCES
    ${_declared_sources}
  COMMON_CAPI_LINK_LIBS
    MLIRPythonCAPI
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
    TARGET MLIRPythonModules
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${r}>
            ${CMAKE_CURRENT_BINARY_DIR}/mlir/_mlir_libs)
endforeach()

install(
  IMPORTED_RUNTIME_ARTIFACTS ${_runtimes}
  LIBRARY DESTINATION "mlir/_mlir_libs"
  COMPONENT MLIRPythonModules
)
