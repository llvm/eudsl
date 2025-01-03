# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_SUBPROJECT_TITLE "EUDSL")
set(EUDSL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message("Building ${LLVM_SUBPROJECT_TITLE} as a standalone project.")
  project(${LLVM_SUBPROJECT_TITLE} CXX C)
  set(EUDSL_STANDALONE_BUILD ON)
else()
  enable_language(CXX C)
  set(EUDSL_STANDALONE_BUILD OFF)
endif()

if(EUDSL_STANDALONE_BUILD)
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG PATHS "${LLVM_BINARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(AddClang)

  set(MLIR_INCLUDE_DIR ${MLIR_INCLUDE_DIRS})
else()
  # turning LLVM -DLLVM_OPTIMIZED_TABLEGEN=ON builds some stuff in the NATIVE dir
  # but not everything so LLVM_BINARY_DIR isn't correct
  string(REPLACE "NATIVE" "" LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
  # Build via external projects mechanism
  set(LLVM_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include)
  set(LLVM_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/include)
  set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIR};${LLVM_GENERATED_INCLUDE_DIR}")

  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

  set(CLANG_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../clang)
  set(CLANG_INCLUDE_DIR ${CLANG_MAIN_SRC_DIR}/include)
  set(CLANG_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/clang/include)
  set(CLANG_INCLUDE_DIRS "${CLANG_INCLUDE_DIR};${CLANG_GENERATED_INCLUDE_DIR}")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(NOT TARGET LLVMSupport)
  message(FATAL_ERROR "LLVMSupport not found")
endif()

add_subdirectory(projects)