# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_SUBPROJECT_TITLE "EUDSL_TBLGEN")
set(EUDSL_TBLGEN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(EUDSL_TBLGEN_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  message("Building ${LLVM_SUBPROJECT_TITLE} as a standalone project.")
  project(${LLVM_SUBPROJECT_TITLE} CXX C)
  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
endif()

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(NOT TARGET LLVMSupport)
  message(FATAL_ERROR "LLVMSupport not found")
endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

set(EUDSL_TBLGEN_SRC_DIR "${EUDSL_TBLGEN_SRC_DIR}/src")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${EUDSL_TBLGEN_SRC_DIR}/eudsl_tblgen)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

nanobind_add_module(eudsl_tblgen_ext NB_STATIC
  src/eudsl_tblgen_ext.cpp
  src/TGParser.cpp
  src/TGLexer.cpp
)
set(eudsl_tblgen_ext_libs
  LLVMTableGenCommon LLVMTableGen LLVMCore MLIRTableGen)
if(APPLE)
  list(APPEND eudsl_tblgen_ext_libs ${LLVM_AVAILABLE_LIBS})
  list(REMOVE_ITEM eudsl_tblgen_ext_libs Remarks LTO LLVM)
  list(REMOVE_DUPLICATES eudsl_tblgen_ext_libs)
  target_link_directories(eudsl_tblgen_ext PRIVATE "${LLVM_LIBRARY_DIR}")
  list(TRANSFORM eudsl_tblgen_ext_libs PREPEND "-Wl,-hidden-l")
endif()
target_link_libraries(eudsl_tblgen_ext PRIVATE ${eudsl_tblgen_ext_libs})

set(nanobind_options
  -Wno-cast-qual
  -Wno-deprecated-literal-operator
  -Wno-covered-switch-default
  -Wno-nested-anon-types
  -Wno-zero-length-array
  -Wno-c++98-compat-extra-semi
  -Wno-c++20-extensions
  $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Darwin>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>)
target_compile_options(eudsl_tblgen_ext PRIVATE ${nanobind_options})
set(_nanobind_tgt)
if(TARGET nanobind-static)
  set(_nanobind_tgt nanobind-static)
elseif(TARGET nanobind-static-abi3)
  set(_nanobind_tgt nanobind-static-abi3)
endif()
target_compile_options(${_nanobind_tgt} PRIVATE ${nanobind_options})

# note WORKING_DIRECTORY
set(NB_STUBGEN_CMD "${Python_EXECUTABLE}" "-m" "nanobind.stubgen"
  --module eudsl_tblgen_ext --recursive --include-private --output-dir .)
set(NB_STUBGEN_OUTPUTS "eudsl_tblgen_ext.pyi")
add_custom_command(
  OUTPUT ${NB_STUBGEN_OUTPUTS}
  COMMAND ${NB_STUBGEN_CMD}
  WORKING_DIRECTORY "${EUDSL_TBLGEN_SRC_DIR}/eudsl_tblgen"
  DEPENDS eudsl_tblgen_ext
)
add_custom_target(eudsl_tblgen_ext_stub ALL DEPENDS ${NB_STUBGEN_OUTPUTS})

install(TARGETS eudsl_tblgen_ext LIBRARY DESTINATION eudsl_tblgen)
install(
  DIRECTORY "${EUDSL_TBLGEN_SRC_DIR}/eudsl_tblgen"
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.so" EXCLUDE
  PATTERN "*.a" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".gitignore" EXCLUDE
)
