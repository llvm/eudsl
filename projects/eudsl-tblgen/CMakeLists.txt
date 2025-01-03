# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_SUBPROJECT_TITLE "EUDSL_TBLGEN")
set(EUDSL_TBLGEN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  message("Building ${LLVM_SUBPROJECT_TITLE} as a standalone project.")
  project(${LLVM_SUBPROJECT_TITLE} CXX C)
  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  # TODO(max): probably don't need this anymore after landing the nanobind fix?
  # technically we need this on windows too but our LLVM is compiled without exception handling
  # and that breaks windows
  if(NOT WIN32)
    include(HandleLLVMOptions)
  endif()
endif()

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

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/eudsl_tblgen)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

nanobind_add_module(eudsl_tblgen_ext NB_STATIC STABLE_ABI
  src/eudsl_tblgen_ext.cpp
  src/TGParser.cpp
  src/TGLexer.cpp
)
target_link_libraries(eudsl_tblgen_ext PRIVATE LLVMTableGenCommon LLVMTableGen)
target_compile_options(eudsl_tblgen_ext
  PUBLIC
  -Wno-cast-qual
  $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Darwin>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>)

nanobind_add_stub(
  eudsl_tblgen_ext_stub
  MODULE eudsl_tblgen_ext
  OUTPUT ${CMAKE_CURRENT_LIST_DIR}/eudsl_tblgen/eudsl_tblgen_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:eudsl_tblgen_ext>
  DEPENDS eudsl_tblgen_ext
)
nanobind_add_stub(
  eudsl_tblgen_stub
  MODULE eudsl_tblgen
  OUTPUT ${CMAKE_CURRENT_LIST_DIR}/eudsl_tblgen/__init__.pyi
  PYTHON_PATH ${CMAKE_CURRENT_LIST_DIR}
  DEPENDS eudsl_tblgen_ext
)

install(TARGETS eudsl_tblgen_ext LIBRARY DESTINATION eudsl_tblgen)
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/eudsl_tblgen
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  PATTERN "*.so" EXCLUDE
  PATTERN "*.a" EXCLUDE
  PATTERN ".gitignore" EXCLUDE
)
