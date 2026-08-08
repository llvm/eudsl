# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_SUBPROJECT_TITLE "EUDSLLLVM")
set(EUDSLLLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message("Building ${LLVM_SUBPROJECT_TITLE} as a standalone project.")
  project(${LLVM_SUBPROJECT_TITLE} CXX C)
  set(EUDSLLLVM_STANDALONE_BUILD ON)
else()
  enable_language(CXX C)
  set(EUDSLLLVM_STANDALONE_BUILD OFF)
endif()

find_package(Python 3.9...<3.15 REQUIRED
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

if(EUDSLLLVM_STANDALONE_BUILD)
  find_package(LLVM REQUIRED CONFIG)

  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# C++ source coverage (llvm-cov). When ON, the extension is built with
# instrumentation; scripts/cpp_coverage.sh runs the test suite (which imports
# the .so) under LLVM_PROFILE_FILE, merges the profraw, and enforces a
# src/IR line-coverage threshold via scripts/check_coverage.py.
option(EUDSL_LLVMPY_ENABLE_COVERAGE "Instrument eudslllvm_ext for llvm-cov" OFF)
if(EUDSL_LLVMPY_ENABLE_COVERAGE)
  message(STATUS "EUDSL_LLVMPY: C++ coverage instrumentation enabled")
  set(EUDSLLLVM_COVERAGE_FLAGS -fprofile-instr-generate -fcoverage-mapping)
endif()

# Which LLVM targets to link and initialize. Defaults to the host target the
# linked LLVM was built for (LLVM_NATIVE_ARCH), so it works against a host-only
# distro on any arch. Extra targets stay reachable by flag rather than by patch,
# e.g. -DEUDSL_LLVMPY_TARGETS="X86;AArch64;AMDGPU" (each must exist in the LLVM
# being built against).
set(EUDSL_LLVMPY_TARGETS "${LLVM_NATIVE_ARCH}" CACHE STRING
    "LLVM targets to link into eudslllvm_ext (semicolon-separated)")
message(STATUS "EUDSL_LLVMPY_TARGETS: ${EUDSL_LLVMPY_TARGETS}")

set(EUDSLLLVM_TARGET_LIBS)
set(EUDSLLLVM_TARGET_INITS)
foreach(_tgt IN LISTS EUDSL_LLVMPY_TARGETS)
  if(NOT TARGET LLVM${_tgt}Info)
    message(FATAL_ERROR
        "EUDSL_LLVMPY_TARGETS names ${_tgt} but LLVM${_tgt}Info does not exist; "
        "the LLVM being built against was configured without that target")
  endif()
  list(APPEND EUDSLLLVM_TARGET_LIBS
      LLVM${_tgt}Info LLVM${_tgt}Desc LLVM${_tgt}CodeGen
      LLVM${_tgt}AsmParser LLVM${_tgt}Disassembler)
  string(APPEND EUDSLLLVM_TARGET_INITS
      "  LLVMInitialize${_tgt}Target();\n"
      "  LLVMInitialize${_tgt}TargetInfo();\n"
      "  LLVMInitialize${_tgt}TargetMC();\n"
      "  LLVMInitialize${_tgt}AsmParser();\n"
      "  LLVMInitialize${_tgt}AsmPrinter();\n")
endforeach()
configure_file("${CMAKE_CURRENT_LIST_DIR}/src/IR/TargetInit.h.in"
               "${EUDSLLLVM_BINARY_DIR}/IR/TargetInit.h" @ONLY)
include_directories("${EUDSLLLVM_BINARY_DIR}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

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
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>
)

set(EUDSLLLVM_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
include_directories(${EUDSLLLVM_SRC_DIR})

# FREE_THREADED marks the module Py_MOD_GIL_NOT_USED so importing under a
# free-threaded interpreter (cp3XXt) doesn't re-enable the GIL. Note: actual
# thread-safety of these bindings has not been verified; this only asserts it.
nanobind_add_module(eudslllvm_ext
  NB_STATIC STABLE_ABI FREE_THREADED
  NB_DOMAIN eudslllvm
  src/eudslllvm_ext.cpp
  src/IR/Ownership.cpp
  src/IR/Context.cpp
  src/IR/Casters.cpp
  src/IR/Types.cpp
  src/IR/Kinds.cpp
  src/IR/Values.cpp
  src/IR/Sequence.cpp
  src/IR/Instructions.cpp
  src/IR/Constants.cpp
  src/IR/Builder.cpp
  src/IR/Attributes.cpp
  src/IR/Metadata.cpp
  src/IR/Errors.cpp
  src/IR/Passes.cpp
  src/IR/Target.cpp
  src/IR/Linker.cpp
  src/IR/JIT.cpp
  src/IR/Intrinsics.cpp
)

set(eudslllvm_ext_libs
  LLVMCore
  LLVMAsmParser
  LLVMBitReader
  LLVMBitWriter
  LLVMPasses
  LLVMAnalysis
  LLVMTransformUtils
  LLVMScalarOpts
  LLVMInstCombine
  LLVMAsmPrinter
  LLVMLinker
  LLVMOrcJIT
  LLVMExecutionEngine
  LLVMJITLink
  LLVMOrcTargetProcess
  LLVMOrcShared
  LLVMRuntimeDyld
  LLVMSupport
  LLVMTarget
  LLVMMC
  LLVMMCParser
  LLVMCodeGen
  LLVMTargetParser
  ${EUDSLLLVM_TARGET_LIBS}
)
target_link_libraries(eudslllvm_ext PRIVATE ${eudslllvm_ext_libs})

set_target_properties(eudslllvm_ext
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${EUDSLLLVM_SRC_DIR}/llvm"
)

target_compile_options(eudslllvm_ext PRIVATE ${nanobind_options})
if(EUDSL_LLVMPY_ENABLE_COVERAGE)
  target_compile_options(eudslllvm_ext PRIVATE ${EUDSLLLVM_COVERAGE_FLAGS})
  target_link_options(eudslllvm_ext PRIVATE ${EUDSLLLVM_COVERAGE_FLAGS})
endif()
# The nanobind helper library name varies with STABLE_ABI (-abi3) and
# FREE_THREADED (-ft), so probe every variant instead of hardcoding two.
set(_nanobind_tgt)
foreach(_cand nanobind-static nanobind-static-abi3 nanobind-static-ft nanobind-static-abi3-ft)
  if(TARGET ${_cand})
    set(_nanobind_tgt ${_cand})
    break()
  endif()
endforeach()
if(_nanobind_tgt)
  target_compile_options(${_nanobind_tgt} PRIVATE ${nanobind_options})
else()
  message(FATAL_ERROR "Could not find a nanobind static helper target")
endif()

# note WORKING_DIRECTORY
set(NB_STUBGEN_CMD "${Python_EXECUTABLE}" "-m" "nanobind.stubgen"
  --module eudslllvm_ext --recursive --include-private --output-dir .)
set(NB_STUBGEN_OUTPUTS "__init__.pyi")
add_custom_command(
  OUTPUT ${NB_STUBGEN_OUTPUTS}
  COMMAND ${NB_STUBGEN_CMD}
  WORKING_DIRECTORY "${EUDSLLLVM_SRC_DIR}/llvm"
  DEPENDS eudslllvm_ext
)
add_custom_target(eudslllvm_ext_stub ALL DEPENDS ${NB_STUBGEN_OUTPUTS})

install(TARGETS eudslllvm_ext LIBRARY DESTINATION llvm)
install(
  DIRECTORY "${EUDSLLLVM_SRC_DIR}/llvm"
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.so" EXCLUDE
  PATTERN "*.a" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".gitignore" EXCLUDE
)
