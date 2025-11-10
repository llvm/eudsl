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

if(NOT TARGET LLVMSupport)
  message(FATAL_ERROR "LLVMSupport not found")
endif()

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
include_directories(${EUDSLLLVM_BINARY_DIR})
include_directories(${EUDSLLLVM_SRC_DIR})

if (EUDSLLLVM_STANDALONE_BUILD)
  execute_process(
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/eudsl-llvmpy-generate.py"
                                        ${LLVM_INCLUDE_DIRS}
                                        "${EUDSLLLVM_BINARY_DIR}/generated"
                                        "${EUDSLLLVM_SRC_DIR}/llvm"
        RESULT_VARIABLE _has_err_generate
        COMMAND_ECHO STDOUT
  )
  if (_has_err_generate AND NOT _has_err_generate EQUAL 0)
    message(FATAL_ERROR "couldn't generate sources: ${_has_err_generate}")
  endif()
  include_directories("${EUDSLLLVM_BINARY_DIR}/generated")
  file(GLOB _gen_src CONFIGURE_DEPENDS "${EUDSLLLVM_BINARY_DIR}/generated/*.cpp")
else()
  set(_gen_src
      Analysis.cpp
      BitReader.cpp
      BitWriter.cpp
      Comdat.cpp
      Core.cpp
      DataTypes.cpp
      DebugInfo.cpp
      Deprecated.cpp
      Disassembler.cpp
      DisassemblerTypes.cpp
      Error.cpp
      ErrorHandling.cpp
      ExecutionEngine.cpp
      ExternC.cpp
      IRReader.cpp
      LLJIT.cpp
      LLJITUtils.cpp
      Linker.cpp
      Object.cpp
      Orc.cpp
      OrcEE.cpp
      PassBuilder.cpp
      Remarks.cpp
      Support.cpp
      Target.cpp
      TargetMachine.cpp
      Types.cpp
      Visibility.cpp
      blake3.cpp
  )
  list(TRANSFORM _gen_src PREPEND "${EUDSLLLVM_BINARY_DIR}/generated/")
  #If we're building against third_party/llvm-project then build/include will also be in this list...
  set(_llvm_include_dirs ${LLVM_INCLUDE_DIRS})
  list(LENGTH _llvm_include_dirs _llvm_include_dirs_len)
  if (_llvm_include_dirs_len GREATER_EQUAL 2)
    list(GET _llvm_include_dirs 0 _llvm_include_dirs)
    set(_llvm_include_dir "llvm/include")
    cmake_path(SET _llvm_include_dir NORMALIZE "${_llvm_include_dir}")
    if(NOT "${_llvm_include_dirs}" MATCHES "${_llvm_include_dir}$")
      message(FATAL_ERROR "Expected 0th entry in ${LLVM_INCLUDE_DIRS} to be ${_llvm_include_dir}")
    endif()
  endif ()
  cmake_path(SET _eudsl_tblgen_path NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../eudsl-tblgen/src")
  add_custom_command(
    OUTPUT ${_gen_src}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${_eudsl_tblgen_path}:$ENV{PYTHONPATH}
                                    "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/eudsl-llvmpy-generate.py"
                                    ${_llvm_include_dirs}
                                    "${EUDSLLLVM_BINARY_DIR}/generated"
                                    "${EUDSLLLVM_SRC_DIR}/llvm"
    DEPENDS eudsl_tblgen_ext
  )
endif ()


include_directories("${EUDSLLLVM_BINARY_DIR}/generated")

nanobind_add_module(eudslllvm_ext
  NB_STATIC STABLE_ABI
  NB_DOMAIN eudslllvm
  src/eudslllvm_ext.cpp
  ${_gen_src}
)
set(eudslllvm_ext_libs
  LLVMCore
  LLVMExecutionEngine
  LLVMOrcJIT
  LLVMOrcDebugging
  LLVMInterpreter
  LLVMMCDisassembler
  LLVMMCJIT
  # AMDGPU
  LLVMAMDGPUCodeGen
  LLVMAMDGPUAsmParser
  LLVMAMDGPUDisassembler
  LLVMAMDGPUTargetMCA
  LLVMAMDGPUDesc
  LLVMAMDGPUInfo
  LLVMAMDGPUUtils
  # NVPTX
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  list(APPEND eudslllvm_ext_libs
    LLVMX86Info
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Disassembler)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  list(APPEND eudslllvm_ext_libs
    LLVMAArch64Info
    LLVMAArch64Desc
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Disassembler)
endif()

if(APPLE)
  list(APPEND eudslllvm_ext_libs ${LLVM_AVAILABLE_LIBS})
  list(REMOVE_ITEM eudslllvm_ext_libs Remarks LTO LLVM LLVMTableGen)
  list(REMOVE_DUPLICATES eudslllvm_ext_libs)
  target_link_directories(eudslllvm_ext PRIVATE "${LLVM_LIBRARY_DIR}")
  list(TRANSFORM eudslllvm_ext_libs PREPEND "-Wl,-hidden-l")
endif()
target_link_libraries(eudslllvm_ext PRIVATE ${eudslllvm_ext_libs})

set_target_properties(eudslllvm_ext
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${EUDSLLLVM_SRC_DIR}/llvm"
)

target_compile_options(eudslllvm_ext PRIVATE ${nanobind_options})
set(_nanobind_tgt)
if(TARGET nanobind-static)
  set(_nanobind_tgt nanobind-static)
elseif(TARGET nanobind-static-abi3)
  set(_nanobind_tgt nanobind-static-abi3)
endif()
target_compile_options(${_nanobind_tgt} PRIVATE ${nanobind_options})

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
