# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.29)
project(orc-pgo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "orc-pgo: using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(HandleLLVMOptions)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(_llvm_defs NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_compile_definitions(${_llvm_defs})

option(ORC_PGO_ENABLE_COVERAGE "Build with source-based coverage" OFF)

# Executables passed to llvm-cov as --objects (via coverage-objects.txt).
# Libraries are instrumented with orc_pgo_add_coverage and measured through
# the executables that link them.
set_property(GLOBAL PROPERTY ORC_PGO_COVERAGE_OBJECTS "")

function(orc_pgo_add_coverage target)
  if(ORC_PGO_ENABLE_COVERAGE)
    target_compile_options(${target} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(${target} PRIVATE -fprofile-instr-generate)
  endif()
endfunction()

function(orc_pgo_coverage_object target)
  set_property(GLOBAL APPEND PROPERTY ORC_PGO_COVERAGE_OBJECTS ${target})
endfunction()

# Static LLVM only (never libLLVM.dylib): one copy of LLVM per process.
# Grow this list as PRs need more components.
llvm_map_components_to_libnames(ORC_PGO_LLVM_LIBS
  core support irreader asmparser bitreader bitwriter linker passes
  transformutils instrumentation ipo analysis profiledata target mc object
  orcjit jitlink lto native)

add_library(OrcPgo STATIC
  lib/CounterLayout.cpp
)
target_include_directories(OrcPgo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(OrcPgo PUBLIC ${ORC_PGO_LLVM_LIBS})
orc_pgo_add_coverage(OrcPgo)

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(unittests)
endif()

get_property(_cov_targets GLOBAL PROPERTY ORC_PGO_COVERAGE_OBJECTS)
set(_cov_content "")
foreach(_t IN LISTS _cov_targets)
  string(APPEND _cov_content "$<TARGET_FILE:${_t}>\n")
endforeach()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/coverage-objects.txt" CONTENT "${_cov_content}")
