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

find_package(Clang REQUIRED CONFIG)
message(STATUS "orc-pgo: using Clang from ${Clang_DIR}")

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
# OrcPgoRuntime links these, and so does every executable orc-pgo builds.
set(ORC_PGO_LLVM_COMPONENTS
  core support irreader asmparser bitreader bitwriter linker passes
  transformutils instrumentation ipo analysis profiledata target mc object
  orcjit jitlink lto native)
llvm_map_components_to_libnames(ORC_PGO_LLVM_LIBS ${ORC_PGO_LLVM_COMPONENTS})

add_library(OrcPgoCommon STATIC
  lib/Analyses.cpp
  lib/Eligibility.cpp
  lib/PgoOptions.cpp
  lib/Snapshot.cpp
)
target_include_directories(OrcPgoCommon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(OrcPgoCommon PUBLIC ${ORC_PGO_LLVM_LIBS})
orc_pgo_add_coverage(OrcPgoCommon)

add_library(OrcPgoRuntime STATIC
  lib/ValueProfRuntime.cpp
  lib/ProfileBuilder.cpp
  lib/Annotate.cpp
  lib/TierUp.cpp
  lib/Runtime.cpp
  lib/RuntimeEntry.cpp
  lib/Stats.cpp
)
target_link_libraries(OrcPgoRuntime PUBLIC OrcPgoCommon)
orc_pgo_add_coverage(OrcPgoRuntime)

add_library(OrcPgoBuild STATIC
  lib/Instrument.cpp
  lib/Thunks.cpp
  lib/SymbolTable.cpp
  lib/Pipeline.cpp
)
target_link_libraries(OrcPgoBuild PUBLIC OrcPgoCommon)
orc_pgo_add_coverage(OrcPgoBuild)

# ---- orc-pgo driver ----

# clang's resource directory in the same distribution: builtin headers and the
# compiler-rt builtins that orc-pgo executables link.
set(ORC_PGO_CLANG_RESOURCE_DIR "${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
if(NOT EXISTS "${ORC_PGO_CLANG_RESOURCE_DIR}/include/stddef.h")
  message(FATAL_ERROR "orc-pgo: no clang resource headers in ${ORC_PGO_CLANG_RESOURCE_DIR}")
endif()

# clang finds the macOS SDK only through -isysroot or SDKROOT.
set(ORC_PGO_DEFAULT_SYSROOT "")
if(APPLE)
  execute_process(COMMAND xcrun --show-sdk-path
                  OUTPUT_VARIABLE ORC_PGO_DEFAULT_SYSROOT
                  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()

# OrcPgoRuntime's static LLVM dependencies in link order, plus the system
# libraries they need, for the executables the driver links.
find_program(ORC_PGO_LLVM_CONFIG llvm-config PATHS "${LLVM_TOOLS_BINARY_DIR}"
             NO_DEFAULT_PATH REQUIRED)
execute_process(COMMAND "${ORC_PGO_LLVM_CONFIG}" --link-static --libfiles ${ORC_PGO_LLVM_COMPONENTS}
                OUTPUT_VARIABLE _llvm_libfiles OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${ORC_PGO_LLVM_CONFIG}" --link-static --system-libs ${ORC_PGO_LLVM_COMPONENTS}
                OUTPUT_VARIABLE _llvm_syslibs OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
separate_arguments(_llvm_link UNIX_COMMAND "${_llvm_libfiles} ${_llvm_syslibs}")
list(TRANSFORM _llvm_link REPLACE "^(.+)$" "\"\\1\"")
list(JOIN _llvm_link ", " ORC_PGO_LLVM_LINK_ARGS)

# In coverage builds the runtime archives are compiled with the coverage
# compiler's -fprofile-instr-generate. So the apps orc-pgo links need that
# compiler's profile runtime; outside coverage builds they never link one.
# Darwin: lib/darwin/libclang_rt.profile_osx.a. Linux:
# lib/<triple>/libclang_rt.profile.a or lib/linux/libclang_rt.profile-<arch>.a.
set(ORC_PGO_EXTRA_LINK_INPUT "")
if(ORC_PGO_ENABLE_COVERAGE)
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
                  OUTPUT_VARIABLE _cov_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                  COMMAND_ERROR_IS_FATAL ANY)
  file(GLOB_RECURSE _cov_profile "${_cov_resource_dir}/lib/*libclang_rt.profile*.a")
  if(APPLE)
    list(FILTER _cov_profile INCLUDE REGEX "libclang_rt\\.profile_osx\\.a$")
  else()
    list(FILTER _cov_profile INCLUDE REGEX "${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  list(LENGTH _cov_profile _n)
  if(NOT _n EQUAL 1)
    message(FATAL_ERROR "orc-pgo: expected one profile runtime under "
                        "${_cov_resource_dir}/lib, found: ${_cov_profile}")
  endif()
  set(ORC_PGO_EXTRA_LINK_INPUT "${_cov_profile}")
endif()

configure_file(cmake/Config.h.in "${CMAKE_CURRENT_BINARY_DIR}/Config.h.in" @ONLY)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/orc-pgo/Config.h"
     INPUT "${CMAKE_CURRENT_BINARY_DIR}/Config.h.in")

# Their imported targets carry the remaining clang and LLVM dependencies.
set(ORC_PGO_CLANG_LIBS
  clangFrontendTool clangCodeGen clangDriver clangFrontend clangSerialization
  clangOptions clangBasic)

add_library(OrcPgoDriver STATIC
  lib/Archive.cpp
  lib/Driver.cpp
)
target_include_directories(OrcPgoDriver
  PUBLIC ${CLANG_INCLUDE_DIRS}
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
target_link_libraries(OrcPgoDriver PUBLIC OrcPgoBuild ${ORC_PGO_CLANG_LIBS})
orc_pgo_add_coverage(OrcPgoDriver)

add_executable(orc-pgo tools/orc-pgo/orc-pgo.cpp)
target_link_libraries(orc-pgo PRIVATE OrcPgoDriver)
# Executables built by orc-pgo link the runtime archives by path (Config.h).
add_dependencies(orc-pgo OrcPgoRuntime OrcPgoCommon)
orc_pgo_add_coverage(orc-pgo)
orc_pgo_coverage_object(orc-pgo)

set(ORC_PGO_DRIVER "$<TARGET_FILE:orc-pgo>")
configure_file(cmake/OrcPgoToolchain.cmake.in
               "${CMAKE_CURRENT_BINARY_DIR}/OrcPgoToolchain.cmake.in" @ONLY)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/OrcPgoToolchain.cmake"
     INPUT "${CMAKE_CURRENT_BINARY_DIR}/OrcPgoToolchain.cmake.in")

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # The tests that run the driver (DriverTest, multi-tu-e2e) get their own
  # TMPDIR, emptied before each ctest run.
  set(ORC_PGO_TEST_TMPDIR "${CMAKE_CURRENT_BINARY_DIR}/tmp")
  add_test(NAME orc-pgo-clean-tmp
           COMMAND ${CMAKE_COMMAND} -DDIR=${ORC_PGO_TEST_TMPDIR}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CleanTemp.cmake)
  set_tests_properties(orc-pgo-clean-tmp PROPERTIES FIXTURES_SETUP orc_pgo_tmp)

  add_subdirectory(unittests)

  add_test(NAME multi-tu-e2e
           COMMAND ${CMAKE_COMMAND}
             -DORC_PGO=$<TARGET_FILE:orc-pgo>
             -DTOOLCHAIN=${CMAKE_CURRENT_BINARY_DIR}/OrcPgoToolchain.cmake
             "-DEXTRA_LINK_INPUT=${ORC_PGO_EXTRA_LINK_INPUT}"
             -DORC_PGO_INCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/include
             -DSRC=${CMAKE_CURRENT_SOURCE_DIR}/test/multi-tu
             -DBIN=${CMAKE_CURRENT_BINARY_DIR}/multi-tu
             -P ${CMAKE_CURRENT_SOURCE_DIR}/test/multi-tu/run.cmake)
  set_tests_properties(multi-tu-e2e PROPERTIES
                       TIMEOUT 900
                       ENVIRONMENT "TMPDIR=${ORC_PGO_TEST_TMPDIR}"
                       FIXTURES_REQUIRED orc_pgo_tmp)
endif()

get_property(_cov_targets GLOBAL PROPERTY ORC_PGO_COVERAGE_OBJECTS)
set(_cov_content "")
foreach(_t IN LISTS _cov_targets)
  string(APPEND _cov_content "$<TARGET_FILE:${_t}>\n")
endforeach()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/coverage-objects.txt" CONTENT "${_cov_content}")
