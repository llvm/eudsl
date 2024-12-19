set(LLVM_ENABLE_PROJECTS "llvm;mlir" CACHE STRING "")

# LLVM options

set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_UTILS ON CACHE BOOL "")
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

# All the tools will use libllvm shared library
# (but doesn't work on windows)
if (NOT WIN32)
  set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
  set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "")
endif()

# useful things
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_WARNINGS ON CACHE BOOL "")
set(LLVM_FORCE_ENABLE_STATS ON CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")

# MLIR options

set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")
set(MLIR_ENABLE_EXECUTION_ENGINE ON CACHE BOOL "")
set(MLIR_ENABLE_SPIRV_CPU_RUNNER ON CACHE BOOL "")

# space savings

set(LLVM_BUILD_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
# When exceptions are disabled, unwind tables are large and useless
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")
set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_GO_TESTS OFF CACHE BOOL "")

# tests

option(RUN_TESTS "" OFF)
set(LLVM_INCLUDE_TESTS ${RUN_TESTS} CACHE BOOL "")
set(LLVM_BUILD_TESTS ${RUN_TESTS} CACHE BOOL "")
set(MLIR_INCLUDE_INTEGRATION_TESTS ${RUN_TESTS} CACHE BOOL "")
set(MLIR_INCLUDE_TESTS ${RUN_TESTS} CACHE BOOL "")

### Distributions ###

set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")

set(LLVM_DISTRIBUTIONS MlirDevelopment CACHE STRING "")
set(LLVM_MlirDevelopment_DISTRIBUTION_COMPONENTS
    clangAPINotes
    clangAST
    clangASTMatchers
    clangAnalysis
    clangBasic
    clangDriver
    clangDriver
    clangEdit
    clangFormat
    clangFrontend
    clangLex
    clangParse
    clangRewrite
    clangSema
    clangSerialization
    clangSupport
    clangTooling
    clangToolingCore
    clangToolingInclusions

    llvm-config
    llvm-headers
    llvm-libraries

    cmake-exports
    FileCheck
    not
    MLIRPythonModules
    # triggers LLVMMlirDevelopmentExports.cmake
    mlirdevelopment-cmake-exports
    # triggers MLIRMlirDevelopmentTargets.cmake
    mlir-mlirdevelopment-cmake-exports
    # triggers MLIRConfig.cmake and etc
    mlir-cmake-exports
    mlir-headers
    mlir-libraries
    mlir-opt
    mlir-python-sources
    mlir-reduce
    mlir-tblgen
    mlir-translate
    CACHE STRING "")

if (NOT WIN32)
  list(APPEND LLVM_MlirDevelopment_DISTRIBUTION_COMPONENTS LLVM)
endif()
