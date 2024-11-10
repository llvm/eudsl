set(LLVM_ENABLE_PROJECTS "llvm;mlir" CACHE STRING "")

# All the tools will use libllvm shared library
set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "")
# When exceptions are disabled, unwind tables are large and useless
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")

# useful things
set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_UTILS ON CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_WARNINGS ON CACHE BOOL "")
set(LLVM_FORCE_ENABLE_STATS ON CACHE BOOL "")
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

# MLIR

set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "")
set(MLIR_ENABLE_EXECUTION_ENGINE ON CACHE BOOL "")
set(MLIR_ENABLE_SPIRV_CPU_RUNNER ON CACHE BOOL "")

# space savings

set(LLVM_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBCXX OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
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
set(LLVM_DISTRIBUTIONS
      Development
      MlirDevelopment
      Toolchain
    CACHE STRING "")

set(LLVM_TOOLCHAIN_TOOLS
  llvm-addr2line
  llvm-ar
  llvm-cxxfilt
  llvm-dis
  llvm-dwarfdump
  llvm-lib
  llvm-link
  llvm-mc
  llvm-nm
  llvm-objcopy
  llvm-objdump
  llvm-ranlib
  llvm-rc
  llvm-readelf
  llvm-readobj
  llvm-size
  llvm-strip
  llvm-symbolizer
  llvm-xray
  CACHE STRING "")

set(LLVM_BUILD_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_UTILITIES
    FileCheck
    count
    not
    CACHE STRING "")

set(LLVM_Toolchain_DISTRIBUTION_COMPONENTS
      LLVM
      LTO
      ${LLVM_TOOLCHAIN_TOOLS}
      ${LLVM_TOOLCHAIN_UTILITIES}
    CACHE STRING "")

set(LLVM_Development_DISTRIBUTION_COMPONENTS
      Remarks
      cmake-exports
      development-cmake-exports
      llc
      llvm-config
      llvm-headers
      llvm-libraries
      opt
      toolchain-cmake-exports
    CACHE STRING "")

set(LLVM_MLIR_TOOLS
      mlir-opt
      mlir-reduce
      mlir-tblgen
      mlir-translate
    CACHE STRING "")

set(LLVM_MLIR_Python_COMPONENTS
      MLIRPythonModules
      mlir-python-sources
    CACHE STRING "")

set(LLVM_MlirDevelopment_DISTRIBUTION_COMPONENTS
      MLIRPythonModules
      mlir-cmake-exports
      mlir-headers
      mlir-libraries
      ${LLVM_MLIR_TOOLS}
      ${LLVM_MLIR_Python_COMPONENTS}
    CACHE STRING "")
