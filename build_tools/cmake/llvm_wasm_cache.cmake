set(LLVM_ENABLE_PROJECTS "mlir;llvm" CACHE STRING "")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_INCLUDE_TOOLS ON CACHE BOOL "")
set(LLVM_TARGETS_TO_BUILD "WebAssembly" CACHE STRING "")
set(LLVM_TARGET_ARCH "wasm32-wasi" CACHE STRING "")
set(LLVM_DEFAULT_TARGET_TRIPLE "wasm32-wasi" CACHE STRING "")

set(LLVM_BUILD_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")
set(LLVM_ENABLE_CRASH_OVERRIDES OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBPFM OFF CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")

set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "")
set(MLIR_BUILD_MLIR_C_DYLIB OFF CACHE BOOL "")
# when building libLLVM
# relocation R_WASM_MEMORY_ADDR_SLEB cannot be used against symbol
set(LLVM_ENABLE_PIC OFF CACHE BOOL "")
set(MLIR_ENABLE_SPIRV_CPU_RUNNER OFF)
set(MLIR_ENABLE_EXECUTION_ENGINE OFF)

set(LLVM_ENABLE_THREADS OFF CACHE BOOL "")
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_UTILS OFF CACHE BOOL "")

### Distributions ###

set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")

set(LLVM_DISTRIBUTIONS MlirDevelopment CACHE STRING "")
set(LLVM_MlirDevelopment_DISTRIBUTION_COMPONENTS
  llvm-config
  llvm-headers
  llvm-libraries
  cmake-exports
  # opt
  # triggers LLVMMlirDevelopmentExports.cmake
  mlirdevelopment-cmake-exports
  # triggers MLIRMlirDevelopmentTargets.cmake
  mlir-mlirdevelopment-cmake-exports
  # triggers MLIRConfig.cmake and etc
  mlir-cmake-exports
  mlir-headers
  mlir-libraries
  mlir-opt
  # mlir-reduce
  # mlir-tblgen
  # mlir-translate
  CACHE STRING "")