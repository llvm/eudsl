# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_SUBPROJECT_TITLE "EUDSLPY")
set(EUDSLPY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message("Building ${LLVM_SUBPROJECT_TITLE} as a standalone project.")
  project(${LLVM_SUBPROJECT_TITLE} CXX C)
  set(EUDSLPY_STANDALONE_BUILD ON)
else()
  enable_language(CXX C)
  set(EUDSLPY_STANDALONE_BUILD OFF)
endif()

find_package(Python 3.9...<3.14 REQUIRED
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

if(EUDSLPY_STANDALONE_BUILD)
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
  # TODO(max): probably don't need this anymore after landing the nanobind fix?
  # technically we need this on windows too but our LLVM is compiled without exception handling
  # and that breaks windows
  if(NOT WIN32)
    include(HandleLLVMOptions)
  endif()
  # for out-of-tree MLIR_INCLUDE_DIR points to the build dir by default
  # and MLIR_INCLUDE_DIRS points to the correct place
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

set(EUDSLPY_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
include_directories(${EUDSLPY_BINARY_DIR})
include_directories(${EUDSLPY_SRC_DIR})
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m eudsl_nbgen --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE eudsl_nbgen_DIR)
find_package(eudsl_nbgen CONFIG REQUIRED)

# too big
# eudslpygen(EUDSLGen_acc
#   ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#   NAMESPACES mlir::acc mlir::acc::detail
# )

eudslpygen(EUDSLGen_affine
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::affine mlir::affine::detail
  EXTRA_INCLUDES
    mlir/IR/IntegerSet.h
    mlir/Dialect/Affine/IR/AffineValueMap.h
)

eudslpygen(EUDSLGen_amdgpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::amdgpu mlir::amdgpu::detail
)

eudslpygen(EUDSLGen_amx
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::amx mlir::amx::detail
)

eudslpygen(EUDSLGen_arith
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::arith mlir::arith::detail
)

eudslpygen(EUDSLGen_arm_neon
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::arm_neon mlir::arm_neon::detail
)

# too big
# eudslpygen(EUDSLGen_arm_sme
#   ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#   NAMESPACES mlir::arm_sme mlir::arm_sme::detail
# )

eudslpygen(EUDSLGen_arm_sve
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::arm_sve mlir::arm_sve::detail
)

eudslpygen(EUDSLGen_async
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::async mlir::async::detail
)

eudslpygen(EUDSLGen_bufferization
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::bufferization mlir::bufferization::detail
  EXTRA_INCLUDES
    mlir/Dialect/Bufferization/Transforms/Bufferize.h
)

eudslpygen(EUDSLGen_cf
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR/ControlFlowOps.h
  NAMESPACES mlir::cf mlir::cf::detail
  EXTRA_INCLUDES
    mlir/Dialect/ControlFlow/IR/ControlFlowOps.h
    mlir/IR/PatternMatch.h
)

eudslpygen(EUDSLGen_complex
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::complex mlir::complex::detail
)

eudslpygen(EUDSLGen_DLTIDialect
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::DLTIDialect mlir::DLTIDialect::detail
)

eudslpygen(EUDSLGen_emitc
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::emitc mlir::emitc::detail
)

eudslpygen(EUDSLGen_func
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::func mlir::func::detail
)

eudslpygen(EUDSLGen_gpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::gpu mlir::gpu::detail
  EXTRA_INCLUDES
    mlir/Target/LLVMIR/ModuleTranslation.h
    llvm/IR/IRBuilder.h
)

eudslpygen(EUDSLGen_index
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Index/IR/IndexOps.h
  NAMESPACES mlir::index mlir::index::detail
  EXTRA_INCLUDES
    mlir/Dialect/Index/IR/IndexOps.h
    mlir/IR/PatternMatch.h
)

#eudslpygen(EUDSLGen_irdl
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  NAMESPACES mlir::irdl mlir::irdl::detail
#)

eudslpygen(EUDSLGen_linalg
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::linalg mlir::linalg::detail
)

eudslpygen(EUDSLGen_LLVM
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::LLVM mlir::LLVM::detail
)

eudslpygen(EUDSLGen_math
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::math mlir::math::detail
)

eudslpygen(EUDSLGen_memref
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::memref mlir::memref::detail
)

eudslpygen(EUDSLGen_mesh
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Mesh/IR/MeshOps.h
  NAMESPACES mlir::mesh mlir::mesh::detail
  EXTRA_INCLUDES
    mlir/Dialect/Mesh/IR/MeshOps.h
)

eudslpygen(EUDSLGen_ml_program
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::ml_program mlir::ml_program::detail
)

eudslpygen(EUDSLGen_mpi
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::mpi mlir::mpi::detail
)

eudslpygen(EUDSLGen_nvgpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::nvgpu mlir::nvgpu::detail
)

eudslpygen(EUDSLGen_NVVM
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::NVVM mlir::NVVM::detail
)

#eudslpygen(EUDSLGen_omp
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  NAMESPACES mlir::omp mlir::omp::detail
#)

eudslpygen(EUDSLGen_pdl
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/PDL/IR/PDLOps.h
  NAMESPACES mlir::pdl mlir::pdl::detail
  EXTRA_INCLUDES
    mlir/Dialect/PDL/IR/PDLOps.h
)

eudslpygen(EUDSLGen_pdl_interp
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::pdl_interp mlir::pdl_interp::detail
)

eudslpygen(EUDSLGen_polynomial
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Polynomial/IR/PolynomialOps.h
  NAMESPACES mlir::polynomial mlir::polynomial::detail
  EXTRA_INCLUDES
    mlir/Dialect/Polynomial/IR/PolynomialOps.h
    mlir/IR/PatternMatch.h
)

eudslpygen(EUDSLGen_ptr
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Ptr/IR/PtrOps.h
  NAMESPACES mlir::ptr mlir::ptr::detail
  EXTRA_INCLUDES
    mlir/Dialect/Ptr/IR/PtrOps.h
    mlir/IR/DialectImplementation.h
)

eudslpygen(EUDSLGen_quant
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::quant mlir::quant::detail
  EXTRA_INCLUDES
    mlir/Dialect/Quant/IR/QuantTypes.h
)

eudslpygen(EUDSLGen_ROCDL
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::ROCDL mlir::ROCDL::detail
)

eudslpygen(EUDSLGen_scf
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::scf mlir::scf::detail
)

eudslpygen(EUDSLGen_shape
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::shape mlir::shape::detail
)

eudslpygen(EUDSLGen_sparse_tensor
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::sparse_tensor mlir::sparse_tensor::detail
)

# nb::detail::nb_func_new("get_vce_triple_attr_name"): mismatched static/instance method flags in function overloads!
# eudslpygen(EUDSLGen_spirv
#   ${MLIR_INCLUDE_DIR}/mlir/Dialect/SPIRV/IR/SPIRVOps.h
#   NAMESPACES mlir::spirv mlir::spirv::detail
#   EXTRA_INCLUDES
#     mlir/Dialect/SPIRV/IR/SPIRVOps.h
#     mlir/IR/PatternMatch.h
# )

eudslpygen(EUDSLGen_tensor
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::tensor mlir::tensor::detail
)

eudslpygen(EUDSLGen_tosa
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::tosa mlir::tosa::detail
)

eudslpygen(EUDSLGen_transform
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::transform mlir::transform::detail
)

eudslpygen(EUDSLGen_ub
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::ub mlir::ub::detail
)

#eudslpygen(EUDSLGen_vector
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  NAMESPACES mlir::vector mlir::vector::detail
#)

eudslpygen(EUDSLGen_x86vector
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::x86vector mlir::x86vector::detail
)

eudslpygen(EUDSLGen_xegpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::xegpu mlir::xegpu::detail
)

nanobind_add_module(eudslpy_ext
  NB_STATIC
  NB_DOMAIN eudsl
  src/eudslpy_ext.cpp
)
target_link_libraries(eudslpy_ext PRIVATE
  # broken
  # EUDSLGen_irdl
  # EUDSLGen_omp
  # EUDSLGen_vector
  # weight reduction
  # EUDSLGen_acc
  # EUDSLGen_amx
  # EUDSLGen_arm_neon
  # EUDSLGen_arm_sme
  # EUDSLGen_arm_sve
  # EUDSLGen_DLTIDialect
  # EUDSLGen_mesh
  # EUDSLGen_ml_program
  # EUDSLGen_mpi
  # EUDSLGen_ptr
  # EUDSLGen_quant
  # EUDSLGen_sparse_tensor
  # EUDSLGen_spirv
  # EUDSLGen_transform
  # EUDSLGen_ub
  # EUDSLGen_x86vector
  # EUDSLGen_xegpu

  EUDSLGen_affine
  EUDSLGen_amdgpu
  EUDSLGen_arith
  EUDSLGen_async
  EUDSLGen_bufferization
  EUDSLGen_cf
  EUDSLGen_complex
  EUDSLGen_emitc
  EUDSLGen_func
  EUDSLGen_gpu
  EUDSLGen_index
  EUDSLGen_linalg
  EUDSLGen_LLVM
  EUDSLGen_math
  EUDSLGen_memref
  EUDSLGen_nvgpu
  EUDSLGen_NVVM
  EUDSLGen_pdl
  EUDSLGen_pdl_interp
  EUDSLGen_polynomial
  EUDSLGen_ROCDL
  EUDSLGen_scf
  EUDSLGen_shape
  EUDSLGen_tensor
  EUDSLGen_tosa
)
set_target_properties(eudslpy_ext
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${EUDSLPY_SRC_DIR}/eudsl"
)
target_link_libraries(eudslpy_ext PRIVATE MLIR)
# hack - on GHA, linux, the build OOMs
if ("$ENV{DISABLE_COMPILE_OPT}" MATCHES "true")
  target_compile_options(eudslpy_ext PRIVATE -O0)
endif()

patch_mlir_llvm_rpath(eudslpy_ext)
# copy libMLIR into the ext dir for wheels
install(IMPORTED_RUNTIME_ARTIFACTS MLIR LLVM LIBRARY DESTINATION eudsl)

target_compile_options(eudslpy_ext PRIVATE ${nanobind_options})
target_compile_options(nanobind-static PRIVATE ${nanobind_options})

# note WORKING_DIRECTORY
set(NB_STUBGEN_CMD "${Python_EXECUTABLE}" "-m" "nanobind.stubgen"
  --module eudslpy_ext --recursive --include-private --output-dir .)
set(NB_STUBGEN_OUTPUTS "__init__.pyi")
add_custom_command(
  OUTPUT ${NB_STUBGEN_OUTPUTS}
  COMMAND ${NB_STUBGEN_CMD}
  WORKING_DIRECTORY "${EUDSLPY_SRC_DIR}/eudsl"
  DEPENDS eudslpy_ext
)
add_custom_target(eudslpy_ext_stub ALL DEPENDS ${NB_STUBGEN_OUTPUTS})

install(TARGETS eudslpy_ext LIBRARY DESTINATION eudsl)
install(
  DIRECTORY "${EUDSLPY_SRC_DIR}/eudsl"
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.so" EXCLUDE
  PATTERN "*.a" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".gitignore" EXCLUDE
)
