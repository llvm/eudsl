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
  $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Darwin>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>
)

set(EUDSLPY_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
set(EUDSLPY_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake")
include_directories(${EUDSLPY_BINARY_DIR})
include_directories(${EUDSLPY_SRC_DIR})
find_package(EUDSLPY CONFIG REQUIRED)

add_eudslpygen(eudslpy-gen EUDSLPY
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  EXPORT ${LLVM_SUBPROJECT_TITLE}
  src/eudslpy-gen.cpp
)

# too big
# eudslpygen(EUDSLGenacc
#   ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#   NAMESPACES mlir::acc mlir::acc::detail
# )

eudslpygen(EUDSLGenaffine
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::affine mlir::affine::detail
  EXTRA_INCLUDES
    mlir/IR/IntegerSet.h
    mlir/Dialect/Affine/IR/AffineValueMap.h
)

eudslpygen(EUDSLGenamdgpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::amdgpu mlir::amdgpu::detail
)

eudslpygen(EUDSLGenamx
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::amx mlir::amx::detail
)

eudslpygen(EUDSLGenarith
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::arith mlir::arith::detail
)

eudslpygen(EUDSLGenarm_neon
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::arm_neon mlir::arm_neon::detail
)

# too big
# eudslpygen(EUDSLGenarm_sme
#   ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#   NAMESPACES mlir::arm_sme mlir::arm_sme::detail
# )

eudslpygen(EUDSLGenarm_sve
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::arm_sve mlir::arm_sve::detail
)

eudslpygen(EUDSLGenasync
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::async mlir::async::detail
)

eudslpygen(EUDSLGenbufferization
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::bufferization mlir::bufferization::detail
  EXTRA_INCLUDES
    mlir/Dialect/Bufferization/Transforms/Bufferize.h
)

eudslpygen(EUDSLGencf
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR/ControlFlowOps.h
  NAMESPACES mlir::cf mlir::cf::detail
  EXTRA_INCLUDES
    mlir/Dialect/ControlFlow/IR/ControlFlowOps.h
    mlir/IR/PatternMatch.h
)

eudslpygen(EUDSLGencomplex
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::complex mlir::complex::detail
)

eudslpygen(EUDSLGenDLTIDialect
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::DLTIDialect mlir::DLTIDialect::detail
)

eudslpygen(EUDSLGenemitc
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::emitc mlir::emitc::detail
)

eudslpygen(EUDSLGenfunc
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::func mlir::func::detail
)

eudslpygen(EUDSLGengpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::gpu mlir::gpu::detail
  EXTRA_INCLUDES
    mlir/Target/LLVMIR/ModuleTranslation.h
    llvm/IR/IRBuilder.h
)

eudslpygen(EUDSLGenindex
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Index/IR/IndexOps.h
  NAMESPACES mlir::index mlir::index::detail
  EXTRA_INCLUDES
    mlir/Dialect/Index/IR/IndexOps.h
    mlir/IR/PatternMatch.h
)

#eudslpygen(EUDSLGenirdl
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  NAMESPACES mlir::irdl mlir::irdl::detail
#)

eudslpygen(EUDSLGenlinalg
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::linalg mlir::linalg::detail
)

eudslpygen(EUDSLGenLLVM
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::LLVM mlir::LLVM::detail
)

eudslpygen(EUDSLGenmath
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::math mlir::math::detail
)

eudslpygen(EUDSLGenmemref
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::memref mlir::memref::detail
)

eudslpygen(EUDSLGenmesh
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Mesh/IR/MeshOps.h
  NAMESPACES mlir::mesh mlir::mesh::detail
  EXTRA_INCLUDES
    mlir/Dialect/Mesh/IR/MeshOps.h
)

eudslpygen(EUDSLGenml_program
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::ml_program mlir::ml_program::detail
)

eudslpygen(EUDSLGenmpi
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::mpi mlir::mpi::detail
)

eudslpygen(EUDSLGennvgpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::nvgpu mlir::nvgpu::detail
)

eudslpygen(EUDSLGenNVVM
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::NVVM mlir::NVVM::detail
)

#eudslpygen(EUDSLGenomp
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  NAMESPACES mlir::omp mlir::omp::detail
#)

eudslpygen(EUDSLGenpdl
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/PDL/IR/PDLOps.h
  NAMESPACES mlir::pdl mlir::pdl::detail
  EXTRA_INCLUDES
    mlir/Dialect/PDL/IR/PDLOps.h
)

eudslpygen(EUDSLGenpdl_interp
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::pdl_interp mlir::pdl_interp::detail
)

eudslpygen(EUDSLGenpolynomial
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Polynomial/IR/PolynomialOps.h
  NAMESPACES mlir::polynomial mlir::polynomial::detail
  EXTRA_INCLUDES
    mlir/Dialect/Polynomial/IR/PolynomialOps.h
    mlir/IR/PatternMatch.h
)

eudslpygen(EUDSLGenptr
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Ptr/IR/PtrOps.h
  NAMESPACES mlir::ptr mlir::ptr::detail
  EXTRA_INCLUDES
    mlir/Dialect/Ptr/IR/PtrOps.h
    mlir/IR/DialectImplementation.h
)

eudslpygen(EUDSLGenquant
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::quant mlir::quant::detail
  EXTRA_INCLUDES
    mlir/Dialect/Quant/IR/QuantTypes.h
)

eudslpygen(EUDSLGenROCDL
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::ROCDL mlir::ROCDL::detail
)

eudslpygen(EUDSLGenscf
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::scf mlir::scf::detail
)

eudslpygen(EUDSLGenshape
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::shape mlir::shape::detail
)

eudslpygen(EUDSLGensparse_tensor
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::sparse_tensor mlir::sparse_tensor::detail
)

# nb::detail::nb_func_new("get_vce_triple_attr_name"): mismatched static/instance method flags in function overloads!
# eudslpygen(EUDSLGenspirv
#   ${MLIR_INCLUDE_DIR}/mlir/Dialect/SPIRV/IR/SPIRVOps.h
#   NAMESPACES mlir::spirv mlir::spirv::detail
#   EXTRA_INCLUDES
#     mlir/Dialect/SPIRV/IR/SPIRVOps.h
#     mlir/IR/PatternMatch.h
# )

eudslpygen(EUDSLGentensor
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::tensor mlir::tensor::detail
)

eudslpygen(EUDSLGentosa
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::tosa mlir::tosa::detail
)

eudslpygen(EUDSLGentransform
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::transform mlir::transform::detail
)

eudslpygen(EUDSLGenub
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::ub mlir::ub::detail
)

#eudslpygen(EUDSLGenvector
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  NAMESPACES mlir::vector mlir::vector::detail
#)

eudslpygen(EUDSLGenx86vector
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  NAMESPACES mlir::x86vector mlir::x86vector::detail
)

eudslpygen(EUDSLGenxegpu
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
  # EUDSLGenirdl
  # EUDSLGenomp
  # EUDSLGenvector
  # weight reduction
  # EUDSLGenacc
  # EUDSLGenamx
  # EUDSLGenarm_neon
  # EUDSLGenarm_sme
  # EUDSLGenarm_sve
  # EUDSLGenDLTIDialect
  # EUDSLGenmesh
  # EUDSLGenml_program
  # EUDSLGenmpi
  # EUDSLGenptr
  # EUDSLGenquant
  # EUDSLGensparse_tensor
  # EUDSLGenspirv
  # EUDSLGentransform
  # EUDSLGenub
  # EUDSLGenx86vector
  # EUDSLGenxegpu

  EUDSLGenaffine
  EUDSLGenamdgpu
  EUDSLGenarith
  EUDSLGenasync
  EUDSLGenbufferization
  EUDSLGencf
  EUDSLGencomplex
  EUDSLGenemitc
  EUDSLGenfunc
  EUDSLGengpu
  EUDSLGenindex
  EUDSLGenlinalg
  EUDSLGenLLVM
  EUDSLGenmath
  EUDSLGenmemref
  EUDSLGennvgpu
  EUDSLGenNVVM
  EUDSLGenpdl
  EUDSLGenpdl_interp
  EUDSLGenpolynomial
  EUDSLGenROCDL
  EUDSLGenscf
  EUDSLGenshape
  EUDSLGentensor
  EUDSLGentosa
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

# hack so we can move libMLIR and libLLVM into the wheel
# see AddLLVM.cmake#llvm_setup_rpath
if(APPLE OR UNIX)
  set(_origin_prefix "\$ORIGIN")
  if(APPLE)
    set(_origin_prefix "@loader_path")
  endif()
  if (EUDSLPY_STANDALONE_BUILD)
    get_target_property(_mlir_loc MLIR LOCATION)
    get_target_property(_llvm_loc LLVM LOCATION)
  else()
    set(_mlir_loc "$<TARGET_FILE:MLIR>")
    set(_llvm_loc "$<TARGET_FILE:LLVM>")
  endif()
  set(_old_rpath "${_origin_prefix}/../lib${LLVM_LIBDIR_SUFFIX}")
  if(APPLE)
    execute_process(COMMAND install_name_tool -rpath "${_old_rpath}" ${_origin_prefix} "${_mlir_loc}" ERROR_VARIABLE rpath_err)
    execute_process(COMMAND install_name_tool -rpath "${_old_rpath}" ${_origin_prefix} "${_llvm_loc}" ERROR_VARIABLE rpath_err)
    # maybe already updated...
    if (rpath_err AND NOT rpath_err MATCHES "no LC_RPATH load command with path: ${_old_rpath}")
      message(FATAL_ERROR "couldn't update rpath because: ${rpath_err}")
    endif()
  else()
    # sneaky sneaky - undocumented
    file(RPATH_CHANGE FILE "${_mlir_loc}" OLD_RPATH "${_old_rpath}" NEW_RPATH "${_origin_prefix}")
    file(RPATH_CHANGE FILE "${_llvm_loc}" OLD_RPATH "${_old_rpath}" NEW_RPATH "${_origin_prefix}")
  endif()
  set_target_properties(eudslpy_ext PROPERTIES INSTALL_RPATH "${_origin_prefix}")
endif()
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
  PATTERN ".gitignore" EXCLUDE
)
