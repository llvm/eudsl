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
include_directories(${EUDSLPY_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(NOT TARGET LLVMSupport)
  message(FATAL_ERROR "LLVMSupport not found")
endif()

set(EUDSLPY_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(EUDSLPY CONFIG REQUIRED)
add_eudslpygen(eudslpy-gen EUDSLPY
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  EXPORT ${LLVM_SUBPROJECT_TITLE}
  src/eudslpy-gen.cpp
)

eudslpygen(EUDSLGenacc
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenacc.cpp.inc
  NAMESPACES mlir::acc mlir::acc::detail
)

eudslpygen(EUDSLGenaffine
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenaffine.cpp.inc
  NAMESPACES mlir::affine mlir::affine::detail
)

eudslpygen(EUDSLGenamdgpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenamdgpu.cpp.inc
  NAMESPACES mlir::amdgpu mlir::amdgpu::detail
)

eudslpygen(EUDSLGenamx
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenamx.cpp.inc
  NAMESPACES mlir::amx mlir::amx::detail
)

eudslpygen(EUDSLGenarith
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenarith.cpp.inc
  NAMESPACES mlir::arith mlir::arith::detail
)

eudslpygen(EUDSLGenarm_neon
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenarm_neon.cpp.inc
  NAMESPACES mlir::arm_neon mlir::arm_neon::detail
)

eudslpygen(EUDSLGenarm_sme
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenarm_sme.cpp.inc
  NAMESPACES mlir::arm_sme mlir::arm_sme::detail
)

eudslpygen(EUDSLGenarm_sve
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenarm_sve.cpp.inc
  NAMESPACES mlir::arm_sve mlir::arm_sve::detail
)

eudslpygen(EUDSLGenasync
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenasync.cpp.inc
  NAMESPACES mlir::async mlir::async::detail
)

eudslpygen(EUDSLGenbufferization
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenbufferization.cpp.inc
  NAMESPACES mlir::bufferization mlir::bufferization::detail
)

eudslpygen(EUDSLGencf
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR/ControlFlowOps.h
  EUDSLGencf.cpp.inc
  NAMESPACES mlir::cf mlir::cf::detail
)

eudslpygen(EUDSLGencomplex
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGencomplex.cpp.inc
  NAMESPACES mlir::complex mlir::complex::detail
)

eudslpygen(EUDSLGenDLTIDialect
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenDLTIDialect.cpp.inc
  NAMESPACES mlir::DLTIDialect mlir::DLTIDialect::detail
)

#eudslpygen(EUDSLGenemitc
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  EUDSLGenemitc.cpp.inc
#  NAMESPACES mlir::emitc mlir::emitc::detail
#)

eudslpygen(EUDSLGenfunc
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenfunc.cpp.inc
  NAMESPACES mlir::func mlir::func::detail
)

eudslpygen(EUDSLGengpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGengpu.cpp.inc
  NAMESPACES mlir::gpu mlir::gpu::detail
)

eudslpygen(EUDSLGenindex
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Index/IR/IndexOps.h
  EUDSLGenindex.cpp.inc
  NAMESPACES mlir::index mlir::index::detail
)

#eudslpygen(EUDSLGenirdl
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  EUDSLGenirdl.cpp.inc
#  NAMESPACES mlir::irdl mlir::irdl::detail
#)

eudslpygen(EUDSLGenlinalg
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenlinalg.cpp.inc
  NAMESPACES mlir::linalg mlir::linalg::detail
)

#eudslpygen(EUDSLGenLLVM
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  EUDSLGenLLVM.cpp.inc
#  NAMESPACES mlir::LLVM mlir::LLVM::detail
#)

eudslpygen(EUDSLGenmath
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenmath.cpp.inc
  NAMESPACES mlir::math mlir::math::detail
)

eudslpygen(EUDSLGenmemref
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenmemref.cpp.inc
  NAMESPACES mlir::memref mlir::memref::detail
)

eudslpygen(EUDSLGenmesh
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Mesh/IR/MeshOps.h
  EUDSLGenmesh.cpp.inc
  NAMESPACES mlir::mesh mlir::mesh::detail
)

eudslpygen(EUDSLGenml_program
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenml_program.cpp.inc
  NAMESPACES mlir::ml_program mlir::ml_program::detail
)

eudslpygen(EUDSLGenmpi
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenmpi.cpp.inc
  NAMESPACES mlir::mpi mlir::mpi::detail
)

eudslpygen(EUDSLGennvgpu
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGennvgpu.cpp.inc
  NAMESPACES mlir::nvgpu mlir::nvgpu::detail
)

eudslpygen(EUDSLGenNVVM
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenNVVM.cpp.inc
  NAMESPACES mlir::NVVM mlir::NVVM::detail
)

#eudslpygen(EUDSLGenomp
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  EUDSLGenomp.cpp.inc
#  NAMESPACES mlir::omp mlir::omp::detail
#)

eudslpygen(EUDSLGenpdl
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/PDL/IR/PDLOps.h
  EUDSLGenpdl.cpp.inc
  NAMESPACES mlir::pdl mlir::pdl::detail
)

eudslpygen(EUDSLGenpdl_interp
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenpdl_interp.cpp.inc
  NAMESPACES mlir::pdl_interp mlir::pdl_interp::detail
)

eudslpygen(EUDSLGenpolynomial
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Polynomial/IR/PolynomialOps.h
  EUDSLGenpolynomial.cpp.inc
  NAMESPACES mlir::polynomial mlir::polynomial::detail
)

eudslpygen(EUDSLGenptr
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Ptr/IR/PtrOps.h
  EUDSLGenptr.cpp.inc
  NAMESPACES mlir::ptr mlir::ptr::detail
)

eudslpygen(EUDSLGenquant
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenquant.cpp.inc
  NAMESPACES mlir::quant mlir::quant::detail
)

eudslpygen(EUDSLGenROCDL
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenROCDL.cpp.inc
  NAMESPACES mlir::ROCDL mlir::ROCDL::detail
)

eudslpygen(EUDSLGenscf
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenscf.cpp.inc
  NAMESPACES mlir::scf mlir::scf::detail
)

eudslpygen(EUDSLGenshape
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenshape.cpp.inc
  NAMESPACES mlir::shape mlir::shape::detail
)

eudslpygen(EUDSLGensparse_tensor
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGensparse_tensor.cpp.inc
  NAMESPACES mlir::sparse_tensor mlir::sparse_tensor::detail
)

eudslpygen(EUDSLGenspirv
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/SPIRV/IR/SPIRVOps.h
  EUDSLGenspirv.cpp.inc
  NAMESPACES mlir::spirv mlir::spirv::detail
)

eudslpygen(EUDSLGentensor
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGentensor.cpp.inc
  NAMESPACES mlir::tensor mlir::tensor::detail
)

eudslpygen(EUDSLGentosa
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGentosa.cpp.inc
  NAMESPACES mlir::tosa mlir::tosa::detail
)

eudslpygen(EUDSLGentransform
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGentransform.cpp.inc
  NAMESPACES mlir::transform mlir::transform::detail
)

eudslpygen(EUDSLGenub
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenub.cpp.inc
  NAMESPACES mlir::ub mlir::ub::detail
)

#eudslpygen(EUDSLGenvector
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  EUDSLGenvector.cpp.inc
#  NAMESPACES mlir::vector mlir::vector::detail
#)

eudslpygen(EUDSLGenx86vector
  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
  EUDSLGenx86vector.cpp.inc
  NAMESPACES mlir::x86vector mlir::x86vector::detail
)

#eudslpygen(EUDSLGenxegpu
#  ${MLIR_INCLUDE_DIR}/mlir/InitAllDialects.h
#  EUDSLGenxegpu.cpp.inc
#  NAMESPACES mlir::xegpu mlir::xegpu::detail
#)

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

nanobind_add_module(eudslpy_ext
  NB_STATIC
  NB_DOMAIN eudsl
  src/eudslpy_ext.cpp
)
add_dependencies(eudslpy_ext
  EUDSLGenacc
  EUDSLGenaffine
  EUDSLGenamdgpu
  EUDSLGenamx
  EUDSLGenarith
  EUDSLGenarm_neon
  EUDSLGenarm_sme
  EUDSLGenarm_sve
  EUDSLGenasync
  EUDSLGenbufferization
  EUDSLGencf
  EUDSLGencomplex
  EUDSLGenDLTIDialect
#  EUDSLGenemitc
  EUDSLGenfunc
  EUDSLGengpu
  EUDSLGenindex
#  EUDSLGenirdl
  EUDSLGenlinalg
#  EUDSLGenLLVM
  EUDSLGenmath
  EUDSLGenmemref
  EUDSLGenmesh
  EUDSLGenml_program
  EUDSLGenmpi
  EUDSLGennvgpu
  EUDSLGenNVVM
#  EUDSLGenomp
  EUDSLGenpdl
  EUDSLGenpdl_interp
  EUDSLGenpolynomial
  EUDSLGenptr
  EUDSLGenquant
  EUDSLGenROCDL
  EUDSLGenscf
  EUDSLGenshape
  EUDSLGensparse_tensor
  EUDSLGenspirv
  EUDSLGentensor
  EUDSLGentosa
  EUDSLGentransform
  EUDSLGenub
#  EUDSLGenvector
  EUDSLGenx86vector
#  EUDSLGenxegpu
)
set_target_properties(eudslpy_ext
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${EUDSLPY_SRC_DIR}/eudsl"
)
target_link_libraries(eudslpy_ext PRIVATE MLIR)

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
set(NB_STUBGEN_CMD "${Python_EXECUTABLE}" "-m" "nanobind.stubgen" --module eudslpy_ext --recursive --include-private --output-dir .)
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
