cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(AddEUDSL)
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

add_eudslpygen(eudslpy-gen EUDSLPY
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  EXPORT ${LLVM_SUBPROJECT_TITLE}
  eudslpy-gen.cpp
)

eudslpygen(EUDSLGenArith
  ${MLIR_INCLUDE_DIR}/mlir/Dialect/Arith/IR/Arith.h
  EUDSLGenArith.cpp.inc
  NAMESPACES mlir::arith mlir::arith::detail
)



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

nanobind_add_module(eudslpy_ext
  NB_STATIC
  NB_DOMAIN eudsl
  eudslpy_ext.cpp
)
set_target_properties(eudslpy_ext
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/eudsl"
)
target_link_libraries(eudslpy_ext PRIVATE MLIR)
target_compile_options(eudslpy_ext PRIVATE ${nanobind_options})
target_compile_options(nanobind-static PRIVATE ${nanobind_options})
add_dependencies(eudslpy_ext EUDSLGenArith)

set(NB_STUBGEN_CMD "${Python_EXECUTABLE}" "-m" "nanobind.stubgen" -m eudsl -r)
set(NB_STUBGEN_OUTPUTS ${CMAKE_CURRENT_LIST_DIR}/eudsl/__init__.pyi)
add_custom_command(
  OUTPUT ${NB_STUBGEN_OUTPUTS}
  COMMAND ${NB_STUBGEN_CMD}
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  DEPENDS eudslpy_ext
)
add_custom_target(eudslpy_ext_stub ALL DEPENDS ${NB_STUBGEN_OUTPUTS})

install(TARGETS eudslpy_ext LIBRARY DESTINATION eudsl)
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/eudsl
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  PATTERN "*.so" EXCLUDE
  PATTERN "*.a" EXCLUDE
  PATTERN ".gitignore" EXCLUDE
)
