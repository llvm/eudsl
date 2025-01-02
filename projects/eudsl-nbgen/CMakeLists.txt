cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLVM_SUBPROJECT_TITLE "EUDSL_NBGEN")
set(EUDSL_NBGEN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message("Building ${LLVM_SUBPROJECT_TITLE} as a standalone project.")
  project(${LLVM_SUBPROJECT_TITLE} CXX C)
  set(EUDSL_NBGEN_STANDALONE_BUILD ON)
else()
  enable_language(CXX C)
  set(EUDSL_NBGEN_STANDALONE_BUILD OFF)
endif()

if(EUDSL_NBGEN_STANDALONE_BUILD)
  find_package(LLVM REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG PATHS "${LLVM_BINARY_DIR}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddClang)
  # TODO(max): probably don't need this anymore after landing the nanobind fix?
  # technically we need this on windows too but our LLVM is compiled without exception handling
  # and that breaks windows
  if(NOT WIN32)
    include(HandleLLVMOptions)
  endif()
else()
  # turning LLVM -DLLVM_OPTIMIZED_TABLEGEN=ON builds some stuff in the NATIVE dir
  # but not everything so LLVM_BINARY_DIR isn't correct
  string(REPLACE "NATIVE" "" LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
  # Build via external projects mechanism
  set(LLVM_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/include)
  set(LLVM_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/include)
  set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIR};${LLVM_GENERATED_INCLUDE_DIR}")

  set(CLANG_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../clang)
  set(CLANG_INCLUDE_DIR ${CLANG_MAIN_SRC_DIR}/include)
  set(CLANG_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/clang/include)
  set(CLANG_INCLUDE_DIRS "${CLANG_INCLUDE_DIR};${CLANG_GENERATED_INCLUDE_DIR}")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

if(NOT TARGET LLVMSupport)
  message(FATAL_ERROR "LLVMSupport not found")
endif()

# this will set the rpath of the exe to be `../lib`, where is where we deposit libLLVM.so below
add_llvm_executable(eudsl-nbgen src/eudsl-nbgen.cpp)
target_link_libraries(eudsl-nbgen
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
)

string(TOLOWER ${LLVM_SUBPROJECT_TITLE} EUDSL_NBGEN_INSTALL_DATADIR)
# https://github.com/scikit-build/scikit-build-core/blob/a887a9b6c057b4ce9d3cfd53ae24e73caf1395a2/docs/build.md?plain=1#L139-L148
# actually installs to venv/bin
install(TARGETS eudsl-nbgen RUNTIME DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}.data/scripts")
# this actually installs into venv/lib
install(IMPORTED_RUNTIME_ARTIFACTS LLVM LIBRARY DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}.data/data/lib")

install(
  DIRECTORY src/
  DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}"
  FILES_MATCHING PATTERN "*\.py"
)

install(
  FILES
  cmake/eudsl_nbgen-config.cmake
  cmake/make_generated_shards.py
  DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}/cmake"
)
