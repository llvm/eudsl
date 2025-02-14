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
  include(HandleLLVMOptions)
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
if (NOT "$ENV{PIP_BUILD_TRACKER}" STREQUAL "")
  # pip install
  # actually installs to venv/bin
  # https://github.com/scikit-build/scikit-build-core/blob/a887a9b6c057b4ce9d3cfd53ae24e73caf1395a2/docs/build.md?plain=1#L139-L148
  install(TARGETS eudsl-nbgen RUNTIME DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}.data/scripts")
  if (NOT WIN32)
    # this actually installs into venv/lib
    install(IMPORTED_RUNTIME_ARTIFACTS LLVM LIBRARY DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}.data/data/lib")
  endif()
else()
  # pip cmake install
  install(TARGETS eudsl-nbgen RUNTIME DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}/bin")
  if (NOT WIN32)
    install(IMPORTED_RUNTIME_ARTIFACTS LLVM LIBRARY DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}/lib")
  endif()
endif()

install(
  # the slash is load-bearing...
  DIRECTORY src/
  DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}"
  FILES_MATCHING PATTERN "*\.py"
)

install(
  DIRECTORY includes/eudsl/
  DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}/includes/eudsl"
  FILES_MATCHING PATTERN "*\.h"
)

install(
  FILES
  cmake/eudsl_nbgen-config.cmake
  DESTINATION "${EUDSL_NBGEN_INSTALL_DATADIR}/cmake"
)
