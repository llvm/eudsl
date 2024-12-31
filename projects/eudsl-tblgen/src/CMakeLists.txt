cmake_minimum_required(VERSION 3.29)
project(eudsl_tblgen CXX C)
set(CMAKE_CXX_STANDARD 17)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})

# technically we need this but our LLVM is compiled without exception handling
# and that breaks windows
if(NOT WIN32)
  include(HandleLLVMOptions)
endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/eudsl_tblgen)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

nanobind_add_module(eudsl_tblgen_ext NB_STATIC STABLE_ABI eudsl_tblgen_ext.cpp TGParser.cpp TGLexer.cpp)
target_link_libraries(eudsl_tblgen_ext PRIVATE LLVMTableGenCommon LLVMTableGen)
target_compile_options(eudsl_tblgen_ext
  PUBLIC
  $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Darwin>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>)

nanobind_add_stub(
  eudsl_tblgen_ext_stub
  MODULE eudsl_tblgen_ext
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/eudsl_tblgen/eudsl_tblgen_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:eudsl_tblgen_ext>
  DEPENDS eudsl_tblgen_ext
)
nanobind_add_stub(
  eudsl_tblgen_stub
  MODULE eudsl_tblgen
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/eudsl_tblgen/__init__.pyi
  PYTHON_PATH ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS eudsl_tblgen_ext
)

install(TARGETS eudsl_tblgen_ext LIBRARY DESTINATION eudsl_tblgen)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/eudsl_tblgen
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  PATTERN "*.so" EXCLUDE
  PATTERN "*.a" EXCLUDE
  PATTERN ".gitignore" EXCLUDE
)
