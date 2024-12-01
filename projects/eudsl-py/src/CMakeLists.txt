cmake_minimum_required(VERSION 3.29)
project(eudslpy CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# technically we need this but our LLVM is compiled without exception handling
# and that breaks windows
if(NOT WIN32)
  include(HandleLLVMOptions)
endif()

if(NOT TARGET LLVMSupport)
  message(FATAL_ERROR "LLVMSupport not found")
endif()

# no clue why but with LLVM_LINK_LLVM_DYLIB even static libs depend on LLVM
get_property(MLIR_ALL_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
foreach(_lib ${MLIR_ALL_LIBS})
  get_target_property(_interface_link_libraries ${_lib} INTERFACE_LINK_LIBRARIES)
  list(REMOVE_DUPLICATES _interface_link_libraries)
  list(REMOVE_ITEM _interface_link_libraries LLVM)
  # for some reason, explicitly adding below as a link library doesn't work - missing symbols...
  list(APPEND _interface_link_libraries LLVMSupport)
  set_target_properties(${_lib} PROPERTIES INTERFACE_LINK_LIBRARIES "${_interface_link_libraries}")
endforeach()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/eudsl)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(-Wno-cast-qual -Wno-deprecated-literal-operator)
nanobind_add_module(eudsl_ext NB_STATIC STABLE_ABI eudsl_ext.cpp ir.cpp arith.cpp)
target_link_libraries(eudsl_ext PRIVATE MLIRIR LLVMSupport MLIRTransforms MLIRArithDialect)
target_compile_options(eudsl_ext
  PRIVATE
  $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Darwin>:-fexceptions -frtti>
  $<$<PLATFORM_ID:Windows>:/EHsc /GR>)

set(NB_STUBGEN_CMD "${Python_EXECUTABLE}" "-m" "nanobind.stubgen" -m eudsl -r)
set(NB_STUBGEN_OUTPUTS ${CMAKE_CURRENT_SOURCE_DIR}/eudsl/__init__.pyi)
add_custom_command(
  OUTPUT ${NB_STUBGEN_OUTPUTS}
  COMMAND ${NB_STUBGEN_CMD}
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS eudsl_ext
)
add_custom_target(eudsl_ext_stub ALL DEPENDS ${NB_STUBGEN_OUTPUTS})

install(TARGETS eudsl_ext LIBRARY DESTINATION eudsl)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/eudsl
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  PATTERN "*.so" EXCLUDE
  PATTERN "*.a" EXCLUDE
  PATTERN ".gitignore" EXCLUDE
)
