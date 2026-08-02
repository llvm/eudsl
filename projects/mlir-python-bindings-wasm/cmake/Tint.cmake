# Build Tint (SPIR-V reader + WGSL writer) without dawn's top-level CMakeLists.
#
# Tint does not actually depend on dawn. It lives inside the dawn repo and its
# CMakeLists reads a handful of DAWN_* variables and dawn's cmake helper
# functions, but it links nothing from dawn's runtime. So instead of adding
# dawn/CMakeLists.txt -- which under Emscripten unconditionally descends into
# generator/ and src/{emdawnwebgpu,dawn} (the guard at its line ~505 ends in
# "OR EMSCRIPTEN"), needs jinja2 to run a code generator we have no use for, and
# defines a dawncpp_module target that forces C++20 module scanning -- we add
# only src/utils and src/tint and provide the variables ourselves.
#
# Requires these to already exist (add them before including this file, or let
# this file add them): SPIRV-Headers, SPIRV-Tools, abseil-cpp.
#
# Defines: tint_lang_spirv_reader, tint_lang_wgsl_writer, and friends.

if(TARGET tint_lang_spirv_reader)
  return()
endif()

if(NOT DEFINED TINT_REPO_ROOT)
  message(FATAL_ERROR "set TINT_REPO_ROOT to the repo root before including Tint.cmake")
endif()

set(_tint_dawn "${TINT_REPO_ROOT}/third_party/dawn")
set(_tint_spirv_tools "${TINT_REPO_ROOT}/third_party/SPIRV-Tools")
set(_tint_spirv_headers "${TINT_REPO_ROOT}/third_party/SPIRV-Headers")
set(_tint_abseil "${TINT_REPO_ROOT}/third_party/abseil-cpp")

foreach(_d "${_tint_dawn}" "${_tint_spirv_tools}" "${_tint_spirv_headers}" "${_tint_abseil}")
  if(NOT EXISTS "${_d}/CMakeLists.txt")
    message(FATAL_ERROR
      "${_d} is empty. Run: git submodule update --init --depth 1 "
      "third_party/{dawn,SPIRV-Tools,SPIRV-Headers,abseil-cpp}")
  endif()
endforeach()

# Tint's headers use concepts and std::span. Save/restore around the
# add_subdirectory calls: CMake variables are directory-scoped, not
# block-scoped, so leaking C++20 into the rest of the build switches on module
# dependency scanning, and emscan-deps rejects emscripten's own
# -sSUPPORT_LONGJMP. Nothing here uses modules, so scanning stays off.
set(_tint_saved_std "${CMAKE_CXX_STANDARD}")
set(_tint_saved_scan "${CMAKE_CXX_SCAN_FOR_MODULES}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# --- dependencies -------------------------------------------------------------
if(NOT TARGET SPIRV-Headers)
  set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
  set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
  add_subdirectory("${_tint_spirv_headers}" "${CMAKE_CURRENT_BINARY_DIR}/spirv-headers"
                   EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET SPIRV-Tools)
  set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
  set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
  set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
  set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
  set(ENABLE_RTTI ON CACHE BOOL "" FORCE)
  set(SPIRV-Headers_SOURCE_DIR "${_tint_spirv_headers}" CACHE PATH "" FORCE)
  add_subdirectory("${_tint_spirv_tools}" "${CMAKE_CURRENT_BINARY_DIR}/spirv-tools"
                   EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET absl_base)
  add_subdirectory("${_tint_abseil}" "${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp"
                   EXCLUDE_FROM_ALL)
endif()

# --- the variables dawn's top-level would have set ----------------------------
set(DAWN_THIRD_PARTY_DIR "${_tint_dawn}/third_party")
set(DAWN_BUILD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")
set(DAWN_INCLUDE_DIR "${_tint_dawn}/include")
set(DAWN_SRC_DIR "${_tint_dawn}/src")
set(DAWN_ABSEIL_DIR "${_tint_abseil}")
set(DAWN_SPIRV_TOOLS_DIR "${_tint_spirv_tools}")
set(DAWN_SPIRV_HEADERS_DIR "${_tint_spirv_headers}")
set(DAWN_WERROR OFF)
set(DAWN_BUILD_PROTOBUF OFF)
set(DAWN_ENABLE_SPIRV_VALIDATION OFF)

# Reader + writer only. Everything else off, following IREE's
# compiler/plugins/target/WebGPUSPIRV/dawn/CMakeLists.txt.
set(TINT_BUILD_SPV_READER ON)
set(TINT_BUILD_WGSL_WRITER ON)
foreach(_off
    TINT_BUILD_SPV_WRITER TINT_BUILD_WGSL_READER TINT_BUILD_GLSL_WRITER
    TINT_BUILD_HLSL_WRITER TINT_BUILD_MSL_WRITER TINT_BUILD_NULL_WRITER
    TINT_BUILD_GLSL_VALIDATOR TINT_BUILD_CMD_TOOLS TINT_BUILD_TESTS
    TINT_BUILD_BENCHMARKS TINT_BUILD_FUZZERS TINT_BUILD_IR_BINARY
    TINT_BUILD_TINTD TINT_BUILD_FUZZER_VULKAN_SUPPORT)
  set(${_off} OFF)
endforeach()

# tint sources include as "src/tint/..." and "source/opt/..." (spirv-tools).
#
# These MUST NOT use include_directories(): that sets the directory property,
# which every target declared later in the including directory inherits. MLIR's
# declare_mlir_python_sources reads a target's INTERFACE_INCLUDE_DIRECTORIES back
# out as the source ROOT_DIR (AddMLIRPython.cmake:933), so polluting the
# directory scope makes it compute nonsense relative paths and try to mkdir
# outside the build tree. Add them to the tint targets after the fact instead.
set(_tint_include_dirs
  "${_tint_dawn}"
  "${_tint_dawn}/include"
  "${DAWN_BUILD_GEN_DIR}/include"
  "${_tint_spirv_tools}"
  "${_tint_spirv_tools}/include"
)

# dawn's cmake helpers (dawn_add_library et al). These are standalone modules;
# including them does not pull in any dawn target.
list(APPEND CMAKE_MODULE_PATH "${_tint_dawn}/src/cmake")
include(DawnSetIfNotDefined)
include(DawnCompilerChecks)
include(DawnCompilerExtraFlags)
include(DawnCompilerPlatformFlags)
include(DawnCompilerWarningFlags)
include(DawnLibrary)

# Two INTERFACE targets that dawn's top-level CMakeLists (lines ~397-410) would
# otherwise define. Tint links dawn_internal_config, but it carries nothing but
# include paths -- no dawn runtime code.
if(NOT TARGET dawn_public_config)
  add_library(dawn_public_config INTERFACE)
  target_include_directories(dawn_public_config INTERFACE
    "${DAWN_INCLUDE_DIR}"
    "${DAWN_BUILD_GEN_DIR}/include"
  )
endif()
if(NOT TARGET dawn_internal_config)
  add_library(dawn_internal_config INTERFACE)
  target_include_directories(dawn_internal_config INTERFACE
    "${_tint_dawn}"
    "${DAWN_BUILD_GEN_DIR}/src"
  )
  target_link_libraries(dawn_internal_config INTERFACE dawn_public_config)
endif()

# src/utils provides dawn_shared_utils, which tint links. It is a small
# standalone utility library, not part of dawn's WebGPU runtime.
add_subdirectory("${_tint_dawn}/src/utils" "${CMAKE_CURRENT_BINARY_DIR}/dawn_utils"
                 EXCLUDE_FROM_ALL)
add_subdirectory("${_tint_dawn}/src/tint" "${CMAKE_CURRENT_BINARY_DIR}/tint"
                 EXCLUDE_FROM_ALL)

# Apply the include dirs to the tint/dawn_utils targets only, now that they
# exist -- see the note above about why include_directories() is unusable here.
get_property(_tint_all_targets DIRECTORY "${_tint_dawn}/src/tint"
             PROPERTY BUILDSYSTEM_TARGETS)
get_property(_tint_utils_targets DIRECTORY "${_tint_dawn}/src/utils"
             PROPERTY BUILDSYSTEM_TARGETS)
foreach(_t ${_tint_all_targets} ${_tint_utils_targets})
  get_target_property(_type ${_t} TYPE)
  if(NOT _type STREQUAL "INTERFACE_LIBRARY" AND NOT _type STREQUAL "UTILITY")
    target_include_directories(${_t} PUBLIC ${_tint_include_dirs})
  endif()
endforeach()

set(CMAKE_CXX_STANDARD "${_tint_saved_std}")
set(CMAKE_CXX_SCAN_FOR_MODULES "${_tint_saved_scan}")

if(NOT TARGET tint_lang_spirv_reader OR NOT TARGET tint_lang_wgsl_writer)
  message(FATAL_ERROR "Tint.cmake ran but the tint targets were not created")
endif()
