# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

# copy-pasta from AddMLIR.cmake/AddLLVM.cmake/TableGen.cmake

function(eudslpygen target inputFile)
  set(EUDSLPYGEN_TARGET_DEFINITIONS ${inputFile})
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES;NAMESPACES" ${ARGN})
  if (IS_ABSOLUTE ${EUDSLPYGEN_TARGET_DEFINITIONS})
    set(EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE ${inputFile})
  else()
    set(EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${inputFile})
  endif()

  if(NOT EXISTS "${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}")
    message(FATAL_ERROR "${inputFile} does not exist")
  endif()

  get_directory_property(eudslpygen_includes INCLUDE_DIRECTORIES)
  list(APPEND eudslpygen_includes ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES} ${ARG_EXTRA_INCLUDES})
  list(REMOVE_ITEM eudslpygen_includes "")
  list(TRANSFORM eudslpygen_includes PREPEND -I)

  set(_gen_target_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${target}")
  file(MAKE_DIRECTORY ${_gen_target_dir})
  set(fullGenFile "${_gen_target_dir}/${target}.cpp.gen")
  file(RELATIVE_PATH fullGenFile_rel "${CMAKE_BINARY_DIR}" "${fullGenFile}")
  set(_depfile ${fullGenFile}.d)

  # hack but most of the time we're loading headers that are downstream of tds anyway
  # this could be smarter by asking people to list td targets or something but that's too onerous
  file(GLOB_RECURSE global_tds "${MLIR_INCLUDE_DIR}/mlir/*.td")
  # use cc -MM  to collect all transitive headers
  set(clang_command ${CMAKE_CXX_COMPILER}
    # -v
    -xc++ "-std=c++${CMAKE_CXX_STANDARD}"
    -MM ${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}
    -MT ${fullGenFile_rel}
    ${eudslpygen_includes}
    -o ${_depfile})
  execute_process(COMMAND ${clang_command} RESULT_VARIABLE had_error
    # COMMAND_ECHO STDERR
  )
  if(had_error OR NOT EXISTS "${_depfile}")
    set(additional_cmdline -o "${fullGenFile_rel}")
  else()
    # Use depfile instead of globbing arbitrary *.td(s) for Ninja.
    if(CMAKE_GENERATOR MATCHES "Ninja")
      # Make output path relative to build.ninja, assuming located on ${CMAKE_BINARY_DIR}.
      # CMake emits build targets as relative paths but Ninja doesn't identify
      # absolute path (in *.d) as relative path (in build.ninja)
      # Note that eudslpygen is executed on ${CMAKE_BINARY_DIR} as working directory.
      set(additional_cmdline -o "${fullGenFile_rel}" DEPFILE "${_depfile}")
    else()
      # the length of the first line in the depfile...
      string(LENGTH "${fullGenFile_rel}: \\" depfile_offset)
      file(READ ${_depfile} local_headers OFFSET ${depfile_offset})
      string(REPLACE "\\" ";" local_headers "${local_headers}")
      string(REGEX REPLACE "[ \t\r\n]" "" local_headers "${local_headers}")
      list(REMOVE_ITEM local_headers "")
      set(additional_cmdline -o "${fullGenFile_rel}")
    endif()
  endif()

  if (IS_ABSOLUTE ${EUDSLPYGEN_TARGET_DEFINITIONS})
    set(EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE ${EUDSLPYGEN_TARGET_DEFINITIONS})
  else()
    set(EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE
      ${CMAKE_CURRENT_SOURCE_DIR}/${EUDSLPYGEN_TARGET_DEFINITIONS})
  endif()

  # We need both _EUDSLPYGEN_TARGET and _EUDSLPYGEN_EXE in the  DEPENDS list
  # (both the target and the file) to have .inc files rebuilt on
  # a eudslpygen change, as cmake does not propagate file-level dependencies
  # of custom targets. See the following ticket for more information:
  # https://cmake.org/Bug/view.php?id=15858
  # The dependency on both, the target and the file, produces the same
  # dependency twice in the result file when
  # ("${EUDSLPY_EUDSLPYGEN_TARGET}" STREQUAL "${EUDSLPY_EUDSLPYGEN_EXE}")
  # but lets us having smaller and cleaner code here.
  set(eudslpygen_exe ${EUDSLPY_EUDSLPYGEN_EXE})
  set(eudslpygen_depends ${EUDSLPY_EUDSLPYGEN_TARGET} ${eudslpygen_exe})

  string(REPLACE " " ";" eudslpygen_defines "${LLVM_DEFINITIONS}")
  list(JOIN ARG_NAMESPACES "," namespaces)

  add_custom_command(OUTPUT ${fullGenFile}
    COMMAND ${eudslpygen_exe} ${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}
      -I${CMAKE_CURRENT_SOURCE_DIR}
      -namespaces=${namespaces}
      ${eudslpygen_includes}
      ${eudslpygen_defines}
      ${additional_cmdline}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    # The file in EUDSLPYGEN_TARGET_DEFINITIONS may be not in the current
    # directory and local_headers may not contain it, so we must
    # explicitly list it here:
    DEPENDS ${ARG_DEPENDS} ${eudslpygen_depends} ${local_headers} ${global_tds}
    COMMENT "EUDSLPY: Generating ${fullGenFile}..."
  )
  # this is the specific thing connected the dependencies...
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${fullGenFile})

  # epic hack to specify all shards that will be generated even though we don't know them before hand
  # TODO(max): refactor eudslpy-gen into its own subproject so that we can do execute_process(CMAKE_COMMAND... )
  set(_byproducts)
  # lol spirv has 260 ops
  set(_max_num_shards 30)
  foreach(i RANGE ${_max_num_shards})
    list(APPEND _byproducts "${fullGenFile}.shard.${i}.cpp")
  endforeach()

  add_custom_command(OUTPUT "${fullGenFile}.sharded.cpp"
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/make_generated_registration.py
      ${fullGenFile} -t ${target} -I ${ARG_EXTRA_INCLUDES} ${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}
      -m ${_max_num_shards}
    BYPRODUCTS ${_byproducts}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${fullGenFile}
    COMMENT "EUDSLPY: Generating ${fullGenFile}.sharded.cpp..."
  )
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${fullGenFile}.sharded.cpp")

  add_library(${target} STATIC "${fullGenFile}.sharded.cpp" ${_byproducts})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --include_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_include_dir)
  target_include_directories(${target} PRIVATE ${eudslpygen_includes}
    ${Python_INCLUDE_DIRS} ${nanobind_include_dir})

  # `make clean' must remove all those generated files:
  # TODO(max): clean up dep files
  set_property(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${_byproducts})
  set_source_files_properties(${_byproducts} PROPERTIES GENERATED 1)
endfunction()

macro(add_eudslpygen target project)
  set(options)
  set(oneValueArgs DESTINATION EXPORT)
  set(multiValueArgs)
  # When used inside a macro, arg might not be a suitable prefix because the code will affect the calling scope.
  cmake_parse_arguments(ARG_EUDSLPYGEN
    "${options}" "${oneValueArgs}" "${multiValueArgs}"
    ${ARGN}
  )

  add_llvm_executable(${target} DISABLE_LLVM_LINK_LLVM_DYLIB
    ${ARG_EUDSLPYGEN_UNPARSED_ARGUMENTS} PARTIAL_SOURCES_INTENDED)
  target_link_libraries(${target}
    PRIVATE
    clangAST
    clangBasic
    clangFrontend
    clangSerialization
    clangTooling
    LLVMSupport
  )

  set(${project}_EUDSLPYGEN_DEFAULT "${target}")
  if (LLVM_NATIVE_TOOL_DIR)
    if (EXISTS "${LLVM_NATIVE_TOOL_DIR}/${target}${LLVM_HOST_EXECUTABLE_SUFFIX}")
      set(${project}_EUDSLPYGEN_DEFAULT "${LLVM_NATIVE_TOOL_DIR}/${target}${LLVM_HOST_EXECUTABLE_SUFFIX}")
    endif()
  endif()

  if(ARG_EUDSLPYGEN_EXPORT)
    set(${project}_EUDSLPYGEN "${${project}_EUDSLPYGEN_DEFAULT}" CACHE
      STRING "Native eudslpy-gen executable. Saves building one when cross-compiling.")
  else()
    set(${project}_EUDSLPYGEN "${${project}_EUDSLPYGEN_DEFAULT}")
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()

  if(PROJECT_IS_TOP_LEVEL)
    set(_parent_scope)
  else()
    set(_parent_scope "PARENT_SCOPE")
  endif()

  # Effective eudslpygen executable to be used:
  set(${project}_EUDSLPYGEN_EXE ${${project}_EUDSLPYGEN} ${_parent_scope})
  set(${project}_EUDSLPYGEN_TARGET ${${project}_EUDSLPYGEN} ${_parent_scope})

  if(LLVM_USE_HOST_TOOLS)
    if( ${${project}_EUDSLPYGEN} STREQUAL "${target}" )
      # The NATIVE eudslpygen executable *must* depend on the current target one
      # otherwise the native one won't get rebuilt when the tablgen sources
      # change, and we end up with incorrect builds.
      build_native_tool(${target} ${project}_EUDSLPYGEN_EXE DEPENDS ${target})
      set(${project}_EUDSLPYGEN_EXE ${${project}_EUDSLPYGEN_EXE} PARENT_SCOPE)

      add_custom_target(${target}-host DEPENDS ${${project}_EUDSLPYGEN_EXE})
      get_subproject_title(subproject_title)
      set_target_properties(${target}-host PROPERTIES FOLDER "${subproject_title}/Native")
      set(${project}_EUDSLPYGEN_TARGET ${target}-host PARENT_SCOPE)

      # If we're using the host eudslpygen, and utils were not requested, we have no
      # need to build this eudslpygen.
      if (NOT LLVM_BUILD_UTILS)
        set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL ON)
      endif()
    endif()
  endif()

  if (ARG_EUDSLPYGEN_DESTINATION AND NOT LLVM_INSTALL_TOOLCHAIN_ONLY AND
      (LLVM_BUILD_UTILS OR ${target} IN_LIST LLVM_DISTRIBUTION_COMPONENTS))
    set(export_arg)
    if(ARG_EUDSLPYGEN_EXPORT)
      get_target_export_arg(${target} ${ARG_EUDSLPYGEN_EXPORT} export_arg)
    endif()
    install(TARGETS ${target}
      ${export_arg}
      COMPONENT ${target}
      RUNTIME DESTINATION "${ARG_EUDSLPYGEN_DESTINATION}")
    if(NOT LLVM_ENABLE_IDE)
      # TODO(max): need my own one of these...
      add_llvm_install_targets("install-${target}"
        DEPENDS ${target}
        COMPONENT ${target})
    endif()
  endif()
  if(ARG_EUDSLPYGEN_EXPORT)
    string(TOUPPER ${ARG_EUDSLPYGEN_EXPORT} export_upper)
    set_property(GLOBAL APPEND PROPERTY ${export_upper}_EXPORTS ${target})
  endif()
endmacro()

function(patch_mlir_llvm_rpath target)
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
      if (EXISTS ${_mlir_loc})
        execute_process(COMMAND install_name_tool -rpath "${_old_rpath}" ${_origin_prefix} "${_mlir_loc}" ERROR_VARIABLE rpath_err)
      endif()
      if (EXISTS ${_llvm_loc})
        execute_process(COMMAND install_name_tool -rpath "${_old_rpath}" ${_origin_prefix} "${_llvm_loc}" ERROR_VARIABLE rpath_err)
      endif()
      # maybe already updated...
      if (rpath_err AND NOT rpath_err MATCHES "no LC_RPATH load command with path: ${_old_rpath}")
        message(FATAL_ERROR "couldn't update rpath because: ${rpath_err}")
      endif()
    else()
      # sneaky sneaky - undocumented
      if (EXISTS ${_mlir_loc})
        file(RPATH_CHANGE FILE "${_mlir_loc}" OLD_RPATH "${_old_rpath}" NEW_RPATH "${_origin_prefix}")
      endif()
      if (EXISTS ${_llvm_loc})
        file(RPATH_CHANGE FILE "${_llvm_loc}" OLD_RPATH "${_old_rpath}" NEW_RPATH "${_origin_prefix}")
      endif()
    endif()
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "${_origin_prefix}")
  endif()
endfunction()