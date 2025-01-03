# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

# copy-pasta from AddMLIR.cmake/AddLLVM.cmake/TableGen.cmake

function(eudsl_nbgen target input_file)
  set(EUDSL_NBGEN_TARGET_DEFINITIONS ${input_file})
  cmake_parse_arguments(ARG "" "" "LINK_LIBS;EXTRA_INCLUDES;NAMESPACES" ${ARGN})
  if (IS_ABSOLUTE ${EUDSL_NBGEN_TARGET_DEFINITIONS})
    set(EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE ${input_file})
  else()
    set(EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${input_file})
  endif()

  if (NOT EXISTS "${EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE}")
    message(FATAL_ERROR "${input_file} does not exist")
  endif()

  get_directory_property(eudsl_nbgen_includes INCLUDE_DIRECTORIES)
  list(TRANSFORM ARG_EXTRA_INCLUDES PREPEND -I)
  list(APPEND eudsl_nbgen_includes ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
  list(REMOVE_ITEM eudsl_nbgen_includes "")
  list(TRANSFORM eudsl_nbgen_includes PREPEND -I)
  list(APPEND eudsl_nbgen_includes ${ARG_EXTRA_INCLUDES})

  set(_gen_target_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/${target}")
  file(MAKE_DIRECTORY ${_gen_target_dir})
  set(_full_gen_file "${_gen_target_dir}/${target}.cpp.gen")
  set(_depfile ${_full_gen_file}.d)

  # hack but most of the time we're loading headers that are downstream of tds anyway
  # this could be smarter by asking people to list td targets or something but that's too onerous
  file(GLOB_RECURSE global_tds "${MLIR_INCLUDE_DIR}/mlir/*.td")
  if (NOT EXISTS ${_depfile})
    # use cc -MM  to collect all transitive headers
    set(clang_command ${CMAKE_CXX_COMPILER}
      # -v
      -xc++ "-std=c++${CMAKE_CXX_STANDARD}"
      -MM ${EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE}
      -MT ${_full_gen_file}
      ${eudsl_nbgen_includes}
      -o ${_depfile}
    )
    execute_process(COMMAND ${clang_command} RESULT_VARIABLE _had_error_depfile
      # COMMAND_ECHO STDOUT
    )
  endif()

  if (IS_ABSOLUTE ${EUDSL_NBGEN_TARGET_DEFINITIONS})
    set(EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE ${EUDSL_NBGEN_TARGET_DEFINITIONS})
  else()
    set(EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE
      ${CMAKE_CURRENT_SOURCE_DIR}/${EUDSL_NBGEN_TARGET_DEFINITIONS})
  endif()

  string(REPLACE " " ";" eudsl_nbgen_defines "${LLVM_DEFINITIONS}")
  list(JOIN ARG_NAMESPACES "," namespaces)

  set(eudsl_nbgen_generate_cmd
    ${EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE}
    -I${CMAKE_CURRENT_LIST_DIR} -namespaces=${namespaces}
    ${eudsl_nbgen_includes} ${eudsl_nbgen_defines}
    -o "${_full_gen_file}"
  )
  set(eudsl_nbgen_shardify_cmd
    -shardify ${_full_gen_file}
    # ARG_EXTRA_INCLUDES has already had -I prepended
    -shard-target ${target} ${ARG_EXTRA_INCLUDES} -I ${EUDSL_NBGEN_TARGET_DEFINITIONS_ABSOLUTE}
  )

  find_program(EUDSL_NBGEN_EXE "eudsl-nbgen")
  if (EUDSL_NBGEN_EXE STREQUAL "EUDSL_NBGEN_EXE-NOTFOUND")
    ##################################
    # not standalone build
    ##################################
    if (WIN32)
      set(EUDSL_NBGEN_EXE "eudsl-nbgen.exe")
    else()
      set(EUDSL_NBGEN_EXE "eudsl-nbgen")
    endif()

    string(REPLACE " " ";" eudsl_nbgen_defines "${LLVM_DEFINITIONS}")
    list(JOIN ARG_NAMESPACES "," namespaces)

    add_custom_command(OUTPUT ${_full_gen_file}
      COMMAND ${EUDSL_NBGEN_EXE} ${eudsl_nbgen_generate_cmd}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPENDS ${EUDSL_NBGEN_EXE} ${global_tds}
      DEPFILE ${_depfile}
      COMMENT "eudsl-nbgen: Generating ${_full_gen_file}..."
    )
    # epic hack to specify all shards that will be generated even though we don't know them before hand
    set(_shards)
    # lol spirv has 260 ops
    set(_max_num_shards 30)
    # note this count [0, 30] <- inclusive
    foreach(i RANGE ${_max_num_shards})
      list(APPEND _shards "${_full_gen_file}.shard.${i}.cpp")
    endforeach()

    add_custom_command(OUTPUT "${_full_gen_file}.sharded.cpp"
      COMMAND ${EUDSL_NBGEN_EXE} ${eudsl_nbgen_shardify_cmd}
      -max-number-shards ${_max_num_shards}
      BYPRODUCTS ${_shards}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      DEPENDS ${_full_gen_file} ${EUDSL_NBGEN_EXE}
      COMMENT "eudsl-nbgen: Generating ${_full_gen_file}.sharded.cpp..."
    )
  else()
    ##################################
    # standalone build
    ##################################
    execute_process(
      COMMAND ${EUDSL_NBGEN_EXE} ${eudsl_nbgen_generate_cmd}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      RESULT_VARIABLE _had_error_gen_cpp
      # COMMAND_ECHO STDOUT
    )
    if ((_had_error_gen_cpp AND NOT _had_error_gen_cpp EQUAL 0) OR NOT EXISTS "${_full_gen_file}")
      message(FATAL_ERROR "failed to create ${_full_gen_file}: ${_had_error_gen_cpp}")
    endif()
    execute_process(
      COMMAND ${EUDSL_NBGEN_EXE} ${eudsl_nbgen_shardify_cmd}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      RESULT_VARIABLE _had_error_gen_sharded
      # COMMAND_ECHO STDOUT
    )
    if ((_had_error_gen_sharded AND NOT _had_error_gen_sharded EQUAL 0) OR NOT EXISTS "${_full_gen_file}.sharded.cpp")
      message(FATAL_ERROR "failed to create ${_full_gen_file}.sharded.cpp: ${_had_error_gen_sharded}")
    endif()
    file(GLOB _shards CONFIGURE_DEPENDS "${_gen_target_dir}/*shard*cpp")
    if (NOT _shards)
      message(FATAL_ERROR "no shards created")
    endif()
  endif()

  add_library(${target} STATIC "${_full_gen_file}.sharded.cpp" ${_shards})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --include_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_include_dir
    RESULT_VARIABLE _has_err_find_nanobind
  )
  if ((_has_err_find_nanobind AND NOT _has_err_find_nanobind EQUAL 0) OR NOT EXISTS "${nanobind_include_dir}")
    message(FATAL_ERROR "couldn't find nanobind include dir: ${_has_err_find_nanobind}")
  endif()
  target_include_directories(${target} PRIVATE
    ${eudsl_nbgen_includes}
    ${Python_INCLUDE_DIRS}
    ${nanobind_include_dir}
  )
  # not sure why unix needs this buy not apple (and really only in root-cmake build...)
  if(UNIX AND NOT APPLE)
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
  endif()
  set(_link_libs ${ARG_LINK_LIBS})
  if (NOT ARG_LINK_LIBS)
    set(_link_libs MLIR LLVM)
  endif()
  target_link_libraries(${target} PUBLIC ${_link_libs})
  target_compile_options(${target} PUBLIC -Wno-cast-qual)

  # `make clean' must remove all those generated files:
  # TODO(max): clean up dep files
  set_property(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${_shards} ${_depfile})
  set_source_files_properties(${_shards} PROPERTIES GENERATED 1)
endfunction()

function(patch_mlir_llvm_rpath target)
  cmake_parse_arguments(ARG "STANDALONE" "" "" ${ARGN})
  # hack so we can move libMLIR and libLLVM into the wheel
  # see AddLLVM.cmake#llvm_setup_rpath
  if (APPLE OR UNIX)
    set(_origin_prefix "\$ORIGIN")
    if (APPLE)
      set(_origin_prefix "@loader_path")
    endif()
    if (STANDALONE)
      get_target_property(_mlir_loc MLIR LOCATION)
      get_target_property(_llvm_loc LLVM LOCATION)
    else()
      set(_mlir_loc "$<TARGET_FILE:MLIR>")
      set(_llvm_loc "$<TARGET_FILE:LLVM>")
    endif()
    set(_old_rpath "${_origin_prefix}/../lib${LLVM_LIBDIR_SUFFIX}")
    if (APPLE)
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
