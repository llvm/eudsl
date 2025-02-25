# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

# copy-pasta from AddMLIR.cmake/AddLLVM.cmake/TableGen.cmake

function(eudsl_tblgen target)
  cmake_parse_arguments(ARG "" "TD_FILE;OUTPUT_DIRECTORY;KIND" "INCLUDES;DEPENDS" ${ARGN})
  if (IS_ABSOLUTE ${ARG_TD_FILE})
    set(EUDSL_TBLGEN_TD_FILE_INPUT_ABSOLUTE ${ARG_TD_FILE})
  else()
    set(EUDSL_TBLGEN_TD_FILE_INPUT_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_TD_FILE})
  endif()

  if (NOT EXISTS "${EUDSL_TBLGEN_TD_FILE_INPUT_ABSOLUTE}")
    message(FATAL_ERROR "${ARG_TD_FILE} @ ${EUDSL_TBLGEN_TD_FILE_INPUT_ABSOLUTE} does not exist")
  endif()

  get_directory_property(eudsl_tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND eudsl_tblgen_includes ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
  list(REMOVE_ITEM eudsl_tblgen_includes "")
  list(APPEND eudsl_tblgen_includes ${ARG_INCLUDES})
  # list(TRANSFORM eudsl_tblgen_includes PREPEND -I)

  set(eudsl_tblgen_generate_cmd
    ${Python_EXECUTABLE} -Wignore -m eudsl_tblgen.mlir ${EUDSL_TBLGEN_TD_FILE_INPUT_ABSOLUTE}
    -k ${ARG_KIND} -I ${eudsl_tblgen_includes}
    -o "${ARG_OUTPUT_DIRECTORY}"
  )

  get_filename_component(_prefix ${EUDSL_TBLGEN_TD_FILE_INPUT_ABSOLUTE} NAME_WE)
  set(_output_files
    "${_prefix}_${ARG_KIND}_decls.h.inc"
    "${_prefix}_${ARG_KIND}_defns.cpp.inc"
    "${_prefix}_${ARG_KIND}_nbclasses.cpp.inc"
  )
  list(TRANSFORM _output_files PREPEND "${ARG_OUTPUT_DIRECTORY}/")

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m eudsl_tblgen.mlir --help
    RESULT_VARIABLE _has_err_generate
    OUTPUT_QUIET
    # COMMAND_ECHO STDOUT
  )
  if (_has_err_generate AND NOT _has_err_generate EQUAL 0)
    message(FATAL_ERROR "couldn't generate sources: ${_has_err_generate}")
  endif()
  if (_has_err_generate AND NOT _has_err_generate EQUAL 0)
    ##################################
    # not standalone build
    ##################################

    add_custom_command(OUTPUT ${_output_files}
      COMMAND ${eudsl_tblgen_generate_cmd}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DEPENDS eudsl_tblgen_ext ${ARG_DEPENDS}
      COMMENT "eudsl-tblgen: Generating ${_output_files}..."
    )
  else()
    message(STATUS "found EUDSL_TBLGEN_EXE")
    ##################################
    # standalone build
    ##################################
    execute_process(
      COMMAND ${eudsl_tblgen_generate_cmd}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE _had_error_gen_cpp
      # COMMAND_ECHO STDOUT
    )
    if (_had_error_gen_cpp AND NOT _had_error_gen_cpp EQUAL 0)
      message(FATAL_ERROR "failed to create ${_output_files}: ${_had_error_gen_cpp}")
    endif()
  endif()

  add_custom_target(${target} ALL DEPENDS ${_output_files})

  # `make clean' must remove all those generated files:
  # TODO(max): clean up dep files
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${_output_files})
  set_source_files_properties(${_output_files} PROPERTIES GENERATED 1)
endfunction()
