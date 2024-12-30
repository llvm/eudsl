# copy-pasta from AddMLIR.cmake/AddLLVM.cmake/TableGen.cmake

# Clear out any pre-existing compile_commands file before processing. This
# allows for generating a clean compile_commands on each configure.
file(REMOVE ${CMAKE_BINARY_DIR}/eudslpygen_compile_commands.yml)

# no clue why but with LLVM_LINK_LLVM_DYLIB even static libs depend on LLVM
get_property(MLIR_ALL_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
foreach(_lib ${MLIR_ALL_LIBS})
  get_target_property(_interface_link_libraries ${_lib} INTERFACE_LINK_LIBRARIES)
  if(NOT _interface_link_libraries)
    continue()
  endif()
  list(REMOVE_DUPLICATES _interface_link_libraries)
  list(REMOVE_ITEM _interface_link_libraries LLVM)
  # for some reason, explicitly adding below as a link library doesn't work - missing symbols...
  list(APPEND _interface_link_libraries LLVMSupport)
  set_target_properties(${_lib} PROPERTIES INTERFACE_LINK_LIBRARIES "${_interface_link_libraries}")
endforeach()

function(add_public_eudslpygen_target target eudslpygen_output)
  if(NOT eudslpygen_output)
    message(FATAL_ERROR "Requires eudslpygen() definitions as EUDSLPYGEN_OUTPUT.")
  endif()
  add_custom_target(${target}
    DEPENDS ${eudslpygen_output})
  if(LLVM_COMMON_DEPENDS)
    add_dependencies(${target} ${LLVM_COMMON_DEPENDS})
  endif()
  get_subproject_title(subproject_title)
  set_target_properties(${target} PROPERTIES FOLDER "${subproject_title}/Eudslpygenning")
endfunction()

function(eudslpygen target inputFile)
  set(EUDSLPYGEN_TARGET_DEFINITIONS ${inputFile})
  # Get the current set of include paths for this source file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES;NAMESPACES" ${ARGN})
  # Build the absolute path for the current input file.
  if (IS_ABSOLUTE ${EUDSLPYGEN_TARGET_DEFINITIONS})
    set(EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE ${inputFile})
  else()
    set(EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${inputFile})
  endif()

  if(NOT EXISTS "${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}")
    message(FATAL_ERROR "${inputFile} does not exist")
  endif()

  # message(FATAL_ERROR "${CMAKE_CXX_COMPILER}")
  get_directory_property(eudslpygen_includes INCLUDE_DIRECTORIES)
  list(APPEND eudslpygen_includes ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES} ${ARG_EXTRA_INCLUDES})
  # Filter out empty items before prepending each entry with -I
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
  set(clang_command ${CMAKE_CXX_COMPILER} -v -xc++ "-std=c++${CMAKE_CXX_STANDARD}"
    -MM ${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}
    -MT ${fullGenFile_rel}
    ${eudslpygen_includes}
    -o ${_depfile})
  execute_process(COMMAND ${clang_command} RESULT_VARIABLE had_error COMMAND_ECHO STDERR)
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

  add_custom_command(OUTPUT "${fullGenFile}.sharded.cpp"
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/make_generated_registration.py
      ${fullGenFile} -t ${target} -I ${ARG_EXTRA_INCLUDES} ${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${fullGenFile}
    COMMENT "EUDSLPY: Generating ${fullGenFile}.sharded.cpp..."
  )

  # this is the specific thing connected the dependencies...
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${fullGenFile}.sharded.cpp")
  file(GLOB _generated_shards "${_gen_target_dir}/*.shard.*")
  list(APPEND _generated_shards "${fullGenFile}.sharded.cpp")
  set(${target}_GENERATED_SHARDS ${_generated_shards} PARENT_SCOPE)

  # `make clean' must remove all those generated files:
  # TODO(max): clean up dep files
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${fullGenFile})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${fullGenFile} PROPERTIES
    GENERATED 1)

  # Append the includes used for this file to the pdll_compilation_commands
  # file.
  file(APPEND ${CMAKE_BINARY_DIR}/eudslpygen_compile_commands.yml
    "--- !FileInfo:\n"
    "  filepath: \"${EUDSLPYGEN_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${eudslpygen_includes}\"\n"
  )

  add_public_eudslpygen_target(${target} "${fullGenFile}.sharded.cpp;${_generated_shards}")
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
    # Internal eudslpygen
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
