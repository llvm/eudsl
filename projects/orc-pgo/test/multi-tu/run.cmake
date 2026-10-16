# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# cmake -DORC_PGO=<orc-pgo> -DEXTRA_LINK_INPUT=<path or empty> -DORC_PGO_INCLUDE=<dir>
#       -DSRC=<dir> -DBIN=<dir> -P run.cmake
#
# EXTRA_LINK_INPUT is Config.h's ORC_PGO_EXTRA_LINK_INPUT. It is non-empty only
# in coverage builds, where apps must link the coverage profile runtime. That
# runtime does initialize in the app: it reads the app's __llvm_prf_* sections
# (orc-pgo's instrumentation, a newer raw version than the coverage compiler's),
# complains about the version mismatch and may write a .profraw at exit. The
# app runs are therefore given their own LLVM_PROFILE_FILE, inside ${BIN}, so
# that nothing they write can reach the coverage gate's profraw directory.
#
# Builds test/multi-tu with orc-pgo as the C++ compiler, then checks the
# program's output, its tier-up and value-profiling stats, its dynamic
# dependencies, and the profile runtime on the link: none, or in coverage
# builds only EXTRA_LINK_INPUT.

foreach(var ORC_PGO EXTRA_LINK_INPUT ORC_PGO_INCLUDE SRC BIN)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "run.cmake: -D${var}=... is required")
  endif()
endforeach()

if(CMAKE_HOST_APPLE)
  find_program(DEPS_TOOL otool REQUIRED)
  set(deps_args -L)
  set(libc_pattern "libSystem")
else()
  find_program(DEPS_TOOL readelf REQUIRED)
  set(deps_args -d)
  set(libc_pattern "libc\\.so")
endif()
find_program(NM_TOOL nm REQUIRED)
file(READ "${SRC}/expected.txt" expected)

function(run_variant variant)
  set(bin "${BIN}/${variant}")
  set(app "${bin}/multi_tu")
  file(REMOVE_RECURSE "${bin}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -G Ninja -S "${SRC}" -B "${bin}"
            -DCMAKE_BUILD_TYPE=Release -DMULTI_TU_ORC_PGO=ON
            "-DORC_PGO_INCLUDE_DIR=${ORC_PGO_INCLUDE}" ${ARGN}
    RESULT_VARIABLE rc OUTPUT_VARIABLE log ERROR_VARIABLE log)
  if(rc)
    message(FATAL_ERROR "[${variant}] configure failed (${rc}):\n${log}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build "${bin}"
                  RESULT_VARIABLE rc OUTPUT_VARIABLE build_log ERROR_VARIABLE build_log)
  if(rc)
    message(FATAL_ERROR "[${variant}] build failed (${rc}):\n${build_log}")
  endif()

  # The native link line (printed by -v) has the orc-pgo runtime.
  string(REGEX MATCH "[^\n]*libOrcPgoRuntime\\.a[^\n]*" ld_line "${build_log}")
  if(ld_line STREQUAL "")
    message(FATAL_ERROR "[${variant}] no link line with libOrcPgoRuntime.a:\n${build_log}")
  endif()
  execute_process(COMMAND "${NM_TOOL}" "${app}" OUTPUT_VARIABLE syms RESULT_VARIABLE rc)
  if(rc OR NOT syms MATCHES "__orc_pgo_start")
    message(FATAL_ERROR "[${variant}] no __orc_pgo_start (rc=${rc}):\n${syms}")
  endif()
  if(EXTRA_LINK_INPUT STREQUAL "")
    # (e) No compiler-rt profile runtime on the link line or in the app.
    if(ld_line MATCHES "libclang_rt\\.profile" OR syms MATCHES "__llvm_profile_write_file")
      message(FATAL_ERROR "[${variant}] linked compiler-rt's profile runtime:\n${ld_line}")
    endif()
  else()
    # Coverage build: (e) doesn't apply. The instrumented runtime archives need
    # the coverage compiler's profile runtime, and it must be the only one.
    string(FIND "${ld_line}" "${EXTRA_LINK_INPUT}" at)
    string(REGEX MATCHALL "libclang_rt\\.profile" profiles "${ld_line}")
    list(LENGTH profiles n_profiles)
    if(at EQUAL -1 OR NOT n_profiles EQUAL 1)
      message(FATAL_ERROR "[${variant}] expected exactly ${EXTRA_LINK_INPUT}:\n${ld_line}")
    endif()
  endif()

  # (d) Only system dylibs: no LLVM or clang shared library.
  execute_process(COMMAND "${DEPS_TOOL}" ${deps_args} "${app}"
                  OUTPUT_VARIABLE deps RESULT_VARIABLE rc)
  if(rc OR deps MATCHES "libLLVM|libclang" OR NOT deps MATCHES "${libc_pattern}")
    message(FATAL_ERROR "[${variant}] unexpected dynamic dependencies (rc=${rc}):\n${deps}")
  endif()

  foreach(mode jit disabled)
    # Hermetic in coverage builds: see the top of this file.
    set(env ORC_PGO_THRESHOLD=1000 "LLVM_PROFILE_FILE=${bin}/app-%p.profraw")
    if(mode STREQUAL "disabled")
      list(APPEND env ORC_PGO_DISABLE=1)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -E env ${env} "${app}"
                    OUTPUT_VARIABLE out ERROR_VARIABLE err RESULT_VARIABLE rc)
    if(rc)
      message(FATAL_ERROR "[${variant}/${mode}] exit ${rc}\nstdout:\n${out}\nstderr:\n${err}")
    endif()
    # (b) Same output as the AOT-only program.
    if(NOT out STREQUAL expected)
      message(FATAL_ERROR "[${variant}/${mode}] stdout:\n${out}\nexpected:\n${expected}")
    endif()
    if(NOT err MATCHES "orc-pgo-stats: tiered_up=([0-9]+) failed=([0-9]+) promoted=([0-9]+)")
      message(FATAL_ERROR "[${variant}/${mode}] no stats line:\n${err}")
    endif()
    set(tiered ${CMAKE_MATCH_1})
    set(failed ${CMAKE_MATCH_2})
    set(promoted ${CMAKE_MATCH_3})
    if(mode STREQUAL "jit")
      # (a) scaled and mulDiv were tiered up, and (c) value profiling promoted
      # the hot indirect calls in scaled.
      if(tiered LESS 2 OR NOT failed EQUAL 0 OR promoted LESS 1)
        message(FATAL_ERROR "[${variant}/jit] stats:\n${err}")
      endif()
    elseif(NOT tiered EQUAL 0 OR NOT failed EQUAL 0 OR NOT promoted EQUAL 0)
      message(FATAL_ERROR "[${variant}/disabled] the runtime ran:\n${err}")
    endif()
  endforeach()
endfunction()

run_variant(object "-DCMAKE_CXX_COMPILER=${ORC_PGO}")
