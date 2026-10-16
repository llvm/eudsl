# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.29)
project(orc_pgo_multi_tu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MULTI_TU_ORC_PGO "Built by orc-pgo: poll the runtime and print its stats" OFF)
set(ORC_PGO_INCLUDE_DIR "" CACHE PATH "orc-pgo's include directory (orc-pgo/Stats.h)")

add_library(shapes OBJECT shapes.cpp)
add_library(stats OBJECT stats.cpp)
add_executable(multi_tu main.cpp $<TARGET_OBJECTS:shapes> $<TARGET_OBJECTS:stats>)

if(MULTI_TU_ORC_PGO)
  target_compile_definitions(multi_tu PRIVATE MULTI_TU_ORC_PGO=1)
  target_include_directories(multi_tu PRIVATE "${ORC_PGO_INCLUDE_DIR}")
  # Prints the native link command, which run.cmake inspects.
  target_link_options(multi_tu PRIVATE -v)
endif()
