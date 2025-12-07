# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2025.

cmake_minimum_required(VERSION 3.29)
project(mlir-webgpu LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0116 OLD)
endif()

set(DAWN_FETCH_DEPENDENCIES ON)
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/dawn"
  EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_BINARY_DIR}/dawn
)
if(NOT Dawn_SOURCE_DIR OR NOT EXISTS ${Dawn_SOURCE_DIR})
  message(FATAL_ERROR "failed to configure Dawn dependency")
endif()
set(DAWN_SOURCE_DIR ${Dawn_SOURCE_DIR})

# https://github.com/kainino0x/webgpu-cross-platform-demo
add_executable(dawn_app shader_test.cpp)
target_compile_definitions(dawn_app PRIVATE DEMO_USE_ASYNCIFY=1)

if(EMSCRIPTEN)
  set_target_properties(dawn_app PROPERTIES SUFFIX ".html")
  target_link_libraries(dawn_app PRIVATE emdawnwebgpu_cpp webgpu_glfw)
  target_link_options(dawn_app PRIVATE
    "-sASYNCIFY=1"
    "-sUSE_GLFW=3"
    "-sASYNCIFY_STACK_SIZE=65536"
    "-sEXPORTED_RUNTIME_METHODS=ccall"
  )
else()
  target_link_libraries(dawn_app PRIVATE webgpu_dawn webgpu_glfw glfw)
endif()
