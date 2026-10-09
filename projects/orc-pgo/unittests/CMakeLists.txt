# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(OrcPgoTestSupport STATIC TestSupport.cpp)
target_link_libraries(OrcPgoTestSupport PUBLIC OrcPgo GTest::gtest)
target_include_directories(OrcPgoTestSupport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

function(orc_pgo_add_unittest name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE OrcPgoTestSupport GTest::gtest_main)
  orc_pgo_add_coverage(${name})
  orc_pgo_coverage_object(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

orc_pgo_add_unittest(CounterLayoutTest CounterLayoutTest.cpp)
