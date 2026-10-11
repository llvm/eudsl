# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(OrcPgoTestSupport STATIC TestSupport.cpp SimulatedAot.cpp)
target_link_libraries(OrcPgoTestSupport PUBLIC OrcPgoCommon OrcPgoBuild OrcPgoRuntime GTest::gtest)
target_include_directories(OrcPgoTestSupport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

function(orc_pgo_add_unittest name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE OrcPgoTestSupport GTest::gtest_main)
  orc_pgo_add_coverage(${name})
  orc_pgo_coverage_object(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

orc_pgo_add_unittest(PgoOptionsTest PgoOptionsTest.cpp)
orc_pgo_add_unittest(SnapshotTest SnapshotTest.cpp)
orc_pgo_add_unittest(ValueProfRuntimeTest ValueProfRuntimeTest.cpp)
orc_pgo_add_unittest(InstrumentTest InstrumentTest.cpp)
orc_pgo_add_unittest(ProfileBuilderTest ProfileBuilderTest.cpp)
orc_pgo_add_unittest(AnnotateTest AnnotateTest.cpp)
orc_pgo_add_unittest(EligibilityTest EligibilityTest.cpp)
orc_pgo_add_unittest(ThunksTest ThunksTest.cpp)
orc_pgo_add_unittest(SymbolTableTest SymbolTableTest.cpp)
orc_pgo_add_unittest(PipelineTest PipelineTest.cpp)
orc_pgo_add_unittest(TierUpTest TierUpTest.cpp)
orc_pgo_add_unittest(RuntimeTest RuntimeTest.cpp)
target_compile_options(RuntimeTest PRIVATE -fexceptions)
orc_pgo_add_unittest(RuntimeEntryTest RuntimeEntryTest.cpp)
