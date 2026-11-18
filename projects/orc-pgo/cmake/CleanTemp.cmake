# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# cmake -DDIR=<dir> -P CleanTemp.cmake
#
# Empties <dir>, creating it if needed. It is the TMPDIR of the tests that run
# the driver, so that what an aborted or killed run leaves behind is removed by
# the next run instead of accumulating in the system's temporary directory.

if(NOT DEFINED DIR)
  message(FATAL_ERROR "CleanTemp.cmake: -DDIR=... is required")
endif()
file(REMOVE_RECURSE "${DIR}")
file(MAKE_DIRECTORY "${DIR}")
