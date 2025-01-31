# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

if(NOT WIN32)
  add_subdirectory(eudsl-py)
endif()
add_subdirectory(eudsl-llvmpy)
add_subdirectory(eudsl-nbgen)
add_subdirectory(eudsl-tblgen)
