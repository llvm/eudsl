#  Copyright (c) 2025.
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path


def cmake_dir() -> str:
    return str(Path(__file__).parent / "cmake")


__all__ = ["cmake_dir"]
