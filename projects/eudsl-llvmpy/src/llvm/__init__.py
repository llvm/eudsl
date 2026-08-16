#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

# MLIR-style submodule layout: nothing is re-exported at the package top level.
# Use llvm.ir.Context, llvm.types.i32, llvm.passmanager.run_passes,
# llvm.jit.LLJIT, llvm.intrinsics.sqrt, llvm.instructions.load, llvm.dsl.function,
# etc.
from . import ir, types, passmanager, jit, intrinsics, instructions, dsl  # noqa: F401

from .dsl.values import install_value_casters as _install_value_casters

_install_value_casters()

