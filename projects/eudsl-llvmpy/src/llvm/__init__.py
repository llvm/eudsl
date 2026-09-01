#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

# MLIR-style submodule layout: nothing is re-exported at the package top level.
# Use llvm.ir.Context, llvm.types.i32, llvm.passmanager.run_passes,
# llvm.jit.LLJIT, llvm.intrinsics.sqrt, llvm.instructions.load, llvm.dsl.function,
# etc.
from . import (
    ir,
    types,
    passmanager,
    jit,
    intrinsics,
    instructions,
    mir,
    dsl,
)  # noqa: F401

from .dsl.values import install_value_casters as _install_value_casters

_install_value_casters()

# Attaches mir.ReadyQueueStrategy onto the mir submodule.
from . import mir_strategies as _mir_strategies  # noqa: F401

# Attaches mir.RAGreedy onto the mir submodule.
from . import mir_greedy as _mir_greedy  # noqa: F401

# Attaches the ILP register allocators onto the mir submodule. The ortools
# dependency is imported lazily, so importing llvm never requires it.
from . import mir_ilp_base as _mir_ilp_base  # noqa: F401
from . import mir_ilp_assign as _mir_ilp_assign  # noqa: F401
from . import mir_ilp_packing as _mir_ilp_packing  # noqa: F401

from .eudslllvm_ext import enable_debug  # noqa: F401
