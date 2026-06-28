# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mlir.dialects import func
from mlir.extras import types as T
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)

pytest.mark.usefixtures("ctx")


def test_testing(ctx: MLIRContext):
    @func.FuncOp.from_py_func(T.i32(), T.i32())
    def foo(a, b):
        return

    # CHECK: func.func @foo(%arg0: i32, %arg1: i32) {
    # CHECK:   return
    # CHECK: }
    filecheck_with_comments(ctx.module)


def test_filecheck_failure(ctx: MLIRContext):
    @func.FuncOp.from_py_func(T.i32())
    def bar(a):
        return

    with pytest.raises(ValueError):
        filecheck("// CHECK: this_will_not_match_anything", ctx.module)


def test_filecheck_with_comments_failure(ctx: MLIRContext):
    """Lines 132-133: filecheck_with_comments raises ValueError on mismatch"""

    @func.FuncOp.from_py_func(T.i32())
    def baz(a):
        return

    # CHECK: this_will_definitely_not_match_the_actual_ir_output_xyz123
    with pytest.raises(ValueError):
        filecheck_with_comments(ctx.module)
