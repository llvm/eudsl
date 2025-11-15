import pytest
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck_with_comments,
    filecheck,
    MLIRContext,
)
from mlir import ir
from mlir.dialects import func
from mlir.extras import types as T

pytest.mark.usefixtures("ctx")


def test_testing(ctx: MLIRContext):
    @func.FuncOp.from_py_func(T.i32(), T.i32())
    def foo(a, b):
        return

    # CHECK: func.func @foo(%arg0: i32, %arg1: i32) {
    # CHECK:   return
    # CHECK: }
    filecheck_with_comments(ctx.module)
