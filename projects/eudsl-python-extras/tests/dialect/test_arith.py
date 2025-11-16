from mlir.extras.dialects.ext import arith
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)

def test_arith_value(ctx: MLIRContext):
    c = arith.constant(1)
    print(ctx.module)

    