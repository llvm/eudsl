# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from mlir.extras.dialects import arith
from mlir.extras.testing import (
    filecheck_with_comments,
    MLIRContext,
    mlir_mod_ctx,
)
from mlir.dialects import func
from mlir.extras import types as T


def test_arith_value(ctx: MLIRContext):
    i32 = T.i32()

    @func.FuncOp.from_py_func(i32, i32)
    def foo(a: i32, b: i32) -> i32:
        c1 = arith.constant(1)
        return c1 * a + b - c1 / a // c1 % b

    # CHECK-LABEL:   func.func @foo(
    # CHECK-SAME:                   %[[ARG0:.*]]: i32,
    # CHECK-SAME:                   %[[ARG1:.*]]: i32) -> i32 {
    # CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
    # CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_0]], %[[ARG0]] : i32
    # CHECK:           %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG1]] : i32
    # CHECK:           %[[DIVSI_0:.*]] = arith.divsi %[[CONSTANT_0]], %[[ARG0]] : i32
    # CHECK:           %[[FLOORDIVSI_0:.*]] = arith.floordivsi %[[DIVSI_0]], %[[CONSTANT_0]] : i32
    # CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[FLOORDIVSI_0]], %[[ARG1]] : i32
    # CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ADDI_0]], %[[REMSI_0]] : i32
    # CHECK:           return %[[SUBI_0]] : i32
    # CHECK:         }
    filecheck_with_comments(ctx.module)


if __name__ == "__main__":
    with mlir_mod_ctx() as c:
        test_arith_value(c)
