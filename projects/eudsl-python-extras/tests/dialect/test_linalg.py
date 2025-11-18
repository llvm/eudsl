# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from typing import TypeVar

import mlir.extras.types as T
import pytest

from mlir.extras.dialects import linalg, memref, tensor
from mlir.ir import ShapedType

from mlir.extras.dialects.func import func

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    MLIRContext,
    filecheck,
    filecheck_with_comments,
    mlir_ctx as ctx,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    x = memref.alloc((10, 10), T.i32())
    linalg.fill(5, x)
    linalg.fill_rng_2d(0.0, 10.0, 1, x)

    x = tensor.empty(10, 10, T.i32())
    y = linalg.fill_rng_2d(0.0, 10.0, 1, x)
    z = linalg.fill(5, x)

    # CHECK:  %[[VAL_0:.*]] = memref.alloc() : memref<10x10xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 5 : i32
    # CHECK:  linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_0]] : memref<10x10xi32>)
    # CHECK:  %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f64
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1.000000e+01 : f64
    # CHECK:  %[[VAL_4:.*]] = arith.constant 1 : i32
    # CHECK:  linalg.fill_rng_2d ins(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : f64, f64, i32) outs(%[[VAL_0]] : memref<10x10xi32>)
    # CHECK:  %[[VAL_5:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:  %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f64
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1.000000e+01 : f64
    # CHECK:  %[[VAL_8:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_9:.*]] = linalg.fill_rng_2d ins(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : f64, f64, i32) outs(%[[VAL_5]] : tensor<10x10xi32>) -> tensor<10x10xi32>
    # CHECK:  %[[VAL_10:.*]] = arith.constant 5 : i32
    # CHECK:  %[[VAL_11:.*]] = linalg.fill ins(%[[VAL_10]] : i32) outs(%[[VAL_5]] : tensor<10x10xi32>) -> tensor<10x10xi32>

    filecheck_with_comments(ctx.module)


def test_pooling_ncdhw_max(ctx: MLIRContext):
    S = ShapedType.get_dynamic_size()

    generics = (
        kernel_size_0,
        kernel_size_1,
        kernel_size_2,
        stride_0,
        stride_1,
        stride_2,
        dilation_0,
        dilation_1,
        dilation_2,
    ) = list(
        map(
            TypeVar,
            [
                "kernel_size_0",
                "kernel_size_1",
                "kernel_size_2",
                "stride_0",
                "stride_1",
                "stride_2",
                "dilation_0",
                "dilation_1",
                "dilation_2",
            ],
        )
    )

    @func(
        generics=(
            kernel_size_0,
            kernel_size_1,
            kernel_size_2,
            stride_0,
            stride_1,
            stride_2,
            dilation_0,
            dilation_1,
            dilation_2,
        )
    )
    def maxpool3d(
        input: T.memref(S, S, S, S, S, T.f32()),
        output: T.memref(S, S, S, S, S, T.f32()),
    ):
        kernel_shape_surrogate = memref.alloca(
            (kernel_size_0, kernel_size_1, kernel_size_2),
            T.f32(),
        )

        linalg.pooling_ncdhw_max(
            input,
            kernel_shape_surrogate,
            output,
            strides=[stride_0, stride_1, stride_2],
            dilations=[dilation_0, dilation_1, dilation_2],
        )

    kernel_sizes = [1, 2, 2]
    strides = [1, 1, 1]
    dilations = [1, 1, 1]
    maxpool3d_k = maxpool3d[
        kernel_sizes[0],
        kernel_sizes[1],
        kernel_sizes[2],
        strides[0],
        strides[1],
        strides[2],
        dilations[0],
        dilations[1],
        dilations[2],
    ].emit()
    # CHECK: func.func @maxpool3d_int_1_int_2_int_2_int_1_int_1_int_1_int_1_int_1_int_1(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>) {
    # CHECK:   %alloca = memref.alloca() : memref<1x2x2xf32>
    # CHECK:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2 + d5, d3 + d6, d4 + d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %alloca : memref<?x?x?x?x?xf32>, memref<1x2x2xf32>) outs(%arg1 : memref<?x?x?x?x?xf32>) {
    # CHECK:   ^bb0(%in: f32, %in_0: f32, %out: f32):
    # CHECK:     %0 = arith.maximumf %in, %out : f32
    # CHECK:     linalg.yield %0 : f32
    # CHECK:   }
    # CHECK:   return
    # CHECK: }
    filecheck_with_comments(maxpool3d_k)
