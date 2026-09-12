# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import mlir.extras.types as T
from mlir.dialects.linalg import ElementwiseKind
from mlir.extras.dialects import linalg, memref, tensor

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


# The unary/binary linalg named ops (abs, add, ...) were removed upstream in
# favor of the generic linalg.elementwise op; the wrappers below now emit that.
# These tests exercise every rewritten wrapper.


def test_elementwise_unary(ctx: MLIRContext):
    x = memref.alloc((4, 4), T.f32())
    y = memref.alloc((4, 4), T.f32())

    linalg.abs(x, y)
    linalg.ceil(x, y)
    linalg.exp(x, y)
    linalg.floor(x, y)
    linalg.log(x, y)
    linalg.negf(x, y)

    # CHECK: linalg.elementwise <abs> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <ceil> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <exp> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <floor> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <log> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <negf> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    filecheck_with_comments(ctx.module)


def test_elementwise_binary(ctx: MLIRContext):
    x = memref.alloc((4, 4), T.f32())
    y = memref.alloc((4, 4), T.f32())
    z = memref.alloc((4, 4), T.f32())

    linalg.add(x, y, z)
    linalg.sub(x, y, z)
    linalg.mul(x, y, z)
    linalg.div(x, y, z)

    # CHECK: linalg.elementwise <add> ins(%{{.*}}, %{{.*}} : memref<4x4xf32>, memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <sub> ins(%{{.*}}, %{{.*}} : memref<4x4xf32>, memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <mul> ins(%{{.*}}, %{{.*}} : memref<4x4xf32>, memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <div> ins(%{{.*}}, %{{.*}} : memref<4x4xf32>, memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    filecheck_with_comments(ctx.module)


def test_elementwise_binary_int(ctx: MLIRContext):
    x = memref.alloc((4, 4), T.i32())
    y = memref.alloc((4, 4), T.i32())
    z = memref.alloc((4, 4), T.i32())

    linalg.div_unsigned(x, y, z)
    linalg.max(x, y, z)

    # CHECK: linalg.elementwise <div_unsigned> ins(%{{.*}}, %{{.*}} : memref<4x4xi32>, memref<4x4xi32>) outs(%{{.*}} : memref<4x4xi32>)
    # CHECK: linalg.elementwise <max_signed> ins(%{{.*}}, %{{.*}} : memref<4x4xi32>, memref<4x4xi32>) outs(%{{.*}} : memref<4x4xi32>)
    filecheck_with_comments(ctx.module)


def test_elementwise_tensor_result(ctx: MLIRContext):
    x = tensor.empty(4, 4, T.f32())
    y = tensor.empty(4, 4, T.f32())
    out = tensor.empty(4, 4, T.f32())

    r = linalg.add(x, y, out)

    # CHECK: %{{.*}} = linalg.elementwise <add> ins(%{{.*}}, %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>) outs(%{{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
    filecheck_with_comments(ctx.module)


def test_elemwise_generic(ctx: MLIRContext):
    x = memref.alloc((4, 4), T.f32())
    y = memref.alloc((4, 4), T.f32())
    z = memref.alloc((4, 4), T.f32())

    # default kinds preserve the old op defaults (unary -> exp, binary -> add)
    linalg.elemwise_unary(x, y)
    linalg.elemwise_binary(x, y, z)
    # explicit kinds
    linalg.elemwise_unary(x, y, fun=ElementwiseKind.log)
    linalg.elemwise_binary(x, y, z, fun=ElementwiseKind.mul)

    # CHECK: linalg.elementwise <exp> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <add> ins(%{{.*}}, %{{.*}} : memref<4x4xf32>, memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <log> ins(%{{.*}} : memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    # CHECK: linalg.elementwise <mul> ins(%{{.*}}, %{{.*}} : memref<4x4xf32>, memref<4x4xf32>) outs(%{{.*}} : memref<4x4xf32>)
    filecheck_with_comments(ctx.module)


# The transposed / unsigned matmul named ops were removed upstream; the wrappers
# now emit the generic linalg.matmul / batch_matmul with transposed indexing
# maps (or an unsigned cast).


def test_matmul_transpose_a(ctx: MLIRContext):
    # A is transposed: [K, M] @ [K, N] -> [M, N]
    a = memref.alloc((8, 4), T.f32())
    b = memref.alloc((8, 6), T.f32())
    c = memref.alloc((4, 6), T.f32())

    linalg.matmul_transpose_a(a, b, c)

    # CHECK: #[[$MAP_A:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
    # CHECK: linalg.matmul indexing_maps = [#[[$MAP_A]], {{.*}}] ins(%{{.*}}, %{{.*}} : memref<8x4xf32>, memref<8x6xf32>) outs(%{{.*}} : memref<4x6xf32>)
    filecheck_with_comments(ctx.module)


def test_matmul_transpose_b(ctx: MLIRContext):
    # B is transposed: [M, K] @ [N, K] -> [M, N]
    a = memref.alloc((4, 8), T.f32())
    b = memref.alloc((6, 8), T.f32())
    c = memref.alloc((4, 6), T.f32())

    linalg.matmul_transpose_b(a, b, c)

    # CHECK: #[[$MAP_B:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
    # CHECK: linalg.matmul indexing_maps = [{{.*}}, #[[$MAP_B]], {{.*}}] ins(%{{.*}}, %{{.*}} : memref<4x8xf32>, memref<6x8xf32>) outs(%{{.*}} : memref<4x6xf32>)
    filecheck_with_comments(ctx.module)


def test_matmul_unsigned(ctx: MLIRContext):
    a = memref.alloc((4, 8), T.i32())
    b = memref.alloc((8, 6), T.i32())
    c = memref.alloc((4, 6), T.i32())

    linalg.matmul_unsigned(a, b, c)

    # CHECK: linalg.matmul
    # CHECK-SAME: cast_unsigned
    filecheck_with_comments(ctx.module)


def test_batch_matmul_transpose_a(ctx: MLIRContext):
    # A is transposed: [B, K, M] @ [B, K, N] -> [B, M, N]
    a = memref.alloc((2, 8, 4), T.f32())
    b = memref.alloc((2, 8, 6), T.f32())
    c = memref.alloc((2, 4, 6), T.f32())

    linalg.batch_matmul_transpose_a(a, b, c)

    # CHECK: #[[$BMAP_A:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
    # CHECK: linalg.batch_matmul indexing_maps = [#[[$BMAP_A]], {{.*}}] ins(%{{.*}}, %{{.*}} : memref<2x8x4xf32>, memref<2x8x6xf32>) outs(%{{.*}} : memref<2x4x6xf32>)
    filecheck_with_comments(ctx.module)


def test_batch_matmul_transpose_b(ctx: MLIRContext):
    # B is transposed: [B, M, K] @ [B, N, K] -> [B, M, N]
    a = memref.alloc((2, 4, 8), T.f32())
    b = memref.alloc((2, 6, 8), T.f32())
    c = memref.alloc((2, 4, 6), T.f32())

    linalg.batch_matmul_transpose_b(a, b, c)

    # CHECK: #[[$BMAP_B:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
    # CHECK: linalg.batch_matmul indexing_maps = [{{.*}}, #[[$BMAP_B]], {{.*}}] ins(%{{.*}}, %{{.*}} : memref<2x4x8xf32>, memref<2x6x8xf32>) outs(%{{.*}} : memref<2x4x6xf32>)
    filecheck_with_comments(ctx.module)
