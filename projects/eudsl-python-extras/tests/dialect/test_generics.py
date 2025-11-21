# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
from textwrap import dedent
from typing import TypeVar

import mlir.extras.types as T
import pytest
from mlir.ir import ShapedType

from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.ast.py_type import PyTypeVarObject
from mlir.extras.dialects import linalg, arith, scf, memref, gpu
from mlir.extras.dialects.func import func
from mlir.extras.dialects.gpu import (
    set_container_module,
    module,
)
from mlir.extras.runtime.passes import run_pipeline, Pipeline

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_func_no_context_2(ctx: MLIRContext):
    @func
    def matmul_i32_i32[M, N](
        A: "T.memref(M, N, T.i32())",
        B: "T.memref(M, N, T.i32())",
        C: "T.memref(M, N, T.i32())",
    ):
        linalg.matmul(A, B, C)

    matmul_i32_i32[16, 16].emit()

    # CHECK:  func.func @matmul_i32_i32_int_16_int_16(%[[VAL_0:.*]]: memref<16x16xi32>, %[[VAL_1:.*]]: memref<16x16xi32>, %[[VAL_2:.*]]: memref<16x16xi32>) {
    # CHECK:    linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%[[VAL_0]], %[[VAL_1]] : memref<16x16xi32>, memref<16x16xi32>) outs(%[[VAL_2]] : memref<16x16xi32>)
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_generics_just_args(ctx: MLIRContext):
    @func
    def mat_product_kernel[M, K, N, dtype](
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        one = arith.constant(1.0, dtype)

    mat_product_kernel[32, 32, 32, T.f32()].emit()

    # CHECK:  func.func @mat_product_kernel_int_32_int_32_int_32_type_f32(%[[VAL_0:.*]]: memref<32x32xf32>, %[[VAL_1:.*]]: memref<32x32xf32>, %[[VAL_2:.*]]: memref<32x32xf32>) {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_generics_closure(ctx: MLIRContext):
    @func
    def mat_product_kernel[M, K, N, dtype](
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        one = arith.constant(1, dtype)

    mat_product_kernel[32, 32, 32, T.i32()].emit()
    mat_product_kernel[32, 32, 32, T.f32()].emit()

    # CHECK:  func.func @mat_product_kernel_int_32_int_32_int_32_type_i32(%[[VAL_0:.*]]: memref<32x32xi32>, %[[VAL_1:.*]]: memref<32x32xi32>, %[[VAL_2:.*]]: memref<32x32xi32>) {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 1 : i32
    # CHECK:    return
    # CHECK:  }
    # CHECK:  func.func @mat_product_kernel_int_32_int_32_int_32_type_f32(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
    # CHECK:    %cst = arith.constant 1.000000e+00 : f32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_generics_callable(ctx: MLIRContext):
    _op = TypeVar("_op")

    @func
    def mat_product_kernel1[_op]():
        one = arith.constant(1, T.f32())
        two = _op(one, one)

    @func
    def mat_product_kernel2[_op]():
        one = arith.constant(1, T.f32())
        two = _op(one, one)

    mat_product_kernel1[arith.maximumf,].emit()
    mat_product_kernel2[arith.minimumf,].emit()
    mat_product_kernel2[arith.maximumf,].emit()

    # CHECK:  func.func @mat_product_kernel1_function_maximumf() {
    # CHECK:    %cst = arith.constant 1.000000e+00 : f32
    # CHECK:    %0 = arith.maximumf %cst, %cst : f32
    # CHECK:    return
    # CHECK:  }
    # CHECK:  func.func @mat_product_kernel2_function_minimumf() {
    # CHECK:    %cst = arith.constant 1.000000e+00 : f32
    # CHECK:    %0 = arith.minimumf %cst, %cst : f32
    # CHECK:    return
    # CHECK:  }
    # CHECK:  func.func @mat_product_kernel2_function_maximumf() {
    # CHECK:    %cst = arith.constant 1.000000e+00 : f32
    # CHECK:    %0 = arith.maximumf %cst, %cst : f32
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_generics_with_canonicalizations(ctx: MLIRContext):
    @func
    @canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
    def mat_product_kernel[M, K, N, dtype](
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        x = arith.constant(1, index=True)
        y = arith.constant(1, index=True)
        one = arith.constant(1.0, type=dtype)
        tmp = arith.constant(0, type=dtype)
        for k, tmp, _ in scf.range_(K, iter_args=[tmp]):
            tmp += A[x, k] * B[k, y]
            tmp = yield tmp
        C[x, y] = tmp + one

    mat_product_kernel[32, 32, 32, T.f32()].emit()

    # CHECK:  func.func @mat_product_kernel_int_32_int_32_int_32_type_f32(%[[VAL_0:.*]]: memref<32x32xf32>, %[[VAL_1:.*]]: memref<32x32xf32>, %[[VAL_2:.*]]: memref<32x32xf32>) {
    # CHECK:    %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_4:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:    %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
    # CHECK:    %[[VAL_7:.*]] = arith.constant 0 : index
    # CHECK:    %[[VAL_8:.*]] = arith.constant 32 : index
    # CHECK:    %[[VAL_9:.*]] = arith.constant 1 : index
    # CHECK:    %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_7]] to %[[VAL_8]] step %[[VAL_9]] iter_args(%[[VAL_12:.*]] = %[[VAL_6]]) -> (f32) {
    # CHECK:      %[[VAL_13:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_11]]] : memref<32x32xf32>
    # CHECK:      %[[VAL_14:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_11]], %[[VAL_4]]] : memref<32x32xf32>
    # CHECK:      %[[VAL_15:.*]] = math.fma %[[VAL_13]], %[[VAL_14]], %[[VAL_12]] : f32
    # CHECK:      scf.yield %[[VAL_15]] : f32
    # CHECK:    }
    # CHECK:    %[[VAL_16:.*]] = arith.addf %[[VAL_17:.*]], %[[VAL_5]] : f32
    # CHECK:    memref.store %[[VAL_16]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_4]]] : memref<32x32xf32>
    # CHECK:    return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_generics_assignment(ctx: MLIRContext):
    @func
    def type_bound[M, K, N: T.i32()](
        A: "T.memref(M, K, T.f32())",
        B: "T.memref(K, N, T.f32())",
        C: "T.memref(M, N, T.f32())",
    ):
        x = arith.constant(1, index=True)
        y = arith.constant(1, index=True)

    @func
    def type_bound_and_default[M, K, N: T.i32() = 10, L: T.f32() = 10.0](
        A: "T.memref(M, K, T.f32())",
        B: "T.memref(K, N, T.f32())",
        C: "T.memref(M, N, T.f32())",
    ):
        x = arith.constant(1, index=True)
        y = arith.constant(1, index=True)
        n = arith.constant(L)

    # CHECK: func.func @type_bound_int_32_int_32_i32_10(%arg0: memref<32x32xf32>, %arg1: memref<32x10xf32>, %arg2: memref<32x10xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   return
    # CHECK: }
    type_bound[32, 32, 10].emit()
    # CHECK: func.func @type_bound_and_default_int_32_int_32_i32_10_f32_10.0(%arg0: memref<32x32xf32>, %arg1: memref<32x10xf32>, %arg2: memref<32x10xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 1.000000e+01 : f32
    # CHECK:   return
    # CHECK: }
    type_bound_and_default[32, 32].emit()


def test_name_mangling(ctx: MLIRContext):
    _S = ShapedType.get_dynamic_size()

    @func
    def maxpool2d[
        kernel_size_0, kernel_size_1, stride_0, stride_1, dilation_0, dilation_1
    ](
        input: T.memref(_S, _S, _S, _S, T.f32()),
        output: T.memref(_S, _S, _S, _S, T.f32()),
    ):
        kernel_shape_surrogate = memref.alloca(
            (kernel_size_0, kernel_size_1),
            T.f32(),
        )

        linalg.pooling_nchw_max(
            input,
            kernel_shape_surrogate,
            output,
            strides=[stride_0, stride_1],
            dilations=[dilation_0, dilation_1],
        )

    maxpool2d_k = maxpool2d[2, 2, 1, 1, 1, 1].emit()
    # CHECK: func.func @maxpool2d_int_2_int_2_int_1_int_1_int_1_int_1(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>) {
    # CHECK:   %alloca = memref.alloca() : memref<2x2xf32>
    # CHECK:   linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %alloca : memref<?x?x?x?xf32>, memref<2x2xf32>) outs(%arg1 : memref<?x?x?x?xf32>)
    # CHECK:   return
    # CHECK: }
    filecheck_with_comments(maxpool2d_k)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="requires python3.12 or higher")
def test_generics(ctx: MLIRContext):
    set_container_module(ctx.module)

    @gpu.func
    def mat_product_kernel[M, K, N, dtype](
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        x = gpu.block_dim.x * gpu.block_idx.x + gpu.thread_idx.x
        y = gpu.block_dim.y * gpu.block_idx.y + gpu.thread_idx.y

        one = arith.constant(1.0, type=dtype)
        tmp = arith.constant(0, type=dtype)
        for k, tmp, _ in scf.range_(K, iter_args=[tmp]):
            tmp += A[x, k] * B[k, y]
            tmp = scf.yield_(tmp)
        C[x, y] = tmp + one

    @module("naive", ["#nvvm.target"])
    def _():
        mat_product_kernel[32, 32, 32, T.f32()].emit()  # noqa: F821

    correct = dedent(
        """\
    module attributes {gpu.container_module} {
      gpu.module @naive [#nvvm.target]  {
        gpu.func @mat_product_kernel_int_32_int_32_int_32_type_f32(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) kernel {
          %block_dim_x = gpu.block_dim  x
          %block_id_x = gpu.block_id  x
          %0 = arith.muli %block_dim_x, %block_id_x : index
          %thread_id_x = gpu.thread_id  x
          %1 = arith.addi %0, %thread_id_x : index
          %block_dim_y = gpu.block_dim  y
          %block_id_y = gpu.block_id  y
          %2 = arith.muli %block_dim_y, %block_id_y : index
          %thread_id_y = gpu.thread_id  y
          %3 = arith.addi %2, %thread_id_y : index
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1 = arith.constant 1 : index
          %4 = scf.for %arg3 = %c0 to %c32 step %c1 iter_args(%arg4 = %cst_0) -> (f32) {
            %6 = memref.load %arg0[%1, %arg3] : memref<32x32xf32>
            %7 = memref.load %arg1[%arg3, %3] : memref<32x32xf32>
            %8 = arith.mulf %6, %7 : f32
            %9 = arith.addf %arg4, %8 : f32
            scf.yield %9 : f32
          }
          %5 = arith.addf %4, %cst : f32
          memref.store %5, %arg2[%1, %3] : memref<32x32xf32>
          gpu.return
        }
      }
    }
    """
    )

    filecheck(correct, ctx.module)


def test_generic_type_var_closure_patching(ctx: MLIRContext):
    def fun2[foo, bar, A: foo + bar]():
        print(A.__bound__)

    A_type_param = fun2.__type_params__[2]

    a = PyTypeVarObject.from_object(A_type_param)
    a_something = a.bound.contents.into_object()
    a_something.__closure__[0].cell_contents = 5
    a_something.__closure__[1].cell_contents = 7

    fun2()


def test_generic_type_var_closure_patching_dependent_generics(ctx: MLIRContext):
    @gpu.func
    def test_plain[M, K, N, dtype, A_t = T.memref(
        M, K, dtype
    ), B_t = T.memref(K, N, dtype), C_t = T.memref(M, N, dtype),](
        A: A_t, B: B_t, C: C_t
    ):
        one = arith.constant(1.0, type=dtype)

    @gpu.func
    @canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
    def test_2_with_rewrite[M, K, N, dtype, A_t = T.memref(
        M, K, dtype
    ), B_t = T.memref(K, N, dtype), C_t = T.memref(M, N, dtype),](
        A: A_t, B: B_t, C: C_t
    ):
        one = arith.constant(1.0, type=dtype)

    @module("mod1", ["#nvvm.target"])
    def _():
        test_plain[1, 2, 3, T.f32()].emit()  # noqa: F821
        test_2_with_rewrite[1, 2, 3, T.f32()].emit()  # noqa: F821

    @module("mod2", ["#nvvm.target"])
    def _():
        test_plain[4, 5, 6, T.f16()].emit()  # noqa: F821
        test_2_with_rewrite[4, 5, 6, T.f16()].emit()  # noqa: F821

    # CHECK: gpu.module @mod1 [#nvvm.target] {
    # CHECK:   gpu.func @"test_plain_int_1_int_2_int_3_type_f32_MemRefType_memref<1x2xf32>_MemRefType_memref<2x3xf32>_MemRefType_memref<1x3xf32>"(%arg0: memref<1x2xf32>, %arg1: memref<2x3xf32>, %arg2: memref<1x3xf32>) kernel {
    # CHECK:     %cst = arith.constant 1.000000e+00 : f32
    # CHECK:     gpu.return
    # CHECK:   }
    # CHECK:   gpu.func @"test_2_with_rewrite_int_1_int_2_int_3_type_f32_MemRefType_memref<1x2xf32>_MemRefType_memref<2x3xf32>_MemRefType_memref<1x3xf32>"(%arg0: memref<1x2xf32>, %arg1: memref<2x3xf32>, %arg2: memref<1x3xf32>) kernel {
    # CHECK:     %cst = arith.constant 1.000000e+00 : f32
    # CHECK:     gpu.return
    # CHECK:   }
    # CHECK: }
    # CHECK: gpu.module @mod2 [#nvvm.target] {
    # CHECK:   gpu.func @"test_plain_int_4_int_5_int_6_type_f16_MemRefType_memref<4x5xf16>_MemRefType_memref<5x6xf16>_MemRefType_memref<4x6xf16>"(%arg0: memref<4x5xf16>, %arg1: memref<5x6xf16>, %arg2: memref<4x6xf16>) kernel {
    # CHECK:     %cst = arith.constant 1.000000e+00 : f16
    # CHECK:     gpu.return
    # CHECK:   }
    # CHECK:   gpu.func @"test_2_with_rewrite_int_4_int_5_int_6_type_f16_MemRefType_memref<4x5xf16>_MemRefType_memref<5x6xf16>_MemRefType_memref<4x6xf16>"(%arg0: memref<4x5xf16>, %arg1: memref<5x6xf16>, %arg2: memref<4x6xf16>) kernel {
    # CHECK:     %cst = arith.constant 1.000000e+00 : f16
    # CHECK:     gpu.return
    # CHECK:   }
    # CHECK: }
    filecheck_with_comments(ctx.module)


def test_pooling_nchw_max(ctx: MLIRContext):
    S = ShapedType.get_dynamic_size()

    @func
    def maxpool2d[
        kernel_size_0, kernel_size_1, stride_0, stride_1, dilation_0, dilation_1
    ](
        input: T.memref(S, S, S, S, T.f32()),
        output: T.memref(S, S, S, S, T.f32()),
    ):
        kernel_shape_surrogate = memref.alloca(
            (kernel_size_0, kernel_size_1),
            T.f32(),
        )

        linalg.pooling_nchw_max(
            input,
            kernel_shape_surrogate,
            output,
            strides=[stride_0, stride_1],
            dilations=[dilation_0, dilation_1],
        )

    kernel_sizes = [2, 3]
    strides = [4, 5]
    dilations = [6, 7]
    maxpool2d_k = maxpool2d[
        kernel_sizes[0],
        kernel_sizes[1],
        strides[0],
        strides[1],
        dilations[0],
        dilations[1],
    ].emit()
    module = run_pipeline(
        ctx.module,
        Pipeline().bufferize().Func(Pipeline().convert_linalg_to_parallel_loops()),
    )
    # CHECK: func.func @maxpool2d_int_2_int_3_int_4_int_5_int_6_int_7(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>) {
    # CHECK:   %c3 = arith.constant 3 : index
    # CHECK:   %c2 = arith.constant 2 : index
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c0 = arith.constant 0 : index
    # CHECK:   %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
    # CHECK:   %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
    # CHECK:   %dim_1 = memref.dim %arg1, %c2 : memref<?x?x?x?xf32>
    # CHECK:   %dim_2 = memref.dim %arg1, %c3 : memref<?x?x?x?xf32>
    # CHECK:   scf.parallel (%arg2, %arg3, %arg4, %arg5) = (%c0, %c0, %c0, %c0) to (%dim, %dim_0, %dim_1, %dim_2) step (%c1, %c1, %c1, %c1) {
    # CHECK:     scf.for %arg6 = %c0 to %c2 step %c1 {
    # CHECK:       scf.for %arg7 = %c0 to %c3 step %c1 {
    # CHECK:         %0 = affine.apply #map(%arg4, %arg6)
    # CHECK:         %1 = affine.apply #map1(%arg5, %arg7)
    # CHECK:         %2 = memref.load %arg0[%arg2, %arg3, %0, %1] : memref<?x?x?x?xf32>
    # CHECK:         %3 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<?x?x?x?xf32>
    # CHECK:         %4 = arith.maximumf %3, %2 : f32
    # CHECK:         memref.store %4, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<?x?x?x?xf32>
    # CHECK:       }
    # CHECK:     }
    # CHECK:     scf.reduce
    # CHECK:   }
    # CHECK:   return
    # CHECK: }
    filecheck_with_comments(module)


def test_pooling_ncdhw_max(ctx: MLIRContext):
    S = ShapedType.get_dynamic_size()

    @func
    def maxpool3d[
        kernel_size_0,
        kernel_size_1,
        kernel_size_2,
        stride_0,
        stride_1,
        stride_2,
        dilation_0,
        dilation_1,
        dilation_2,
    ](
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

    kernel_sizes = [1, 2, 3]
    strides = [5, 6, 7]
    dilations = [7, 8, 9]
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
    # CHECK: func.func @maxpool3d_int_1_int_2_int_3_int_5_int_6_int_7_int_7_int_8_int_9(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>) {
    # CHECK:   %alloca = memref.alloca() : memref<1x2x3xf32>
    # CHECK:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2 * 5 + d5 * 7, d3 * 6 + d6 * 8, d4 * 7 + d7 * 9)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %alloca : memref<?x?x?x?x?xf32>, memref<1x2x3xf32>) outs(%arg1 : memref<?x?x?x?x?xf32>) {
    # CHECK:   ^bb0(%in: f32, %in_0: f32, %out: f32):
    # CHECK:     %0 = arith.maximumf %in, %out : f32
    # CHECK:     linalg.yield %0 : f32
    # CHECK:   }
    # CHECK:   return
    # CHECK: }
    filecheck_with_comments(maxpool3d_k)


def test_pooling_ncdhw_max_parallel(ctx: MLIRContext):
    S = ShapedType.get_dynamic_size()

    @func
    def maxpool3d[
        kernel_size_0,
        kernel_size_1,
        kernel_size_2,
        stride_0,
        stride_1,
        stride_2,
        dilation_0,
        dilation_1,
        dilation_2,
    ](
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

    kernel_sizes = [1, 2, 3]
    strides = [4, 5, 6]
    dilations = [7, 8, 9]
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
    module = run_pipeline(
        ctx.module,
        Pipeline().bufferize().Func(Pipeline().convert_linalg_to_parallel_loops()),
    )
    # CHECK: #map = affine_map<(d0, d1) -> (d0 * 4 + d1 * 7)>
    # CHECK: #map1 = affine_map<(d0, d1) -> (d0 * 5 + d1 * 8)>
    # CHECK: #map2 = affine_map<(d0, d1) -> (d0 * 6 + d1 * 9)>
    # CHECK: module {
    # CHECK:   func.func @maxpool3d_int_1_int_2_int_3_int_4_int_5_int_6_int_7_int_8_int_9(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>) {
    # CHECK:     %c4 = arith.constant 4 : index
    # CHECK:     %c3 = arith.constant 3 : index
    # CHECK:     %c2 = arith.constant 2 : index
    # CHECK:     %c1 = arith.constant 1 : index
    # CHECK:     %c0 = arith.constant 0 : index
    # CHECK:     %dim = memref.dim %arg0, %c0 : memref<?x?x?x?x?xf32>
    # CHECK:     %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?x?xf32>
    # CHECK:     %dim_1 = memref.dim %arg1, %c2 : memref<?x?x?x?x?xf32>
    # CHECK:     %dim_2 = memref.dim %arg1, %c3 : memref<?x?x?x?x?xf32>
    # CHECK:     %dim_3 = memref.dim %arg1, %c4 : memref<?x?x?x?x?xf32>
    # CHECK:     scf.parallel (%arg2, %arg3, %arg4, %arg5, %arg6) = (%c0, %c0, %c0, %c0, %c0) to (%dim, %dim_0, %dim_1, %dim_2, %dim_3) step (%c1, %c1, %c1, %c1, %c1) {
    # CHECK:       scf.for %arg7 = %c0 to %c1 step %c1 {
    # CHECK:         scf.for %arg8 = %c0 to %c2 step %c1 {
    # CHECK:           scf.for %arg9 = %c0 to %c3 step %c1 {
    # CHECK:             %0 = affine.apply #map(%arg4, %arg7)
    # CHECK:             %1 = affine.apply #map1(%arg5, %arg8)
    # CHECK:             %2 = affine.apply #map2(%arg6, %arg9)
    # CHECK:             %3 = memref.load %arg0[%arg2, %arg3, %0, %1, %2] : memref<?x?x?x?x?xf32>
    # CHECK:             %4 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?x?xf32>
    # CHECK:             %5 = arith.maximumf %3, %4 : f32
    # CHECK:             memref.store %5, %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<?x?x?x?x?xf32>
    # CHECK:           }
    # CHECK:         }
    # CHECK:       }
    # CHECK:       scf.reduce
    # CHECK:     }
    # CHECK:     return
    # CHECK:   }
    # CHECK: }
    filecheck_with_comments(module)
