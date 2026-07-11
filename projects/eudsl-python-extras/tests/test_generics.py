# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
from typing import TypeVar

import pytest

import mlir.extras.types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.ast.py_type import PyTypeVarObject
from mlir.extras.dialects import linalg, arith, scf, memref, gpu
from mlir.extras.dialects.func import func
from mlir.extras.dialects.gpu import block_idx
from mlir.extras.dialects.llvm import func as llvm_func, mlir_constant
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
from mlir.ir import ShapedType

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


@func
def matmul_i32_i32[M, N](
    A: "T.memref(M, N, T.i32())",
    B: "T.memref(M, N, T.i32())",
    C: "T.memref(M, N, T.i32())",
):
    linalg.matmul(A, B, C)


def test_func_no_context_2(ctx: MLIRContext):

    # CHECK: func.func @matmul_i32_i32_int_16_int_16(%[[VAL_0:.*]]: memref<16x16xi32>, %[[VAL_1:.*]]: memref<16x16xi32>, %[[VAL_2:.*]]: memref<16x16xi32>) {
    # CHECK:   linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%[[VAL_0]], %[[VAL_1]] : memref<16x16xi32>, memref<16x16xi32>) outs(%[[VAL_2]] : memref<16x16xi32>)
    # CHECK:   return
    # CHECK: }
    matmul_i32_i32[16, 16].emit()

    filecheck_with_comments(ctx.module)


def test_generics_just_args(ctx: MLIRContext):
    @func
    def mat_product_kernel[M, K, N, dtype](
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        one = arith.constant(1.0, dtype)

    # CHECK: func.func @mat_product_kernel_int_32_int_64_int_96_type_f32(%arg0: memref<32x64xf32>, %arg1: memref<64x96xf32>, %arg2: memref<32x96xf32>) {
    # CHECK:   %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:   return
    # CHECK: }
    mat_product_kernel[32, 64, 96, T.f32()].emit()

    filecheck_with_comments(ctx.module)


def test_generics_closure(ctx: MLIRContext):
    @func
    def mat_product_kernel[M, K, N, dtype](
        A: "T.memref(M, K, dtype)",
        B: "T.memref(K, N, dtype)",
        C: "T.memref(M, N, dtype)",
    ):
        one = arith.constant(1, dtype)

    # CHECK: func.func @mat_product_kernel_int_32_int_64_int_96_type_i32(%arg0: memref<32x64xi32>, %arg1: memref<64x96xi32>, %arg2: memref<32x96xi32>) {
    # CHECK:   %[[VAL_3:.*]] = arith.constant 1 : i32
    # CHECK:   return
    # CHECK: }
    mat_product_kernel[32, 64, 96, T.i32()].emit()

    # CHECK: func.func @mat_product_kernel_int_32_int_64_int_96_type_f32(%arg0: memref<32x64xf32>, %arg1: memref<64x96xf32>, %arg2: memref<32x96xf32>) {
    # CHECK:   %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:   return
    # CHECK: }
    mat_product_kernel[32, 64, 96, T.f32()].emit()

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

    # CHECK: func.func @mat_product_kernel1_function_maximumf() {
    # CHECK:   %cst = arith.constant 1.000000e+00 : f32
    # CHECK:   %0 = arith.maximumf %cst, %cst : f32
    # CHECK:   return
    # CHECK: }
    mat_product_kernel1[arith.maximumf].emit()

    # CHECK: func.func @mat_product_kernel2_function_minimumf() {
    # CHECK:   %cst = arith.constant 1.000000e+00 : f32
    # CHECK:   %0 = arith.minimumf %cst, %cst : f32
    # CHECK:   return
    # CHECK: }
    mat_product_kernel2[arith.minimumf].emit()

    # CHECK: func.func @mat_product_kernel2_function_maximumf() {
    # CHECK:   %cst = arith.constant 1.000000e+00 : f32
    # CHECK:   %0 = arith.maximumf %cst, %cst : f32
    # CHECK:   return
    # CHECK: }
    mat_product_kernel2[arith.maximumf].emit()

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

    # CHECK: func.func @mat_product_kernel_int_32_int_64_int_96_type_f32(%arg0: memref<32x64xf32>, %arg1: memref<64x96xf32>, %arg2: memref<32x96xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 1.000000e+00 : f32
    # CHECK:   %cst_1 = arith.constant 0.000000e+00 : f32
    # CHECK:   %c0 = arith.constant 0 : index
    # CHECK:   %c64 = arith.constant 64 : index
    # CHECK:   %c1_2 = arith.constant 1 : index
    # CHECK:   %0 = scf.for %arg3 = %c0 to %c64 step %c1_2 iter_args(%arg4 = %cst_1) -> (f32) {
    # CHECK:     %2 = memref.load %arg0[%c1, %arg3] : memref<32x64xf32>
    # CHECK:     %3 = memref.load %arg1[%arg3, %c1_0] : memref<64x96xf32>
    # CHECK:     %4 = math.fma %2, %3, %arg4 : f32
    # CHECK:     scf.yield %4 : f32
    # CHECK:   }
    # CHECK:   %1 = arith.addf %0, %cst : f32
    # CHECK:   memref.store %1, %arg2[%c1, %c1_0] : memref<32x96xf32>
    # CHECK:   return
    # CHECK: }
    mat_product_kernel[32, 64, 96, T.f32()].emit()

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

    # CHECK: func.func @type_bound_int_32_int_64_i32_10(%arg0: memref<32x64xf32>, %arg1: memref<64x10xf32>, %arg2: memref<32x10xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   return
    # CHECK: }
    type_bound[32, 64, 10].emit()

    # CHECK: func.func @type_bound_and_default_int_32_int_64_i32_10_f32_10.0(%arg0: memref<32x64xf32>, %arg1: memref<64x10xf32>, %arg2: memref<32x10xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 1.000000e+01 : f32
    # CHECK:   return
    # CHECK: }
    type_bound_and_default[32, 64].emit()

    # CHECK: func.func @type_bound_and_default_int_33_int_11_i32_10_f32_10.0(%arg0: memref<33x11xf32>, %arg1: memref<11x10xf32>, %arg2: memref<33x10xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 1.000000e+01 : f32
    # CHECK:   return
    # CHECK: }
    type_bound_and_default[33, 11].emit()

    # CHECK: func.func @type_bound_and_default_int_66_int_22_i32_10_f32_10.0(%arg0: memref<66x22xf32>, %arg1: memref<22x10xf32>, %arg2: memref<66x10xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 1.000000e+01 : f32
    # CHECK:   return
    # CHECK: }
    type_bound_and_default[66, 22, None, None].emit()

    filecheck_with_comments(ctx.module)


def test_partial_specialization(ctx: MLIRContext):
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

    partial = type_bound[32, 64]
    # CHECK: func.func @type_bound_int_32_int_64_i32_10(%arg0: memref<32x64xf32>, %arg1: memref<64x10xf32>, %arg2: memref<32x10xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   return
    # CHECK: }
    partial[10].emit()

    # CHECK: func.func @type_bound_int_32_int_64_i32_16(%arg0: memref<32x64xf32>, %arg1: memref<64x16xf32>, %arg2: memref<32x16xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   return
    # CHECK: }
    partial[16].emit()

    partial_with_defaults = type_bound_and_default[32, 64]

    # CHECK: func.func @type_bound_and_default_int_32_int_64_i32_20_f32_10.0(%arg0: memref<32x64xf32>, %arg1: memref<64x20xf32>, %arg2: memref<32x20xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 1.000000e+01 : f32
    # CHECK:   return
    # CHECK: }
    partial_with_defaults[20].emit()

    # CHECK: func.func @type_bound_and_default_int_32_int_64_i32_26_f32_10.0(%arg0: memref<32x64xf32>, %arg1: memref<64x26xf32>, %arg2: memref<32x26xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 1.000000e+01 : f32
    # CHECK:   return
    # CHECK: }
    partial_with_defaults[26].emit()

    # CHECK: func.func @type_bound_and_default_int_32_int_64_i32_20_f32_66.0(%arg0: memref<32x64xf32>, %arg1: memref<64x20xf32>, %arg2: memref<32x20xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 6.600000e+01 : f32
    # CHECK:   return
    # CHECK: }
    partial_with_defaults[20][66.0].emit()

    # CHECK: func.func @type_bound_and_default_int_32_int_64_i32_26_f32_88.0(%arg0: memref<32x64xf32>, %arg1: memref<64x26xf32>, %arg2: memref<32x26xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 8.800000e+01 : f32
    # CHECK:   return
    # CHECK: }
    partial_with_defaults[26][88.0].emit()

    # CHECK: func.func @type_bound_and_default_int_32_int_64_i32_30_f32_76.0(%arg0: memref<32x64xf32>, %arg1: memref<64x30xf32>, %arg2: memref<32x30xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 7.600000e+01 : f32
    # CHECK:   return
    # CHECK: }
    partial_with_defaults[30][76.0].emit()

    # CHECK: func.func @type_bound_and_default_int_32_int_64_i32_36_f32_98.0(%arg0: memref<32x64xf32>, %arg1: memref<64x36xf32>, %arg2: memref<32x36xf32>) {
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c1_0 = arith.constant 1 : index
    # CHECK:   %cst = arith.constant 9.800000e+01 : f32
    # CHECK:   return
    # CHECK: }
    partial_with_defaults[36][98.0].emit()

    filecheck_with_comments(ctx.module)


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

    # CHECK: func.func @maxpool2d_int_2_int_2_int_1_int_1_int_1_int_1(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>) {
    # CHECK:   %alloca = memref.alloca() : memref<2x2xf32>
    # CHECK:   linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %alloca : memref<?x?x?x?xf32>, memref<2x2xf32>) outs(%arg1 : memref<?x?x?x?xf32>)
    # CHECK:   return
    # CHECK: }
    maxpool2d_k = maxpool2d[2, 2, 1, 1, 1, 1].emit()

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
        mat_product_kernel[32, 64, 96, T.f32()].emit()  # noqa: F821

    # CHECK: gpu.module @naive [#nvvm.target] {
    # CHECK:   gpu.func @mat_product_kernel_int_32_int_64_int_96_type_f32(%arg0: memref<32x64xf32>, %arg1: memref<64x96xf32>, %arg2: memref<32x96xf32>) kernel {
    # CHECK:     %block_dim_x = gpu.block_dim  x
    # CHECK:     %block_id_x = gpu.block_id  x
    # CHECK:     %0 = arith.muli %block_dim_x, %block_id_x : index
    # CHECK:     %thread_id_x = gpu.thread_id  x
    # CHECK:     %1 = arith.addi %0, %thread_id_x : index
    # CHECK:     %block_dim_y = gpu.block_dim  y
    # CHECK:     %block_id_y = gpu.block_id  y
    # CHECK:     %2 = arith.muli %block_dim_y, %block_id_y : index
    # CHECK:     %thread_id_y = gpu.thread_id  y
    # CHECK:     %3 = arith.addi %2, %thread_id_y : index
    # CHECK:     %cst = arith.constant 1.000000e+00 : f32
    # CHECK:     %cst_0 = arith.constant 0.000000e+00 : f32
    # CHECK:     %c0 = arith.constant 0 : index
    # CHECK:     %c64 = arith.constant 64 : index
    # CHECK:     %c1 = arith.constant 1 : index
    # CHECK:     %4 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %cst_0) -> (f32) {
    # CHECK:       %6 = memref.load %arg0[%1, %arg3] : memref<32x64xf32>
    # CHECK:       %7 = memref.load %arg1[%arg3, %3] : memref<64x96xf32>
    # CHECK:       %8 = arith.mulf %6, %7 : f32
    # CHECK:       %9 = arith.addf %arg4, %8 : f32
    # CHECK:       scf.yield %9 : f32
    # CHECK:     }
    # CHECK:     %5 = arith.addf %4, %cst : f32
    # CHECK:     memref.store %5, %arg2[%1, %3] : memref<32x96xf32>
    # CHECK:     gpu.return
    # CHECK:   }
    # CHECK: }

    filecheck_with_comments(ctx.module)


def test_generic_type_var_closure_patching(ctx: MLIRContext):
    def fun2[foo, bar, A: foo + bar]():
        print(A.__bound__)

    A_type_param = fun2.__type_params__[2]

    a = PyTypeVarObject.from_object(A_type_param)
    a_something = a.bound.contents.into_object()
    a_something.__closure__[0].cell_contents = 5
    a_something.__closure__[1].cell_contents = 7


def test_generic_type_var_closure_patching_dependent_generics(ctx: MLIRContext):
    # fmt: off
    @gpu.func
    def test_plain[
    M,
    K,
    N,
    dtype,
    A_t = T.memref(M, K, dtype),
    B_t = T.memref(K, N, dtype),
    C_t = T.memref(M, N, dtype)
    ](
            A: A_t, B: B_t, C: C_t
    ):
        one = arith.constant(1.0, type=dtype)

    # fmt: on

    # fmt: off
    @gpu.func
    @canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
    def test_2_with_rewrite[
    M,
    K,
    N,
    dtype,
    A_t = T.memref(M, K, dtype),
    B_t = T.memref(K, N, dtype),
    C_t = T.memref(M, N, dtype)
    ](
            A: A_t, B: B_t, C: C_t
    ):
        one = arith.constant(1.0, type=dtype)

    # fmt: on

    @module("mod1", ["#nvvm.target"])
    def _():
        # CHECK: gpu.func @"test_plain_int_1_int_2_int_3_type_f32_MemRefType_memref<1x2xf32>_MemRefType_memref<2x3xf32>_MemRefType_memref<1x3xf32>"(%arg0: memref<1x2xf32>, %arg1: memref<2x3xf32>, %arg2: memref<1x3xf32>) kernel {
        # CHECK:   %cst = arith.constant 1.000000e+00 : f32
        # CHECK:   gpu.return
        # CHECK: }
        test_plain[1, 2, 3, T.f32()].emit()  # noqa: F821

        # CHECK: gpu.func @"test_2_with_rewrite_int_1_int_2_int_3_type_f32_MemRefType_memref<1x2xf32>_MemRefType_memref<2x3xf32>_MemRefType_memref<1x3xf32>"(%arg0: memref<1x2xf32>, %arg1: memref<2x3xf32>, %arg2: memref<1x3xf32>) kernel {
        # CHECK:   %cst = arith.constant 1.000000e+00 : f32
        # CHECK:   gpu.return
        # CHECK: }
        test_2_with_rewrite[1, 2, 3, T.f32()].emit()  # noqa: F821

    @module("mod2", ["#nvvm.target"])
    def _():
        # CHECK: gpu.func @"test_plain_int_4_int_5_int_6_type_f16_MemRefType_memref<4x5xf16>_MemRefType_memref<5x6xf16>_MemRefType_memref<4x6xf16>"(%arg0: memref<4x5xf16>, %arg1: memref<5x6xf16>, %arg2: memref<4x6xf16>) kernel {
        # CHECK:   %cst = arith.constant 1.000000e+00 : f16
        # CHECK:   gpu.return
        # CHECK: }
        test_plain[4, 5, 6, T.f16()].emit()  # noqa: F821

        # CHECK: gpu.func @"test_2_with_rewrite_int_4_int_5_int_6_type_f16_MemRefType_memref<4x5xf16>_MemRefType_memref<5x6xf16>_MemRefType_memref<4x6xf16>"(%arg0: memref<4x5xf16>, %arg1: memref<5x6xf16>, %arg2: memref<4x6xf16>) kernel {
        # CHECK:   %cst = arith.constant 1.000000e+00 : f16
        # CHECK:   gpu.return
        # CHECK: }
        test_2_with_rewrite[4, 5, 6, T.f16()].emit()  # noqa: F821

        # CHECK: gpu.func @"test_2_with_rewrite_int_4_int_5_int_6_type_f16_MemRefType_memref<7x8xf32>_MemRefType_memref<5x6xf16>_MemRefType_memref<4x6xf16>"(%arg0: memref<7x8xf32>, %arg1: memref<5x6xf16>, %arg2: memref<4x6xf16>) kernel {
        # CHECK:   %cst = arith.constant 1.000000e+00 : f16
        # CHECK:   gpu.return
        # CHECK: }
        test_2_with_rewrite[
            4, 5, 6, T.f16(), T.memref(7, 8, T.f32())
        ].emit()  # noqa: F821

    filecheck_with_comments(ctx.module)


def test_pooling_nchw_max(ctx: MLIRContext):
    S = ShapedType.get_dynamic_size()

    @func
    def maxpool2d[
        kernel_size_0, kernel_size_1, stride_0, stride_1, dilation_0, dilation_1, dtype
    ](
        input: "T.memref(S, S, S, S, dtype)",
        output: "T.memref(S, S, S, S, dtype)",
    ):
        kernel_shape_surrogate = memref.alloca((kernel_size_0, kernel_size_1), dtype)

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
    maxpool2d_k_dtype = maxpool2d[
        kernel_sizes[0],
        kernel_sizes[1],
        strides[0],
        strides[1],
        dilations[0],
        dilations[1],
    ]

    # CHECK: func.func @maxpool2d_int_2_int_3_int_4_int_5_int_6_int_7_type_f32(%arg0: memref<?x?x?x?xf32>, %arg1: memref<?x?x?x?xf32>) {
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
    maxpool2d_k_f32 = maxpool2d_k_dtype[T.f32()].emit()

    # CHECK: func.func @maxpool2d_int_2_int_3_int_4_int_5_int_6_int_7_type_f64(%arg0: memref<?x?x?x?xf64>, %arg1: memref<?x?x?x?xf64>) {
    # CHECK:   %c3 = arith.constant 3 : index
    # CHECK:   %c2 = arith.constant 2 : index
    # CHECK:   %c1 = arith.constant 1 : index
    # CHECK:   %c0 = arith.constant 0 : index
    # CHECK:   %dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf64>
    # CHECK:   %dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf64>
    # CHECK:   %dim_1 = memref.dim %arg1, %c2 : memref<?x?x?x?xf64>
    # CHECK:   %dim_2 = memref.dim %arg1, %c3 : memref<?x?x?x?xf64>
    # CHECK:   scf.parallel (%arg2, %arg3, %arg4, %arg5) = (%c0, %c0, %c0, %c0) to (%dim, %dim_0, %dim_1, %dim_2) step (%c1, %c1, %c1, %c1) {
    # CHECK:     scf.for %arg6 = %c0 to %c2 step %c1 {
    # CHECK:       scf.for %arg7 = %c0 to %c3 step %c1 {
    # CHECK:         %0 = affine.apply #map(%arg4, %arg6)
    # CHECK:         %1 = affine.apply #map1(%arg5, %arg7)
    # CHECK:         %2 = memref.load %arg0[%arg2, %arg3, %0, %1] : memref<?x?x?x?xf64>
    # CHECK:         %3 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<?x?x?x?xf64>
    # CHECK:         %4 = arith.maximumf %3, %2 : f64
    # CHECK:         memref.store %4, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<?x?x?x?xf64>
    # CHECK:       }
    # CHECK:     }
    # CHECK:     scf.reduce
    # CHECK:   }
    # CHECK:   return
    # CHECK: }
    maxpool2d_k_f64 = maxpool2d_k_dtype[T.f64()].emit()

    module = run_pipeline(
        ctx.module,
        Pipeline().bufferize().Func(Pipeline().convert_linalg_to_parallel_loops()),
    )

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

    # CHECK: func.func @maxpool3d_int_1_int_2_int_3_int_5_int_6_int_7_int_7_int_8_int_9(%arg0: memref<?x?x?x?x?xf32>, %arg1: memref<?x?x?x?x?xf32>) {
    # CHECK:   %alloca = memref.alloca() : memref<1x2x3xf32>
    # CHECK:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2 * 5 + d5 * 7, d3 * 6 + d6 * 8, d4 * 7 + d7 * 9)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %alloca : memref<?x?x?x?x?xf32>, memref<1x2x3xf32>) outs(%arg1 : memref<?x?x?x?x?xf32>) {
    # CHECK:   ^bb0(%in: f32, %in_0: f32, %out: f32):
    # CHECK:     %0 = arith.maximumf %in, %out : f32
    # CHECK:     linalg.yield %0 : f32
    # CHECK:   }
    # CHECK:   return
    # CHECK: }
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

    filecheck_with_comments(module)


def test_wrong_generics_types(ctx: MLIRContext):
    # fmt: off
    @gpu.func
    @canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
    def sgemm_shared_mem_1d_block_tiling[
        M, K, N, dtype,
        BM, BN, BK, TM,
        A_t = T.memref(M, K, dtype),
        B_t = T.memref(K, N, dtype),
        C_t = T.memref(M, N, dtype)
    ](
        A: A_t, B: B_t, C: C_t
    ):
    # fmt: on
        base = gpu.dynamic_shared_memory()
        A_shared = memref.view(base, (BM, BK), dtype=dtype)
        B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

        c_row = block_idx.y * BM
        c_col = block_idx.x * BN

    # the last 8 sets the reified param for A_t (which is wrong)
    with pytest.raises(TypeError, match="Generic parameter 'A_t'"):
        sgemm_shared_mem_1d_block_tiling[128, 128, 128, T.f32(), 64, 64, 8, 8, 8].emit()


def test_generics_basic(ctx: MLIRContext):
    """Basic generics with type params, ReifiedTypeParam, and __getitem__."""

    @func
    def generic_func[M, N](
        A: "T.memref(M, N, T.f32())",
    ):
        one = arith.constant(1.0)

    generic_func[4, 8].emit()

    ctx.module.operation.verify()

    # CHECK: func.func @generic_func_int_4_int_8(%[[ARG:.*]]: memref<4x8xf32>)
    filecheck_with_comments(ctx.module)


def test_generics_with_defaults(ctx: MLIRContext):
    """Generics with default values."""

    @func
    def with_defaults[M, K, N: T.i32() = 10](
        A: "T.memref(M, K, T.f32())",
        B: "T.memref(K, N, T.f32())",
    ):
        one = arith.constant(1.0)

    with_defaults[4, 8].emit()

    ctx.module.operation.verify()

    # CHECK: func.func @with_defaults_int_4_int_8_i32_10(%[[A:.*]]: memref<4x8xf32>, %[[B:.*]]: memref<8x10xf32>)
    filecheck_with_comments(ctx.module)


def test_partial_specialization_with_default_emit(ctx: MLIRContext):
    """Partial specialization where remaining generics use defaults."""

    @func
    def partial_default[M, N: T.i32() = 10](
        A: "T.memref(M, N, T.f32())",
    ):
        one = arith.constant(1.0)

    # Specialize M only, N uses default
    partial = partial_default[4]
    partial.emit()

    ctx.module.operation.verify()

    # CHECK: func.func @partial_default_int_4_i32_10(%[[ARG:.*]]: memref<4x10xf32>)
    filecheck_with_comments(ctx.module)


def test_generics_callable_concrete_val(ctx: MLIRContext):
    """v = v.__name__ when concrete_val is callable in __getitem__."""
    _op = TypeVar("_op")

    @func
    def callable_generic[_op]():
        one = arith.constant(1.0, T.f32())
        two = _op(one, one)

    callable_generic[arith.maximumf].emit()

    ctx.module.operation.verify()

    # CHECK: func.func @callable_generic_function_maximumf()
    # CHECK:   arith.maximumf
    filecheck_with_comments(ctx.module)


def test_generics_dependent_closure(ctx: MLIRContext):
    """Dependent generics with closures.
    Tests add_replace_in_closure and maybe_eval_type_data_closure_vals."""

    @func
    def dependent[M, K, dtype, A_t = T.memref(M, K, dtype)](
        A: A_t,
    ):
        one = arith.constant(1.0, type=dtype)

    dependent[4, 8, T.f32()].emit()

    ctx.module.operation.verify()

    # CHECK: func.func @"dependent_int_4_int_8_type_f32_MemRefType_memref<4x8xf32>"(%[[ARG:.*]]: memref<4x8xf32>)
    filecheck_with_comments(ctx.module)


def test_reserved_generic_name_T(ctx: MLIRContext):
    """ValueError for T reserved generic name."""
    from mlir.extras.dialects.func import FuncBase, ReifiedTypeParam
    from mlir.dialects.func import FuncOp, ReturnOp, CallOp

    T_var = TypeVar("T")

    def body(x):
        return x

    body.__annotations__ = {"x": T.i32()}
    reified_T = ReifiedTypeParam(T_var, concrete_val=42)
    with pytest.raises(ValueError, match="T is a reserved generic name"):
        fb = FuncBase(
            body,
            FuncOp.__base__,
            ReturnOp,
            CallOp.__base__,
            generics=[reified_T],
        )
        fb._build_input_types()


def test_reserved_generic_name_S(ctx: MLIRContext):
    """ValueError for S reserved generic name."""
    from mlir.extras.dialects.func import FuncBase, ReifiedTypeParam
    from mlir.dialects.func import FuncOp, ReturnOp, CallOp

    S_var = TypeVar("S")

    def body(x):
        return x

    body.__annotations__ = {"x": T.i32()}
    reified_S = ReifiedTypeParam(S_var, concrete_val=42)
    with pytest.raises(ValueError, match="S is a reserved generic name"):
        fb = FuncBase(
            body,
            FuncOp.__base__,
            ReturnOp,
            CallOp.__base__,
            generics=[reified_S],
        )
        fb._build_input_types()


def test_reified_type_param_no_bound_type_infer(ctx: MLIRContext):
    """ReifiedTypeParam infers type_name when no bound and no default."""
    from mlir.extras.dialects.func import ReifiedTypeParam

    tvar = TypeVar("MyVar")
    # Pass a Type as concrete_val - should set type_name to "type"
    r = ReifiedTypeParam(tvar, concrete_val=T.i32())
    assert r.type_name == "type"

    # Pass an int as concrete_val - should set type_name to "int"
    r2 = ReifiedTypeParam(tvar, concrete_val=42)
    assert r2.type_name == "int"


def test_llvm_generics_type_param(ctx: MLIRContext):
    """llvm.func generic where a type param is used as an argument type."""

    @llvm_func
    def gen_ty[dtype](a: "dtype"):
        one = mlir_constant(1, T.i32())

    gen_ty[T.i32()].emit()
    gen_ty[T.i64()].emit()

    ctx.module.operation.verify()

    # CHECK: llvm.func @gen_ty_type_i32(%[[A:.*]]: i32) {
    # CHECK:   llvm.mlir.constant(1 : i32) : i32
    # CHECK:   llvm.return
    # CHECK: }
    # CHECK: llvm.func @gen_ty_type_i64(%[[B:.*]]: i64) {
    # CHECK:   llvm.mlir.constant(1 : i32) : i32
    # CHECK:   llvm.return
    # CHECK: }

    filecheck_with_comments(ctx.module)


def test_llvm_generics_dimension(ctx: MLIRContext):
    """llvm.func generic where a dimension param is reified into the body."""

    @llvm_func
    def gen_dim[N](a: "T.i32()"):
        c = mlir_constant(N, T.i32())

    gen_dim[8].emit()

    ctx.module.operation.verify()

    # CHECK: llvm.func @gen_dim_int_8(%[[A:.*]]: i32) {
    # CHECK:   llvm.mlir.constant(8 : i32) : i32
    # CHECK:   llvm.return
    # CHECK: }

    filecheck_with_comments(ctx.module)


def test_llvm_generics_partial_specialization(ctx: MLIRContext):
    """llvm.func generic specialized one type param at a time via chained __getitem__."""

    @llvm_func
    def gen_two[A_t, B_t](a: "A_t", b: "B_t"):
        one = mlir_constant(1, T.i32())

    # Specialize A_t, then B_t
    partial = gen_two[T.i32()]
    partial[T.i64()].emit()

    ctx.module.operation.verify()

    # CHECK: llvm.func @gen_two_type_i32_type_i64(%[[A:.*]]: i32, %[[B:.*]]: i64) {
    # CHECK:   llvm.mlir.constant(1 : i32) : i32
    # CHECK:   llvm.return
    # CHECK: }

    filecheck_with_comments(ctx.module)


def test_llvm_generics_preserves_subclass(ctx: MLIRContext):
    """Specializing an LLVMFunc via __getitem__ must return an LLVMFunc (not FuncBase)."""
    from mlir.extras.dialects.llvm import LLVMFunc

    @llvm_func
    def gen[N](a: "T.i32()"):
        c = mlir_constant(N, T.i32())

    specialized = gen[4]
    assert isinstance(specialized, LLVMFunc)
    specialized.emit()

    ctx.module.operation.verify()

    # CHECK: llvm.func @gen_int_4(%[[A:.*]]: i32) {
    filecheck_with_comments(ctx.module)
