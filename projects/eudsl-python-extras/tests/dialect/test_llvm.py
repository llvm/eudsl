# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import inspect
from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.dialects import llvm
from mlir.extras.dialects.func import func
from mlir.extras.dialects.llvm import mlir_constant, llvm_ptr_t, func as llvm_func

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    MLIRContext,
    filecheck,
    filecheck_with_comments,
    mlir_ctx as ctx,
)
from mlir.ir import Attribute, FunctionType, IndexType, IntegerType
from util import llvm_bindings_not_installed, llvm_amdgcn_bindings_not_installed

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


@pytest.mark.skipif(
    llvm_bindings_not_installed() or llvm_amdgcn_bindings_not_installed(),
    reason="llvm bindings not installed or llvm_amdgcn bindings not installed",
)
def test_call_instrinsic(ctx: MLIRContext):
    @func(emit=True)
    def sum(a: T.i32(), b: T.i32(), c: T.f32()):
        e = llvm.amdgcn.cvt_pk_i16(a, b)
        f = llvm.amdgcn.frexp_mant(c)

    correct = dedent("""
    module {
      func.func @sum(%arg0: i32, %arg1: i32, %arg2: f32) {
        %0 = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.i16"(%arg0, %arg1) : (i32, i32) -> vector<2xi16>
        %1 = llvm.call_intrinsic "llvm.amdgcn.frexp.mant"(%arg2) : (f32) -> f32
        return
      }
    }
    """)
    filecheck(correct, ctx.module)


def test_mlir_constant_int(ctx: MLIRContext):
    # Test mlir_constant with int value and no explicit type (lines 29-33)
    c = mlir_constant(42)
    assert c is not None

    # Test mlir_constant with explicit type
    c2 = mlir_constant(7, type=T.i32())
    assert c2 is not None

    correct = dedent("""\
    module {
      %0 = llvm.mlir.constant(42 : i32) : i32
      %1 = llvm.mlir.constant(7 : i32) : i32
    }
    """)
    filecheck(correct, ctx.module)


def test_mlir_constant_float(ctx: MLIRContext):
    # Test mlir_constant with float value (lines 34-35)
    c = mlir_constant(3.14)
    assert c is not None

    c2 = mlir_constant(2.5, type=T.f32())
    assert c2 is not None

    correct = dedent("""\
    module {
      %0 = llvm.mlir.constant(3.140000e+00 : f32) : f32
      %1 = llvm.mlir.constant(2.500000e+00 : f32) : f32
    }
    """)
    filecheck(correct, ctx.module)


def test_mlir_constant_unsupported_type(ctx: MLIRContext):
    # Test mlir_constant with unsupported type (lines 36-37)
    # Need to pass type explicitly to bypass infer_mlir_type
    with pytest.raises(NotImplementedError, match="is not a valid type"):
        mlir_constant("hello", type=T.i32())


def test_llvm_emit(ctx: MLIRContext):
    @llvm_func
    def demo_fun1() -> T.i32():
        one = mlir_constant(1, T.i32())
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()

    # CHECK:  llvm.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
    # CHECK:    llvm.return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_llvm_declare(ctx: MLIRContext):
    @llvm_func
    def demo_fun1() -> T.i32(): ...

    @llvm_func
    def demo_fun2(x: T.i32()) -> T.i32(): ...

    @llvm_func
    def demo_fun3(x: T.i32(), p: llvm_ptr_t()) -> T.i32(): ...

    ctx.module.operation.verify()

    # CHECK:  llvm.func @demo_fun1() -> i32 attributes {sym_visibility = "private"}
    # CHECK:  llvm.func @demo_fun2(i32) -> i32 attributes {sym_visibility = "private"}
    # CHECK:  llvm.func @demo_fun3(i32, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}

    filecheck_with_comments(ctx.module)


def test_llvm_void_return(ctx: MLIRContext):
    @llvm_func
    def no_result(x: T.i32()):
        one = mlir_constant(1, T.i32())
        return

    no_result.emit()

    ctx.module.operation.verify()

    # CHECK:  llvm.func @no_result(%[[VAL_0:.*]]: i32) {
    # CHECK:    %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
    # CHECK:    llvm.return
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_llvm_integer_index_type_annotations(ctx: MLIRContext):
    @llvm_func
    def f_i32(x: IntegerType[32]) -> IntegerType[32]: ...

    @llvm_func
    def f_i64(x: IntegerType[64]) -> IntegerType[64]: ...

    @llvm_func
    def f_index(x: IndexType) -> IndexType: ...

    ctx.module.operation.verify()

    # CHECK: llvm.func @f_i32(i32) -> i32 attributes {sym_visibility = "private"}
    # CHECK: llvm.func @f_i64(i64) -> i64 attributes {sym_visibility = "private"}
    # CHECK: llvm.func @f_index(index) -> index attributes {sym_visibility = "private"}

    filecheck_with_comments(ctx.module)


def test_llvm_float_type_annotations(ctx: MLIRContext):
    @llvm_func
    def f_f32(x: T.f32()) -> T.f32(): ...

    @llvm_func
    def f_f64(x: T.f64()) -> T.f64(): ...

    ctx.module.operation.verify()

    # CHECK: llvm.func @f_f32(f32) -> f32 attributes {sym_visibility = "private"}
    # CHECK: llvm.func @f_f64(f64) -> f64 attributes {sym_visibility = "private"}

    filecheck_with_comments(ctx.module)


def test_llvm_pointer_type_annotation(ctx: MLIRContext):
    @llvm_func
    def f_ptr(p: llvm_ptr_t()) -> llvm_ptr_t(): ...

    ctx.module.operation.verify()

    # CHECK: llvm.func @f_ptr(!llvm.ptr) -> !llvm.ptr attributes {sym_visibility = "private"}

    filecheck_with_comments(ctx.module)


def test_llvm_multiple_arg_with_body(ctx: MLIRContext):
    @llvm_func
    def add_first(a: T.i32(), b: T.i32(), c: T.i32()) -> T.i32():
        return a

    add_first.emit()

    ctx.module.operation.verify()

    # CHECK: llvm.func @add_first(%[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: i32) -> i32 {
    # CHECK:   llvm.return %[[A]] : i32
    # CHECK: }

    filecheck_with_comments(ctx.module)


def test_llvm_explicit_function_type(ctx: MLIRContext):
    func_type = llvm.FunctionType.get(T.i32(), [T.i32(), T.i32()])

    @llvm_func(function_type=func_type)
    def demo_fun1(a, b):
        one = mlir_constant(1, T.i32())
        return one

    demo_fun1.emit()

    ctx.module.operation.verify()

    # CHECK:  llvm.func @demo_fun1(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i32 {
    # CHECK:    %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
    # CHECK:    llvm.return %[[VAL_2]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_llvm_func_with_func_attrs(ctx: MLIRContext):
    @llvm_func(func_attrs={"llvm.emit_c_interface": Attribute.parse("unit")})
    def with_attrs() -> T.i32():
        one = mlir_constant(1, T.i32())
        return one

    with_attrs.emit()

    ctx.module.operation.verify()

    # CHECK: llvm.func @with_attrs() -> i32 attributes {llvm.emit_c_interface}
    filecheck_with_comments(ctx.module)


def test_llvm_func_emit_true(ctx: MLIRContext):
    @llvm_func(emit=True)
    def emitted_immediately() -> T.i32():
        one = mlir_constant(1, T.i32())
        return one

    ctx.module.operation.verify()

    # CHECK: llvm.func @emitted_immediately() -> i32
    filecheck_with_comments(ctx.module)


def test_llvm_multiple_results_unsupported(ctx: MLIRContext):
    with pytest.raises(ValueError, match="llvm.func supports at most one result type"):

        @llvm_func
        def two_results() -> (T.i32(), T.i32()): ...


def test_llvm_call(ctx: MLIRContext):
    @llvm_func
    def callee(a: T.i32(), b: T.i32()) -> T.i32():
        return a

    callee.emit()

    @llvm_func
    def caller() -> T.i32():
        one = mlir_constant(1, T.i32())
        return callee(one, one)

    caller.emit()

    ctx.module.operation.verify()

    # CHECK:  llvm.func @callee(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i32 {
    # CHECK:    llvm.return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  llvm.func @caller() -> i32 {
    # CHECK:    %[[VAL_2:.*]] = llvm.mlir.constant(1 : i32) : i32
    # CHECK:    %[[VAL_3:.*]] = llvm.call @callee(%[[VAL_2]], %[[VAL_2]]) : (i32, i32) -> i32
    # CHECK:    llvm.return %[[VAL_3]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_llvm_call_void(ctx: MLIRContext):
    @llvm_func
    def sink(a: T.i32()):
        one = mlir_constant(1, T.i32())
        return

    sink.emit()

    @llvm_func
    def driver():
        one = mlir_constant(1, T.i32())
        sink(one)
        return

    driver.emit()

    ctx.module.operation.verify()

    # CHECK:  llvm.func @driver() {
    # CHECK:    %[[VAL_0:.*]] = llvm.mlir.constant(1 : i32) : i32
    # CHECK:    llvm.call @sink(%[[VAL_0]]) : (i32) -> ()
    # CHECK:    llvm.return
    # CHECK:  }

    filecheck_with_comments(ctx.module)
