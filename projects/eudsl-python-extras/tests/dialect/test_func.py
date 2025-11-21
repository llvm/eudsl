# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import inspect
import sys

import mlir.extras.types as T
import pytest
from mlir.ir import FunctionType

from mlir.extras.context import mlir_mod_ctx, RAIIMLIRContextModule
from mlir.extras.dialects.arith import constant
from mlir.extras.dialects.func import func

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_emit(ctx: MLIRContext):
    @func
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()

    # CHECK:  func.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_declare_byte_rep(ctx: MLIRContext):
    def demo_fun1(): ...

    if sys.version_info.minor == 14:
        assert demo_fun1.__code__.co_code == b"\x80\x00R\x00#\x00"
    elif sys.version_info.minor == 13:
        assert demo_fun1.__code__.co_code == b"\x95\x00g\x00"
    elif sys.version_info.minor == 12:
        assert demo_fun1.__code__.co_code == b"\x97\x00y\x00"
    elif sys.version_info.minor == 11:
        assert demo_fun1.__code__.co_code == b"\x97\x00d\x00S\x00"
    elif sys.version_info.minor in {8, 9, 10}:
        assert demo_fun1.__code__.co_code == b"d\x00S\x00"
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")


def test_declare(ctx: MLIRContext):
    @func
    def demo_fun1() -> T.i32(): ...

    @func
    def demo_fun2() -> (T.i32(), T.i32()): ...

    @func
    def demo_fun3(x: T.i32()) -> (T.i32(), T.i32()): ...

    @func
    def demo_fun4(x: T.i32(), y: T.i32()) -> (T.i32(), T.i32()): ...

    demo_fun1()
    demo_fun2()
    one = constant(1)
    demo_fun3(one)
    demo_fun4(one, one)

    ctx.module.operation.verify()

    # CHECK:  func.func private @demo_fun1() -> i32
    # CHECK:  func.func private @demo_fun2() -> (i32, i32)
    # CHECK:  func.func private @demo_fun3(i32) -> (i32, i32)
    # CHECK:  func.func private @demo_fun4(i32, i32) -> (i32, i32)
    # CHECK:  %[[VAL_0:.*]] = func.call @demo_fun1() : () -> i32
    # CHECK:  %[[VAL_1:.*]]:2 = func.call @demo_fun2() : () -> (i32, i32)
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_3:.*]]:2 = func.call @demo_fun3(%[[VAL_2]]) : (i32) -> (i32, i32)
    # CHECK:  %[[VAL_4:.*]]:2 = func.call @demo_fun4(%[[VAL_2]], %[[VAL_2]]) : (i32, i32) -> (i32, i32)

    filecheck_with_comments(ctx.module)


def test_func_base_meta(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1.emit()
    foo1()

    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_func_base_meta2(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1()

    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_func_no_context():
    @func
    def foo1():
        one = constant(1)
        return one

    with mlir_mod_ctx() as mod_ctx:
        foo1()

        # CHECK:  func.func @foo1() -> i32 {
        # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
        # CHECK:    return %[[VAL_0]] : i32
        # CHECK:  }
        # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

        filecheck_with_comments(mod_ctx.module)


def test_raii_mlir_context_module():
    ctx = RAIIMLIRContextModule()

    @func
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()

    # CHECK:  func.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_explicit_function_type(ctx: MLIRContext):
    input_types = [T.i32(), T.i32()]
    result_types = [T.i32()]
    func_type = FunctionType.get(input_types, result_types)

    @func(function_type=func_type)
    def demo_fun1(a, b):
        one = constant(1)
        return one

    demo_fun1.emit()

    # CHECK:  func.func @demo_fun1(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i32 {
    # CHECK:    %[[VAL_2:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_2]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)
