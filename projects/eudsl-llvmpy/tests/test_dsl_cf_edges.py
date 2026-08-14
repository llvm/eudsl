#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Edge cases and error branches of the DSL control-flow lowering."""
import ctypes

import pytest

import llvm
from llvm.ast.canonicalize import canonicalize, Canonicalizer, FunctionPatcher
from llvm.dsl.cf import LLVMCanonicalizer
from llvm.dsl.cf import range_
from llvm.dsl.values import with_element_type
from llvm.testing import assert_no_leaks


def test_range_single_arg():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def total(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for i in range_(n):  # single-arg range_: start defaults to 0
            acc = acc + i
            yield acc
        return acc

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    del jit, mod, ctx, total, i32, fn
    assert_no_leaks()


def test_for_side_effect_bare_yield():
    # No loop-carried values: body ends with a bare `yield`; stores into a ptr.
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def fill(p: llvm.types.ptr, n: i32) -> i32:
        tp = with_element_type(p, i32)
        for i in range_(0, n):
            tp[i] = i
            yield
        return n

    printed = str(mod)
    assert "while.header" in printed and "store i32" in printed
    del mod, ctx, fill, i32
    assert_no_leaks()


def test_range_too_many_args_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="range_ takes 1-3"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n, 1, 2):
                    acc = acc + i
                    yield acc
                return acc

        del mod


def test_for_non_name_target_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="single name"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for (i, j) in range_(0, n):
                    acc = acc + acc
                    yield acc
                return acc

        del mod


def test_for_body_without_trailing_yield_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="must end with"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n):
                    acc = acc + i
                return acc

        del mod


def test_while_body_without_trailing_yield_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="must end with"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                i = llvm.const_int(i32, 0)
                while i.ne(n):
                    i = i + 1
                return i

        del mod


def test_loop_yield_non_name_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="loop-carried variable names"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n):
                    acc = acc + i
                    yield acc + i  # not a plain name
                return acc

        del mod


def test_elif_multiple_carried_results():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def pick2(x: i32, a: i32, b: i32) -> i32:
        if x < 0:
            p, q = yield a, b
        elif x.eq(llvm.const_int(i32, 0)):
            p, q = yield b, a
        else:
            p, q = yield a, a
        return p - q

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("pick2"))
    assert fn(-1, 10, 3) == 10 - 3
    assert fn(0, 10, 3) == 3 - 10
    assert fn(5, 10, 3) == 10 - 10
    del jit, mod, ctx, pick2, i32, fn
    assert_no_leaks()


def test_range_runtime_callable():
    # range_ is a marker consumed by the transform, but also runtime-callable.
    assert range_(3) == (0, 3, 1)
    assert range_(1, 4) == (1, 4, 1)
    assert range_(1, 10, 2) == (1, 10, 2)


def test_for_over_python_list_unrolls():
    # A non-range_ `for` is left as a Python loop and unrolls at trace time.
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def sum3(a: i32, b: i32, c: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for x in [a, b, c]:
            acc = acc + x
        return acc

    printed = str(mod)
    assert "while.header" not in printed  # unrolled, no loop
    assert printed.count("add i32") == 3
    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("sum3"))
    assert fn(2, 3, 4) == 9
    del jit, mod, ctx, sum3, i32, fn
    assert_no_leaks()


def test_nested_if_in_then_branch():
    # A nested if in the THEN branch that yields forces the body-forward path
    # of CanonicalizeElIfs.
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(c: llvm.types.i1, d: llvm.types.i1, a: i32, b: i32) -> i32:
        if c:
            if d:
                r = yield a
            else:
                r = yield b
        else:
            r = yield b
        return r

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("f"))
    assert fn(True, True, 10, 20) == 10
    assert fn(True, False, 10, 20) == 20
    assert fn(False, True, 10, 20) == 20
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()
