#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes

import llvm
from llvm.testing import assert_no_leaks


def test_if_else_produces_phi():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod)
        def pick(c: llvm.types.i1, a: i32, b: i32) -> i32:
            if c:
                r = yield a + 1
            else:
                r = yield b
            return r

        printed = str(mod)
        assert "br i1" in printed
        assert "phi i32" in printed
        assert "add i32" in printed
        del mod
    assert_no_leaks()


def test_if_else_jits_correctly():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def pick(c: llvm.types.i1, a: i32, b: i32) -> i32:
        if c:
            r = yield a
        else:
            r = yield b
        return r

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("pick"))
    assert fn(True, 10, 20) == 10
    assert fn(False, 10, 20) == 20
    del jit, mod, ctx, pick, i32, fn
    assert_no_leaks()


def test_elif_chain_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def classify(x: i32) -> i32:
        if x < 0:
            r = yield llvm.const_int(i32, -1)
        elif x.eq(llvm.const_int(i32, 0)):
            r = yield llvm.const_int(i32, 0)
        else:
            r = yield llvm.const_int(i32, 1)
        return r

    printed = str(mod)
    assert printed.count("phi i32") == 2  # nested elif phis
    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("classify"))
    assert fn(-5) == -1
    assert fn(0) == 0
    assert fn(7) == 1
    del jit, mod, ctx, classify, i32, fn
    assert_no_leaks()


def test_while_countdown_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def sum_to(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        i = llvm.const_int(i32, 0)
        while i.ne(n):
            acc = acc + i
            i = i + 1
            yield acc, i
        return acc

    printed = str(mod)
    assert "while.header" in printed
    assert printed.count("phi i32") == 2  # acc, i loop-carried
    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sum_to"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    assert fn(0) == 0
    del jit, mod, ctx, sum_to, i32, fn
    assert_no_leaks()


def test_for_range_sum_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def total(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for i in range_(0, n):
            acc = acc + i
            yield acc
        return acc

    printed = str(mod)
    assert "while.header" in printed
    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    assert fn(1) == 0
    del jit, mod, ctx, total, i32, fn
    assert_no_leaks()


def test_if_else_multiple_results():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def swap_if(c: llvm.i1, a: i32, b: i32) -> i32:
        if c:
            x, y = yield a, b
        else:
            x, y = yield b, a
        return x - y

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("swap_if"))
    assert fn(True, 10, 3) == 10 - 3
    assert fn(False, 10, 3) == 3 - 10
    del jit, mod, ctx, swap_if, i32, fn
    assert_no_leaks()


def test_if_no_else_side_effect_only():
    # A single-branch if with no yielded result: side effect via store.
    from llvm.dsl.values import with_element_type

    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def clamp0(c: llvm.i1, p: llvm.ptr_t) -> i32:
        tp = with_element_type(p, i32)
        if c:
            tp[0] = llvm.const_int(i32, 0)
        return tp[0]

    printed = str(mod)
    assert "br i1" in printed
    assert "store i32 0" in printed
    del mod, ctx, clamp0, i32
    assert_no_leaks()


def test_elif_elif_three_way():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def sign(x: i32) -> i32:
        if x < 0:
            r = yield llvm.const_int(i32, -1)
        elif x.eq(llvm.const_int(i32, 0)):
            r = yield llvm.const_int(i32, 0)
        elif x < 10:
            r = yield llvm.const_int(i32, 1)
        else:
            r = yield llvm.const_int(i32, 2)
        return r

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sign"))
    assert fn(-4) == -1
    assert fn(0) == 0
    assert fn(5) == 1
    assert fn(99) == 2
    del jit, mod, ctx, sign, i32, fn
    assert_no_leaks()


def test_while_two_carried_values():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    # Fibonacci-ish: iterate n times advancing (a, b) -> (b, a+b).
    @llvm.function(module=mod)
    def fib(n: i32) -> i32:
        a = llvm.const_int(i32, 0)
        b = llvm.const_int(i32, 1)
        i = llvm.const_int(i32, 0)
        while i.ne(n):
            a, b = b, a + b
            i = i + 1
            yield a, b, i
        return a

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("fib"))
    assert fn(0) == 0
    assert fn(1) == 1
    assert fn(7) == 13  # 0,1,1,2,3,5,8,13
    del jit, mod, ctx, fib, i32, fn
    assert_no_leaks()


def test_for_with_step_and_two_carried():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    # Sum every other value in [0, n) and count how many; step 2.
    @llvm.function(module=mod)
    def strided(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        cnt = llvm.const_int(i32, 0)
        for i in range_(0, n, 2):
            acc = acc + i
            cnt = cnt + 1
            yield acc, cnt
        return acc

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("strided"))
    assert fn(10) == 0 + 2 + 4 + 6 + 8
    assert fn(1) == 0
    del jit, mod, ctx, strided, i32, fn
    assert_no_leaks()


def test_for_result_used_after_loop():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def total_plus_one(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for i in range_(0, n):
            acc = acc + i
            yield acc
        return acc + 1

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total_plus_one"))
    assert fn(5) == (0 + 1 + 2 + 3 + 4) + 1
    del jit, mod, ctx, total_plus_one, i32, fn
    assert_no_leaks()


def test_for_mixed_start_stop():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def sum_range(lo: i32, hi: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for i in range_(lo, hi):
            acc = acc + i
            yield acc
        return acc

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
        jit.lookup("sum_range")
    )
    assert fn(2, 5) == 2 + 3 + 4
    assert fn(5, 5) == 0
    del jit, mod, ctx, sum_range, i32, fn
    assert_no_leaks()
