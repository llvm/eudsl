#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes

import pytest

import llvm
from llvm.dsl.values import with_element_type
from llvm.ast.canonicalize import canonicalize
from llvm.dsl.cf import LLVMCanonicalizer
from llvm.testing import assert_no_leaks, filecheck_with_comments


def test_if_else_produces_phi():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def pick(c: llvm.types.i1, a: i32, b: i32) -> i32:
            if c:
                r = yield a + 1
            else:
                r = yield b
            return r

        # The if/else with `yield` lowers to a conditional branch and a 2-input
        # phi that merges the then/else values -- ordered and SSA-bound, which a
        # substring or edge-count regex cannot express.
        # CHECK: define i32 @pick(i1 %[[C:.*]], i32 %[[A:.*]], i32 %[[B:.*]])
        # CHECK: br i1 %[[C]], label %if.then, label %if.else
        # CHECK: if.then:
        # CHECK: %[[ADD:.*]] = add i32 %[[A]], 1
        # CHECK: if.end:
        # CHECK: %[[PHI:.*]] = phi i32 [ %[[ADD]], %if.then ], [ %[[B]], %if.else ]
        # CHECK: ret i32 %[[PHI]]
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_if_else_jits_correctly():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def pick(c: llvm.types.i1, a: i32, b: i32) -> i32:
        if c:
            r = yield a
        else:
            r = yield b
        return r

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("pick"))
    assert fn(True, 10, 20) == 10
    assert fn(False, 10, 20) == 20
    del jit, mod, ctx, pick, i32, fn
    assert_no_leaks()


def test_elif_chain_jits():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def classify(x: i32) -> i32:
        if x < 0:
            r = yield llvm.ir.const_int(i32, -1, signed=True)
        elif x.eq(llvm.ir.const_int(i32, 0)):
            r = yield llvm.ir.const_int(i32, 0)
        else:
            r = yield llvm.ir.const_int(i32, 1)
        return r

    printed = str(mod)
    assert printed.count("phi i32") == 2  # nested elif phis
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("classify"))
    assert fn(-5) == -1
    assert fn(0) == 0
    assert fn(7) == 1
    del jit, mod, ctx, classify, i32, fn
    assert_no_leaks()


def test_while_countdown_jits():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def sum_to(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        i = llvm.ir.const_int(i32, 0)
        while i.ne(n):
            acc = acc + i
            i = i + 1
            yield acc, i
        return acc

    printed = str(mod)
    assert "while.header" in printed
    assert printed.count("phi i32") == 2  # acc, i loop-carried
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sum_to"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    assert fn(0) == 0
    del jit, mod, ctx, sum_to, i32, fn
    assert_no_leaks()


def test_for_range_sum_jits():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def total(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            acc = acc + i
            yield acc
        return acc

    printed = str(mod)
    assert "while.header" in printed
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    assert fn(1) == 0
    del jit, mod, ctx, total, i32, fn
    assert_no_leaks()


def test_function_void_no_explicit_return():
    # A non-empty body (an empty/pass/docstring-only body is a declaration,
    # not a definition) that falls off the end without a DSL `return` still
    # needs the entry block terminated with `ret void`.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        def store_and_fall_through(p: llvm.types.ptr) -> llvm.types.void:
            from llvm.dsl.values import with_element_type

            tp = with_element_type(p, i32)
            tp[0] = llvm.ir.const_int(i32, 0)

        assert "ret void" in str(mod)
        del mod
    assert_no_leaks()


def test_function_rejects_unresolvable_annotation():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(TypeError, match="cannot resolve type annotation"):

            @llvm.dsl.function(module=mod)
            def bad(x: "not a type") -> i32:
                return x

        del mod
    assert_no_leaks()


def test_if_else_multiple_results():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def swap_if(c: llvm.types.i1, a: i32, b: i32) -> i32:
        if c:
            x, y = yield a, b
        else:
            x, y = yield b, a
        return x - y

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("swap_if"))
    assert fn(True, 10, 3) == 10 - 3
    assert fn(False, 10, 3) == 3 - 10
    del jit, mod, ctx, swap_if, i32, fn
    assert_no_leaks()


def test_while_single_carried_value():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def count_to(n: i32) -> i32:
        i = llvm.ir.const_int(i32, 0)
        while i.ne(n):
            i = i + 1
            yield i
        return i

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("count_to"))
    assert fn(5) == 5
    assert fn(0) == 0
    del jit, mod, ctx, count_to, i32, fn
    assert_no_leaks()


def test_for_range_single_and_stepped_args():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def sum_upto(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(n):
            acc = acc + i
            yield acc
        return acc

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def sum_stepped(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n, 2):
            acc = acc + i
            yield acc
        return acc

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn1 = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sum_upto"))
    fn2 = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sum_stepped"))
    assert fn1(5) == 0 + 1 + 2 + 3 + 4
    assert fn2(10) == 0 + 2 + 4 + 6 + 8
    del jit, mod, ctx, sum_upto, sum_stepped, i32, fn1, fn2
    assert_no_leaks()


def test_for_non_range_iterable_is_left_untouched():
    # ForToForLoop only rewrites `for x in range_(...)`; any other iterable
    # is a plain Python loop that unrolls at trace/build time.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    def sum_consts(a: i32) -> i32:
        acc = a
        for v in (1, 2, 3):
            acc = acc + llvm.ir.const_int(i32, v)
        return acc

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sum_consts"))
    assert fn(10) == 10 + 1 + 2 + 3
    del jit, mod, ctx, sum_consts, i32, fn
    assert_no_leaks()


def test_break_inside_dsl_control_flow_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="`break`"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                i = llvm.ir.const_int(i32, 0)
                while i.ne(n):
                    break
                    i = i + 1
                    yield i
                return i

        del mod
    assert_no_leaks()


def test_continue_inside_dsl_control_flow_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="`continue`"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                i = llvm.ir.const_int(i32, 0)
                while i.ne(n):
                    continue
                    i = i + 1
                    yield i
                return i

        del mod
    assert_no_leaks()


def test_early_return_inside_dsl_control_flow_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="early `return`"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(c: llvm.types.i1, a: i32) -> i32:
                if c:
                    return a
                r = yield a
                return r

        del mod
    assert_no_leaks()


def test_while_body_without_trailing_yield_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="must end with `yield"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                i = llvm.ir.const_int(i32, 0)
                while i.ne(n):
                    i = i + 1
                return i

        del mod
    assert_no_leaks()


def test_for_body_without_trailing_yield_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="must end with `yield"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                acc = llvm.ir.const_int(i32, 0)
                for i in range_(0, n):
                    acc = acc + i
                return acc

        del mod
    assert_no_leaks()


def test_for_target_must_be_single_name_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="single name"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                acc = llvm.ir.const_int(i32, 0)
                for i, j in range_(0, n):
                    acc = acc + i
                    yield acc
                return acc

        del mod
    assert_no_leaks()


def test_for_range_wrong_arg_count_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="1-3 arguments"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                acc = llvm.ir.const_int(i32, 0)
                for i in range_(0, n, 1, 1):
                    acc = acc + i
                    yield acc
                return acc

        del mod
    assert_no_leaks()


def test_loop_yield_non_name_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="plain loop-carried"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                acc = llvm.ir.const_int(i32, 0)
                for i in range_(0, n):
                    yield acc + i
                return acc

        del mod
    assert_no_leaks()


def test_nested_control_flow_inside_loop_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="not supported"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32, c: llvm.types.i1) -> i32:
                i = llvm.ir.const_int(i32, 0)
                while i.ne(n):
                    if c:
                        i = i + 1
                    yield i
                return i

        del mod
    assert_no_leaks()


def test_while_loop_bare_yield_has_no_carried_values():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def spin(n: i32) -> i32:
        while n.eq(llvm.ir.const_int(i32, 0)):
            yield
        return n

    printed = str(mod)
    assert "while.header" in printed
    del mod, ctx, spin, i32
    assert_no_leaks()


def test_range_marker_is_callable_directly():
    from llvm.dsl.cf import range_

    assert range_(5) == (0, 5, 1)
    assert range_(2, 5) == (2, 5, 1)
    assert range_(2, 5, 3) == (2, 5, 3)


def test_for_negative_step_countdown_jits():
    # Exercises the descending-loop condition (iv > stop) for negative step.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def countdown(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(n, 0, -1):
            acc = acc + i
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn_ptr = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("countdown"))
    # sum(range(5, 0, -1)) = 5 + 4 + 3 + 2 + 1 = 15
    assert fn_ptr(5) == 15
    assert fn_ptr(1) == 1
    assert fn_ptr(0) == 0
    del jit, mod, fn_ptr, ctx


def test_if_no_else_side_effect_only():
    # A single-branch if with no yielded result: side effect via store.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def clamp0(c: llvm.types.i1, p: llvm.types.ptr) -> i32:
        tp = with_element_type(p, i32)
        if c:
            tp[0] = llvm.ir.const_int(i32, 0)
        return tp[0]

    printed = str(mod)
    assert "br i1" in printed
    assert "store i32 0" in printed
    mod.verify()
    del mod, ctx, clamp0, i32
    assert_no_leaks()


def test_elif_elif_three_way():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def sign(x: i32) -> i32:
        if x < 0:
            r = yield llvm.ir.const_int(i32, -1, signed=True)
        elif x.eq(llvm.ir.const_int(i32, 0)):
            r = yield llvm.ir.const_int(i32, 0)
        elif x < 10:
            r = yield llvm.ir.const_int(i32, 1)
        else:
            r = yield llvm.ir.const_int(i32, 2)
        return r

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sign"))
    assert fn(-4) == -1
    assert fn(0) == 0
    assert fn(5) == 1
    assert fn(99) == 2
    del jit, mod, ctx, sign, i32, fn
    assert_no_leaks()


def test_while_two_carried_values():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    # Fibonacci-ish: iterate n times advancing (a, b) -> (b, a+b).
    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def fib(n: i32) -> i32:
        a = llvm.ir.const_int(i32, 0)
        b = llvm.ir.const_int(i32, 1)
        i = llvm.ir.const_int(i32, 0)
        while i.ne(n):
            a, b = b, a + b
            i = i + 1
            yield a, b, i
        return a

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("fib"))
    assert fn(0) == 0
    assert fn(1) == 1
    assert fn(7) == 13  # 0,1,1,2,3,5,8,13
    del jit, mod, ctx, fib, i32, fn
    assert_no_leaks()


def test_for_with_step_and_two_carried():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    # Sum every other value in [0, n) and count how many; step 2.
    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def strided(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        cnt = llvm.ir.const_int(i32, 0)
        for i in range_(0, n, 2):
            acc = acc + i
            cnt = cnt + 1
            yield acc, cnt
        return acc

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("strided"))
    assert fn(10) == 0 + 2 + 4 + 6 + 8
    assert fn(1) == 0
    del jit, mod, ctx, strided, i32, fn
    assert_no_leaks()


def test_for_result_used_after_loop():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def total_plus_one(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            acc = acc + i
            yield acc
        return acc + 1

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total_plus_one"))
    assert fn(5) == (0 + 1 + 2 + 3 + 4) + 1
    del jit, mod, ctx, total_plus_one, i32, fn
    assert_no_leaks()


def test_for_mixed_start_stop():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def sum_range(lo: i32, hi: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(lo, hi):
            acc = acc + i
            yield acc
        return acc

    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
        jit.lookup("sum_range")
    )
    assert fn(2, 5) == 2 + 3 + 4
    assert fn(5, 5) == 0
    del jit, mod, ctx, sum_range, i32, fn
    assert_no_leaks()


def test_early_return_inside_while_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="early `return`"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                acc = llvm.ir.const_int(i32, 0)
                while acc.ne(n):
                    return acc
                    yield acc
                return acc

        del mod
    assert_no_leaks()


def test_early_return_inside_for_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="early `return`"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f(n: i32) -> i32:
                acc = llvm.ir.const_int(i32, 0)
                for i in range_(0, n):
                    return acc
                    yield acc
                return acc

        del mod
    assert_no_leaks()


def test_if_else_phi_has_correct_incomings():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def pick(c: llvm.types.i1, a: i32, b: i32) -> i32:
            if c:
                r = yield a + 1
            else:
                r = yield b
            return r

        mod.verify()
        # Exactly one phi, with two incoming edges from the then/else preds.
        assert str(mod).count("phi i32") == 1
        # CHECK: %{{.*}} = phi i32 [ {{.*}}, %if.then ], [ {{.*}}, %if.else ]
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_while_loop_verifies_cleanly():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def sum_to(n: i32) -> i32:
            acc = llvm.ir.const_int(i32, 0)
            i = llvm.ir.const_int(i32, 0)
            while i.ne(n):
                acc = acc + i
                i = i + 1
                yield acc, i
            return acc

        mod.verify()
        del mod
    assert_no_leaks()


def test_for_loop_verifies_cleanly():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def total(n: i32) -> i32:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                acc = acc + i
                yield acc
            return acc

        mod.verify()
        del mod
    assert_no_leaks()
