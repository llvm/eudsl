#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes

import pytest

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
            r = yield llvm.const_int(i32, -1, signed=True)
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


def test_function_void_no_explicit_return():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)

        @llvm.function(module=mod)
        def noop() -> llvm.types.void:
            pass

        assert "ret void" in str(mod)
        del mod
    assert_no_leaks()


def test_function_rejects_unresolvable_annotation():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(TypeError, match="cannot resolve type annotation"):
            @llvm.function(module=mod)
            def bad(x: "not a type") -> i32:
                return x
        del mod
    assert_no_leaks()


def test_if_else_multiple_results():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def swap_if(c: llvm.types.i1, a: i32, b: i32) -> i32:
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


def test_yield_outside_if_stack_returns_directly():
    from llvm.dsl.cf import yield_

    assert yield_(5) == 5
    assert yield_(1, 2) == (1, 2)


def test_while_single_carried_value():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def count_to(n: i32) -> i32:
        i = llvm.const_int(i32, 0)
        while i.ne(n):
            i = i + 1
            yield i
        return i

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("count_to"))
    assert fn(5) == 5
    assert fn(0) == 0
    del jit, mod, ctx, count_to, i32, fn
    assert_no_leaks()


def test_for_range_single_and_stepped_args():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def sum_upto(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for i in range_(n):
            acc = acc + i
            yield acc
        return acc

    @llvm.function(module=mod)
    def sum_stepped(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for i in range_(0, n, 2):
            acc = acc + i
            yield acc
        return acc

    jit = llvm.LLJIT()
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
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def sum_consts(a: i32) -> i32:
        acc = a
        for v in (1, 2, 3):
            acc = acc + llvm.const_int(i32, v)
        return acc

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sum_consts"))
    assert fn(10) == 10 + 1 + 2 + 3
    del jit, mod, ctx, sum_consts, i32, fn
    assert_no_leaks()


def test_break_inside_dsl_control_flow_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="`break`"):
            @llvm.function(module=mod)
            def f(n: i32) -> i32:
                i = llvm.const_int(i32, 0)
                while i.ne(n):
                    break
                    i = i + 1
                    yield i
                return i
        del mod
    assert_no_leaks()


def test_continue_inside_dsl_control_flow_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="`continue`"):
            @llvm.function(module=mod)
            def f(n: i32) -> i32:
                i = llvm.const_int(i32, 0)
                while i.ne(n):
                    continue
                    i = i + 1
                    yield i
                return i
        del mod
    assert_no_leaks()


def test_early_return_inside_dsl_control_flow_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="early `return`"):
            @llvm.function(module=mod)
            def f(c: llvm.types.i1, a: i32) -> i32:
                if c:
                    return a
                r = yield a
                return r
        del mod
    assert_no_leaks()


def test_while_body_without_trailing_yield_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="must end with `yield"):
            @llvm.function(module=mod)
            def f(n: i32) -> i32:
                i = llvm.const_int(i32, 0)
                while i.ne(n):
                    i = i + 1
                return i
        del mod
    assert_no_leaks()


def test_for_body_without_trailing_yield_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="must end with `yield"):
            @llvm.function(module=mod)
            def f(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n):
                    acc = acc + i
                return acc
        del mod
    assert_no_leaks()


def test_for_target_must_be_single_name_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="single name"):
            @llvm.function(module=mod)
            def f(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i, j in range_(0, n):
                    acc = acc + i
                    yield acc
                return acc
        del mod
    assert_no_leaks()


def test_for_range_wrong_arg_count_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="1-3 arguments"):
            @llvm.function(module=mod)
            def f(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n, 1, 1):
                    acc = acc + i
                    yield acc
                return acc
        del mod
    assert_no_leaks()


def test_loop_yield_non_name_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="plain loop-carried"):
            @llvm.function(module=mod)
            def f(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n):
                    yield acc + i
                return acc
        del mod
    assert_no_leaks()

def test_nested_control_flow_inside_loop_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="not supported"):
            @llvm.function(module=mod)
            def f(n: i32, c: llvm.types.i1) -> i32:
                i = llvm.const_int(i32, 0)
                while i.ne(n):
                    if c:
                        i = i + 1
                    yield i
                return i
        del mod
    assert_no_leaks()


def test_while_loop_bare_yield_has_no_carried_values():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def spin(n: i32) -> i32:
        while n.eq(llvm.const_int(i32, 0)):
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


def test_while_loop_wraps_non_tuple_body_result():
    # Direct call to the runtime primitive (bypassing the AST transform, which
    # always returns a real tuple) to exercise the non-tuple wrap branch.
    from llvm.dsl.cf import while_loop
    from llvm.dsl.context import building

    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [i32]), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b, fn):
            n = fn.arg(0)

            def cond(i):
                return i.ne(n)

            def body(i):
                return i + 1  # bare Value, not a tuple

            (result,) = while_loop(cond, body, (llvm.const_int(i32, 0),))
            b.ret(result)

        jit = llvm.LLJIT()
        jit.add_module(mod)
        fn_ptr = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("f"))
        assert fn_ptr(5) == 5
        del jit, mod, fn, fn_ptr
    assert_no_leaks()


def test_for_loop_wraps_none_and_non_tuple_body_result():
    # Direct calls to the runtime primitive to exercise the None- and
    # non-tuple-result wrap branches that the AST transform never produces
    # (its generated body always returns a real tuple).
    from llvm.dsl.cf import for_loop
    from llvm.dsl.context import building

    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [i32]), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b, fn):
            n = fn.arg(0)

            def body_none(i):
                return None  # no carried values

            for_loop(llvm.const_int(i32, 0), n, llvm.const_int(i32, 1), body_none, ())

            def body_bare(i, acc):
                return acc + i  # bare non-tuple, one carried value

            (result,) = for_loop(
                llvm.const_int(i32, 0), n, llvm.const_int(i32, 1), body_bare,
                (llvm.const_int(i32, 0),),
            )
            b.ret(result)

        jit = llvm.LLJIT()
        jit.add_module(mod)
        fn_ptr = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("f"))
        assert fn_ptr(5) == 0 + 1 + 2 + 3 + 4
        del jit, mod, fn, fn_ptr
    assert_no_leaks()
