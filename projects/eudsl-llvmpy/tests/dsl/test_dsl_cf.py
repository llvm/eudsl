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
    assert "for.header" in printed
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


def test_nested_if_inside_for_loop_jits():
    # An `if` nested inside a loop body now lowers to real control flow: the
    # body stays inline, so the if/else passes reach it. The if yields a value
    # (a phi) that feeds the loop-carried accumulator.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def sum_selected(n: i32, c: llvm.types.i1) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            if c:
                r = yield i
            else:
                r = yield i + i
            acc = acc + r
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("sum_selected")
    )
    assert fn(5, True) == 0 + 1 + 2 + 3 + 4
    assert fn(5, False) == 2 * (0 + 1 + 2 + 3 + 4)
    del jit, mod, ctx, sum_selected, i32, fn
    assert_no_leaks()


def test_for_loop_nested_inside_if_jits():
    # The dual: a loop nested inside an `if` branch. The loop's blocks are
    # emitted inside the then-branch and the branch's yielded value is the
    # loop result (a header phi valid at the loop exit).
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def maybe_sum(n: i32, c: llvm.types.i1) -> i32:
        if c:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                acc = acc + i
                yield acc
            r = yield acc
        else:
            r = yield llvm.ir.const_int(i32, -1, signed=True)
        return r

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("maybe_sum")
    )
    assert fn(5, True) == 0 + 1 + 2 + 3 + 4
    assert fn(5, False) == -1
    del jit, mod, ctx, maybe_sum, i32, fn
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


def test_if_within_if_within_for_jits():
    # ifs nested inside ifs, inside a loop body.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32, c: llvm.types.i1, d: llvm.types.i1) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            if c:
                if d:
                    r = yield i
                else:
                    r = yield i + i
            else:
                r = yield llvm.ir.const_int(i32, 0)
            acc = acc + r
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool, ctypes.c_bool)(
        jit.lookup("f")
    )
    assert fn(5, True, True) == 0 + 1 + 2 + 3 + 4
    assert fn(5, True, False) == 2 * (0 + 1 + 2 + 3 + 4)
    assert fn(5, False, False) == 0
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_nested_for_loops_jits():
    # A loop directly nested in another loop (both carry `acc`): acc += 1 for
    # each (i, j) pair -> n*n. Exercises the inner loop's result feeding the
    # outer loop-carried phi via the back-edge.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def grid(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            for j in range_(0, n):
                acc = acc + llvm.ir.const_int(i32, 1)
                yield acc
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("grid"))
    assert fn(3) == 9
    assert fn(0) == 0  # zero iterations: result is the initial carried value
    del jit, mod, ctx, grid, i32, fn
    assert_no_leaks()


def test_nested_loops_inside_if_jits():
    # Loop-in-loop nested inside an `if` branch; the branch yields the loop
    # result, which must be a header phi valid at the branch merge.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32, c: llvm.types.i1) -> i32:
        if c:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                for j in range_(0, n):
                    acc = acc + llvm.ir.const_int(i32, 1)
                    yield acc
                yield acc
            r = yield acc
        else:
            r = yield llvm.ir.const_int(i32, -1, signed=True)
        return r

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("f")
    )
    assert fn(3, True) == 9
    assert fn(3, False) == -1
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_while_nested_inside_for_jits():
    # Mixed loop kinds: a `while` nested in a `for` body (inner while runs `i`
    # times on outer iteration i -> total sum(0..n-1)).
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            j = llvm.ir.const_int(i32, 0)
            while j.ne(i):
                acc = acc + llvm.ir.const_int(i32, 1)
                j = j + llvm.ir.const_int(i32, 1)
                yield acc, j
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("f"))
    assert fn(4) == 0 + 1 + 2 + 3
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_for_nested_inside_while_jits():
    # Mixed loop kinds: a `for` nested in a `while` body.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        i = llvm.ir.const_int(i32, 0)
        while i.ne(n):
            for k in range_(0, i):
                acc = acc + llvm.ir.const_int(i32, 1)
                yield acc
            i = i + llvm.ir.const_int(i32, 1)
            yield acc, i
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("f"))
    assert fn(4) == 0 + 1 + 2 + 3
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_triple_nested_loops_jits():
    # Three levels of loop nesting: acc += 1 per (i, j, k) -> n**3.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def cube(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            for j in range_(0, n):
                for k in range_(0, n):
                    acc = acc + llvm.ir.const_int(i32, 1)
                    yield acc
                yield acc
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("cube"))
    assert fn(2) == 8
    assert fn(3) == 27
    del jit, mod, ctx, cube, i32, fn
    assert_no_leaks()


def test_elif_chain_inside_loop_jits():
    # An elif chain nested inside a loop body.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def bucket(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            if i < llvm.ir.const_int(i32, 2):
                r = yield i
            elif i < llvm.ir.const_int(i32, 4):
                r = yield i + llvm.ir.const_int(i32, 10)
            else:
                r = yield i + llvm.ir.const_int(i32, 100)
            acc = acc + r
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("bucket"))
    # i: 0->0, 1->1, 2->12, 3->13, 4->104, 5->105
    assert fn(6) == 0 + 1 + 12 + 13 + 104 + 105
    del jit, mod, ctx, bucket, i32, fn
    assert_no_leaks()


def test_while_with_nested_if_jits():
    # `while` (not `for`) carrying values, with an `if` in its body whose phi
    # feeds the loop-carried accumulator.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def cond_sum(n: i32, c: llvm.types.i1) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        i = llvm.ir.const_int(i32, 0)
        while i.ne(n):
            if c:
                r = yield acc + i
            else:
                r = yield acc
            acc = r
            i = i + llvm.ir.const_int(i32, 1)
            yield acc, i
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("cond_sum")
    )
    assert fn(5, True) == 0 + 1 + 2 + 3 + 4
    assert fn(5, False) == 0
    del jit, mod, ctx, cond_sum, i32, fn
    assert_no_leaks()


def test_loop_carried_plain_int_init_jits():
    # The loop-carried init is a plain Python int (not const_int); the loop
    # coerces it to a constant of the inferred type.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def total(n: i32) -> i32:
        acc = 0  # plain int, not llvm.ir.const_int(...)
        for i in range_(0, n):
            acc = acc + i
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    del jit, mod, ctx, total, i32, fn
    assert_no_leaks()


@pytest.mark.xfail(
    reason="branch-reassignment leakage: both if branches are traced in the same "
    "Python frame, so a variable reassigned in the then-branch leaks into the "
    "else-branch. Pre-existing if-lowering limitation; see README Limitations.",
    strict=True,
)
def test_branch_reassign_read_in_else_unsupported():
    # Catch the VerifyError, clean up, then fail via assert -- so the xfail's
    # traceback pins nothing live (a raised VerifyError would keep the module
    # alive past the per-test leak gate).
    verified = False
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def f(c: llvm.types.i1, a: i32, b: i32) -> i32:
            acc = a
            if c:
                acc = acc + b  # reassigned in then; leaks into else's `acc`
                r = yield acc
            else:
                r = yield acc
            return r

        try:
            mod.verify()  # invalid IR: else's acc references the then value
            verified = True
        except llvm.ir.VerifyError:
            verified = False
        del mod, i32, f
    assert verified


def test_branch_divergence_via_yield_jits():
    # The supported way to express the above: each branch yields its own value
    # (distinct names), so nothing reassigned in then is read in else.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(c: llvm.types.i1, a: i32, b: i32) -> i32:
        if c:
            r = yield a + b
        else:
            r = yield a
        return r

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("f"))
    assert fn(True, 10, 3) == 13
    assert fn(False, 10, 3) == 10
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


@pytest.mark.xfail(
    reason="branch-reassignment leakage: the then-branch's inner loop reassigns "
    "the outer carried `acc`, which leaks into the else-branch. Pre-existing "
    "if-lowering limitation; see README Limitations.",
    strict=True,
)
def test_loop_in_if_in_loop_reassign_unsupported():
    verified = False
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def f(n: i32, c: llvm.types.i1) -> i32:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                if c:
                    for j in range_(0, i):
                        acc = acc + llvm.ir.const_int(i32, 1)
                        yield acc
                    r = yield acc  # then: acc is the inner loop result
                else:
                    r = yield acc  # else reads acc -> leaked then value
                acc = r
                yield acc
            return acc

        try:
            mod.verify()
            verified = True
        except llvm.ir.VerifyError:
            verified = False
        del mod, i32, f
    assert verified


def test_loop_in_if_in_loop_distinct_names_jits():
    # Same shape as the xfail above, written correctly: the then-branch's inner
    # loop accumulates into a distinct name (`inner`), so the outer carried
    # `acc` is never reassigned inside a branch.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32, c: llvm.types.i1) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            if c:
                inner = acc
                for j in range_(0, i):
                    inner = inner + llvm.ir.const_int(i32, 1)
                    yield inner
                r = yield inner
            else:
                r = yield acc
            acc = r
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("f")
    )
    assert fn(4, True) == 0 + 1 + 2 + 3
    assert fn(5, True) == 0 + 1 + 2 + 3 + 4
    assert fn(4, False) == 0
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_nested_while_loops_jits():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        i = llvm.ir.const_int(i32, 0)
        while i.ne(n):
            j = llvm.ir.const_int(i32, 0)
            while j.ne(i):
                acc = acc + llvm.ir.const_int(i32, 1)
                j = j + llvm.ir.const_int(i32, 1)
                yield acc, j
            i = i + llvm.ir.const_int(i32, 1)
            yield acc, i
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("f"))
    assert fn(4) == 0 + 1 + 2 + 3
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_if_with_loops_in_both_branches_jits():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32, c: llvm.types.i1) -> i32:
        if c:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                acc = acc + i
                yield acc
            r = yield acc
        else:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                acc = acc + llvm.ir.const_int(i32, 1)
                yield acc
            r = yield acc
        return r

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("f")
    )
    assert fn(4, True) == 0 + 1 + 2 + 3
    assert fn(4, False) == 4
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_multi_carried_updated_in_nested_if_jits():
    # Two carried values (acc, cnt); acc is updated via a phi produced by a
    # nested if inside the loop body -- an if-merge phi feeding a loop phi.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32, c: llvm.types.i1) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        cnt = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            if c:
                a2 = yield acc + i
            else:
                a2 = yield acc
            acc = a2
            cnt = cnt + llvm.ir.const_int(i32, 1)
            yield acc, cnt
        return acc + cnt

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("f")
    )
    assert fn(5, True) == (0 + 1 + 2 + 3 + 4) + 5
    assert fn(5, False) == 0 + 5
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_if_nested_in_while_jits():
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32, c: llvm.types.i1) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        i = llvm.ir.const_int(i32, 0)
        while i.ne(n):
            if c:
                r = yield acc + i
            else:
                r = yield acc + llvm.ir.const_int(i32, 1)
            acc = r
            i = i + llvm.ir.const_int(i32, 1)
            yield acc, i
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("f")
    )
    assert fn(5, True) == 0 + 1 + 2 + 3 + 4
    assert fn(5, False) == 5
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


@pytest.mark.xfail(
    reason="branch-reassignment leakage: the then-branch's inner loop reassigns "
    "`acc`, which leaks into the else-branch's `yield acc`. Pre-existing "
    "if-lowering limitation; see README Limitations.",
    strict=True,
)
def test_four_level_for_if_for_if_reassign_unsupported():
    verified = False
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def f(n: i32, c: llvm.types.i1) -> i32:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                if c:
                    for j in range_(0, n):
                        if j < i:
                            r = yield llvm.ir.const_int(i32, 1)
                        else:
                            r = yield llvm.ir.const_int(i32, 0)
                        acc = acc + r
                        yield acc
                    s = yield acc  # then: acc reassigned by the inner loop
                else:
                    s = yield acc  # else reads acc -> leaked
                acc = s
                yield acc
            return acc

        try:
            mod.verify()
            verified = True
        except llvm.ir.VerifyError:
            verified = False
        del mod, i32, f
    assert verified


def test_four_level_for_if_for_if_distinct_names_jits():
    # Four levels (for > if > for > if), written correctly with a distinct
    # inner accumulator so the outer carried `acc` is not reassigned in a branch.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.dsl.function(module=mod)
    @canonicalize(using=LLVMCanonicalizer())
    def f(n: i32, c: llvm.types.i1) -> i32:
        acc = llvm.ir.const_int(i32, 0)
        for i in range_(0, n):
            if c:
                inner = acc
                for j in range_(0, n):
                    if j < i:
                        r = yield llvm.ir.const_int(i32, 1)
                    else:
                        r = yield llvm.ir.const_int(i32, 0)
                    inner = inner + r
                    yield inner
                s = yield inner
            else:
                s = yield acc
            acc = s
            yield acc
        return acc

    mod.verify()
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_bool)(
        jit.lookup("f")
    )
    # c=True: inner adds `i` per outer iteration -> sum(0..n-1)
    assert fn(4, True) == 0 + 1 + 2 + 3
    assert fn(5, True) == 0 + 1 + 2 + 3 + 4
    assert fn(4, False) == 0
    del jit, mod, ctx, f, i32, fn
    assert_no_leaks()


def test_fully_constant_loop_needs_a_value_raises():
    # A loop with only Python-int bounds and carries (no Value anywhere) has no
    # type to infer for its phis; it is rejected with a clear error.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="at least one Value"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def f() -> i32:
                acc = 0
                for i in range_(0, 3):
                    acc = acc + i
                    yield acc
                return acc

        del mod
    assert_no_leaks()


def test_if_in_for_ir_structure():
    # Pin the nested IR shape: the if-merge phi (if.end) feeds the loop-carried
    # header phi through the loop back-edge -- the core if-inside-loop interaction
    # that a substring check cannot express.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def nest(n: i32, c: llvm.types.i1) -> i32:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                if c:
                    r = yield i
                else:
                    r = yield i + i
                acc = acc + r
                yield acc
            return acc

        # CHECK: define i32 @nest(i32 %[[N:.*]], i1 %[[C:.*]])
        # CHECK: for.header:
        # CHECK: %[[IV:.*]] = phi i32 [ 0, %entry ], [ {{.*}}, %if.end ]
        # CHECK: %[[ACC:.*]] = phi i32 [ 0, %entry ], [ {{.*}}, %if.end ]
        # CHECK: br i1 {{.*}}, label %for.body, label %for.end
        # CHECK: for.body:
        # CHECK: br i1 %[[C]], label %if.then, label %if.else
        # CHECK: for.end:
        # CHECK: ret i32 %[[ACC]]
        # CHECK: if.end:
        # CHECK: %[[PHI:.*]] = phi i32 [ %[[IV]], %if.then ], [ {{.*}}, %if.else ]
        # CHECK: add i32 %[[ACC]], %[[PHI]]
        # CHECK: br label %for.header
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_nested_loops_ir_structure():
    # Two loop headers: the inner carried phi is seeded from the outer carried
    # phi, and its value flows back through the outer loop's back-edge.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def grid(n: i32) -> i32:
            acc = llvm.ir.const_int(i32, 0)
            for i in range_(0, n):
                for j in range_(0, n):
                    acc = acc + llvm.ir.const_int(i32, 1)
                    yield acc
                yield acc
            return acc

        # CHECK: define i32 @grid(i32 %[[N:.*]])
        # CHECK: for.header:
        # CHECK: phi i32 [ 0, %entry ], [ {{.*}}, %for.end3 ]
        # CHECK: %[[OACC:.*]] = phi i32 [ 0, %entry ], [ %[[INNER:.*]], %for.end3 ]
        # CHECK: br i1 {{.*}}, label %for.body, label %for.end
        # CHECK: for.body:
        # CHECK: br label %for.header1
        # CHECK: for.end:
        # CHECK: ret i32 %[[OACC]]
        # CHECK: for.header1:
        # CHECK: %[[INNER]] = phi i32 [ %[[OACC]], %for.body ], [ {{.*}}, %for.body2 ]
        # CHECK: for.end3:
        # CHECK: br label %for.header
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


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
