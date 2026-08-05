#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.dsl.context import building
from llvm.testing import assert_no_leaks


def _entry(ctx, mod, ret_ty, arg_tys, name="f"):
    fn = llvm.Function.create(llvm.function_t(ret_ty, arg_tys), name, mod)
    bb = fn.append_basic_block("entry")
    return fn, bb


def test_integer_add_and_mul():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb = _entry(ctx, mod, i32, [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            r = fn.arg(0) * fn.arg(1) + 1
            b.ret(r)
        printed = str(mod)
        assert "mul i32" in printed
        assert "add i32" in printed
        del b, fn, mod
    assert_no_leaks()


def test_float_add_uses_fadd():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.f32(ctx)
        fn, bb = _entry(ctx, mod, f32, [f32, f32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0) + fn.arg(1))
        assert "fadd float" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_scalar_coercion():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb = _entry(ctx, mod, i32, [i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0) + 7)
        assert "add i32 %0, 7" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_comparison_signed():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb = _entry(ctx, mod, llvm.i1(ctx), [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0) < fn.arg(1))
        assert "icmp slt i32" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_comparison_ordered():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.f32(ctx)
        fn, bb = _entry(ctx, mod, llvm.i1(ctx), [f32, f32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0) > fn.arg(1))
        assert "fcmp ogt float" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_eq_ne_named_methods():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb = _entry(ctx, mod, llvm.i1(ctx), [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0).eq(fn.arg(1)))
        assert "icmp eq i32" in str(mod)
        # __eq__ stays identity so Value is still hashable.
        assert fn.arg(0) == fn.arg(0)
        assert len({fn.arg(0), fn.arg(0), fn.arg(1)}) == 2
        del b, fn, mod
    assert_no_leaks()
