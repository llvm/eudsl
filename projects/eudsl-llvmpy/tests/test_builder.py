#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

import llvm
from llvm.testing import assert_no_leaks, filecheck_with_comments


def test_build_add_function():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [i32, i32]), "add2", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            s = b.add(fn.arg(0), fn.arg(1), "s")
            b.ret(s)
        # CHECK: define i32 @add2(i32 %0, i32 %1)
        # CHECK:   %s = add i32 %0, %1
        # CHECK-NEXT:   ret i32 %s
        filecheck_with_comments(mod)
        del b, fn, bb, mod
    assert_no_leaks()


def test_build_conditional_with_phi():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        i1 = llvm.types.i1(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [i1]), "sel", mod)
        entry = fn.append_basic_block("entry")
        a = fn.append_basic_block("a")
        b_ = fn.append_basic_block("b")
        join = fn.append_basic_block("join")
        bld = llvm.IRBuilder(ctx)
        with bld.at_end_of(entry):
            bld.cond_br(fn.arg(0), a, b_)
        with bld.at_end_of(a):
            bld.br(join)
        with bld.at_end_of(b_):
            bld.br(join)
        with bld.at_end_of(join):
            p = bld.phi(i32, "p")
            p.add_incoming(llvm.const_int(i32, 1), a)
            p.add_incoming(llvm.const_int(i32, 2), b_)
            bld.ret(p)
        # CHECK:   br i1 %0, label %a, label %b
        # CHECK:   %p = phi i32 [ 1, %a ], [ 2, %b ]
        # CHECK-NEXT:   ret i32 %p
        filecheck_with_comments(mod)
        del bld, fn, entry, a, b_, join, p, mod
    assert_no_leaks()


def test_alloca_load_store():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            slot = b.alloca(i32, "slot")
            b.store(llvm.const_int(i32, 5), slot)
            loaded = b.load(i32, slot, "loaded")
            b.ret(loaded)
        printed = str(mod)
        assert "%slot = alloca i32" in printed
        assert "store i32 5, ptr %slot" in printed
        assert "%loaded = load i32, ptr %slot" in printed
        del b, fn, bb, mod
    assert_no_leaks()


def test_void_ret():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        void = llvm.types.void(ctx)
        fn = llvm.Function.create(llvm.types.function(void, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            b.ret()
        printed = str(mod)
        assert "ret void" in printed
        del b, fn, bb, mod
    assert_no_leaks()


def test_binary_ops_int():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(
            llvm.types.function(i32, [i32, i32]), "binops", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        x, y = fn.arg(0), fn.arg(1)
        with b.at_end_of(bb):
            r_sub = b.sub(x, y, "r_sub")
            r_mul = b.mul(x, y, "r_mul")
            r_sdiv = b.sdiv(x, y, "r_sdiv")
            r_udiv = b.udiv(x, y, "r_udiv")
            b.ret(r_sub)
        printed = str(mod)
        assert "%r_sub = sub i32 %0, %1" in printed
        assert "%r_mul = mul i32 %0, %1" in printed
        assert "%r_sdiv = sdiv i32 %0, %1" in printed
        assert "%r_udiv = udiv i32 %0, %1" in printed
        del b, fn, bb, mod
    assert_no_leaks()


def test_binary_ops_float():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.types.f32(ctx)
        fn = llvm.Function.create(
            llvm.types.function(f32, [f32, f32]), "fbinops", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        x, y = fn.arg(0), fn.arg(1)
        with b.at_end_of(bb):
            r_fadd = b.fadd(x, y, "r_fadd")
            r_fsub = b.fsub(x, y, "r_fsub")
            r_fmul = b.fmul(x, y, "r_fmul")
            r_fdiv = b.fdiv(x, y, "r_fdiv")
            b.ret(r_fadd)
        printed = str(mod)
        assert "%r_fadd = fadd float %0, %1" in printed
        assert "%r_fsub = fsub float %0, %1" in printed
        assert "%r_fmul = fmul float %0, %1" in printed
        assert "%r_fdiv = fdiv float %0, %1" in printed
        del b, fn, bb, mod
    assert_no_leaks()


def test_icmp():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        i1 = llvm.types.i1(ctx)
        fn = llvm.Function.create(
            llvm.types.function(i1, [i32, i32]), "cmp", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            result = b.icmp(llvm.CmpPredicate.SLT, fn.arg(0), fn.arg(1), "lt")
            b.ret(result)
        printed = str(mod)
        assert "%lt = icmp slt i32 %0, %1" in printed
        assert isinstance(result, llvm.ICmpInst)
        del b, fn, bb, mod
    assert_no_leaks()


def test_fcmp():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f64 = llvm.types.f64(ctx)
        i1 = llvm.types.i1(ctx)
        fn = llvm.Function.create(
            llvm.types.function(i1, [f64, f64]), "fcmp", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            result = b.fcmp(llvm.CmpPredicate.OGT, fn.arg(0), fn.arg(1), "gt")
            b.ret(result)
        printed = str(mod)
        assert "%gt = fcmp ogt double %0, %1" in printed
        assert isinstance(result, llvm.FCmpInst)
        del b, fn, bb, mod
    assert_no_leaks()


def test_gep():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        ptr = llvm.types.ptr(context=ctx)
        fn = llvm.Function.create(
            llvm.types.function(ptr, [ptr, i32]), "gep_test", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            result = b.gep(i32, fn.arg(0), [fn.arg(1)], "elem")
            b.ret(result)
        printed = str(mod)
        assert "%elem = getelementptr i32, ptr %0, i32 %1" in printed
        assert isinstance(result, llvm.GetElementPtrInst)
        del b, fn, bb, mod
    assert_no_leaks()


def test_call():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        callee = llvm.Function.create(
            llvm.types.function(i32, [i32]), "callee", mod
        )
        fn = llvm.Function.create(
            llvm.types.function(i32, [i32]), "caller", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            result = b.call(callee, [fn.arg(0)], "r")
            b.ret(result)
        printed = str(mod)
        assert "%r = call i32 @callee(i32 %0)" in printed
        assert isinstance(result, llvm.CallInst)
        del b, fn, bb, callee, mod
    assert_no_leaks()


def test_i32_const_and_i64_const():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i64 = llvm.types.i64(ctx)
        fn = llvm.Function.create(
            llvm.types.function(i64, []), "consts", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            c32 = b.i32_const(42)
            c64 = b.i64_const(100)
            b.ret(c64)
        assert isinstance(c32, llvm.ConstantInt)
        assert isinstance(c64, llvm.ConstantInt)
        printed = str(mod)
        assert "ret i64 100" in printed
        del b, fn, bb, mod
    assert_no_leaks()


def test_set_insert_point():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [i32]), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        b.set_insert_point(bb)
        assert b.insert_block is bb
        b.ret(fn.arg(0))
        assert "ret i32 %0" in str(mod)
        del b, fn, bb, mod
    assert_no_leaks()


def test_context_manager_does_not_restore():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        void = llvm.types.void(ctx)
        fn = llvm.Function.create(llvm.types.function(void, []), "f", mod)
        outer = fn.append_basic_block("outer")
        inner = fn.append_basic_block("inner")
        b = llvm.IRBuilder(ctx)
        b.set_insert_point(outer)
        assert b.insert_block is outer
        with b.at_end_of(inner):
            assert b.insert_block is inner
        assert b.insert_block is inner
        del b, fn, outer, inner, mod
    assert_no_leaks()


def test_context_manager_propagates_exceptions():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        void = llvm.types.void(ctx)
        fn = llvm.Function.create(llvm.types.function(void, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with pytest.raises(ValueError, match="test error"):
            with b.at_end_of(bb):
                raise ValueError("test error")
        del b, fn, bb, mod
    assert_no_leaks()


def test_context_manager_enter_returns_none():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        void = llvm.types.void(ctx)
        fn = llvm.Function.create(llvm.types.function(void, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb) as guard:
            assert guard is None
        del b, fn, bb, mod
    assert_no_leaks()
