#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exercises DSL API surface and error paths for coverage + correctness."""
import ctypes

import pytest

import llvm
from llvm.dsl import casters as _c
from llvm.dsl.casters import maybe_downcast, register_value_caster
from llvm.dsl.context import building, current_builder, current_function
from llvm.dsl.values import ArithValue, with_element_type, extract
from llvm.testing import assert_no_leaks


def _entry(ctx, mod, ret_ty, arg_tys, name="f"):
    fn = llvm.ir.Function.create(llvm.types.function(ret_ty, arg_tys), name, mod)
    bb = fn.append_basic_block("entry")
    args = [maybe_downcast(fn.arg(i), fn) for i in range(len(arg_tys))]
    return fn, bb, args


def test_all_binary_dunders():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32, f32 = llvm.types.i32(ctx), llvm.types.f32(ctx)
        fn, bb, (a, b) = _entry(ctx, mod, i32, [i32, i32])
        bld = llvm.ir.IRBuilder(ctx)
        with bld.at_end_of(bb), building(bld):
            _ = a - b
            _ = a * b
            _ = a / b
            _ = 3 + a       # __radd__
            _ = 3 * a       # __rmul__
            bld.ret(a + b)
        p = str(mod)
        assert "sub i32" in p and "mul i32" in p and "sdiv i32" in p
        del bld, fn, mod
    assert_no_leaks()


def test_float_binary_and_compare():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32(ctx)
        fn, bb, (a, b) = _entry(ctx, mod, f32, [f32, f32])
        bld = llvm.ir.IRBuilder(ctx)
        with bld.at_end_of(bb), building(bld):
            _ = a - b
            _ = a * b
            _ = a / b
            _ = a <= b
            _ = a >= b
            _ = a.eq(b)
            _ = a.ne(b)
            bld.ret(a + b)
        p = str(mod)
        assert "fsub" in p and "fmul" in p and "fdiv" in p
        assert "fcmp ole" in p and "fcmp oge" in p
        assert "fcmp oeq" in p and "fcmp one" in p
        del bld, fn, mod
    assert_no_leaks()


def test_int_le_ge_eq_ne():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn, bb, (a, b) = _entry(ctx, mod, llvm.types.i1(ctx), [i32, i32])
        bld = llvm.ir.IRBuilder(ctx)
        with bld.at_end_of(bb), building(bld):
            _ = a <= b
            _ = a >= b
            _ = a.ne(b)
            bld.ret(a.eq(b))
        p = str(mod)
        assert "icmp sle" in p and "icmp sge" in p
        assert "icmp eq" in p and "icmp ne" in p
        del bld, fn, mod
    assert_no_leaks()


def test_float_scalar_coercion():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32(ctx)
        fn, bb, (a,) = _entry(ctx, mod, f32, [f32])
        bld = llvm.ir.IRBuilder(ctx)
        with bld.at_end_of(bb), building(bld):
            bld.ret(a + 1.5)  # float ArithValue + Python float -> const_fp
        assert "fadd float" in str(mod)
        del bld, fn, mod
    assert_no_leaks()


def test_insert_value_and_extract():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        st = llvm.types.struct([i32, i32], context=ctx)
        fn = llvm.ir.Function.create(llvm.types.function(st, [st, i32]), "f", mod)
        bb = fn.append_basic_block("entry")
        bld = llvm.ir.IRBuilder(ctx)
        with bld.at_end_of(bb), building(bld):
            agg = fn.arg(0)
            v = maybe_downcast(fn.arg(1), fn)
            agg2 = bld.insert_value(agg, v, 1)
            first = extract(agg2, 0)
            assert isinstance(first, ArithValue)
            bld.ret(agg2)
        p = str(mod)
        assert "insertvalue" in p and "extractvalue" in p
        del bld, fn, mod
    assert_no_leaks()


def test_insert_extract_value_index_via_jit():
    # Pin the index argument of insert_value/extract_value at BOTH 0 and 1 by
    # executing: a wrong or dropped index would return the other field.
    ctx = llvm.ir.Context()
    mod = llvm.ir.Module("m", ctx)
    i32 = llvm.types.i32(ctx)
    st = llvm.types.struct([i32, i32], context=ctx)

    def build(name, idx):
        fn = llvm.ir.Function.create(llvm.types.function(i32, [i32, i32]), name, mod)
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(fn.append_basic_block("entry")):
            agg = llvm.ir.undef(st)
            agg = b.insert_value(agg, fn.arg(0), 0)
            agg = b.insert_value(agg, fn.arg(1), 1)
            b.ret(b.extract_value(agg, idx))

    build("get0", 0)
    build("get1", 1)
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    sig = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
    g0 = sig(jit.lookup("get0"))
    g1 = sig(jit.lookup("get1"))
    assert g0(10, 20) == 10  # field 0
    assert g1(10, 20) == 20  # field 1
    del jit, mod, ctx, g0, g1, sig, st, i32


def test_typed_pointer_setitem_with_value_index():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.ir.Function.create(
            llvm.types.function(llvm.types.void(ctx), [llvm.types.ptr(context=ctx), i32]), "f", mod
        )
        bb = fn.append_basic_block("entry")
        bld = llvm.ir.IRBuilder(ctx)
        with bld.at_end_of(bb), building(bld):
            tp = with_element_type(fn.arg(0), i32)
            idx = maybe_downcast(fn.arg(1), fn)
            tp[idx] = llvm.ir.const_int(i32, 9)  # Value index (not int)
            bld.ret(None)
        assert "getelementptr" in str(mod)
        del bld, fn, mod
    assert_no_leaks()


def test_maybe_downcast_no_caster_passthrough():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        # A void-typed value has no registered caster: returned unchanged.
        fn = llvm.ir.Function.create(llvm.types.function(llvm.types.void(ctx), []), "f", mod)
        v = maybe_downcast(fn, fn)  # Function's type kind has no caster
        assert not isinstance(v, ArithValue)
        del fn, mod
    assert_no_leaks()


def test_current_builder_and_function_raise_outside_context():
    with pytest.raises(RuntimeError, match="no current IRBuilder"):
        current_builder()
    with pytest.raises(RuntimeError, match="no current function"):
        current_function()


def test_register_value_caster_decorator_form():
    # Exercise the decorator form and unregister afterward.
    sentinel = object()

    @register_value_caster(llvm.types.TypeID.Void)
    def _caster(v):  # pragma: no cover - not invoked, just registered
        return sentinel

    assert _c._casters[llvm.types.TypeID.Void] is _caster
    del _c._casters[llvm.types.TypeID.Void]
