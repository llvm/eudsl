#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_GLOBALS_SRC = dedent(
    """\
    @g_const  = constant i32 42
    @g_var    = global i32 7
    @g_noinit = external global i32
    """
)

_AGGREGATE_SRC = dedent(
    """\
    @g_var  = global i32 7
    @arr    = constant [6 x i8] c"hello\\00"
    @zero   = constant [4 x i32] zeroinitializer
    @vec    = constant <4 x i32> <i32 1, i32 2, i32 3, i32 4>
    @strukt = constant { i32, float } { i32 1, float 2.0 }
    @expr   = constant i32* getelementptr (i32, i32* @g_var, i32 1)
    """
)


def test_const_int():
    with llvm.Context() as ctx:
        c = llvm.const_int(llvm.types.i32(ctx), 42)
        assert type(c).__name__ == "ConstantInt"
        assert c.value == 42
        assert str(c) == "i32 42"
        neg = llvm.const_int(llvm.types.i32(ctx), -1, signed=True)
        assert neg.value == -1
        assert neg.zext_value == 4294967295
    assert_no_leaks()


def test_const_int_out_of_range_raises():
    with llvm.Context() as ctx:
        i8 = llvm.types.i8(ctx)
        with pytest.raises(ValueError):
            llvm.const_int(i8, 300, signed=False)
        with pytest.raises(ValueError):
            llvm.const_int(i8, -1, signed=False)
        with pytest.raises(ValueError):
            llvm.const_int(i8, 200, signed=True)
        # in-range values on both sides of zero still work
        assert llvm.const_int(i8, 255, signed=False).zext_value == 255
        assert llvm.const_int(i8, -128, signed=True).value == -128
    assert_no_leaks()


def test_const_bool_and_fp():
    with llvm.Context() as ctx:
        t = llvm.const_bool(ctx, True)
        assert type(t).__name__ == "ConstantInt"
        assert str(t) == "i1 true"
        f = llvm.const_fp(llvm.types.f64(ctx), 1.5)
        assert type(f).__name__ == "ConstantFP"
        assert f.double_value == 1.5
    assert_no_leaks()


def test_undef_poison_null():
    with llvm.Context() as ctx:
        assert type(llvm.undef(llvm.types.i32(ctx))).__name__ == "UndefValue"
        assert type(llvm.poison(llvm.types.i32(ctx))).__name__ == "PoisonValue"
        assert type(llvm.null(llvm.types.ptr(ctx))).__name__ == "ConstantPointerNull"
        assert llvm.null is llvm.const_null
        assert str(llvm.undef(llvm.types.i32(ctx))) == "i32 undef"
    assert_no_leaks()


def test_global_variable_is_constant_and_initializer():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_GLOBALS_SRC, ctx, "m")

        g_const = mod.get_global_variable("g_const")
        assert g_const.is_constant is True
        assert type(g_const.initializer).__name__ == "ConstantInt"
        assert g_const.initializer.value == 42

        g_var = mod.get_global_variable("g_var")
        assert g_var.is_constant is False
        assert type(g_var.initializer).__name__ == "ConstantInt"
        assert g_var.initializer.value == 7

        g_noinit = mod.get_global_variable("g_noinit")
        assert g_noinit.is_constant is False
        assert g_noinit.initializer is None

        assert {g.name for g in mod.globals} == {"g_const", "g_var", "g_noinit"}
        del mod
    assert_no_leaks()


def test_aggregate_and_expr_constant_downcasts():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_AGGREGATE_SRC, ctx, "m")

        assert type(mod.get_global_variable("arr").initializer).__name__ == (
            "ConstantDataArray"
        )
        assert type(mod.get_global_variable("zero").initializer).__name__ == (
            "ConstantAggregateZero"
        )
        assert type(mod.get_global_variable("vec").initializer).__name__ == (
            "ConstantDataVector"
        )
        assert type(mod.get_global_variable("strukt").initializer).__name__ == (
            "ConstantStruct"
        )
        assert type(mod.get_global_variable("expr").initializer).__name__ == (
            "ConstantExpr"
        )
        del mod
    assert_no_leaks()
