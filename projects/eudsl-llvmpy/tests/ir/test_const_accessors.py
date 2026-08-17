#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Accessors for the constant/global kinds that were bare registrations:
ConstantDataArray/Vector element and string access, ConstantExpr.opcode_name,
BlockAddress.function/basic_block, GlobalAlias.aliasee, GlobalIFunc.resolver.
"""

from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent("""\
    @g = global i32 7
    @darr = global [4 x i8] c"abcd"
    @sbytes = global [2 x i8] c"\\FF\\7F"
    @dvec = global <4 x i32> <i32 10, i32 20, i32 30, i32 40>
    @fvec = global <2 x double> <double 1.5, double 2.5>
    @ce = global i64 ptrtoint (ptr @g to i64)
    @al = alias i32, ptr @g
    @res = global ptr null
    @if = ifunc i32 (), ptr @res
    @ba = global ptr blockaddress(@f, %b)

    define i32 @f() {
    entry:
      br label %b
    b:
      %d = load i8, ptr @darr
      %sb = load i8, ptr @sbytes
      %v = load i32, ptr @dvec
      %fv = load double, ptr @fvec
      %c = load i64, ptr @ce
      %a = load i32, ptr @al
      %i = load ptr, ptr @if
      %bp = load ptr, ptr @ba
      ret i32 0
    }
    """)


def _globals(ctx):
    """Parse _SRC and map each global's name to the pointer operand (the
    global/alias/ifunc) of the load that reads it."""
    mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
    f = mod.get_function("f")
    ptr_of = {}
    for i in f.walk():
        if isinstance(i, llvm.ir.LoadInst):
            g = i.pointer_operand
            ptr_of[g.name] = g
    return mod, ptr_of


def test_constant_data_array_string():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        darr = ptr_of["darr"].initializer  # ConstantDataArray
        assert darr.num_elements == 4
        assert darr.get_element_as_int(0) == ord("a")
        assert darr.is_string
        assert darr.as_string == "abcd"
        del mod, ptr_of, darr
    assert_no_leaks()


def test_get_element_as_int_is_unsigned():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        # get_element_as_int has only the unsigned form (unlike ConstantInt's
        # value/zext_value): a high-bit byte reads unsigned, not sign-extended.
        sbytes = ptr_of["sbytes"].initializer
        assert sbytes.get_element_as_int(0) == 0xFF  # 255, not -1
        assert sbytes.get_element_as_int(1) == 0x7F
        del mod, ptr_of, sbytes
    assert_no_leaks()


def test_constant_data_vector_int():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        dvec = ptr_of["dvec"].initializer  # ConstantDataVector (ints)
        assert dvec.num_elements == 4
        assert dvec.get_element_as_int(1) == 20
        del mod, ptr_of, dvec
    assert_no_leaks()


def test_constant_data_vector_double():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        fvec = ptr_of["fvec"].initializer  # ConstantDataVector (doubles)
        assert fvec.get_element_as_double(0) == 1.5
        assert not fvec.is_string
        with pytest.raises(ValueError, match="not a string"):
            _ = fvec.as_string
        del mod, ptr_of, fvec
    assert_no_leaks()


def test_get_element_out_of_range_raises():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        dvec = ptr_of["dvec"].initializer  # 4 elements
        fvec = ptr_of["fvec"].initializer  # 2 elements
        # A bad index would trip LLVM's assert and abort the interpreter; the
        # binding must raise IndexError instead.
        with pytest.raises(IndexError):
            dvec.get_element_as_int(4)
        with pytest.raises(IndexError):
            dvec.get_element_as_int(-1)
        with pytest.raises(IndexError):
            fvec.get_element_as_double(2)
        del mod, ptr_of, dvec, fvec
    assert_no_leaks()


def test_get_element_wrong_type_raises():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        dvec = ptr_of["dvec"].initializer  # int elements
        fvec = ptr_of["fvec"].initializer  # double elements
        # Each accessor asserts its element type in LLVM; guard with ValueError.
        with pytest.raises(ValueError, match="not double"):
            dvec.get_element_as_double(0)
        with pytest.raises(ValueError, match="not an integer"):
            fvec.get_element_as_int(0)
        del mod, ptr_of, dvec, fvec
    assert_no_leaks()


def test_constant_expr_opcode_name():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        ce = ptr_of["ce"].initializer  # ConstantExpr
        assert ce.opcode_name == "ptrtoint"
        del mod, ptr_of, ce
    assert_no_leaks()


def test_block_address():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        ba = ptr_of["ba"].initializer  # BlockAddress
        assert ba.function.name == "f"
        assert ba.basic_block.name == "b"
        del mod, ptr_of, ba
    assert_no_leaks()


def test_global_alias():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        al = ptr_of["al"]  # GlobalAlias
        assert al.aliasee.name == "g"
        del mod, ptr_of, al
    assert_no_leaks()


def test_global_ifunc():
    with llvm.ir.Context() as ctx:
        mod, ptr_of = _globals(ctx)
        ifu = ptr_of["if"]  # GlobalIFunc
        assert ifu.resolver.name == "res"
        del mod, ptr_of, ifu
    assert_no_leaks()
