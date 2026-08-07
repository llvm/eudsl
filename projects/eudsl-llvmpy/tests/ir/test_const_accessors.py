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

_SRC = dedent(
    """\
    @g = global i32 7
    @darr = global [4 x i8] c"abcd"
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
      %v = load i32, ptr @dvec
      %fv = load double, ptr @fvec
      %c = load i64, ptr @ce
      %a = load i32, ptr @al
      %i = load ptr, ptr @if
      %bp = load ptr, ptr @ba
      ret i32 0
    }
    """
)


def test_constant_and_global_accessors():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        ptr_of = {}  # global name -> its pointer operand (the global/alias/ifunc)
        for i in f.walk():
            if isinstance(i, llvm.LoadInst):
                g = i.pointer_operand
                ptr_of[g.name] = g

        darr = ptr_of["darr"].initializer  # ConstantDataArray
        assert darr.num_elements == 4
        assert darr.get_element_as_int(0) == ord("a")
        assert darr.is_string
        assert darr.as_string == "abcd"

        dvec = ptr_of["dvec"].initializer  # ConstantDataVector (ints)
        assert dvec.num_elements == 4
        assert dvec.get_element_as_int(1) == 20

        fvec = ptr_of["fvec"].initializer  # ConstantDataVector (doubles)
        assert fvec.get_element_as_double(0) == 1.5
        assert not fvec.is_string
        with pytest.raises(ValueError, match="not a string"):
            _ = fvec.as_string

        ce = ptr_of["ce"].initializer  # ConstantExpr
        assert ce.opcode_name == "ptrtoint"

        ba = ptr_of["ba"].initializer  # BlockAddress
        assert ba.function.name == "f"
        assert ba.basic_block.name == "b"

        al = ptr_of["al"]  # GlobalAlias
        assert al.aliasee.name == "g"

        ifu = ptr_of["if"]  # GlobalIFunc
        assert ifu.resolver.name == "res"

        del f, ptr_of, darr, dvec, fvec, ce, ba, al, ifu, mod
    assert_no_leaks()
