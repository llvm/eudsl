#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_primitive_types_print():
    with llvm.Context() as ctx:
        assert str(llvm.void_t(ctx)) == "void"
        assert str(llvm.i1(ctx)) == "i1"
        assert str(llvm.i32(ctx)) == "i32"
        assert str(llvm.f32(ctx)) == "float"
        assert str(llvm.f64(ctx)) == "double"
        assert str(llvm.f16(ctx)) == "half"
    assert_no_leaks()


def test_type_predicates():
    with llvm.Context() as ctx:
        assert llvm.void_t(ctx).is_void
        assert not llvm.void_t(ctx).is_sized
        assert llvm.i32(ctx).is_integer
        assert not llvm.i32(ctx).is_floating_point
        assert llvm.f64(ctx).is_floating_point
        assert llvm.i32(ctx).is_sized
    assert_no_leaks()


def test_types_are_uniqued_and_hashable():
    with llvm.Context() as a, llvm.Context() as b:
        assert llvm.i32(a) == llvm.i32(a)
        assert llvm.i32(a) != llvm.i64(a)
        # Types are interned per context, so two contexts give distinct types.
        assert llvm.i32(a) != llvm.i32(b)
        assert len({llvm.i32(a), llvm.i32(a), llvm.i64(a)}) == 2
    assert_no_leaks()
