#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Types as nb::is_generic GenericAliases: subscript with no live context, and
.get(...) factories that evaluate against a context."""
import types

import llvm
from llvm.testing import assert_no_leaks
from llvm.types import (
    ArrayType,
    FunctionType,
    IntegerType,
    PointerType,
    StructType,
    VectorType,
)

# Subscripting a type class needs NO live context: it produces an unevaluated
# GenericAlias. These are built at import time, before any Context exists.
I32 = IntegerType[32]
PTR = PointerType[0]
FN = FunctionType[IntegerType[32], [PointerType[0], IntegerType[64]]]


def test_subscript_yields_generic_alias_without_context():
    assert type(I32) is types.GenericAlias
    assert I32.__origin__ is IntegerType
    assert type(FN) is types.GenericAlias
    # Nested aliases stay unevaluated too.
    assert type(FN.__origin__) is type(FunctionType)


def test_get_evaluates_against_explicit_context():
    with llvm.ir.Context() as ctx:
        assert str(IntegerType.get(32, context=ctx)) == "i32"
        assert str(PointerType.get(context=ctx)) == "ptr"
        assert str(PointerType.get(3, context=ctx)) == "ptr addrspace(3)"
        i32 = llvm.types.i32(ctx)
        f64 = llvm.types.f64(ctx)
        assert str(ArrayType.get(i32, 4, context=ctx)) == "[4 x i32]"
        assert str(VectorType.get(i32, 4, context=ctx)) == "<4 x i32>"
        assert str(StructType.get([i32, f64], context=ctx)) == "{ i32, double }"
        assert (
            str(FunctionType.get(i32, [PointerType.get(context=ctx)], context=ctx))
            == "i32 (ptr)"
        )
    assert_no_leaks()


def test_get_uses_current_context_when_omitted():
    with llvm.ir.Context() as ctx:
        # No explicit context= -> the current `with Context():` is used.
        assert IntegerType.get(32) is llvm.types.i32(ctx)
        assert str(PointerType.get()) == "ptr"
    assert_no_leaks()


def test_get_is_interned_identical_to_primitive_factory():
    with llvm.ir.Context() as ctx:
        assert IntegerType.get(32, context=ctx) is llvm.types.i32(ctx)
        assert PointerType.get(context=ctx) is llvm.types.ptr(context=ctx)
    assert_no_leaks()
