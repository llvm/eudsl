#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Types as nb::is_generic GenericAliases: subscript with no live context, and
.get(...) factories that evaluate against a context."""

import types
from typing import get_args

import pytest

import llvm
from llvm.dsl.func import _resolve
from llvm.testing import assert_no_leaks
from llvm.types import (
    ArrayType,
    FixedVectorType,
    FunctionType,
    IntegerType,
    PointerType,
    ScalableVectorType,
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
    # Nested aliases stay unevaluated: the return-type arg and each element of
    # the params list are still GenericAliases, not eagerly built Types. (A
    # tautology like `type(FN.__origin__) is type(FunctionType)` would pass even
    # if nesting were evaluated eagerly, so inspect the args instead.)
    ret_arg, params_arg = get_args(FN)
    assert type(ret_arg) is types.GenericAlias
    assert [type(p) for p in params_arg] == [types.GenericAlias, types.GenericAlias]


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


def test_get_covers_packed_struct_and_var_arg_function():
    # The optional surfaced flags (StructType packed, FunctionType var_arg) are
    # otherwise uncovered; pin their printed forms.
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32(ctx)
        assert str(StructType.get([i32], packed=True, context=ctx)) == "<{ i32 }>"
        assert (
            str(FunctionType.get(i32, [i32], var_arg=True, context=ctx))
            == "i32 (i32, ...)"
        )
    assert_no_leaks()


def test_element_deriving_get_rejects_foreign_context():
    # ArrayType/VectorType/FunctionType (and the concrete vector subtypes) derive
    # their context from the element or return type. Passing a *different* context
    # is a mistake, not a silent no-op: it is rejected rather than quietly
    # building in the element's context.
    with llvm.ir.Context() as ctx_a, llvm.ir.Context() as ctx_b:
        i32_a = llvm.types.i32(ctx_a)
        i32_b = llvm.types.i32(ctx_b)
        with pytest.raises(ValueError, match="different context"):
            ArrayType.get(i32_a, 4, context=ctx_b)
        with pytest.raises(ValueError, match="different context"):
            VectorType.get(i32_a, 4, context=ctx_b)
        with pytest.raises(ValueError, match="different context"):
            FixedVectorType.get(i32_a, 4, context=ctx_b)
        with pytest.raises(ValueError, match="different context"):
            ScalableVectorType.get(i32_a, 4, context=ctx_b)
        # FunctionType checks both the return type and every parameter.
        with pytest.raises(ValueError, match="return type belongs"):
            FunctionType.get(i32_a, [], context=ctx_b)
        with pytest.raises(ValueError, match="parameter type belongs"):
            # Return type matches ctx_b, but a parameter is from ctx_a.
            FunctionType.get(i32_b, [i32_a], context=ctx_b)
        # Same context is accepted.
        assert str(ArrayType.get(i32_a, 4, context=ctx_a)) == "[4 x i32]"
    assert_no_leaks()


def test_bare_get_missing_required_args_raises():
    # Bare `.get()` without its required arguments fails cleanly rather than
    # producing a half-built type.
    with llvm.ir.Context():
        with pytest.raises(TypeError):
            IntegerType.get()  # missing bits
        with pytest.raises(TypeError):
            ArrayType.get()  # missing element_type and count
    assert_no_leaks()


def test_concrete_vector_subtypes_get_returns_right_subtype():
    # FixedVectorType/ScalableVectorType inherit VectorType's subscript, so each
    # carries its own .get returning the concrete subtype (not a bare
    # VectorType). (Foreign-context rejection is covered in
    # test_element_deriving_get_rejects_foreign_context.)
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32(ctx)
        fv = llvm.types.FixedVectorType.get(i32, 4, context=ctx)
        assert type(fv) is llvm.types.FixedVectorType
        assert str(fv) == "<4 x i32>"
        sv = llvm.types.ScalableVectorType.get(i32, 4, context=ctx)
        assert type(sv) is llvm.types.ScalableVectorType
        assert str(sv) == "<vscale x 4 x i32>"
    assert_no_leaks()


# --- Square-bracket parallels of the .get creation tests above: the same
# constructions written as Type[...] aliases and evaluated (via the same
# _resolve the @function decorator uses) against a context. ---


def test_subscript_evaluates_against_explicit_context():
    with llvm.ir.Context() as ctx:
        assert str(_resolve(IntegerType[32], ctx)) == "i32"
        assert str(_resolve(PointerType[0], ctx)) == "ptr"
        assert str(_resolve(PointerType[3], ctx)) == "ptr addrspace(3)"
        assert str(_resolve(ArrayType[IntegerType[32], 4], ctx)) == "[4 x i32]"
        assert str(_resolve(VectorType[IntegerType[32], 4], ctx)) == "<4 x i32>"
        assert (
            str(_resolve(StructType[IntegerType[32], IntegerType[64]], ctx))
            == "{ i32, i64 }"
        )
        assert (
            str(_resolve(FunctionType[IntegerType[32], [PointerType[0]]], ctx))
            == "i32 (ptr)"
        )
    assert_no_leaks()


def test_subscript_uses_current_context_when_omitted():
    with llvm.ir.Context() as ctx:
        # Resolved against the live context; interns to the same type.
        assert _resolve(IntegerType[32], ctx) is llvm.types.i32(ctx)
        assert str(_resolve(PointerType[0], ctx)) == "ptr"
    assert_no_leaks()


def test_subscript_is_interned_identical_to_primitive_factory():
    with llvm.ir.Context() as ctx:
        assert _resolve(IntegerType[32], ctx) is llvm.types.i32(ctx)
        assert _resolve(PointerType[0], ctx) is llvm.types.ptr(context=ctx)
    assert_no_leaks()


def test_subscript_concrete_vector_subtypes_return_right_subtype():
    with llvm.ir.Context() as ctx:
        fv = _resolve(FixedVectorType[IntegerType[32], 4], ctx)
        assert type(fv) is FixedVectorType
        assert str(fv) == "<4 x i32>"
        sv = _resolve(ScalableVectorType[IntegerType[32], 4], ctx)
        assert type(sv) is ScalableVectorType
        assert str(sv) == "<vscale x 4 x i32>"
    assert_no_leaks()


def test_subscript_rejects_foreign_context_element():
    with llvm.ir.Context() as ctx, llvm.ir.Context() as other:
        i32_other = llvm.types.i32(other)
        with pytest.raises(ValueError, match="different context"):
            _resolve(ArrayType[i32_other, 4], ctx)
    assert_no_leaks()
