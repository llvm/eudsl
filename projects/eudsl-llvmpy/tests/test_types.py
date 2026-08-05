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


# Derived-type str() round-trips. Concrete-subclass accessors (.bit_width etc.)
# and downcasting are validated in test_types_downcast (added with the Type
# type_hook), since until the hook lands the factories return base Type objects.
def test_derived_types_print():
    with llvm.Context() as ctx:
        assert str(llvm.int_t(ctx, 7)) == "i7"
        assert str(llvm.ptr_t(ctx)) == "ptr"
        assert str(llvm.ptr_t(ctx, 3)) == "ptr addrspace(3)"
        assert str(llvm.array_t(llvm.i32(ctx), 4)) == "[4 x i32]"
        assert str(llvm.vector_t(llvm.f32(ctx), 8)) == "<8 x float>"
        assert str(llvm.vector_t(llvm.f32(ctx), 8, scalable=True)) == "<vscale x 8 x float>"
        assert str(llvm.struct_t(ctx, [llvm.i32(ctx), llvm.f64(ctx)])) == "{ i32, double }"
        assert (
            str(llvm.struct_t(ctx, [llvm.i8(ctx), llvm.i32(ctx)], packed=True))
            == "<{ i8, i32 }>"
        )
        assert (
            str(llvm.function_t(llvm.i32(ctx), [llvm.i32(ctx), llvm.f32(ctx)]))
            == "i32 (i32, float)"
        )
        assert (
            str(llvm.function_t(llvm.void_t(ctx), [llvm.ptr_t(ctx)], var_arg=True))
            == "void (ptr, ...)"
        )
    assert_no_leaks()


def test_named_struct_prints_opaque():
    with llvm.Context() as ctx:
        named = llvm.named_struct_t(ctx, "Pair")
        # An opaque named struct prints its full definition. set_body and the
        # concrete-subclass accessors are validated in test_types_downcast,
        # once the Type type_hook makes named_struct_t return a StructType.
        assert str(named) == "%Pair = type opaque"
    assert_no_leaks()
