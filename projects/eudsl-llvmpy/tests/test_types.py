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


def test_types_downcast_to_concrete_classes():
    with llvm.Context() as ctx:
        assert type(llvm.i32(ctx)).__name__ == "IntegerType"
        assert type(llvm.ptr_t(ctx)).__name__ == "PointerType"
        assert type(llvm.array_t(llvm.i32(ctx), 2)).__name__ == "ArrayType"
        assert type(llvm.vector_t(llvm.i32(ctx), 2)).__name__ == "VectorType"
        assert type(llvm.struct_t(ctx, [llvm.i32(ctx)])).__name__ == "StructType"
        assert (
            type(llvm.function_t(llvm.void_t(ctx), [])).__name__ == "FunctionType"
        )
        # Types with no concrete subclass stay Type.
        assert type(llvm.void_t(ctx)).__name__ == "Type"
        assert type(llvm.f64(ctx)).__name__ == "Type"
    assert_no_leaks()


def test_concrete_type_accessors():
    with llvm.Context() as ctx:
        assert llvm.int_t(ctx, 7).bit_width == 7
        assert llvm.ptr_t(ctx, 3).address_space == 3
        a = llvm.array_t(llvm.i32(ctx), 4)
        assert a.num_elements == 4
        assert a.element_type == llvm.i32(ctx)
        v = llvm.vector_t(llvm.f32(ctx), 8)
        assert v.min_num_elements == 8
        assert not v.is_scalable
        assert llvm.vector_t(llvm.f32(ctx), 8, scalable=True).is_scalable
        s = llvm.struct_t(ctx, [llvm.i32(ctx), llvm.f64(ctx)])
        assert s.num_elements == 2
        assert s.element_type(1) == llvm.f64(ctx)
        assert not s.is_packed
        assert llvm.struct_t(ctx, [llvm.i8(ctx)], packed=True).is_packed
        ft = llvm.function_t(llvm.i32(ctx), [llvm.i32(ctx), llvm.f32(ctx)])
        assert ft.return_type == llvm.i32(ctx)
        assert ft.num_params == 2
        assert ft.param_type(1) == llvm.f32(ctx)
        assert ft.params == [llvm.i32(ctx), llvm.f32(ctx)]
        assert not ft.is_var_arg
    assert_no_leaks()


def test_named_struct_set_body_and_name():
    with llvm.Context() as ctx:
        named = llvm.named_struct_t(ctx, "Pair")
        assert named.name == "Pair"
        assert named.is_opaque
        named.set_body([llvm.i32(ctx), llvm.i32(ctx)])
        assert not named.is_opaque
        assert named.num_elements == 2
    assert_no_leaks()
