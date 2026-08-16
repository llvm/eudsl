#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_primitive_types_print():
    with llvm.Context() as ctx:
        assert str(llvm.types.void(ctx)) == "void"
        assert str(llvm.types.label(ctx)) == "label"
        assert str(llvm.types.i1(ctx)) == "i1"
        assert str(llvm.types.i8(ctx)) == "i8"
        assert str(llvm.types.i16(ctx)) == "i16"
        assert str(llvm.types.i32(ctx)) == "i32"
        assert str(llvm.types.i64(ctx)) == "i64"
        assert str(llvm.types.f16(ctx)) == "half"
        assert str(llvm.types.f32(ctx)) == "float"
        assert str(llvm.types.f64(ctx)) == "double"
    assert_no_leaks()


def test_type_predicates():
    with llvm.Context() as ctx:
        assert llvm.types.void(ctx).is_void
        assert not llvm.types.void(ctx).is_sized
        assert llvm.types.i32(ctx).is_integer
        assert not llvm.types.i32(ctx).is_floating_point
        assert llvm.types.i32(ctx).type_id == llvm.types.TypeID.Integer
        assert llvm.types.f64(ctx).is_floating_point
        assert llvm.types.i32(ctx).is_sized
        assert llvm.types.ptr(context=ctx).is_pointer
        assert not llvm.types.i32(ctx).is_pointer
        assert llvm.types.label(ctx).is_label
        assert not llvm.types.i32(ctx).is_label
        assert not llvm.types.void(ctx).is_integer
    assert_no_leaks()


def test_types_are_uniqued_and_hashable():
    with llvm.Context() as a, llvm.Context() as b:
        assert llvm.types.i32(a) == llvm.types.i32(a)
        assert llvm.types.i32(a) != llvm.types.i64(a)
        # Types are interned per context, so two contexts give distinct types.
        assert llvm.types.i32(a) != llvm.types.i32(b)
        assert len({llvm.types.i32(a), llvm.types.i32(a), llvm.types.i64(a)}) == 2
        # __eq__ falls back to False (rather than raising) against non-Type
        # operands, so equality against unrelated Python objects works normally.
        assert llvm.types.i32(a) != 5
        assert llvm.types.i32(a) != "i32"
    assert_no_leaks()


# Derived-type str() round-trips. Concrete-subclass accessors (.bit_width etc.)
# and downcasting are validated in test_types_downcast_to_concrete_classes and
# test_concrete_type_accessors, since the Type type_hook makes the factories
# below return concrete-subclass instances rather than base Type objects.
def test_derived_types_print():
    with llvm.Context() as ctx:
        assert str(llvm.types.int(7, context=ctx)) == "i7"
        assert str(llvm.types.ptr(context=ctx)) == "ptr"
        assert str(llvm.types.ptr(3, context=ctx)) == "ptr addrspace(3)"
        assert str(llvm.types.array(llvm.types.i32(ctx), 4)) == "[4 x i32]"
        assert str(llvm.types.vector(llvm.types.f32(ctx), 8)) == "<8 x float>"
        assert str(llvm.types.vector(llvm.types.f32(ctx), 8, scalable=True)) == "<vscale x 8 x float>"
        assert str(llvm.types.struct([llvm.types.i32(ctx), llvm.types.f64(ctx)], context=ctx)) == "{ i32, double }"
        assert (
            str(llvm.types.struct([llvm.types.i8(ctx), llvm.types.i32(ctx)], packed=True, context=ctx))
            == "<{ i8, i32 }>"
        )
        assert (
            str(llvm.types.function(llvm.types.i32(ctx), [llvm.types.i32(ctx), llvm.types.f32(ctx)]))
            == "i32 (i32, float)"
        )
        assert (
            str(llvm.types.function(llvm.types.void(ctx), [llvm.types.ptr(context=ctx)], var_arg=True))
            == "void (ptr, ...)"
        )
    assert_no_leaks()


def test_named_struct_prints_opaque():
    with llvm.Context() as ctx:
        named = llvm.types.named_struct("Pair", context=ctx)
        # An opaque named struct prints its full definition. set_body and the
        # concrete-subclass accessors are validated in
        # test_named_struct_set_body_and_name.
        assert str(named) == "%Pair = type opaque"
    assert_no_leaks()


def test_types_downcast_to_concrete_classes():
    with llvm.Context() as ctx:
        assert type(llvm.types.i32(ctx)).__name__ == "IntegerType"
        assert type(llvm.types.ptr(context=ctx)).__name__ == "PointerType"
        assert type(llvm.types.array(llvm.types.i32(ctx), 2)).__name__ == "ArrayType"
        # A non-scalable vector downcasts to FixedVectorType, a scalable one to
        # ScalableVectorType; both are VectorType subclasses.
        assert (
            type(llvm.types.vector(llvm.types.i32(ctx), 2)).__name__
            == "FixedVectorType"
        )
        assert (
            type(llvm.types.vector(llvm.types.i32(ctx), 2, scalable=True)).__name__
            == "ScalableVectorType"
        )
        assert isinstance(
            llvm.types.vector(llvm.types.i32(ctx), 2), llvm.types.VectorType
        )
        assert type(llvm.types.struct([llvm.types.i32(ctx)], context=ctx)).__name__ == "StructType"
        assert (
            type(llvm.types.function(llvm.types.void(ctx), [])).__name__ == "FunctionType"
        )
        # Types with no concrete subclass stay Type.
        assert type(llvm.types.void(ctx)).__name__ == "Type"
        assert type(llvm.types.f64(ctx)).__name__ == "Type"
    assert_no_leaks()


def test_concrete_type_accessors():
    with llvm.Context() as ctx:
        assert llvm.types.int(7, context=ctx).bit_width == 7
        assert llvm.types.ptr(3, context=ctx).address_space == 3
        a = llvm.types.array(llvm.types.i32(ctx), 4)
        assert a.num_elements == 4
        assert a.element_type == llvm.types.i32(ctx)
        v = llvm.types.vector(llvm.types.f32(ctx), 8)
        assert v.min_num_elements == 8
        assert not v.is_scalable
        assert v.element_type == llvm.types.f32(ctx)
        fv = llvm.types.vector(llvm.types.i32(ctx), 3)
        assert fv.num_elements == 3
        sv = llvm.types.vector(llvm.types.f32(ctx), 8, scalable=True)
        assert sv.is_scalable
        assert sv.min_num_elements == 8
        # A scalable vector has no fixed element count, so the FixedVectorType
        # accessor is absent on the ScalableVectorType downcast.
        assert not hasattr(sv, "num_elements")
        s = llvm.types.struct([llvm.types.i32(ctx), llvm.types.f64(ctx)], context=ctx)
        assert s.num_elements == 2
        assert s.element_type(1) == llvm.types.f64(ctx)
        assert not s.is_packed
        assert llvm.types.struct([llvm.types.i8(ctx)], packed=True, context=ctx).is_packed
        ft = llvm.types.function(llvm.types.i32(ctx), [llvm.types.i32(ctx), llvm.types.f32(ctx)])
        assert ft.return_type == llvm.types.i32(ctx)
        assert ft.num_params == 2
        assert ft.param_type(1) == llvm.types.f32(ctx)
        assert ft.params == [llvm.types.i32(ctx), llvm.types.f32(ctx)]
        assert not ft.is_var_arg
        assert llvm.types.function(llvm.types.void(ctx), [], var_arg=True).is_var_arg
    assert_no_leaks()


def test_named_struct_set_body_and_name():
    with llvm.Context() as ctx:
        named = llvm.types.named_struct("Pair", context=ctx)
        assert named.name == "Pair"
        assert named.is_opaque
        named.set_body([llvm.types.i32(ctx), llvm.types.i32(ctx)])
        assert not named.is_opaque
        assert named.num_elements == 2
        assert not named.is_packed
        packed = llvm.types.named_struct("Packed", context=ctx)
        packed.set_body([llvm.types.i8(ctx), llvm.types.i32(ctx)], packed=True)
        assert packed.is_packed
    assert_no_leaks()
