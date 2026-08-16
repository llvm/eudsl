#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

import llvm
from llvm.dsl.casters import maybe_downcast, register_value_caster
from llvm.dsl.context import building, current_builder, current_function
from llvm.dsl.values import ArithValue, extract, with_element_type
from llvm.testing import assert_no_leaks, filecheck_with_comments


def _entry(ctx, mod, ret_ty, arg_tys, name="f"):
    fn = llvm.ir.Function.create(llvm.types.function(ret_ty, arg_tys), name, mod)
    bb = fn.append_basic_block("entry")
    # Wrap args the way @function will, so integer/float args are ArithValue.
    args = [maybe_downcast(fn.arg(i), fn) for i in range(len(arg_tys))]
    return fn, bb, args


def test_args_are_arithvalue():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        assert isinstance(args[0], ArithValue)
        del fn, mod
    assert_no_leaks()


def test_integer_add_and_mul():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            r = args[0] * args[1] + 1
            assert isinstance(r, ArithValue)  # result stays typed
            b.ret(r)
        # CHECK: %[[M:.*]] = mul i32 %0, %1
        # CHECK: %[[A:.*]] = add i32 %[[M]], 1
        # CHECK: ret i32 %[[A]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_add_uses_fadd():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, f32, [f32, f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] + args[1])
        # CHECK: %[[R:.*]] = fadd float %0, %1
        # CHECK: ret float %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_sub_uses_fsub():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, f32, [f32, f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] - args[1])
        # CHECK: %[[R:.*]] = fsub float %0, %1
        # CHECK: ret float %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_mul_uses_fmul():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, f32, [f32, f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] * args[1])
        # CHECK: %[[R:.*]] = fmul float %0, %1
        # CHECK: ret float %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_div_uses_fdiv():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, f32, [f32, f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] / args[1])
        # CHECK: %[[R:.*]] = fdiv float %0, %1
        # CHECK: ret float %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_scalar_coercion():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, f32, [f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] + 1.5)
        # CHECK: %[[R:.*]] = fadd float %0, 1.500000e+00
        # CHECK: ret float %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_scalar_coercion():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] + 7)
        # CHECK: %[[R:.*]] = add i32 %0, 7
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_sub():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] - args[1])
        # CHECK: %[[R:.*]] = sub i32 %0, %1
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_div_uses_sdiv():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] / args[1])
        # CHECK: %[[R:.*]] = sdiv i32 %0, %1
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_radd():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(7 + args[0])
        # CHECK: %[[R:.*]] = add i32 %0, 7
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_rmul():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(7 * args[0])
        # CHECK: %[[R:.*]] = mul i32 %0, 7
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_rsub():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(7 - args[0])
        # CHECK: %[[R:.*]] = sub i32 7, %0
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_rdiv():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(7 / args[0])
        # CHECK: %[[R:.*]] = sdiv i32 7, %0
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_mismatched_types_raises_typeerror():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, i32, [i32, f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            with pytest.raises(TypeError, match="mismatched types"):
                args[0] + args[1]
            with pytest.raises(TypeError, match="mismatched types"):
                args[0] < args[1]
        del b, fn, mod
    assert_no_leaks()


def test_le_uses_sle():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] <= args[1])
        # CHECK: %[[R:.*]] = icmp sle i32 %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_ge_uses_sge():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] >= args[1])
        # CHECK: %[[R:.*]] = icmp sge i32 %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_ne_named_method():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0].ne(args[1]))
        # CHECK: %[[R:.*]] = icmp ne i32 %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_comparison_signed():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] < args[1])
        # CHECK: %[[R:.*]] = icmp slt i32 %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_comparison_ordered():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [f32, f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] > args[1])
        # CHECK: %[[R:.*]] = fcmp ogt float %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_gt_uses_sgt():
    # The int path of __gt__ is otherwise only used on floats (-> OGT); pin SGT.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] > args[1])
        # CHECK: %[[R:.*]] = icmp sgt i32 %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, bb, args, mod
    assert_no_leaks()


def test_float_lt_uses_olt():
    # The float path of __lt__ is otherwise only used on ints (-> SLT); pin OLT.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f32 = llvm.types.f32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [f32, f32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] < args[1])
        # CHECK: %[[R:.*]] = fcmp olt float %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, bb, args, mod
    assert_no_leaks()


def test_double_is_arithvalue():
    # The float caster is registered for Double too, not just Float.
    with llvm.ir.Context() as ctx:
        f64 = llvm.types.f64()
        mod = llvm.ir.Module("m", ctx)
        fn, bb, args = _entry(ctx, mod, f64, [f64, f64])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            assert isinstance(args[0], ArithValue)
            b.ret(args[0] + args[1])
        # CHECK: %[[R:.*]] = fadd double %0, %1
        # CHECK: ret double %[[R]]
        filecheck_with_comments(mod)
        del b, fn, bb, args, mod
    assert_no_leaks()


def test_half_is_arithvalue():
    # The float caster is registered for Half too, not just Float.
    with llvm.ir.Context() as ctx:
        f16 = llvm.types.f16()
        mod = llvm.ir.Module("m", ctx)
        fn, bb, args = _entry(ctx, mod, f16, [f16, f16])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            assert isinstance(args[0], ArithValue)
            b.ret(args[0] + args[1])
        # CHECK: %[[R:.*]] = fadd half %0, %1
        # CHECK: ret half %[[R]]
        filecheck_with_comments(mod)
        del b, fn, bb, args, mod
    assert_no_leaks()


def test_eq_ne_named_methods():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(), [i32, i32])
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0].eq(args[1]))
        # CHECK: %[[R:.*]] = icmp eq i32 %0, %1
        # CHECK: ret i1 %[[R]]
        filecheck_with_comments(mod)
        # __eq__ stays identity so the value is still hashable.
        assert type(args[0] == args[0]) is bool
        assert args[0] == args[0]
        assert len({args[0], args[0]}) == 1
        del b, fn, mod
    assert_no_leaks()


def test_gep_load_store_via_alloca():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn = llvm.ir.Function.create(llvm.types.function(i32, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            slot = b.alloca(i32, "slot")
            p = with_element_type(slot, i32)
            p[0] = llvm.ir.const_int(i32, 5)
            v = p[0]
            b.ret(v)
        # CHECK: %[[P0:.*]] = getelementptr i32, ptr %slot, i64 0
        # CHECK: store i32 5, ptr %[[P0]]
        # CHECK: %[[P1:.*]] = getelementptr i32, ptr %slot, i64 0
        # CHECK: %[[V:.*]] = load i32, ptr %[[P1]]
        # CHECK: ret i32 %[[V]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_pointer_subscript_returns_arithvalue():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn = llvm.ir.Function.create(llvm.types.function(i32, [llvm.types.ptr()]), "g", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            p = with_element_type(fn.arg(0), i32)
            v = p[2]
            assert isinstance(v, ArithValue)
            b.ret(v + 1)  # typed chaining works off the loaded value
        # CHECK: %[[P:.*]] = getelementptr i32, ptr %0, i64 2
        # CHECK: %[[V:.*]] = load i32, ptr %[[P]]
        # CHECK: %[[R:.*]] = add i32 %[[V]], 1
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_pointer_subscript_accepts_value_index():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn = llvm.ir.Function.create(llvm.types.function(i32, [llvm.types.ptr()]), "g", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            p = with_element_type(fn.arg(0), i32)
            # An already-built index Value takes the pass-through branch of
            # TypedPointer._idx instead of the python-int -> i64_const branch.
            idx = b.i64_const(2)
            v = p[idx]
            assert isinstance(v, ArithValue)
            b.ret(v)
        # CHECK: %[[P:.*]] = getelementptr i32, ptr %0, i64 2
        # CHECK: %[[V:.*]] = load i32, ptr %[[P]]
        # CHECK: ret i32 %[[V]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_extract_value_from_struct_arg():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        st = llvm.types.struct([i32, i32])
        fn = llvm.ir.Function.create(llvm.types.function(i32, [st]), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            first = extract(fn.arg(0), 0)
            assert isinstance(first, ArithValue)
            b.ret(first + 1)
        # CHECK: %[[E:.*]] = extractvalue { i32, i32 } %0, 0
        # CHECK: %[[R:.*]] = add i32 %[[E]], 1
        # CHECK: ret i32 %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_insert_value_into_struct():
    # insert_value has no DSL sugar wrapper (only extract_value does, via
    # extract()), so exercise the raw IRBuilder binding directly.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        st = llvm.types.struct([i32, i32])
        fn = llvm.ir.Function.create(llvm.types.function(st, [st, i32]), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb):
            updated = b.insert_value(fn.arg(0), fn.arg(1), 1)
            b.ret(updated)
        # CHECK: %[[R:.*]] = insertvalue { i32, i32 } %0, i32 %1, 1
        # CHECK: ret { i32, i32 } %[[R]]
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_maybe_downcast_passes_through_unregistered_type():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        ptr_ty = llvm.types.ptr()
        fn = llvm.ir.Function.create(llvm.types.function(ptr_ty, [ptr_ty]), "h", mod)
        # ptr has no registered caster, so maybe_downcast must return the
        # same Value unchanged rather than wrapping it.
        arg = fn.arg(0)
        assert maybe_downcast(arg, fn) is arg
        del fn, mod
    assert_no_leaks()


def test_register_value_caster_as_decorator():
    marker_type_id = llvm.types.TypeID.Label
    decorator = register_value_caster(marker_type_id)

    class _Marker:
        pass

    result = decorator(_Marker)
    assert result is _Marker
    # Registration took effect: re-register with the original (no caster) to
    # clean up. We can't inspect the C++ map directly, but we verified the
    # decorator returned the class unchanged.


def test_current_builder_raises_without_context():
    with pytest.raises(RuntimeError, match="no current IRBuilder"):
        current_builder()


def test_current_function_raises_without_context():
    with pytest.raises(RuntimeError, match="no current function"):
        current_function()


def test_building_sets_current_function():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        fn = llvm.ir.Function.create(llvm.types.function(i32, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb), building(b, function=fn):
            assert current_function() is fn
        with pytest.raises(RuntimeError, match="no current function"):
            current_function()
        del b, fn, mod
    assert_no_leaks()
