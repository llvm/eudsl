#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.dsl.casters import maybe_downcast
from llvm.dsl.context import building
from llvm.dsl.values import ArithValue, extract, with_element_type
from llvm.testing import assert_no_leaks


def _entry(ctx, mod, ret_ty, arg_tys, name="f"):
    fn = llvm.Function.create(llvm.types.function(ret_ty, arg_tys), name, mod)
    bb = fn.append_basic_block("entry")
    # Wrap args the way @function will, so integer/float args are ArithValue.
    args = [maybe_downcast(fn.arg(i), fn) for i in range(len(arg_tys))]
    return fn, bb, args


def test_args_are_arithvalue():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        assert isinstance(args[0], ArithValue)
        del fn, mod
    assert_no_leaks()


def test_integer_add_and_mul():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn, bb, args = _entry(ctx, mod, i32, [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            r = args[0] * args[1] + 1
            assert isinstance(r, ArithValue)  # result stays typed
            b.ret(r)
        printed = str(mod)
        assert "mul i32" in printed
        assert "add i32" in printed
        del b, fn, mod
    assert_no_leaks()


def test_float_add_uses_fadd():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.types.f32(ctx)
        fn, bb, args = _entry(ctx, mod, f32, [f32, f32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] + args[1])
        assert "fadd float" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_scalar_coercion():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn, bb, args = _entry(ctx, mod, i32, [i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] + 7)
        assert "add i32 %0, 7" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_comparison_signed():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(ctx), [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] < args[1])
        assert "icmp slt i32" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_comparison_ordered():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.types.f32(ctx)
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(ctx), [f32, f32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] > args[1])
        assert "fcmp ogt float" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_integer_gt_uses_sgt():
    # The int path of __gt__ is otherwise only used on floats (-> OGT); pin SGT.
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb, args = _entry(ctx, mod, llvm.i1(ctx), [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] > args[1])
        assert "icmp sgt i32" in str(mod)
        del b, fn, bb, args, mod
    assert_no_leaks()


def test_float_lt_uses_olt():
    # The float path of __lt__ is otherwise only used on ints (-> SLT); pin OLT.
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.f32(ctx)
        fn, bb, args = _entry(ctx, mod, llvm.i1(ctx), [f32, f32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0] < args[1])
        assert "fcmp olt float" in str(mod)
        del b, fn, bb, args, mod
    assert_no_leaks()


def test_double_and_half_are_arithvalue():
    # The float caster is registered for Half/Double too, not just Float.
    with llvm.Context() as ctx:
        for ty, want in ((llvm.f64(ctx), "fadd double"), (llvm.f16(ctx), "fadd half")):
            mod = llvm.Module("m", ctx)
            fn, bb, args = _entry(ctx, mod, ty, [ty, ty])
            b = llvm.IRBuilder(ctx)
            with b.at_end_of(bb), building(b):
                assert isinstance(args[0], ArithValue)
                b.ret(args[0] + args[1])
            assert want in str(mod)
            del b, fn, bb, args, mod
    assert_no_leaks()


def test_eq_ne_named_methods():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn, bb, args = _entry(ctx, mod, llvm.types.i1(ctx), [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(args[0].eq(args[1]))
        assert "icmp eq i32" in str(mod)
        # __eq__ stays identity so the value is still hashable.
        assert args[0] == args[0]
        assert len({args[0], args[0]}) == 1
        del b, fn, mod
    assert_no_leaks()


def test_gep_load_store_via_alloca():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            slot = b.alloca(i32, "slot")
            p = with_element_type(slot, i32)
            p[0] = llvm.const_int(i32, 5)
            v = p[0]
            b.ret(v)
        printed = str(mod)
        assert "getelementptr i32" in printed
        assert "store i32 5" in printed
        assert "load i32" in printed
        del b, fn, mod
    assert_no_leaks()


def test_pointer_subscript_returns_arithvalue():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [llvm.types.ptr(ctx)]), "g", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            p = with_element_type(fn.arg(0), i32)
            v = p[2]
            assert isinstance(v, ArithValue)
            b.ret(v + 1)  # typed chaining works off the loaded value
        printed = str(mod)
        assert "getelementptr i32" in printed
        assert "load i32" in printed
        del b, fn, mod
    assert_no_leaks()


def test_extract_value_from_struct_arg():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        st = llvm.types.struct(ctx, [i32, i32])
        fn = llvm.Function.create(llvm.types.function(i32, [st]), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            first = extract(fn.arg(0), 0)
            assert isinstance(first, ArithValue)
            b.ret(first + 1)
        assert "extractvalue { i32, i32 }" in str(mod)
        del b, fn, mod
    assert_no_leaks()
