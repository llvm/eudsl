#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes

import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_declaration_has_no_body():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod)
        def extern(a: i32) -> i32: ...

        printed = str(mod)
        assert "declare i32 @extern(i32)" in printed
        assert extern.name == "extern"
        del mod
    assert_no_leaks()


def test_declaration_with_pass_body():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod)
        def extern(a: i32) -> i32:
            pass

        assert "declare i32 @extern(i32)" in str(mod)
        del mod
    assert_no_leaks()


def test_call_between_functions_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def inc(x: i32) -> i32:
        return x + 1

    @llvm.function(module=mod)
    def inc2(x: i32) -> i32:
        return inc(inc(x))

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("inc2"))
    assert fn(40) == 42
    del jit, mod, ctx, inc, inc2, i32, fn
    assert_no_leaks()


def test_function_options():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(
            module=mod,
            linkage=llvm.Linkage.INTERNAL,
            attrs={"target-cpu": "znver3"},
        )
        def f(x: i32) -> i32:
            return x

        printed = str(mod)
        assert "define internal i32 @f" in printed
        assert 'target-cpu"="znver3' in printed
        del mod
    assert_no_leaks()


def test_function_calling_conv():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod, calling_conv=llvm.CallingConv.FAST)
        def f(x: i32) -> i32:
            return x

        assert "define fastcc i32 @f" in str(mod)
        del mod
    assert_no_leaks()


def test_varargs_declaration():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod, var_arg=True)
        def printf_like(fmt: llvm.types.ptr(context=ctx)) -> i32: ...

        assert "declare i32 @printf_like(ptr, ...)" in str(mod)
        del mod
    assert_no_leaks()


def test_declaration_with_docstring_only():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod)
        def extern(a: i32) -> i32:
            """An external function."""

        assert "declare i32 @extern(i32)" in str(mod)
        del mod
    assert_no_leaks()


def test_bad_annotation_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(TypeError, match="cannot resolve type annotation"):

            @llvm.function(module=mod)
            def bad(x: "not a type") -> i32:  # noqa: F821
                return x

        del mod
    assert_no_leaks()


def test_void_return_no_explicit_return():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod)
        def store_it(p: llvm.types.ptr(context=ctx), v: i32) -> llvm.types.void:
            from llvm.dsl.values import with_element_type

            tp = with_element_type(p, i32)
            tp[0] = v

        printed = str(mod)
        assert "ret void" in printed
        del mod
    assert_no_leaks()


def test_real_body_is_not_treated_as_empty():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod)
        def f(x: i32) -> i32:
            return x + 1

        assert "define i32 @f" in str(mod)
        assert "add i32" in str(mod)
        del mod
    assert_no_leaks()


def test_call_declared_extern_from_function_body():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def add_one(x: i32) -> i32:
        return x + 1

    @llvm.function(module=mod)
    def extern(x: i32) -> i32: ...

    @llvm.function(module=mod)
    def caller(x: i32) -> i32:
        return add_one(extern(x))

    printed = str(mod)
    assert "call i32 @extern" in printed
    assert "call i32 @add_one" in printed
    del mod, ctx, add_one, extern, caller, i32
    assert_no_leaks()


def test_zero_arg_function():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.types.i32(ctx)

    @llvm.function(module=mod)
    def constant() -> i32:
        return llvm.const_int(i32, 42)

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32)(jit.lookup("constant"))
    assert fn() == 42
    del jit, mod, ctx, constant, i32, fn
    assert_no_leaks()


def test_name_override():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod, name="custom_name")
        def f(x: i32) -> i32:
            return x

        assert "define i32 @custom_name" in str(mod)
        assert f.name == "custom_name"
def test_calling_conv_option():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod, calling_conv=llvm.CallingConv.FAST)
        def f(x: i32) -> i32:
            return x

        assert f.fn.calling_conv == llvm.CallingConv.FAST
        del mod
    assert_no_leaks()


def test_function_without_closure():
    # Body references only its argument and the `llvm` global (no enclosing
    # locals), so the compiled function has no __closure__ -> exercises the
    # no-closure path in transform_ast.
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)

        @llvm.function(module=mod)
        def ident(x: llvm.types.i1) -> llvm.types.i1:
            return x

        assert "define i1 @ident(i1 %0)" in str(mod)
        del mod
    assert_no_leaks()


def test_declaration_with_pass_body():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.function(module=mod)
        def extern2(a: i32) -> i32:
            pass  # empty body -> declaration

        assert "declare i32 @extern2(i32)" in str(mod)
        del mod
    assert_no_leaks()
