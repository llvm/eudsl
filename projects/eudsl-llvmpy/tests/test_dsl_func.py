#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes

import llvm
from llvm.testing import assert_no_leaks


def test_declaration_has_no_body():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)

        @llvm.function(module=mod)
        def extern(a: i32) -> i32: ...

        printed = str(mod)
        assert "declare i32 @extern(i32)" in printed
        assert extern.name == "extern"
        del mod
    assert_no_leaks()


def test_call_between_functions_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

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
        i32 = llvm.i32(ctx)

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


def test_varargs_declaration():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)

        @llvm.function(module=mod, var_arg=True)
        def printf_like(fmt: llvm.ptr_t(ctx)) -> i32: ...

        assert "declare i32 @printf_like(ptr, ...)" in str(mod)
        del mod
    assert_no_leaks()
