#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def _fn(ctx, mod):
    return llvm.ir.Function.create(llvm.types.function(llvm.types.void(ctx), []), "f", mod)


def test_linkage_internal():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.linkage = llvm.Linkage.INTERNAL
        assert f.linkage == llvm.Linkage.INTERNAL
        assert "declare internal void @f()" in str(mod)
        del f, mod
    assert_no_leaks()


def test_linkage_weak():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.linkage = llvm.Linkage.WEAK
        assert "declare weak void @f()" in str(mod)
        del f, mod
    assert_no_leaks()


def test_linkage_linkonce_odr():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.linkage = llvm.Linkage.LINKONCE_ODR
        assert "declare linkonce_odr void @f()" in str(mod)
        del f, mod
    assert_no_leaks()


def test_calling_conv_fast():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.calling_conv = llvm.CallingConv.FAST
        assert f.calling_conv == llvm.CallingConv.FAST
        assert "fastcc" in str(mod)
        del f, mod
    assert_no_leaks()


def test_calling_conv_cold():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.calling_conv = llvm.CallingConv.COLD
        assert f.calling_conv == llvm.CallingConv.COLD
        assert "coldcc" in str(mod)
        del f, mod
    assert_no_leaks()


def test_calling_conv_c():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.calling_conv = llvm.CallingConv.C
        assert f.calling_conv == llvm.CallingConv.C
        printed = str(mod)
        assert "fastcc" not in printed
        assert "coldcc" not in printed
        del f, mod
    assert_no_leaks()


def test_visibility_hidden():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.visibility = llvm.Visibility.HIDDEN
        assert f.visibility == llvm.Visibility.HIDDEN
        assert "declare hidden void @f()" in str(mod)
        del f, mod
    assert_no_leaks()


def test_visibility_protected():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.visibility = llvm.Visibility.PROTECTED
        assert f.visibility == llvm.Visibility.PROTECTED
        assert "declare protected void @f()" in str(mod)
        del f, mod
    assert_no_leaks()


def test_visibility_default():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.visibility = llvm.Visibility.DEFAULT
        assert f.visibility == llvm.Visibility.DEFAULT
        printed = str(mod)
        assert "hidden" not in printed
        assert "protected" not in printed
        del f, mod
    assert_no_leaks()


def test_string_fn_attribute():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f = _fn(ctx, mod)
        f.add_fn_attr("target-cpu", "znver3")
        assert f.has_fn_attr("target-cpu")
        assert f.fn_attr_value("target-cpu") == "znver3"
        assert 'target-cpu"="znver3' in str(mod)
        del f, mod
    assert_no_leaks()
