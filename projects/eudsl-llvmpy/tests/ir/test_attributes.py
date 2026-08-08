#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def _fn(ctx, mod):
    return llvm.ir.Function.create(llvm.types.function(llvm.types.void(ctx), []), "f", mod)


def test_linkage_and_calling_conv():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f = _fn(ctx, mod)
        f.linkage = llvm.ir.Linkage.INTERNAL
        assert f.linkage == llvm.ir.Linkage.INTERNAL
        # A body-less function prints as `declare`; linkage still shows.
        assert "declare internal void @f()" in str(mod)
        f.calling_conv = llvm.ir.CallingConv.FAST
        assert f.calling_conv == llvm.ir.CallingConv.FAST
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
