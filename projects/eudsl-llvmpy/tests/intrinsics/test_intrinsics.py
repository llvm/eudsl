#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

import llvm
import llvm.intrinsics
from llvm.testing import assert_no_leaks


def test_lookup_intrinsic_id():
    assert llvm.intrinsics.lookup_intrinsic_id("llvm.sqrt") != 0
    assert llvm.intrinsics.lookup_intrinsic_id("llvm.not.a.real.intrinsic") == 0


def test_get_overloaded_declaration():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        sqrt_id = llvm.intrinsics.lookup_intrinsic_id("llvm.sqrt")
        assert llvm.intrinsics.intrinsic_is_overloaded(sqrt_id)
        f32 = llvm.types.f32(ctx)
        decl = llvm.intrinsics.get_intrinsic_declaration(mod, sqrt_id, [f32])
        assert decl.name == "llvm.sqrt.f32"
        assert "declare float @llvm.sqrt.f32(float)" in str(mod)
        del decl, mod
    assert_no_leaks()


def test_intrinsics_getattr_shim():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        f64 = llvm.types.f64(ctx)
        decl = llvm.intrinsics.sqrt(mod, [f64])
        assert decl.name == "llvm.sqrt.f64"
        del decl, mod
    assert_no_leaks()


def test_intrinsics_shim_unknown_raises():
    with pytest.raises(AttributeError, match="unknown intrinsic"):
        llvm.intrinsics.definitely_not_an_intrinsic
