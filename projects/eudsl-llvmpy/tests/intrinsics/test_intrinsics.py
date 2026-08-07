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


def test_underscore_to_dot_mangling():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        vec4i32 = llvm.types.vector(i32, 4)
        decl = llvm.intrinsics.vector_reduce_add(mod, [vec4i32])
        assert decl.name == "llvm.vector.reduce.add.v4i32"
        del decl, mod
    assert_no_leaks()


def test_non_overloaded_intrinsic():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        trap_id = llvm.lookup_intrinsic_id("llvm.trap")
        assert not llvm.intrinsic_is_overloaded(trap_id)
        decl = llvm.get_intrinsic_declaration(mod, trap_id)
        assert decl.name == "llvm.trap"
        assert "declare void @llvm.trap()" in str(mod)
        del decl, mod
    assert_no_leaks()


def test_non_overloaded_via_shim():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        decl = llvm.intrinsics.trap(mod)
        assert decl.name == "llvm.trap"
        del decl, mod
    assert_no_leaks()


def test_unknown_intrinsic_raises():
    with pytest.raises(AttributeError, match="unknown intrinsic"):
        llvm.intrinsics.not_a_real_intrinsic_xyz
