#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_add_global_with_initializer():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        g = mod.add_global(i32, "counter", llvm.ir.const_int(i32, 7))
        assert isinstance(g, llvm.ir.GlobalVariable)
        assert g.name == "counter"
        assert "@counter = global i32 7" in str(mod)
        assert mod.get_global("counter") == g
        assert [x.name for x in mod.globals] == ["counter"]
        del g, mod
    assert_no_leaks()


def test_add_global_external_and_get_global_miss():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        # No initializer -> external declaration (the init=None default path).
        g = mod.add_global(i32, "ext")
        assert g.initializer is None
        assert "@ext = external global i32" in str(mod)
        # get_global miss path returns None.
        assert mod.get_global("nope") is None
        del g, mod
    assert_no_leaks()
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        g = mod.add_global(
            i32, "ro", llvm.ir.const_int(i32, 1), constant=True, address_space=1
        )
        printed = str(mod)
        assert "addrspace(1)" in printed
        assert "constant i32 1" in printed
        del g, mod
    assert_no_leaks()
