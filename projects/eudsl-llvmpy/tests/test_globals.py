#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_add_global_with_initializer():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        g = mod.add_global(i32, "counter", llvm.const_int(i32, 7))
        assert type(g).__name__ == "GlobalVariable"
        assert g.name == "counter"
        assert "@counter = global i32 7" in str(mod)
        assert mod.get_global("counter") == g
        assert [x.name for x in mod.globals] == ["counter"]
        del g, mod
    assert_no_leaks()


def test_constant_global_in_address_space():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        g = mod.add_global(
            i32, "ro", llvm.const_int(i32, 1), constant=True, address_space=1
        )
        printed = str(mod)
        assert "addrspace(1)" in printed
        assert "constant i32 1" in printed
        del g, mod
    assert_no_leaks()
