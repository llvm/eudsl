#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks

_GOOD = dedent(
    """\
    define i32 @f(i32 %x) {
    entry:
      ret i32 %x
    }
    """
)


def test_verify_accepts_good_module():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_GOOD, ctx, "m")
        assert mod.verify() is None
        del mod
    assert_no_leaks()


def test_bitcode_round_trip():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_GOOD, ctx, "m")
        data = mod.to_bitcode()
        assert isinstance(data, bytes)
        assert data[:2] == b"BC"
        del mod
    with llvm.ir.Context() as ctx2:
        mod2 = llvm.ir.parse_bitcode(data, ctx2)
        assert "define i32 @f(i32 %x)" in str(mod2)
        del mod2
    assert_no_leaks()
