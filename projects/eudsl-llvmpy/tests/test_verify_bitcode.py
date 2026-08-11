#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

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
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_GOOD, ctx, "m")
        mod.verify()
        del mod
    assert_no_leaks()


def test_verify_rejects_bad_module():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            dedent(
                """\
                define i32 @f(i32 %x) {
                entry:
                  br label %body
                body:
                  ret i32 %y
                unreachable:
                  %y = add i32 %x, 1
                  br label %body
                }
                """
            ),
            ctx,
            "m",
        )
        with pytest.raises(llvm.VerifyError):
            mod.verify()
        del mod
    assert_no_leaks()


def test_parse_bitcode_rejects_garbage():
    with llvm.Context() as ctx:
        with pytest.raises(llvm.ParseError, match="Invalid bitcode signature"):
            llvm.parse_bitcode(b"not bitcode", ctx)
    assert_no_leaks()


def test_bitcode_round_trip():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_GOOD, ctx, "m")
        data = mod.to_bitcode()
        assert isinstance(data, bytes)
        assert data[:2] == b"BC"
        del mod
    with llvm.Context() as ctx2:
        mod2 = llvm.parse_bitcode(data, ctx2)
        mod2.verify()
        printed = str(mod2)
        assert "define i32 @f(i32 %x)" in printed
        assert "ret i32 %x" in printed
        del mod2
    assert_no_leaks()
