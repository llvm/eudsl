#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @f(i32 %x) {
    entry:
      %a = add i32 %x, 0
      ret i32 %a
    }
    """
)


def test_instcombine_removes_add_zero():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        assert "add i32 %x, 0" in str(mod)
        llvm.run_passes(mod, "instcombine")
        printed = str(mod)
        assert "add i32 %x, 0" not in printed
        assert "ret i32 %x" in printed
        del mod
    assert_no_leaks()


def test_bad_pipeline_raises():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        with pytest.raises(RuntimeError, match="unknown pass name"):
            llvm.run_passes(mod, "not-a-real-pass")
        del mod
    assert_no_leaks()


def test_empty_pipeline_raises():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        with pytest.raises(RuntimeError, match="unknown pass name"):
            llvm.run_passes(mod, "")
        del mod
    assert_no_leaks()


def test_verify_pipeline_is_noop():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        before = str(mod)
        llvm.run_passes(mod, "verify")
        assert str(mod) == before
        del mod
    assert_no_leaks()
