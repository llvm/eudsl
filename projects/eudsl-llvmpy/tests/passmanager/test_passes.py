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
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        assert "add i32 %x, 0" in str(mod)
        llvm.passmanager.run_passes(mod, "instcombine")
        assert "add i32 %x, 0" not in str(mod)
        del mod
    assert_no_leaks()


def test_bad_pipeline_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        with pytest.raises(RuntimeError, match="unknown pass name"):
            llvm.passmanager.run_passes(mod, "not-a-real-pass")
        del mod
    assert_no_leaks()
