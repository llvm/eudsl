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
        printed = str(mod)
        assert "add i32 %x, 0" not in printed
        assert "ret i32 %x" in printed
        del mod
    assert_no_leaks()


def test_bad_pipeline_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        with pytest.raises(RuntimeError, match="unknown pass name"):
            llvm.passmanager.run_passes(mod, "not-a-real-pass")
        del mod
    assert_no_leaks()


def test_empty_pipeline_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        with pytest.raises(RuntimeError, match="unknown pass name"):
            llvm.passmanager.run_passes(mod, "")
        del mod
    assert_no_leaks()


def test_verify_pipeline_is_noop():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        before = str(mod)
        llvm.passmanager.run_passes(mod, "verify")
        assert str(mod) == before
        del mod
    assert_no_leaks()


def test_default_pipeline_o2():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_default_pipeline(mod, llvm.passmanager.OptLevel.O2)
        printed = str(mod)
        assert "add i32 %x, 0" not in printed
        assert "ret i32 %x" in printed
        del mod
    assert_no_leaks()


def test_default_pipeline_o0():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_default_pipeline(mod, llvm.passmanager.OptLevel.O0)
        assert "define" in str(mod)
        del mod
    assert_no_leaks()


def test_default_pipeline_with_tuning():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        pto = llvm.passmanager.PipelineTuningOptions()
        pto.loop_vectorization = False
        pto.slp_vectorization = False
        llvm.passmanager.run_default_pipeline(mod, llvm.passmanager.OptLevel.O2, tuning=pto)
        printed = str(mod)
        assert "add i32 %x, 0" not in printed
        assert "ret i32 %x" in printed
        del mod
    assert_no_leaks()


def test_pipeline_tuning_options_defaults():
    pto = llvm.passmanager.PipelineTuningOptions()
    assert pto.loop_unrolling is True
    assert isinstance(pto.loop_vectorization, bool)
    assert isinstance(pto.slp_vectorization, bool)
    assert isinstance(pto.loop_interleaving, bool)
    assert isinstance(pto.merge_functions, bool)


def test_run_passes_with_debug(capsys):
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_passes(mod, "instcombine", debug=True)
        assert "add i32 %x, 0" not in str(mod)
        del mod
    assert_no_leaks()


def test_run_passes_with_verify_each():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_passes(mod, "instcombine", verify_each=True)
        assert "ret i32 %x" in str(mod)
        del mod
    assert_no_leaks()


def test_default_pipeline_with_debug():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_default_pipeline(mod, llvm.passmanager.OptLevel.O1, debug=True)
        assert "define" in str(mod)
        del mod
    assert_no_leaks()
