#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import platform
import sys
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """
)


def test_host_triple_is_nonempty():
    assert isinstance(llvm.host_triple(), str)
    assert llvm.host_triple()


def test_target_machine_triple_roundtrips():
    triple = llvm.host_triple()
    tm = llvm.TargetMachine(triple)
    assert tm.triple == triple
    assert isinstance(tm.data_layout_str, str)
    assert tm.data_layout_str


def test_target_machine_bad_triple_raises():
    with pytest.raises(RuntimeError, match="No available targets"):
        llvm.TargetMachine("not-a-real-triple-xyz")


def test_target_machine_features_as_list():
    tm = llvm.TargetMachine(features=["+sse2", "+avx"])
    assert tm.triple == llvm.host_triple()


def test_set_data_layout_from_mutates_module():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        tm = llvm.TargetMachine()
        mod.set_data_layout_from(tm)
        assert str(mod).startswith(f'; ModuleID = ')
        assert "target datalayout" in str(mod)
        del tm, mod
    assert_no_leaks()


def test_emit_assembly_and_object():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        tm = llvm.TargetMachine()
        mod.set_data_layout_from(tm)
        asm_out = tm.emit_assembly(mod)
        assert "%s = add" not in asm_out
        assert "ret" in asm_out.lower() or "bx" in asm_out.lower() or "blr" in asm_out.lower()
        obj = tm.emit_object(mod)
        assert isinstance(obj, bytes)
        assert len(obj) > 4
        if sys.platform == "darwin":
            assert obj[:4] == b"\xcf\xfa\xed\xfe" or obj[:4] == b"\xfe\xed\xfa\xcf"
        elif sys.platform == "linux":
            assert obj[:4] == b"\x7fELF"
        del tm, mod
    assert_no_leaks()
