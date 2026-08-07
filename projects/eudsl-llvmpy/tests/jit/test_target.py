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

# Same symbol name @add, different body, so any difference in emitted assembly
# comes from codegen of the body rather than the symbol/label text.
_SRC_CONST = dedent(
    """\
    define i32 @add(i32 %a, i32 %b) {
    entry:
      ret i32 12345
    }
    """
)


def test_host_triple_is_nonempty():
    assert isinstance(llvm.jit.host_triple(), str)
    assert llvm.jit.host_triple()


def test_target_machine_triple_roundtrips():
    triple = llvm.jit.host_triple()
    tm = llvm.jit.TargetMachine(triple)
    assert tm.triple == triple
    assert isinstance(tm.data_layout_str, str)
    assert tm.data_layout_str


def test_target_machine_bad_triple_raises():
    with pytest.raises(RuntimeError, match="No available targets"):
        llvm.jit.TargetMachine("not-a-real-triple-xyz")


def test_target_machine_features_as_list():
    tm = llvm.jit.TargetMachine(features=["+sse2", "+avx"])
    assert tm.triple == llvm.jit.host_triple()


def test_set_data_layout_from_mutates_module():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        tm = llvm.jit.TargetMachine()
        mod.set_data_layout_from(tm)
        assert "target datalayout" in str(mod)
        del tm, mod
    assert_no_leaks()


def test_emit_assembly_and_object():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        tm = llvm.jit.TargetMachine()
        mod.set_data_layout_from(tm)
        asm = tm.emit_assembly(mod)
        const_mod = llvm.ir.parse_assembly(_SRC_CONST, ctx, "m2")
        const_mod.set_data_layout_from(tm)
        const_asm = tm.emit_assembly(const_mod)
        assert asm != const_asm
        assert "12345" in const_asm and "12345" not in asm
        obj = tm.emit_object(mod)
        assert isinstance(obj, bytes)
        assert len(obj) > 4
        if sys.platform == "darwin":
            assert obj[:4] == b"\xcf\xfa\xed\xfe" or obj[:4] == b"\xfe\xed\xfa\xcf"
        elif sys.platform == "linux":
            assert obj[:4] == b"\x7fELF"
        del tm, mod, const_mod
    assert_no_leaks()
