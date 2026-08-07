#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

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


def test_target_machine_triple():
    tm = llvm.jit.TargetMachine(llvm.jit.host_triple())
    # TargetMachine.triple round-trips a normalized host triple.
    assert isinstance(tm.triple, str)
    assert "-" in tm.triple


def test_emit_assembly_and_object():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        tm = llvm.jit.TargetMachine(llvm.jit.host_triple())
        assert tm.data_layout_str
        mod.set_data_layout_from(tm)
        asm = tm.emit_assembly(mod)
        # "add" alone is vacuous: the symbol @add puts it in the output no matter
        # what the body compiles to. Prove emit reflects the body by emitting a
        # same-named function with a different body and asserting they differ,
        # and that the constant-returning body shows its immediate.
        const_mod = llvm.ir.parse_assembly(_SRC_CONST, ctx, "m2")
        const_mod.set_data_layout_from(tm)
        const_asm = tm.emit_assembly(const_mod)
        assert asm != const_asm
        assert "12345" in const_asm and "12345" not in asm
        obj = tm.emit_object(mod)
        assert isinstance(obj, bytes)
        assert len(obj) > 0
        del tm, mod, const_mod
    assert_no_leaks()
