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


def test_host_triple_is_nonempty():
    assert isinstance(llvm.host_triple(), str)
    assert llvm.host_triple()


def test_emit_assembly_and_object():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        tm = llvm.TargetMachine(llvm.host_triple())
        assert tm.data_layout_str
        mod.set_data_layout_from(tm)
        asm = tm.emit_assembly(mod)
        assert "add" in asm
        obj = tm.emit_object(mod)
        assert isinstance(obj, bytes)
        assert len(obj) > 0
        del tm, mod
    assert_no_leaks()
