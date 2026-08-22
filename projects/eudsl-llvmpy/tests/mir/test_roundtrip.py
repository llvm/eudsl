#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Round-tripping MIR through its textual serialization format.

`to_mir()` prints a MachineModuleInfo as `.mir` text; `parse_mir` reads it back.
The reference MIR is produced by lowering @add with run_codegen_to_mir rather
than hand-written, so the test stays honest about the real serialization.
"""

from textwrap import dedent

import pytest

import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# AArch64-specific (asserts ADDWrr/RET_ReallyLR); needs the AArch64 backend.
pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)

_ADD_SRC = dedent("""\
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """)

_TRIPLE = "aarch64-unknown-linux-gnu"

_EXPECTED_OPCODES = ["COPY", "COPY", "ADDWrr", "COPY", "RET_ReallyLR"]


def _add_mir_text(ctx):
    mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
    return mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE)).to_mir()


def test_to_mir_emits_machine_function_text():
    with ir.Context() as ctx:
        text = _add_mir_text(ctx)
        assert "name:" in text and "add" in text
        assert "ADDWrr" in text
        assert "RET_ReallyLR" in text
    assert_no_leaks()


def test_parse_mir_recovers_the_machine_function():
    with ir.Context() as ctx:
        text = _add_mir_text(ctx)
    with ir.Context() as ctx:
        mmi = mir.parse_mir(text, ctx, jit.TargetMachine(triple=_TRIPLE))
        block = mmi.machine_function("add").blocks[0]
        assert [i.opcode_name for i in block.instructions] == _EXPECTED_OPCODES
    assert_no_leaks()


def test_mir_print_parse_is_idempotent():
    with ir.Context() as ctx:
        text1 = _add_mir_text(ctx)
    with ir.Context() as ctx:
        text2 = mir.parse_mir(text1, ctx, jit.TargetMachine(triple=_TRIPLE)).to_mir()
    with ir.Context() as ctx:
        text3 = mir.parse_mir(text2, ctx, jit.TargetMachine(triple=_TRIPLE)).to_mir()
    assert text2 == text3
    assert_no_leaks()


def test_parse_mir_rejects_invalid_text():
    with ir.Context() as ctx:
        with pytest.raises(RuntimeError) as ei:
            mir.parse_mir(
                "@@@ not valid mir @@@", ctx, jit.TargetMachine(triple=_TRIPLE)
            )
        # The real parser diagnostic is threaded into the message, not swallowed.
        assert "failed to parse" in str(ei.value)
        assert ":" in str(ei.value)
    assert_no_leaks()


_BAD_MF = dedent("""\
    --- |
      define void @f() {
        ret void
      }
    ...
    ---
    name: f
    body: |
      bb.0:
        NOT_A_REAL_OPCODE
    ...
    """)


def test_parse_mir_rejects_bad_machine_function_body():
    with ir.Context() as ctx:
        with pytest.raises(RuntimeError) as ei:
            mir.parse_mir(_BAD_MF, ctx, jit.TargetMachine(triple=_TRIPLE))
        assert "failed to parse machine functions" in str(ei.value)
        assert ":" in str(ei.value)
    assert_no_leaks()
