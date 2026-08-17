#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Inspecting MIR produced by the codegen pipeline.

`run_codegen_to_mir` runs instruction selection on an IR module and hands back
the MachineModuleInfo that owns the resulting MachineFunctions, so they can be
inspected. The reference shapes asserted here were produced with
`llc -mtriple=aarch64-unknown-linux-gnu -stop-after=finalize-isel`.
"""

from textwrap import dedent

import pytest

import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# These assert AArch64 opcodes (ADDWrr, RET_ReallyLR, ...), so they need the
# AArch64 backend, which eudsl-llvmpy only links when EUDSL_LLVMPY_TARGETS
# includes it (the default is the native target). Skip otherwise.
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

_CONST_SRC = dedent("""\
    define i32 @c() {
      ret i32 42
    }
    """)

_DECL_SRC = dedent("""\
    declare i32 @ext()
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """)

_TRIPLE = "aarch64-unknown-linux-gnu"


def test_run_codegen_to_mir_yields_named_machine_function():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.run_codegen_to_mir(mod, tm)
        assert mmi.machine_function("add").name == "add"
    assert_no_leaks()


def _add_machine_module(ctx):
    """Lower @add to MIR; returns the MachineModuleInfo that owns it."""
    mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
    tm = jit.TargetMachine(triple=_TRIPLE)
    return mir.run_codegen_to_mir(mod, tm)


def test_machine_function_prints_mir_text():
    with ir.Context() as ctx:
        text = str(_add_machine_module(ctx).machine_function("add"))
        assert "ADDWrr" in text
        assert "RET_ReallyLR" in text
    assert_no_leaks()


def test_machine_function_has_single_entry_block():
    with ir.Context() as ctx:
        assert len(_add_machine_module(ctx).machine_function("add").blocks) == 1
    assert_no_leaks()


def test_instruction_opcode_names():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        assert [i.opcode_name for i in block.instructions] == [
            "COPY",
            "COPY",
            "ADDWrr",
            "COPY",
            "RET_ReallyLR",
        ]
    assert_no_leaks()


def test_addwrr_operands_are_one_def_two_uses():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        assert add.num_operands == 3
        assert add.operand(0).is_reg and add.operand(0).is_def
        assert add.operand(1).is_use
        assert add.operand(2).is_use
        assert add.operand(0).reg.is_virtual
    assert_no_leaks()


def test_object_model_accessors():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        assert block.number == 0
        assert isinstance(block.name, str)
        assert "ADDWrr" in str(block)

        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        assert isinstance(add.opcode, int) and add.opcode > 0
        assert "ADDWrr" in str(add)

        # The last COPY is `$w0 = COPY %2`: its def is a physical register.
        last_copy = [i for i in block.instructions if i.opcode_name == "COPY"][-1]
        phys = last_copy.operand(0)
        assert phys.reg.is_physical
        assert phys.reg.id > 0
        assert "w0" in str(phys)
    assert_no_leaks()


def test_machine_function_missing_name_raises():
    with ir.Context() as ctx:
        mmi = _add_machine_module(ctx)
        with pytest.raises(KeyError):
            mmi.machine_function("nope")
    assert_no_leaks()


def test_declared_function_has_no_machine_function():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_DECL_SRC, ctx, "m")
        mmi = mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE))
        with pytest.raises(KeyError):
            mmi.machine_function("ext")  # a declaration has no MachineFunction
    assert_no_leaks()


def test_operand_index_out_of_range_raises():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        with pytest.raises(IndexError):
            add.operand(99)
    assert_no_leaks()


def test_immediate_operand_and_kind_guards():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_CONST_SRC, ctx, "m")
        mmi = mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE))
        block = mmi.machine_function("c").blocks[0]
        mov = next(i for i in block.instructions if i.opcode_name == "MOVi32imm")
        assert mov.operand(1).is_imm
        assert mov.operand(1).imm == 42
        with pytest.raises(ValueError):
            mov.operand(1).reg  # an immediate operand has no register
        with pytest.raises(ValueError):
            mov.operand(0).imm  # a register operand has no immediate
    assert_no_leaks()
