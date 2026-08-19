#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Target-specific MIR: build real target opcodes via runtime name lookup.

Target opcode enums (e.g. AArch64 ADDWrr) live in generated headers that are not
installed with LLVM, so opcodes are resolved by name at runtime through the
function's TargetInstrInfo. build_instr + the operand appenders are the BuildMI
analogue for constructing an arbitrary target instruction.
"""

import pytest

import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# Target-specific (AArch64 opcodes ADDWrr/MOVi32imm/B); needs the AArch64 backend.
pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)

_TRIPLE = "aarch64-unknown-linux-gnu"


def _new_function(ctx):
    mod = ir.Module("m", ctx)
    tm = jit.TargetMachine(triple=_TRIPLE)
    mmi = mir.create_machine_function(mod, tm, "f")
    return mmi, mmi.machine_function("f")


def test_opcode_name_round_trips():
    with ir.Context() as ctx:
        _, mf = _new_function(ctx)
        addwrr = mf.opcode("ADDWrr")
        assert isinstance(addwrr, int) and addwrr > 0
        assert mf.opcode_name(addwrr) == "ADDWrr"
    assert_no_leaks()


def test_unknown_opcode_name_raises():
    with ir.Context() as ctx:
        _, mf = _new_function(ctx)
        with pytest.raises(KeyError):
            mf.opcode("NOT_A_REAL_OPCODE_XYZ")
    assert_no_leaks()


def test_opcode_name_out_of_range_raises():
    with ir.Context() as ctx:
        _, mf = _new_function(ctx)
        # The number->name direction must validate like name->number does, not
        # index out of bounds into an assert-only (NDEBUG) table.
        with pytest.raises(IndexError):
            mf.opcode_name(999999)
    assert_no_leaks()


def test_build_instr_out_of_range_opcode_raises():
    with ir.Context() as ctx:
        _, mf = _new_function(ctx)
        b = mir.MachineIRBuilder(mf)
        with pytest.raises(IndexError):
            b.build_instr(999999)
    assert_no_leaks()


def test_build_target_register_instruction():
    with ir.Context() as ctx:
        _, mf = _new_function(ctx)
        s32 = mir.LLT.scalar(32)
        b = mir.MachineIRBuilder(mf)
        d = mf.create_generic_virtual_register(s32)
        x = mf.create_generic_virtual_register(s32)
        y = mf.create_generic_virtual_register(s32)
        mi = b.build_instr(mf.opcode("ADDWrr"))
        mi.add_def(d)
        mi.add_use(x)
        mi.add_use(y)
        assert mi.opcode_name == "ADDWrr"
        assert mi.num_operands == 3
        # Operand identity and order: def d, then uses x, y in that order.
        assert mi.operand(0).is_def and mi.operand(0).reg.id == d.id
        assert mi.operand(1).is_use and mi.operand(1).reg.id == x.id
        assert mi.operand(2).is_use and mi.operand(2).reg.id == y.id
        assert "ADDWrr" in str(mi)
    assert_no_leaks()


def test_build_instruction_with_immediate():
    with ir.Context() as ctx:
        _, mf = _new_function(ctx)
        s32 = mir.LLT.scalar(32)
        b = mir.MachineIRBuilder(mf)
        d = mf.create_generic_virtual_register(s32)
        mi = b.build_instr(mf.opcode("MOVi32imm"))
        mi.add_def(d)
        mi.add_imm(42)
        assert mi.opcode_name == "MOVi32imm"
        assert mi.operand(0).is_def and mi.operand(0).reg.id == d.id
        assert mi.operand(1).imm == 42
    assert_no_leaks()


def test_build_branch_with_mbb_operand():
    with ir.Context() as ctx:
        _, mf = _new_function(ctx)
        b = mir.MachineIRBuilder(mf)
        target = mf.create_block()
        mi = b.build_instr(mf.opcode("B"))  # AArch64 unconditional branch
        mi.add_mbb(target)
        # Keep the CFG consistent with the terminator so the function verifies.
        mf.blocks[0].add_successor(target)
        assert mi.opcode_name == "B"
        assert mi.num_operands == 1
    assert_no_leaks()
