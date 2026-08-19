#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Target-neutral coverage for the Route-B operand/property primitives.

test_build_selected.py exercises the real AArch64 shape but skips entirely on
non-AArch64 legs, leaving the target-agnostic guards -- add_reg's flag matrix
and validation, set_property/has_property, and add_livein's physical-register
check -- uncovered there. These use only target-independent opcodes (COPY) and
registers, so they run on every runner (triple=None builds against the host).
"""

import pytest

from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks


def test_add_reg_flag_matrix_roundtrips():
    """Every virtual-register-legal add_reg flag is set and read back, catching
    a mis-wired arg. (is_renamable is physreg-only -- see test_build_selected.py
    for the physical-register variant.)"""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=None)
        mf = mir.create_machine_function(mod, tm, "f").machine_function("f")
        b = mir.MachineIRBuilder(mf)
        d = mf.create_generic_virtual_register(mir.LLT.scalar(32))
        u = mf.create_generic_virtual_register(mir.LLT.scalar(32))

        instr = b.build_instr(mf.opcode("COPY"))
        instr.add_reg(d, is_def=True, is_dead=True, is_early_clobber=True, sub_reg=1)
        instr.add_reg(u, implicit=True, is_kill=True, is_undef=True)

        defop = instr.operand(0)
        assert defop.is_def and defop.is_dead and defop.is_early_clobber
        assert defop.sub_reg == 1
        assert not defop.is_kill and not defop.is_renamable

        useop = instr.operand(1)
        assert useop.is_use and useop.is_implicit
        assert useop.is_kill and useop.is_undef
        assert not useop.is_dead
    assert_no_leaks()


def test_add_reg_rejects_contradictory_flags():
    """kill is use-only, dead is def-only, is_renamable is physreg-only, and
    sub_reg is bounds-checked -- LLVM asserts these, compiled out under NDEBUG
    in the release wheel."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=None)
        mf = mir.create_machine_function(mod, tm, "f").machine_function("f")
        b = mir.MachineIRBuilder(mf)
        v = mf.create_generic_virtual_register(mir.LLT.scalar(32))
        instr = b.build_instr(mf.opcode("COPY"))
        with pytest.raises(ValueError):
            instr.add_reg(v, is_def=True, is_kill=True)
        with pytest.raises(ValueError):
            instr.add_reg(v, is_def=False, is_dead=True)
        with pytest.raises(ValueError):
            instr.add_reg(v, is_renamable=True)  # virtual register
        with pytest.raises(IndexError):
            instr.add_reg(v, sub_reg=1_000_000_000)
    assert_no_leaks()


def test_add_livein_rejects_virtual_register():
    """add_livein truncates via asMCReg() (assert-only), so a virtual register
    would silently corrupt the livein list -- reject it instead."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=None)
        mf = mir.create_machine_function(mod, tm, "f").machine_function("f")
        v = mf.create_generic_virtual_register(mir.LLT.scalar(32))
        with pytest.raises(ValueError):
            mf.blocks[0].add_livein(v)
    assert_no_leaks()


def test_has_property_reads_back_set_property():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=None)
        mf = mir.create_machine_function(mod, tm, "f").machine_function("f")
        assert mf.has_property(mir.MachineFunctionProperty.Selected) is False
        mf.set_property(mir.MachineFunctionProperty.Selected)
        assert mf.has_property(mir.MachineFunctionProperty.Selected) is True
    assert_no_leaks()
