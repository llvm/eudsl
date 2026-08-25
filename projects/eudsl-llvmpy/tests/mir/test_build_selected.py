#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Building already-selected target MIR by hand (Route B).

Rather than build generic G_* MIR and rely on GlobalISel selection (which falls
back to SelectionDAG -- and thus to IR -- when the target can't select it), the
DSL can emit fully-selected target instructions directly. The resulting MIR is
well-formed target MIR (verify() is True), unlike an incomplete generic
function that never went through instruction selection.

The reference shape is `llc -stop-after=finalize-isel` for a 32-bit add on
AArch64: liveins $w0/$w1, two COPYs in, ADDWrr, a COPY to $w0, RET_ReallyLR.
"""

from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

import pytest
import llvm

# Target-specific (AArch64 GPR32/ADDWrr/RET_ReallyLR); needs the AArch64 backend.
pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)

_TRIPLE = "aarch64-unknown-linux-gnu"


def _build_selected_add(mmi, declare_liveins=True):
    """Hand-build a fully-selected AArch64 `add(i32,i32)->i32` MachineFunction.

    Returns (mf, {"v0", "v1", "v2", "w0"} register ids) so callers can assert on
    the operand wiring. With declare_liveins=False the live-ins are omitted,
    which (given TracksLiveness) makes the same MIR fail verify().
    """
    mf = mmi.machine_function("add")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]

    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    w1 = mf.physreg("W1")
    if declare_liveins:
        entry.add_livein(w0)
        entry.add_livein(w1)

    v0 = mf.create_vreg(gpr32)
    v1 = mf.create_vreg(gpr32)
    v2 = mf.create_vreg(gpr32)
    copy = mf.opcode("COPY")

    c0 = b.build_instr(copy)
    c0.add_reg(v0, is_def=True)
    c0.add_reg(w0)
    c1 = b.build_instr(copy)
    c1.add_reg(v1, is_def=True)
    c1.add_reg(w1)
    add = b.build_instr(mf.opcode("ADDWrr"))
    add.add_reg(v2, is_def=True)
    add.add_reg(v0)
    add.add_reg(v1)
    c2 = b.build_instr(copy)
    c2.add_reg(w0, is_def=True)
    c2.add_reg(v2)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)

    mf.set_property(mir.MachineFunctionProperty.IsSSA)
    mf.set_property(mir.MachineFunctionProperty.TracksLiveness)
    mf.set_property(mir.MachineFunctionProperty.NoPHIs)
    return mf, {"v0": v0.id, "v1": v1.id, "v2": v2.id, "w0": w0.id}


def test_reg_class_and_physreg_lookup():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "add")
        mf = mmi.machine_function("add")
        assert mf.reg_class("GPR32") is not None
        assert mf.create_vreg(mf.reg_class("GPR32")).is_virtual
        assert mf.physreg("W0").is_physical
    assert_no_leaks()


def test_unknown_reg_class_and_physreg_raise():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mf = mir.create_machine_function(mod, tm, "add").machine_function("add")
        with pytest.raises(KeyError):
            mf.reg_class("NOPE")
        with pytest.raises(KeyError):
            mf.physreg("NOPE")
    assert_no_leaks()


def test_hand_built_selected_add_verifies():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "add")
        mf, regs = _build_selected_add(mmi)

        instrs = list(mf.blocks[0].instructions)
        opcodes = [i.opcode_name for i in instrs]
        assert opcodes == ["COPY", "COPY", "ADDWrr", "COPY", "RET_ReallyLR"]
        # Operand wiring: ADDWrr defs v2 and uses v0, v1.
        addwrr = instrs[2]
        assert addwrr.operand(0).is_def and addwrr.operand(0).reg.id == regs["v2"]
        assert addwrr.operand(1).is_use and addwrr.operand(1).reg.id == regs["v0"]
        assert addwrr.operand(2).is_use and addwrr.operand(2).reg.id == regs["v1"]
        # The terminal RET reads $w0 as an implicit use.
        ret = instrs[4]
        assert ret.operand(0).is_implicit and ret.operand(0).reg.id == regs["w0"]
        # A well-formed, fully-selected function -- unlike incomplete generic MIR.
        assert mf.verify() is True
    assert_no_leaks()


def test_hand_built_add_without_liveins_fails_verify():
    """Dropping the live-in declarations makes the same MIR fail verify(),
    showing add_livein is load-bearing (the verifier checks physreg liveness
    because TracksLiveness is set)."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "add")
        mf, _ = _build_selected_add(mmi, declare_liveins=False)
        assert mf.verify() is False
        assert mf.verify_diagnostic() != ""
    assert_no_leaks()


def test_add_reg_flag_matrix_roundtrips():
    """Every add_reg flag is set and read back, so an argument-position
    mis-wiring in the 11-arg operand builder would be caught. Uses physical
    registers because is_renamable is a physreg-only flag (the target-neutral
    virtual-register subset is covered in test_build_selected_generic.py)."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mf = mir.create_machine_function(mod, tm, "add").machine_function("add")
        b = mir.MachineIRBuilder(mf)
        w0 = mf.physreg("W0")
        w1 = mf.physreg("W1")

        instr = b.build_instr(mf.opcode("COPY"))
        instr.add_reg(
            w0,
            is_def=True,
            is_dead=True,
            is_early_clobber=True,
            is_renamable=True,
            sub_reg=1,
        )
        instr.add_reg(w1, implicit=True, is_kill=True, is_undef=True)

        defop = instr.operand(0)
        assert defop.is_def and defop.is_dead and defop.is_early_clobber
        assert defop.is_renamable and defop.sub_reg == 1
        assert not defop.is_kill

        useop = instr.operand(1)
        assert useop.is_use and useop.is_implicit
        assert useop.is_kill and useop.is_undef
        assert not useop.is_dead and not useop.is_renamable
    assert_no_leaks()


def test_selected_add_survives_mir_roundtrip():
    """The hand-built selected MIR prints to .mir and parses back well-formed."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        text = mmi.to_mir()
        mmi2 = mir.parse_mir(text, ctx, jit.TargetMachine(triple=_TRIPLE))
        assert mmi2.machine_function("add").verify() is True
    assert_no_leaks()


def test_build_brcond_rejects_physical_register():
    """A generic builder op like build_brcond needs a *generic vreg* condition.
    A physical register belongs to no function (it is target-static), so it
    passes the cross-function owner guard -- but it is still not a generic
    virtual register of this function, so it is rejected on that ground."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "add")
        mf = mmi.machine_function("add")
        b = mir.MachineIRBuilder(mf)
        with pytest.raises(ValueError, match="generic virtual register"):
            b.build_brcond(mf.physreg("W0"), mf.blocks[0])
    assert_no_leaks()
