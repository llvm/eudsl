#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Building already-selected target MIR by hand (Route B).

Rather than build generic G_* MIR and rely on GlobalISel selection (which falls
back to SelectionDAG -- and thus to IR -- when the target can't select it), the
DSL can emit fully-selected target instructions directly. This is the substrate
for lowering to machine code without instruction selection: the resulting MIR is
well-formed (verify() is True), unlike an incomplete generic function.

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


def _build_selected_add(mmi):
    """Hand-build a fully-selected AArch64 `add(i32,i32)->i32` MachineFunction."""
    mf = mmi.machine_function("add")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]

    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    w1 = mf.physreg("W1")
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
    return mf


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
        import pytest

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
        mf = _build_selected_add(mmi)

        opcodes = [i.opcode_name for i in mf.blocks[0].instructions]
        assert opcodes == ["COPY", "COPY", "ADDWrr", "COPY", "RET_ReallyLR"]
        # A well-formed, fully-selected function -- unlike incomplete generic MIR.
        assert mf.verify() is True
    assert_no_leaks()
