#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Building generic (GlobalISel) MIR with MachineIRBuilder.

create_machine_function makes a fresh, empty MachineFunction to build into;
MachineIRBuilder emits target-independent G_* instructions. The built MIR is
inspected with the Phase 1 object model / printer.
"""

from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# Generic (G_*) MIR is target-independent, so build against the host target
# (triple=None) -- these run on any runner, no specific backend required.
_TRIPLE = None


def test_create_machine_function_is_empty_and_named():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        mf = mmi.machine_function("f")
        assert mf.name == "f"
        assert len(mf.blocks) == 1  # a single empty entry block
        assert mf.blocks[0].instructions == []
    assert_no_leaks()


def test_build_generic_arithmetic():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        b = mir.MachineIRBuilder(mmi.machine_function("f"))
        s32 = mir.LLT.scalar(32)
        a = b.build_constant(s32, 3)
        c = b.build_constant(s32, 4)
        assert a.is_virtual and c.is_virtual
        s = b.build_add(s32, a, c)
        d = b.build_sub(s32, s, a)
        p = b.build_mul(s32, d, c)
        b.build_copy(s32, p)

        opcodes = [
            i.opcode_name for i in mmi.machine_function("f").blocks[0].instructions
        ]
        assert opcodes == [
            "G_CONSTANT",
            "G_CONSTANT",
            "G_ADD",
            "G_SUB",
            "G_MUL",
            "COPY",
        ]
    assert_no_leaks()


def test_create_generic_vreg_has_requested_type():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        reg = mmi.machine_function("f").create_generic_vreg(mir.LLT.scalar(64))
        assert reg.is_virtual
    assert_no_leaks()
