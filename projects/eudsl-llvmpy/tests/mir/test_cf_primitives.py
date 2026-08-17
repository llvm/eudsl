#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Control-flow primitives for generic MIR: extra blocks, branches, compares,
and G_PHI. These are the pieces the @machine_function control-flow runtime is
built on; here they are driven directly to hand-build an if-diamond.
"""

from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# Generic (G_*) control-flow MIR is target-independent; host target.
_TRIPLE = None


def _new_function(ctx):
    mod = ir.Module("m", ctx)
    tm = jit.TargetMachine(triple=_TRIPLE)
    mmi = mir.create_machine_function(mod, tm, "f")
    return mmi, mmi.machine_function("f")


def test_create_block_and_set_block():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        b = mir.MachineIRBuilder(mf)
        assert b.insert_block.number == mf.blocks[0].number
        bb1 = mf.create_block()
        assert len(mf.blocks) == 2
        b.set_block(bb1)
        assert b.insert_block.number == bb1.number
    assert_no_leaks()


def test_build_if_diamond_with_phi():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        s32 = mir.LLT.scalar(32)
        s1 = mir.LLT.scalar(1)
        b = mir.MachineIRBuilder(mf)
        entry = mf.blocks[0]

        x = mf.create_generic_virtual_register(s32)
        one = b.build_constant(s32, 1)
        two = b.build_constant(s32, 2)
        cond = b.build_icmp(ir.ICmpPredicate.SLT, s1, x, one)  # x < 1

        then_bb = mf.create_block()
        else_bb = mf.create_block()
        join_bb = mf.create_block()
        entry.add_successor(then_bb)
        entry.add_successor(else_bb)
        b.build_brcond(cond, then_bb)
        b.build_br(else_bb)

        b.set_block(then_bb)
        tv = b.build_add(s32, x, one)
        b.build_br(join_bb)
        then_bb.add_successor(join_bb)

        b.set_block(else_bb)
        ev = b.build_sub(s32, x, two)
        b.build_br(join_bb)
        else_bb.add_successor(join_bb)

        b.set_block(join_bb)
        r = b.build_phi(s32, [(tv, then_bb), (ev, else_bb)])
        assert r.is_virtual

        text = str(mf)
        assert len(mf.blocks) == 4
        for opc in ("G_ICMP", "G_BRCOND", "G_BR", "G_PHI"):
            assert opc in text
    assert_no_leaks()
