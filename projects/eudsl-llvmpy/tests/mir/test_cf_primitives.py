#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Control-flow primitives for generic MIR: extra blocks, branches, compares,
and G_PHI. Here they are driven directly to hand-build an if-diamond.
"""

import pytest

from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# Generic (G_*) control-flow MIR is target-independent; host target.
_TRIPLE = None


def _new_function(ctx, name="f"):
    mod = ir.Module("m", ctx)
    tm = jit.TargetMachine(triple=_TRIPLE)
    mmi = mir.create_machine_function(mod, tm, name)
    return mmi, mmi.machine_function(name)


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

        assert len(mf.blocks) == 4

        # Structural check of the G_ICMP: def is `cond`, operands (after the
        # predicate operand) are x and one, in order. build_* return a raw
        # Register, so compare .id directly.
        icmp = next(i for i in entry.instructions if i.opcode_name == "G_ICMP")
        assert icmp.operand(0).reg.id == cond.id
        assert icmp.operand(2).reg.id == x.id
        assert icmp.operand(3).reg.id == one.id

        # Structural check of the G_PHI: def is `r`, and the value operands are
        # tv then ev in that order (register operands are 1 and 3; 2 and 4 are
        # the predecessor blocks). Swapping the pair order would fail here.
        phi = next(i for i in join_bb.instructions if i.opcode_name == "G_PHI")
        assert phi.num_operands == 5
        assert phi.operand(0).reg.id == r.id
        assert phi.operand(1).reg.id == tv.id
        assert phi.operand(3).reg.id == ev.id
    assert_no_leaks()


def test_create_block_accepts_ir_basic_block():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        # Passing an IR BasicBlock links the MBB to it (debug info/naming); the
        # default (None) is used everywhere else. Just exercise the arg path.
        fn = ir.parse_assembly(
            "define void @g() {\nentry:\n  ret void\n}\n", ctx, "g2"
        ).functions[0]
        bb = fn.entry_block
        before = len(mf.blocks)
        mbb = mf.create_block(bb)
        assert len(mf.blocks) == before + 1
        assert mbb.number == before
    assert_no_leaks()


def test_build_icmp_rejects_float_predicate():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        s32, s1 = mir.LLT.scalar(32), mir.LLT.scalar(1)
        b = mir.MachineIRBuilder(mf)
        x = b.build_constant(s32, 1)
        with pytest.raises(ValueError, match="integer comparison predicate"):
            b.build_icmp(ir.FCmpPredicate.OLT, s1, x, x)
    assert_no_leaks()


def test_build_icmp_rejects_non_s1_result():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        s32 = mir.LLT.scalar(32)
        b = mir.MachineIRBuilder(mf)
        x = b.build_constant(s32, 1)
        with pytest.raises(ValueError, match="s1"):
            b.build_icmp(ir.ICmpPredicate.SLT, s32, x, x)  # result must be s1
    assert_no_leaks()


def test_build_icmp_rejects_mismatched_operands():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        s32, s64, s1 = mir.LLT.scalar(32), mir.LLT.scalar(64), mir.LLT.scalar(1)
        b = mir.MachineIRBuilder(mf)
        a = b.build_constant(s32, 1)
        c = b.build_constant(s64, 1)
        with pytest.raises(ValueError, match="same type"):
            b.build_icmp(ir.ICmpPredicate.SLT, s1, a, c)
    assert_no_leaks()


def test_build_phi_rejects_empty_incomings():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        b = mir.MachineIRBuilder(mf)
        with pytest.raises(ValueError, match="at least one"):
            b.build_phi(mir.LLT.scalar(32), [])
    assert_no_leaks()


def test_build_phi_rejects_mistyped_incoming():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        s32, s64 = mir.LLT.scalar(32), mir.LLT.scalar(64)
        b = mir.MachineIRBuilder(mf)
        wrong = b.build_constant(s64, 1)  # not s32
        entry = mf.blocks[0]
        with pytest.raises(ValueError, match="incoming value"):
            b.build_phi(s32, [(wrong, entry)])
    assert_no_leaks()


def test_cross_function_block_rejected():
    with ir.Context() as ctx:
        mmi_f, mf = _new_function(ctx, "f")
        mmi_g, mg = _new_function(ctx, "g")
        foreign = mg.create_block()  # a block owned by g
        b = mir.MachineIRBuilder(mf)
        entry = mf.blocks[0]
        with pytest.raises(ValueError, match="different MachineFunction"):
            b.set_block(foreign)
        with pytest.raises(ValueError, match="different MachineFunction"):
            b.build_br(foreign)
        with pytest.raises(ValueError, match="different MachineFunction"):
            entry.add_successor(foreign)
    assert_no_leaks()


def test_cross_function_register_rejected():
    with ir.Context() as ctx:
        mmi_f, mf = _new_function(ctx, "f")
        mmi_g, mg = _new_function(ctx, "g")
        s32, s1 = mir.LLT.scalar(32), mir.LLT.scalar(1)
        foreign = mir.MachineIRBuilder(mg).build_constant(s32, 1)  # g's vreg
        bf = mir.MachineIRBuilder(mf)
        with pytest.raises(ValueError):
            bf.build_brcond(foreign, mf.blocks[0])
    assert_no_leaks()


def test_set_branch_target_on_non_branch_raises():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        b = mir.MachineIRBuilder(mf)
        b.build_constant(mir.LLT.scalar(32), 7)
        const_mi = mf.blocks[0].instructions[0]
        with pytest.raises(ValueError):
            const_mi.set_branch_target(mf.create_block())
    assert_no_leaks()


def test_replace_successor_requires_an_existing_successor():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        entry = mf.blocks[0]
        a = mf.create_block()
        b = mf.create_block()
        # `a` is not a successor of entry yet -- replacing it is a no-op in
        # LLVM's assert-only path (UB in release); guarded to raise.
        with pytest.raises(ValueError, match="not a successor"):
            entry.replace_successor(a, b)
        # After adding the edge, the replacement succeeds.
        entry.add_successor(a)
        entry.replace_successor(a, b)
    assert_no_leaks()


def test_replace_successor_rejects_cross_function_block():
    with ir.Context() as ctx:
        mmi_f, mf = _new_function(ctx, "f")
        mmi_g, mg = _new_function(ctx, "g")
        entry = mf.blocks[0]
        a = mf.create_block()
        entry.add_successor(a)
        foreign = mg.create_block()  # belongs to g
        with pytest.raises(ValueError, match="different MachineFunction"):
            entry.replace_successor(a, foreign)
    assert_no_leaks()


def test_add_phi_incoming_requires_a_phi():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        s32 = mir.LLT.scalar(32)
        b = mir.MachineIRBuilder(mf)
        b.build_constant(s32, 1)
        const_mi = mf.blocks[0].instructions[0]  # a G_CONSTANT, not a G_PHI
        with pytest.raises(ValueError, match="requires a G_PHI"):
            const_mi.add_phi_incoming(
                mf.create_generic_virtual_register(s32), mf.blocks[0]
            )
    assert_no_leaks()


def test_build_empty_phi_then_add_incomings():
    with ir.Context() as ctx:
        mmi, mf = _new_function(ctx)
        s32 = mir.LLT.scalar(32)
        b = mir.MachineIRBuilder(mf)
        pred = mf.blocks[0]
        v = b.build_constant(s32, 1)
        phi = b.build_empty_phi(s32)  # def-only
        assert phi.opcode_name == "G_PHI"
        assert phi.num_operands == 1  # just the def
        phi.add_phi_incoming(v, pred)
        assert phi.num_operands == 3  # def + (value, block)
        assert phi.operand(1).reg.id == v.id
    assert_no_leaks()
