#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generic-MIR loop lowering: for/while -> header G_PHIs.

A `for i in range_(...)` / `while COND` in a @machine_function body lowers to a
preheader -> header(phis) -> body -> exit shape, mirroring the IR DSL. Asserted
on the printed MIR structure. The loop body's trailing `yield` is rewritten to
the loop-carried update by the AST transformer.
"""

from llvm import ir, jit, mir
from llvm.ast.canonicalize import canonicalize
from llvm.dsl import machine_function
from llvm.dsl.machine_cf import MIRCanonicalizer, range_, while_, loop_yield
from llvm.testing import assert_no_leaks

import pytest

# Generic (G_*) MIR is target-independent; host target.
_TRIPLE = None


def _header_phis(mf):
    """The G_PHIs of the loop header (the block that has them)."""
    return [
        i for blk in mf.blocks for i in blk.instructions if i.opcode_name == "G_PHI"
    ]


def test_for_loop_lowers_to_header_body_exit_with_phis():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def total(n: s32, acc: s32):
            for i in range_(0, n):
                acc = acc + i
                yield acc
            return acc

        # entry (preheader), for.header, for.body, for.end
        assert len(total.machine_function.blocks) == 4
        text = total.to_mir()
        assert "G_PHI" in text
        assert "G_BRCOND" in text
    assert_no_leaks()


def test_while_loop_lowers_with_carried_phi():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def countdown(n: s32):
            while n > 0:
                n = n - 1
                yield n
            return n

        assert len(countdown.machine_function.blocks) == 4
        text = countdown.to_mir()
        assert "G_PHI" in text
        assert "G_ICMP" in text  # the while condition
    assert_no_leaks()


def test_for_loop_single_arg_range():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def total(n: s32, acc: s32):
            for i in range_(n):  # single-arg form: 0..n
                acc = acc + i
                yield acc
            return acc

        assert len(total.machine_function.blocks) == 4
        # 0..n ascending -> signed-less-than induction compare
        assert "intpred(slt)" in total.to_mir()
    assert_no_leaks()


def test_for_loop_descending_step_compares_sgt():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def total(n: s32, acc: s32):
            for i in range_(n, 0, -1):
                acc = acc + i
                yield acc
            return acc

        # A negative step makes the induction compare signed-greater-than.
        assert "intpred(sgt)" in total.to_mir()
    assert_no_leaks()


def test_induction_phi_has_two_incomings():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def total(n: s32, acc: s32):
            for i in range_(0, n):
                acc = acc + i
                yield acc
            return acc

        # An induction phi and a carried (acc) phi, each with exactly two
        # incomings: def + (preheader value, block) + (back-edge value, block).
        phis = _header_phis(total.machine_function)
        assert len(phis) == 2
        for phi in phis:
            assert phi.num_operands == 5  # def, v0, bb0, v1, bb1
    assert_no_leaks()


def test_loop_carried_result_is_the_header_phi():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        captured = {}

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def total(n: s32, acc: s32):
            for i in range_(0, n):
                acc = acc + i
                yield acc
            captured["acc_out"] = acc  # the loop's live-out carried value
            return acc

        # The value used after the loop is a carried header phi's def, so its
        # register is one of the header phi defs (the back-edge was wired).
        phi_defs = {p.operand(0).reg.id for p in _header_phis(total.machine_function)}
        assert captured["acc_out"].reg.id in phi_defs
    assert_no_leaks()


def test_if_in_loop_body_wires_back_edge_from_moved_block():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def f(n: s32, acc: s32):
            for i in range_(0, n):
                if i < n:
                    acc = yield acc + i
                else:
                    acc = yield acc
                yield acc
            return acc

        # The if inside the loop body moves the builder, so the back-edge
        # predecessor is the if's merge block, not the loop body entry. It must
        # still build valid MIR with the loop's carried phi present.
        mf = f.machine_function
        assert len(_header_phis(mf)) >= 1
        assert len(mf.blocks) >= 6  # preheader, header, body+if diamond, exit
    assert_no_leaks()


def test_range_step_zero_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(ValueError, match="must not be zero"):

            @machine_function(module=mod, target=tm)
            @canonicalize(using=MIRCanonicalizer())
            def f(n: s32, acc: s32):
                for i in range_(0, n, 0):
                    acc = acc + i
                    yield acc
                return acc

    assert_no_leaks()


def test_range_non_int_step_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(TypeError, match="step must be an int"):

            @machine_function(module=mod, target=tm)
            @canonicalize(using=MIRCanonicalizer())
            def f(n: s32, acc: s32):
                for i in range_(0, n, 1.5):
                    acc = acc + i
                    yield acc
                return acc

    assert_no_leaks()


def test_loop_needs_a_machinevalue_for_llt():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        # All-int bounds and no carried MachineValue -> no LLT to infer from.
        with pytest.raises(NotImplementedError, match="at least one MachineValue"):

            @machine_function(module=mod, target=tm)
            def f(a: s32):
                with range_(0, 10):
                    loop_yield()

    assert_no_leaks()


def test_while_condition_must_be_i1_machinevalue():
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        # Drive while_ directly with a cond_fn that returns a non-MachineValue.
        with pytest.raises(TypeError, match="i1 MachineValue"):

            @machine_function(module=ir.Module("m1", ctx), target=tm)
            def f(a: s32):
                with while_(lambda c: 5, iter_args=(a,)):
                    loop_yield(a)

        # ...and one that returns a MachineValue of the wrong (non-i1) type.
        with pytest.raises(TypeError, match="i1 MachineValue"):

            @machine_function(module=ir.Module("m2", ctx), target=tm)
            def g(a: s32):
                with while_(lambda c: c, iter_args=(a,)):  # returns s32, not i1
                    loop_yield(a)

    assert_no_leaks()


def test_loop_yield_outside_loop_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(RuntimeError, match="outside a loop body"):

            @machine_function(module=mod, target=tm)
            def f(a: s32):
                loop_yield(a)

    assert_no_leaks()


def test_loop_carried_arity_mismatch_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        # Drive a loop directly, yielding fewer carried values than iter_args.
        with pytest.raises(ValueError, match="carries 1"):

            @machine_function(module=mod, target=tm)
            def f(a: s32):
                with range_(0, a, iter_args=(a,)):
                    loop_yield()  # zero carried values, but the loop carries 1

    assert_no_leaks()


def test_loop_body_exception_propagates_and_pops_stack():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(ZeroDivisionError):

            @machine_function(module=mod, target=tm)
            def f(a: s32):
                with range_(0, a, iter_args=(a,)):
                    raise ZeroDivisionError("boom")

        # __exit__ popped the loop stack even though the body raised, so a
        # later loop_yield reports "outside a loop body" rather than mis-firing.
        with pytest.raises(RuntimeError, match="outside a loop body"):
            loop_yield()
    assert_no_leaks()
