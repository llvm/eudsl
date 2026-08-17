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
from llvm.dsl.machine_cf import MIRCanonicalizer
from llvm.testing import assert_no_leaks

# Generic (G_*) MIR is target-independent; host target.
_TRIPLE = None


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
