#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The generic-MIR if/else control-flow runtime.

Mirrors the IR DSL's if/else lowering (llvm.dsl.cf) but emits MachineBasicBlocks
and G_PHI through MachineIRBuilder. The same AST canonicalizer rewrites `if/else`
+ `yield` into if_ctx_manager/else_ctx_manager/yield_; only the injected runtime
differs (MIRCanonicalizer).
"""

import pytest

from llvm import ir, jit, mir
from llvm.ast.canonicalize import canonicalize
from llvm.dsl import machine_function
from llvm.dsl.machine_cf import (
    MIRCanonicalizer,
    if_ctx_manager,
    else_ctx_manager,
    yield_,
)
from llvm.testing import assert_no_leaks

# Generic (G_*) control-flow MIR is target-independent; host target.
_TRIPLE = None


def test_if_else_builds_a_phi_diamond():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def f(a: s32, b: s32):
            if a < b:
                r = yield a + 1
            else:
                r = yield b
            return r

        text = f.to_mir()
        assert "G_ICMP" in text
        assert "G_BRCOND" in text
        assert "G_PHI" in text
        # entry, if.then, if.else, if.end
        assert len(f.machine_function.blocks) == 4
    assert_no_leaks()


def test_if_without_else_has_no_phi():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def f(a: s32, b: s32):
            if a < b:
                r = yield a * b
            return r

        text = f.to_mir()
        assert "G_BRCOND" in text
        assert "G_PHI" not in text
        # entry, if.then, if.end
        assert len(f.machine_function.blocks) == 3
    assert_no_leaks()


def test_comparison_produces_i1():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        captured = {}

        @machine_function(module=mod, target=tm)
        def f(a: s32, b: s32):
            captured["c"] = a < b

        assert captured["c"].llt == mir.LLT.scalar(1)
    assert_no_leaks()


def test_all_comparison_predicates_build_g_icmp():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        i1 = mir.LLT.scalar(1)
        cmps = {}

        @machine_function(module=mod, target=tm)
        def f(a: s32, b: s32):
            cmps["lt"] = a < b
            cmps["le"] = a <= b
            cmps["gt"] = a > b
            cmps["ge"] = a >= b
            cmps["eq"] = a.eq(b)
            cmps["ne"] = a.ne(b)

        assert all(v.llt == i1 for v in cmps.values())
        opcodes = [i.opcode_name for i in f.machine_function.blocks[0].instructions]
        assert opcodes.count("G_ICMP") == 6
    assert_no_leaks()


def test_if_else_with_two_carried_values_builds_two_phis():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def f(a: s32, b: s32):
            if a < b:
                x, y = yield a, b
            else:
                x, y = yield b, a
            return x + y

        assert f.to_mir().count("G_PHI") == 2

        # The branches yield (a, b) then (b, a) -- a classic swap. The two phis
        # must route the values in that swapped order: whatever is phi0's
        # then-value (operand 1) must be phi1's else-value (operand 3), and vice
        # versa. A then/else swap or wrong-predecessor bug breaks this.
        phis = [
            i
            for blk in f.machine_function.blocks
            for i in blk.instructions
            if i.opcode_name == "G_PHI"
        ]
        assert len(phis) == 2
        p0, p1 = phis
        assert p0.operand(1).reg.id == p1.operand(3).reg.id
        assert p0.operand(3).reg.id == p1.operand(1).reg.id
    assert_no_leaks()


def test_nested_if_else_captures_moved_predecessor():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        @canonicalize(using=MIRCanonicalizer())
        def f(a: s32, b: s32):
            if a < b:
                if b < a:
                    r = yield a
                else:
                    r = yield b
                s = yield r + a
            else:
                s = yield b
            return s

        # Outer + inner diamonds each build a phi; the outer phi's then-edge
        # predecessor is the inner merge block (the builder moved), so this
        # exercises the moved-predecessor capture.
        assert f.to_mir().count("G_PHI") == 2
        # entry + (inner then/else/merge) + (outer then-tail already = inner
        # merge) + outer else + outer merge.
        assert len(f.machine_function.blocks) >= 6
    assert_no_leaks()


def test_equality_is_python_identity_not_icmp():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        captured = {}

        @machine_function(module=mod, target=tm)
        def f(a: s32, b: s32):
            # `==`/`!=` are Python identity (documented), NOT a G_ICMP: two
            # distinct MachineValues compare unequal, and no instruction is
            # emitted. Value comparison is a.eq(b) / a.ne(b).
            captured["eq"] = a == b
            captured["ne"] = a != b

        assert captured["eq"] is False
        assert captured["ne"] is True
        opcodes = [i.opcode_name for i in f.machine_function.blocks[0].instructions]
        assert "G_ICMP" not in opcodes
    assert_no_leaks()


def test_unsigned_comparisons_build_g_icmp():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        i1 = mir.LLT.scalar(1)
        cmps = {}

        @machine_function(module=mod, target=tm)
        def f(a: s32, b: s32):
            cmps["ult"] = a.ult(b)
            cmps["ule"] = a.ule(b)
            cmps["ugt"] = a.ugt(b)
            cmps["uge"] = a.uge(b)

        assert all(v.llt == i1 for v in cmps.values())
        opcodes = [i.opcode_name for i in f.machine_function.blocks[0].instructions]
        assert opcodes.count("G_ICMP") == 4
    assert_no_leaks()


def test_reflected_comparison_produces_i1():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        captured = {}

        @machine_function(module=mod, target=tm)
        def f(a: s32):
            captured["c"] = 1 < a  # int.__lt__ defers -> a.__gt__(1) -> SGT

        assert captured["c"].llt == mir.LLT.scalar(1)
    assert_no_leaks()


def test_mismatched_comparison_types_raise():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32, s64 = mir.LLT.scalar(32), mir.LLT.scalar(64)

        with pytest.raises(TypeError, match="mismatched types"):

            @machine_function(module=mod, target=tm)
            def f(a: s32, b: s64):
                return a < b

    assert_no_leaks()


def test_mismatched_branch_arity_raises():
    # Drive the if/else runtime directly (bypassing the canonicalizer) to feed
    # branches that yield different numbers of values.
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(ValueError, match="different numbers of values"):

            @machine_function(module=mod, target=tm)
            def f(a: s32, b: s32):
                with if_ctx_manager(a < b) as op:
                    yield_(a, b)  # two values
                with else_ctx_manager(op):
                    yield_(b)  # one value

    assert_no_leaks()


def test_mismatched_branch_types_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32, s64 = mir.LLT.scalar(32), mir.LLT.scalar(64)

        with pytest.raises(TypeError, match="mismatched types"):

            @machine_function(module=mod, target=tm)
            def f(a: s32, c: s64):
                with if_ctx_manager(a < a) as op:
                    yield_(a)  # s32
                with else_ctx_manager(op):
                    yield_(c)  # s64 -> phi type mismatch

    assert_no_leaks()


def test_yield_outside_if_body_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(RuntimeError, match="outside a lowered if/else"):

            @machine_function(module=mod, target=tm)
            def f(a: s32):
                yield_(a)  # no active if op

    assert_no_leaks()
