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
from llvm.dsl.machine_cf import MIRCanonicalizer
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
    assert_no_leaks()
