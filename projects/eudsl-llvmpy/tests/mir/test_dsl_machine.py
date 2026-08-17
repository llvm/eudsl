#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The @machine_function DSL: build generic MIR with Pythonic operators.

Mirrors the IR DSL's @function/ArithValue: parameters annotated with an LLT
become MachineValues over fresh generic vregs, and `+ - *` emit G_ADD/G_SUB/
G_MUL through a contextual MachineIRBuilder.
"""

import pytest

from llvm import ir, jit, mir
from llvm.dsl import machine_function
from llvm.dsl.machine import MachineValue, current_machine_builder
from llvm.testing import assert_no_leaks

# Generic (G_*) MIR is target-independent; build against the host
# target so these run on any runner.
_TRIPLE = None


def test_machine_function_builds_generic_add():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        def f(a: s32, b: s32):
            return a + b

        assert f.name == "f"
        opcodes = [i.opcode_name for i in f.machine_function.blocks[0].instructions]
        assert opcodes == ["G_ADD"]
    assert_no_leaks()


def test_operators_and_int_coercion():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        def g(a: s32):
            return (a + 1) * a - a

        ops = [i.opcode_name for i in g.machine_function.blocks[0].instructions]
        assert ops == ["G_CONSTANT", "G_ADD", "G_MUL", "G_SUB"]
    assert_no_leaks()


def test_reflected_operators():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        def g(a: s32):
            return 1 + a, 2 * a, 10 - a

        ops = [i.opcode_name for i in g.machine_function.blocks[0].instructions]
        # three constants (1, 2, 10) and one each of add/mul/sub
        assert ops.count("G_CONSTANT") == 3
        assert "G_ADD" in ops and "G_MUL" in ops and "G_SUB" in ops
    assert_no_leaks()


def test_to_mir_from_dsl():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm)
        def f(a: s32, b: s32):
            return a * b

        assert "G_MUL" in f.to_mir()
    assert_no_leaks()


def test_unannotated_parameter_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)

        with pytest.raises(TypeError):

            @machine_function(module=mod, target=tm)
            def f(a):  # no LLT annotation
                return a

    assert_no_leaks()


def test_current_machine_builder_outside_function_raises():
    with pytest.raises(RuntimeError):
        current_machine_builder()


def test_machine_value_holds_register_and_type():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        captured = {}

        @machine_function(module=mod, target=tm)
        def f(a: s32):
            captured["a"] = a
            return a

        assert isinstance(captured["a"], MachineValue)
        assert captured["a"].reg.is_virtual
        assert captured["a"].llt == s32
    assert_no_leaks()
