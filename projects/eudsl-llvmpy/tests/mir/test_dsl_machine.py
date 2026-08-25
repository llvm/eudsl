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

        instrs = g.machine_function.blocks[0].instructions
        ops = [i.opcode_name for i in instrs]
        # one each of add/mul/sub, plus three constants whose literal values are
        # exactly 1, 2, 10 -- read straight off the G_CONSTANT's CImm operand
        # (operand 1; operand 0 is the def) rather than inferred from structure.
        assert "G_ADD" in ops and "G_MUL" in ops and "G_SUB" in ops
        consts = sorted(
            i.operand(1).cimm.value for i in instrs if i.opcode_name == "G_CONSTANT"
        )
        assert consts == [1, 2, 10]
    assert_no_leaks()


def test_reflected_subtraction_operand_order():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        captured = {}

        @machine_function(module=mod, target=tm)
        def g(a: s32):
            captured["a"] = a.reg.id
            return 10 - a

        instrs = g.machine_function.blocks[0].instructions
        const = next(i for i in instrs if i.opcode_name == "G_CONSTANT")
        sub = next(i for i in instrs if i.opcode_name == "G_SUB")
        # The constant operand literally holds 10 (read via its CImm), and
        # `10 - a` must emit G_SUB(const, a): lhs (operand 1) is the constant,
        # rhs (operand 2) is a. operand(0) is the def. If the reflected flag were
        # dropped, these two would be swapped.
        assert const.operand(1).cimm.value == 10
        assert sub.operand(1).reg.id == const.operand(0).reg.id
        assert sub.operand(2).reg.id == captured["a"]
    assert_no_leaks()


def test_forward_subtraction_operand_order():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        captured = {}

        @machine_function(module=mod, target=tm)
        def g(a: s32):
            captured["a"] = a.reg.id
            return a - 10

        instrs = g.machine_function.blocks[0].instructions
        const = next(i for i in instrs if i.opcode_name == "G_CONSTANT")
        sub = next(i for i in instrs if i.opcode_name == "G_SUB")
        # The constant operand literally holds 10 (read via its CImm), and
        # `a - 10` -> G_SUB(a, const): a is the lhs, the constant is the rhs.
        assert const.operand(1).cimm.value == 10
        assert sub.operand(1).reg.id == captured["a"]
        assert sub.operand(2).reg.id == const.operand(0).reg.id
    assert_no_leaks()


def test_explicit_name_override():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        @machine_function(module=mod, target=tm, name="custom")
        def f(a: s32):
            return a + a

        assert f.name == "custom"
        assert f.machine_function.name == "custom"
    assert_no_leaks()


def test_mismatched_operand_types_raise():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32, s64 = mir.LLT.scalar(32), mir.LLT.scalar(64)

        with pytest.raises(TypeError, match="mismatched types"):

            @machine_function(module=mod, target=tm)
            def f(a: s32, b: s64):
                return a + b

    assert_no_leaks()


def test_non_int_coercion_raises():
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        for bad in (3.7, True, "5"):
            with pytest.raises(TypeError, match="cannot coerce"):

                @machine_function(module=ir.Module("m", ctx), target=tm)
                def f(a: s32):
                    return a + bad  # noqa: B023 (traced eagerly in this loop)

    assert_no_leaks()


def test_value_from_other_function_raises():
    """A value from one @machine_function body used in another's is rejected:
    its Register carries function g as its owner, so f's builder refuses it
    (ValueError from the C++ owner check, not a Python-side anchor)."""
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        leaked = {}

        @machine_function(module=ir.Module("g", ctx), target=tm)
        def g(a: s32):
            leaked["a"] = a
            return a

        with pytest.raises(ValueError, match="different MachineFunction"):

            @machine_function(module=ir.Module("f", ctx), target=tm)
            def f(b: s32):
                return b + leaked["a"]

    assert_no_leaks()


def test_value_reused_under_second_builder_of_same_function():
    """Dropping the per-builder anchor ties a MachineValue to its *function*,
    not to the specific builder instance that was current when it was made: a
    value built under one MachineIRBuilder is still usable under a second
    builder of the same MachineFunction (the C++ owner check keys on the
    function). Contrast test_value_from_other_function_raises."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        mf = mir.create_machine_function(mod, tm, "f").machine_function("f")
        with mir.MachineIRBuilder(mf):
            a = MachineValue(mf.create_generic_virtual_register(s32), s32)
        with mir.MachineIRBuilder(mf):  # a fresh builder over the same function
            out = a + a
        assert out.reg.is_virtual
        assert any(i.opcode_name == "G_ADD" for i in mf.blocks[0].instructions)
    assert_no_leaks()


def test_builder_stack_cleared_when_body_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(ZeroDivisionError):

            @machine_function(module=mod, target=tm)
            def f(a: s32):
                raise ZeroDivisionError("boom")

        # The `with MachineIRBuilder(mf):` __exit__ popped the builder even
        # though the body raised.
        with pytest.raises(RuntimeError):
            current_machine_builder()
    assert_no_leaks()


def test_keyword_only_parameter_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        with pytest.raises(TypeError, match="must be positional"):

            @machine_function(module=mod, target=tm)
            def f(*, a: s32):
                return a

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

        with pytest.raises(TypeError, match="missing an LLT annotation"):

            @machine_function(module=mod, target=tm)
            def f(a):  # no LLT annotation
                return a

    assert_no_leaks()


def test_non_llt_annotation_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)

        with pytest.raises(TypeError, match="must be annotated with an LLT"):

            @machine_function(module=mod, target=tm)
            def f(a: int):  # annotated, but not an LLT
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
