#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Building generic (GlobalISel) MIR with MachineIRBuilder.

create_machine_function makes a fresh, empty MachineFunction to build into;
MachineIRBuilder emits target-independent G_* instructions. The built MIR is
inspected with the MIR object model / printer.
"""

import gc

import pytest

from llvm import ir, jit, mir, types
from llvm.testing import assert_no_leaks

# Generic (G_*) MIR is target-independent, so build against the host target
# (triple=None) -- these run on any runner, no non-host backend required.
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


def test_create_machine_function_defaults_to_void_external():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        f = mmi.machine_function("f").function
        # Default stub: void() with external linkage.
        assert f.function_type.return_type.is_void
        assert f.function_type.num_params == 0
        assert f.linkage == ir.Linkage.EXTERNAL
    assert_no_leaks()


def test_create_machine_function_honors_function_type_and_linkage():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        i32 = types.i32(ctx)
        fnTy = types.FunctionType.get(i32, [i32, i32], context=ctx)
        mmi = mir.create_machine_function(
            mod, tm, "f", function_type=fnTy, linkage=ir.Linkage.INTERNAL
        )
        f = mmi.machine_function("f").function
        assert f.function_type.return_type == i32
        assert f.function_type.num_params == 2
        assert f.linkage == ir.Linkage.INTERNAL
    assert_no_leaks()


def test_create_machine_function_consumes_module():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mir.create_machine_function(mod, tm, "f")
        assert mod._is_consumed
        with pytest.raises(RuntimeError, match="has been consumed"):
            mod.functions  # the module moved into the wrapper; it can't be used
    assert_no_leaks()


def test_create_machine_function_rejects_empty_name():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        with pytest.raises(ValueError, match="must not be empty"):
            mir.create_machine_function(mod, tm, "")
        # A rejected call must leave the module usable (not consumed).
        assert not mod._is_consumed
    assert_no_leaks()


def test_create_machine_function_rejects_duplicate_name():
    with ir.Context() as ctx:
        mod = ir.parse_assembly("define void @f() {\n  ret void\n}\n", ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        with pytest.raises(ValueError, match="already has a function named"):
            mir.create_machine_function(mod, tm, "f")
        assert not mod._is_consumed
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
        cp = b.build_copy(s32, p)
        assert s.is_virtual and d.is_virtual and p.is_virtual and cp.is_virtual

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


def test_build_constant_rejects_non_scalar_type():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        b = mir.MachineIRBuilder(
            mir.create_machine_function(mod, tm, "f").machine_function("f")
        )
        with pytest.raises(ValueError, match="scalar or fixed-vector"):
            b.build_constant(mir.LLT.pointer(0, 64), 0)
    assert_no_leaks()


def test_build_add_rejects_type_mismatch():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        b = mir.MachineIRBuilder(
            mir.create_machine_function(mod, tm, "f").machine_function("f")
        )
        s32, s64 = mir.LLT.scalar(32), mir.LLT.scalar(64)
        a = b.build_constant(s32, 1)
        # Result type s64 disagrees with the s32 operands.
        with pytest.raises(ValueError, match="result type"):
            b.build_add(s64, a, a)
    assert_no_leaks()


def test_build_add_rejects_register_from_another_function():
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        mmi_g = mir.create_machine_function(ir.Module("g", ctx), tm, "g")
        foreign = mir.MachineIRBuilder(mmi_g.machine_function("g")).build_constant(
            s32, 7
        )

        mmi_f = mir.create_machine_function(ir.Module("f", ctx), tm, "f")
        bf = mir.MachineIRBuilder(mmi_f.machine_function("f"))
        # The register carries function g as its owner, so f's builder rejects
        # it by provenance (see test_cross_function_vreg_collision_rejected for
        # why the owner, not just the id/type, is what makes this sound).
        with pytest.raises(ValueError, match="different MachineFunction"):
            bf.build_add(s32, foreign, foreign)
    assert_no_leaks()


def test_builder_outlives_machine_function_handle():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        b = mir.MachineIRBuilder(mmi.machine_function("f"))
        del mmi
        gc.collect()
        # keep_alive<1,2> pins the MachineFunction (and transitively its owning
        # MirModule) to the builder, so building still works.
        assert b.build_constant(mir.LLT.scalar(32), 5).is_virtual
    assert_no_leaks()


def test_create_generic_virtual_register_is_virtual():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        mf = mmi.machine_function("f")
        s64 = mir.LLT.scalar(64)
        reg = mf.create_generic_virtual_register(s64)
        assert reg.is_virtual
        # The vreg carries the requested LLT: it is accepted as an s64 operand
        # (the builder validates operand type against the result type).
        b = mir.MachineIRBuilder(mf)
        assert b.build_add(s64, reg, reg).is_virtual
    assert_no_leaks()


def test_build_typed_instr_mints_and_reuses_destinations():
    """`build` is the typed buildInstr(opcode, DstOps, SrcOps): an LLT dst mints
    a fresh generic vreg and defines it; a Register dst defines that existing
    vreg. Both are exercised here on a G_ADD, plus the Register source uses."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        mf = mmi.machine_function("f")
        s32 = mir.LLT.scalar(32)
        b = mir.MachineIRBuilder(mf)
        a = mf.create_generic_virtual_register(s32)
        c = mf.create_generic_virtual_register(s32)
        g_add = mf.opcode("G_ADD")

        # LLT dst: a fresh generic vreg is minted for the def.
        minted = b.build(g_add, [s32], [a, c])
        assert minted.opcode_name == "G_ADD"
        assert minted.operand(0).is_def and minted.operand(0).reg.is_virtual
        assert minted.operand(0).reg.id not in (a.id, c.id)  # a new register
        assert minted.operand(1).reg.id == a.id
        assert minted.operand(2).reg.id == c.id

        # Register dst: the caller's existing vreg is defined instead.
        dst = mf.create_generic_virtual_register(s32)
        reused = b.build(g_add, [dst], [a, c])
        assert reused.operand(0).reg.id == dst.id
    assert_no_leaks()


def test_build_rejects_out_of_range_opcode():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        mf = mmi.machine_function("f")
        b = mir.MachineIRBuilder(mf)
        with pytest.raises(IndexError, match="opcode number out of range"):
            b.build(10**9, [mir.LLT.scalar(32)], [])
    assert_no_leaks()


def test_build_emits_multiple_defs():
    """build's headline over the single-def helpers is multiple defs: G_UADDO
    yields a (result, carry) pair, so both operand 0 and operand 1 are defs."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        mf = mmi.machine_function("f")
        s32, s1 = mir.LLT.scalar(32), mir.LLT.scalar(1)
        b = mir.MachineIRBuilder(mf)
        a = mf.create_generic_virtual_register(s32)
        c = mf.create_generic_virtual_register(s32)
        uaddo = b.build(mf.opcode("G_UADDO"), [s32, s1], [a, c])
        assert uaddo.opcode_name == "G_UADDO"
        assert uaddo.num_defs == 2
        assert uaddo.operand(0).is_def and uaddo.operand(1).is_def
    assert_no_leaks()


def test_build_rejects_registers_from_another_function():
    """build guards both destinations and sources: a foreign vreg used as a dst
    or a src is rejected by owner (its own guard call sites, not just the
    shared helper reached via other builders)."""
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)
        mmi_g = mir.create_machine_function(ir.Module("g", ctx), tm, "g")
        foreign = mir.MachineIRBuilder(mmi_g.machine_function("g")).build_constant(
            s32, 1
        )
        mmi_f = mir.create_machine_function(ir.Module("f", ctx), tm, "f")
        mf = mmi_f.machine_function("f")
        b = mir.MachineIRBuilder(mf)
        own = mf.create_generic_virtual_register(s32)
        g_add = mf.opcode("G_ADD")
        with pytest.raises(ValueError, match="dst .*different MachineFunction"):
            b.build(g_add, [foreign], [own, own])
        with pytest.raises(ValueError, match="src .*different MachineFunction"):
            b.build(g_add, [s32], [foreign, own])
    assert_no_leaks()


def test_register_read_off_operand_feeds_back_into_builder():
    """A register inspected off an instruction's operand carries its owning
    function, so it is accepted when fed back into that function's builder --
    the round-trip the owner field must not wrongly reject."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.create_machine_function(mod, tm, "f")
        mf = mmi.machine_function("f")
        s32 = mir.LLT.scalar(32)
        b = mir.MachineIRBuilder(mf)
        a = mf.create_generic_virtual_register(s32)
        c = mf.create_generic_virtual_register(s32)
        add = b.build(mf.opcode("G_ADD"), [s32], [a, c])
        read_back = add.operand(0).reg  # the def vreg, read off the operand
        assert read_back.is_virtual
        # Accepted by both the typed helper and the raw operand appender.
        assert b.build_add(s32, read_back, read_back).is_virtual
        instr = b.build_instr(mf.opcode("G_ADD"))
        instr.add_def(read_back)  # no "different MachineFunction" error
        assert instr.operand(0).reg.id == read_back.id
    assert_no_leaks()


def test_machine_ir_builder_context_manager_tracks_current():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        outer = mir.MachineIRBuilder(
            mir.create_machine_function(mod, tm, "f").machine_function("f")
        )
        inner = mir.MachineIRBuilder(
            mir.create_machine_function(ir.Module("g", ctx), tm, "g").machine_function(
                "g"
            )
        )
        with outer as entered:
            # __enter__ returns the builder itself, and it becomes current.
            assert entered is outer
            assert mir.current_machine_builder() is outer
            with inner:
                # Nested: the innermost builder is current.
                assert mir.current_machine_builder() is inner
            # Popping restores the outer builder.
            assert mir.current_machine_builder() is outer
        # Fully unwound: no current builder.
        with pytest.raises(RuntimeError, match="no current MachineIRBuilder"):
            mir.current_machine_builder()
    assert_no_leaks()


def test_current_machine_builder_without_context_raises():
    with pytest.raises(RuntimeError, match="no current MachineIRBuilder"):
        mir.current_machine_builder()
    assert_no_leaks()


def test_machine_ir_builder_unbalanced_exit_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        b = mir.MachineIRBuilder(
            mir.create_machine_function(mod, tm, "f").machine_function("f")
        )
        # __exit__ without a matching __enter__ (empty stack) is unbalanced.
        with pytest.raises(ValueError, match="unbalanced"):
            b.__exit__(None, None, None)
    assert_no_leaks()


# A function with no `body:` parses to a MachineFunction with zero blocks.
_BODYLESS_MIR = """\
--- |
  define void @f() {
    ret void
  }
...
---
name:            f
...
"""


def test_machine_ir_builder_rejects_block_less_machine_function():
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        mf = mir.parse_mir(_BODYLESS_MIR, ctx, tm).machine_function("f")
        assert len(mf.blocks) == 0
        with pytest.raises(ValueError, match="no basic block"):
            mir.MachineIRBuilder(mf)
    assert_no_leaks()
