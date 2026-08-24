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
        with pytest.raises(ValueError, match="virtual register of this"):
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
