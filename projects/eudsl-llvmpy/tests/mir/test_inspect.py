#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Inspecting MIR produced by the codegen pipeline.

`run_codegen_to_mir` runs instruction selection on an IR module and hands back
the MachineModuleInfo that owns the resulting MachineFunctions, so they can be
inspected. The reference shapes asserted here were produced with
`llc -mtriple=aarch64-unknown-linux-gnu -stop-after=finalize-isel`.
"""

from textwrap import dedent

import gc

import pytest

import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# These assert AArch64 opcodes (ADDWrr, RET_ReallyLR, ...), so they need the
# AArch64 backend, which eudsl-llvmpy only links when EUDSL_LLVMPY_TARGETS
# includes it (the default is the native target). Skip otherwise.
pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)

_ADD_SRC = dedent("""\
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """)

_CONST_SRC = dedent("""\
    define i32 @c() {
      ret i32 42
    }
    """)

_DECL_SRC = dedent("""\
    declare i32 @ext()
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """)

# A loop cannot collapse to a single block (a plain conditional branch would be
# if-converted to a CSEL), so its machine blocks get distinct, non-zero numbers.
_LOOP_SRC = dedent("""\
    define void @count(i32 %n) {
    entry:
      br label %head
    head:
      %i = phi i32 [ 0, %entry ], [ %i.next, %body ]
      %c = icmp slt i32 %i, %n
      br i1 %c, label %body, label %exit
    body:
      %i.next = add i32 %i, 1
      br label %head
    exit:
      ret void
    }
    """)

# Two definitions, so to_mir's per-function loop runs more than once.
_TWO_FN_SRC = dedent("""\
    define i32 @add(i32 %a, i32 %b) {
      %s = add i32 %a, %b
      ret i32 %s
    }
    define i32 @sub(i32 %a, i32 %b) {
      %d = sub i32 %a, %b
      ret i32 %d
    }
    """)

_TRIPLE = "aarch64-unknown-linux-gnu"


# A load from a global -> a machine operand that isGlobal (the @g address).
_GLOBAL_SRC = dedent("""\
    @g = global i32 0
    define i32 @f() {
      %v = load i32, ptr @g
      ret i32 %v
    }
    """)

# An alloca -> a stack slot, i.e. a frame-index (isFI) machine operand. The
# accesses are volatile so they survive as real load/store machine instructions
# (a plain alloca would be promoted away).
_ALLOCA_SRC = dedent("""\
    define i32 @a(i32 %x) {
      %p = alloca i32
      store volatile i32 %x, ptr %p
      %v = load volatile i32, ptr %p
      ret i32 %v
    }
    """)

# Generic (pre-ISel) MIR parsed directly: G_CONSTANT carries a CImm operand and
# G_FCONSTANT an FPImm operand -- the operand kinds selected MIR doesn't have.
_GENERIC_MIR = dedent("""\
    --- |
      define i32 @c() { ret i32 0 }
      define float @fc() { ret float 0.0 }
    ...
    ---
    name: c
    body: |
      bb.0:
        %0:_(s32) = G_CONSTANT i32 42
        $w0 = COPY %0(s32)
        RET_ReallyLR implicit $w0
    ...
    ---
    name: fc
    body: |
      bb.0:
        %0:_(s32) = G_FCONSTANT float 1.000000e+00
        RET_ReallyLR
    ...
    """)

# An external-symbol (isSymbol) machine operand, via an ADRP of a runtime symbol.
_SYMBOL_MIR = dedent("""\
    --- |
      define void @s() { ret void }
    ...
    ---
    name: s
    body: |
      bb.0:
        $x0 = ADRP target-flags(aarch64-page) &__stack_chk_guard
        RET_ReallyLR
    ...
    """)


def test_run_codegen_to_mir_yields_named_machine_function():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.run_codegen_to_mir(mod, tm)
        assert mmi.machine_function("add").name == "add"
    assert_no_leaks()


def _add_machine_module(ctx):
    """Lower @add to MIR; returns the MachineModuleInfo that owns it."""
    mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
    tm = jit.TargetMachine(triple=_TRIPLE)
    return mir.run_codegen_to_mir(mod, tm)


def test_machine_function_prints_mir_text():
    with ir.Context() as ctx:
        text = str(_add_machine_module(ctx).machine_function("add"))
        assert "ADDWrr" in text
        assert "RET_ReallyLR" in text
    assert_no_leaks()


def test_machine_function_has_single_entry_block():
    with ir.Context() as ctx:
        assert len(_add_machine_module(ctx).machine_function("add").blocks) == 1
    assert_no_leaks()


def test_instruction_opcode_names():
    with ir.Context() as ctx:
        mf = _add_machine_module(ctx).machine_function("add")
        block = mf.blocks[0]
        assert [i.opcode_name for i in block.instructions] == [
            "COPY",
            "COPY",
            "ADDWrr",
            "COPY",
            "RET_ReallyLR",
        ]
        # finalize-isel MIR is well-formed (backs the README example's
        # `print(mf.verify())  # True`).
        assert mf.verify() is True
    assert_no_leaks()


def test_addwrr_operands_are_one_def_two_uses():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        assert add.num_operands == 3
        assert add.operand(0).is_reg and add.operand(0).is_def
        assert add.operand(1).is_use
        assert add.operand(2).is_use
        assert add.operand(0).reg.is_virtual
        assert not add.operand(0).reg.is_physical
    assert_no_leaks()


def test_object_model_accessors():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        assert block.number == 0
        assert isinstance(block.name, str)
        assert "ADDWrr" in str(block)

        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        assert isinstance(add.opcode, int) and add.opcode > 0
        assert "ADDWrr" in str(add)

        # The last COPY is `$w0 = COPY %2`: its def is a physical register.
        last_copy = [i for i in block.instructions if i.opcode_name == "COPY"][-1]
        phys = last_copy.operand(0)
        assert phys.reg.is_physical
        assert not phys.reg.is_virtual
        assert phys.reg.id > 0
        assert "w0" in str(phys)
    assert_no_leaks()


def test_machine_function_missing_name_raises():
    with ir.Context() as ctx:
        mmi = _add_machine_module(ctx)
        with pytest.raises(KeyError):
            mmi.machine_function("nope")
    assert_no_leaks()


def test_declared_function_has_no_machine_function():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_DECL_SRC, ctx, "m")
        mmi = mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE))
        with pytest.raises(KeyError):
            mmi.machine_function("ext")  # a declaration has no MachineFunction
    assert_no_leaks()


def test_operand_index_out_of_range_raises():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        with pytest.raises(IndexError):
            add.operand(99)
    assert_no_leaks()


def test_immediate_operand_and_kind_guards():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_CONST_SRC, ctx, "m")
        mmi = mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE))
        block = mmi.machine_function("c").blocks[0]
        mov = next(i for i in block.instructions if i.opcode_name == "MOVi32imm")
        assert mov.operand(1).is_imm
        assert mov.operand(1).imm == 42
        # is_def/is_use are register-only; a non-register operand reports False
        # rather than tripping LLVM's isReg() assert.
        assert not mov.operand(1).is_def
        assert not mov.operand(1).is_use
        with pytest.raises(ValueError):
            mov.operand(1).reg  # an immediate operand has no register
        with pytest.raises(ValueError):
            mov.operand(0).imm  # a register operand has no immediate
    assert_no_leaks()


def test_run_codegen_to_mir_consumes_module():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE))
        assert mod._is_consumed
        with pytest.raises(RuntimeError):
            mod.functions  # the module moved into the wrapper; it can't be used
    assert_no_leaks()


def test_machine_function_outlives_dropped_mmi_handle():
    with ir.Context() as ctx:
        mmi = _add_machine_module(ctx)
        mf = mmi.machine_function("add")
        block = mf.blocks[0]
        del mmi
        gc.collect()
        # reference_internal pins the owning MachineModuleInfo through mf/block,
        # so the machine function stays valid after the Python handle is dropped.
        assert mf.name == "add"
        assert any(i.opcode_name == "ADDWrr" for i in block.instructions)
    assert_no_leaks()


def test_multi_block_function_numbers_its_blocks():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_LOOP_SRC, ctx, "m")
        mmi = mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE))
        blocks = mmi.machine_function("count").blocks
        numbers = [b.number for b in blocks]
        assert len(blocks) >= 2
        assert numbers == sorted(numbers)
        assert max(numbers) >= 1  # not every block is number 0
    assert_no_leaks()


def test_to_mir_emits_every_defined_function():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_TWO_FN_SRC, ctx, "m")
        mmi = mir.run_codegen_to_mir(mod, jit.TargetMachine(triple=_TRIPLE))
        assert mmi.machine_function("add").name == "add"
        assert mmi.machine_function("sub").name == "sub"
        text = mmi.to_mir()
        assert "name:            add" in text
        assert "name:            sub" in text
    assert_no_leaks()


def test_register_accessors():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        vreg = add.operand(0).reg  # a def: virtual
        assert vreg.is_valid
        assert vreg.is_virtual
        assert vreg.virt_reg_index >= 0
        # Equality + hashing (by register id).
        assert vreg == add.operand(0).reg
        assert vreg != add.operand(1).reg
        assert hash(vreg) == vreg.id
        assert (vreg == "not a register") is False
        # virt_reg_index is virtual-only.
        phys = (
            next(i for i in block.instructions if i.opcode_name == "RET_ReallyLR")
            .operand(0)
            .reg
        )
        assert phys.is_physical
        with pytest.raises(ValueError):
            phys.virt_reg_index
    assert_no_leaks()


def test_operand_kind_predicates_are_all_false_on_a_register():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        reg = next(i for i in block.instructions if i.opcode_name == "ADDWrr").operand(
            0
        )
        assert reg.is_reg
        # Every non-register kind predicate is False for a register operand
        # (this exercises each predicate's binding line).
        for pred in (
            "is_cimm",
            "is_fpimm",
            "is_mbb",
            "is_fi",
            "is_cpi",
            "is_jti",
            "is_target_index",
            "is_global",
            "is_symbol",
            "is_block_address",
            "is_reg_mask",
            "is_metadata",
            "is_predicate",
        ):
            assert getattr(reg, pred) is False
        # Register-flag reads + target_flags.
        assert reg.is_debug is False
        assert reg.is_internal_read is False
        assert isinstance(reg.is_tied, bool)
        assert reg.target_flags == 0
    assert_no_leaks()


def test_operand_getters_raise_on_wrong_kind():
    with ir.Context() as ctx:
        block = _add_machine_module(ctx).machine_function("add").blocks[0]
        reg = next(i for i in block.instructions if i.opcode_name == "ADDWrr").operand(
            0
        )
        for attr in (
            "cimm",
            "fpimm",
            "mbb",
            "index",
            "global_value",
            "symbol_name",
            "offset",
        ):
            with pytest.raises(ValueError):
                getattr(reg, attr)
    assert_no_leaks()


def test_branch_operands_expose_target_blocks():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_LOOP_SRC, ctx, "m")
        mf = mir.run_codegen_to_mir(
            mod, jit.TargetMachine(triple=_TRIPLE)
        ).machine_function("count")
        mbb_ops = [
            i.operand(k)
            for b in mf.blocks
            for i in b.instructions
            for k in range(i.num_operands)
            if i.operand(k).is_mbb
        ]
        assert mbb_ops  # the loop has conditional/unconditional branches
        assert all(o.mbb.number >= 0 for o in mbb_ops)
    assert_no_leaks()


def test_constant_operands_read_their_values():
    with ir.Context() as ctx:
        mmi = mir.parse_mir(_GENERIC_MIR, ctx, jit.TargetMachine(triple=_TRIPLE))
        cimm = next(
            i.operand(k)
            for b in mmi.machine_function("c").blocks
            for i in b.instructions
            for k in range(i.num_operands)
            if i.operand(k).is_cimm
        )
        assert cimm.cimm.zext_value == 42  # the G_CONSTANT's ConstantInt
        fpimm = next(
            i.operand(k)
            for b in mmi.machine_function("fc").blocks
            for i in b.instructions
            for k in range(i.num_operands)
            if i.operand(k).is_fpimm
        )
        assert type(fpimm.fpimm).__name__ == "ConstantFP"
    assert_no_leaks()


def test_global_operand_reads_the_global_value():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_GLOBAL_SRC, ctx, "m")
        mf = mir.run_codegen_to_mir(
            mod, jit.TargetMachine(triple=_TRIPLE)
        ).machine_function("f")
        gop = next(
            i.operand(k)
            for b in mf.blocks
            for i in b.instructions
            for k in range(i.num_operands)
            if i.operand(k).is_global
        )
        assert gop.global_value.name == "g"
        assert gop.offset == 0
        assert isinstance(gop.target_flags, int)
    assert_no_leaks()


def test_frame_index_operand_has_an_index():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ALLOCA_SRC, ctx, "m")
        mf = mir.run_codegen_to_mir(
            mod, jit.TargetMachine(triple=_TRIPLE)
        ).machine_function("a")
        fop = next(
            i.operand(k)
            for b in mf.blocks
            for i in b.instructions
            for k in range(i.num_operands)
            if i.operand(k).is_fi
        )
        assert isinstance(fop.index, int)
    assert_no_leaks()


def test_external_symbol_operand_reads_its_name():
    with ir.Context() as ctx:
        mmi = mir.parse_mir(_SYMBOL_MIR, ctx, jit.TargetMachine(triple=_TRIPLE))
        sop = next(
            i.operand(k)
            for b in mmi.machine_function("s").blocks
            for i in b.instructions
            for k in range(i.num_operands)
            if i.operand(k).is_symbol
        )
        assert sop.symbol_name == "__stack_chk_guard"
        assert sop.offset == 0
    assert_no_leaks()


def test_machine_instr_navigation_and_classification():
    with ir.Context() as ctx:
        mf = _add_machine_module(ctx).machine_function("add")
        block = mf.blocks[0]
        add = next(i for i in block.instructions if i.opcode_name == "ADDWrr")
        # Navigation.
        assert add.parent.number == block.number
        assert add.num_defs == 1
        assert add.num_explicit_operands == 3
        # Classification on known instructions.
        copy = next(i for i in block.instructions if i.opcode_name == "COPY")
        ret = next(i for i in block.instructions if i.opcode_name == "RET_ReallyLR")
        assert copy.is_copy
        assert ret.is_return and ret.is_terminator
        assert not add.is_terminator
        # Exercise every remaining predicate line (values vary by instr).
        for pred in (
            "is_branch",
            "is_conditional_branch",
            "is_unconditional_branch",
            "is_indirect_branch",
            "is_barrier",
            "is_call",
            "is_phi",
            "is_implicit_def",
            "may_load",
            "may_store",
            "is_debug_instr",
        ):
            assert isinstance(getattr(add, pred), bool)
    assert_no_leaks()


def test_branch_instruction_classification():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_LOOP_SRC, ctx, "m")
        mf = mir.run_codegen_to_mir(
            mod, jit.TargetMachine(triple=_TRIPLE)
        ).machine_function("count")
        instrs = [i for b in mf.blocks for i in b.instructions]
        assert any(i.is_branch for i in instrs)
        assert any(i.is_conditional_branch for i in instrs)
        assert any(i.is_phi for i in instrs)
    assert_no_leaks()


def test_memory_instruction_classification():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ALLOCA_SRC, ctx, "m")
        mf = mir.run_codegen_to_mir(
            mod, jit.TargetMachine(triple=_TRIPLE)
        ).machine_function("a")
        instrs = [i for b in mf.blocks for i in b.instructions]
        assert any(i.may_load for i in instrs)
        assert any(i.may_store for i in instrs)
    assert_no_leaks()


def test_machine_basic_block_cfg_accessors():
    with ir.Context() as ctx:
        mf = mir.run_codegen_to_mir(
            ir.parse_assembly(_LOOP_SRC, ctx, "m"),
            jit.TargetMachine(triple=_TRIPLE),
        ).machine_function("count")
        entry = mf.blocks[0]
        assert entry.is_entry_block
        assert entry.parent.name == "count"
        # Some block has a successor, and some block has a predecessor.
        assert any(b.successors for b in mf.blocks)
        assert any(b.predecessors for b in mf.blocks)
        # Successor/predecessor edges are consistent.
        for b in mf.blocks:
            for s in b.successors:
                assert b in s.predecessors
    assert_no_leaks()


def test_machine_function_accessors():
    with ir.Context() as ctx:
        mf = _add_machine_module(ctx).machine_function("add")
        assert mf.num_blocks == 1
        assert mf.function.name == "add"  # the IR Function it was lowered from
    assert_no_leaks()


def test_machine_module_info_lists_machine_functions():
    with ir.Context() as ctx:
        mmi = mir.run_codegen_to_mir(
            ir.parse_assembly(_TWO_FN_SRC, ctx, "m"),
            jit.TargetMachine(triple=_TRIPLE),
        )
        names = sorted(mf.name for mf in mmi.machine_functions)
        assert names == ["add", "sub"]
    assert_no_leaks()
