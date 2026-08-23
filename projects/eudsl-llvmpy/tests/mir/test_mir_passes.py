#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Running MIR passes: the GlobalISel lowering pipeline and the verifier.

run_codegen_to_mir(global_isel=True) runs the GlobalISel pipeline (IRTranslator
-> Legalizer -> RegBankSelect -> InstructionSelect) instead of SelectionDAG ISel,
so the retained MIR is fully selected target MIR. MachineFunction.verify runs the
machine verifier on any MIR, returning whether it is well-formed.

(Running individual GlobalISel passes on DSL-built generic MIR is not exposed:
the pass factories and an MMI-injecting pass ctor are not in the installed LLVM
headers, so lowering hand-built MIR would need those. Lowering from IR via the
pipeline above is the supported route.)
"""

from textwrap import dedent

import pytest

import llvm
from llvm import ir, jit, mir
from llvm.dsl import machine_function
from llvm.testing import assert_no_leaks

# Target-specific (AArch64 GISel selection to ADDWrr, etc.); needs the backend.
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

_TRIPLE = "aarch64-unknown-linux-gnu"


def test_global_isel_skips_declared_functions():
    with ir.Context() as ctx:
        # A declaration has no body -> no MachineFunction; the post-run
        # residual-generic scan must skip it and still select the definition.
        src = dedent("""\
            declare i32 @ext()
            define i32 @add(i32 %a, i32 %b) {
              %s = add i32 %a, %b
              ret i32 %s
            }
            """)
        mod = ir.parse_assembly(src, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.run_codegen_to_mir(mod, tm, global_isel=True)
        opcodes = [
            i.opcode_name for i in mmi.machine_function("add").blocks[0].instructions
        ]
        assert "ADDWrr" in opcodes
    assert_no_leaks()


def test_global_isel_selects_target_instructions():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.run_codegen_to_mir(mod, tm, global_isel=True)
        opcodes = [
            i.opcode_name for i in mmi.machine_function("add").blocks[0].instructions
        ]
        assert "ADDWrr" in opcodes
        assert "RET_ReallyLR" in opcodes
    assert_no_leaks()


def test_globally_selected_mir_verifies():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.run_codegen_to_mir(mod, tm, global_isel=True)
        assert mmi.machine_function("add").verify() is True
    assert_no_leaks()


def test_selection_dag_selected_mir_verifies():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.run_codegen_to_mir(mod, tm)  # default SelectionDAG ISel
        assert mmi.machine_function("add").verify() is True
    assert_no_leaks()


def test_incomplete_generic_function_does_not_verify():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_TRIPLE)
        s32 = mir.LLT.scalar(32)

        # Uses of the parameter vregs that are never defined -> not well-formed.
        @machine_function(module=mod, target=tm)
        def f(a: s32, b: s32):
            return a + b

        mf = f.machine_function
        assert mf.verify() is False
        # verify_diagnostic surfaces *why* (the detail verify() throws away).
        diag = mf.verify_diagnostic()
        assert diag != ""
        assert "Bad machine code" in diag or "not defined" in diag.lower()
    assert_no_leaks()


def test_verify_diagnostic_empty_when_well_formed():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.run_codegen_to_mir(mod, tm)
        mf = mmi.machine_function("add")
        assert mf.verify() is True
        assert mf.verify_diagnostic() == ""
    assert_no_leaks()


def test_global_isel_flag_is_not_sticky_on_reused_target_machine():
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        # global_isel is a pure function of the argument on each call, so a
        # global_isel=True run does not leave the shared TargetMachine in
        # GlobalISel mode for a subsequent default (SelectionDAG) run.
        mir.run_codegen_to_mir(
            ir.parse_assembly(_ADD_SRC, ctx, "m1"), tm, global_isel=True
        )
        assert tm.global_isel is True
        mir.run_codegen_to_mir(
            ir.parse_assembly(_ADD_SRC, ctx, "m2"), tm, global_isel=False
        )
        assert tm.global_isel is False
        mir.run_codegen_to_mir(ir.parse_assembly(_ADD_SRC, ctx, "m3"), tm)  # default
        assert tm.global_isel is False
    assert_no_leaks()


def test_global_isel_verified_mir_survives_mir_roundtrip():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_SRC, ctx, "m")
        tm = jit.TargetMachine(triple=_TRIPLE)
        text = mir.run_codegen_to_mir(mod, tm, global_isel=True).to_mir()
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_TRIPLE)
        mmi = mir.parse_mir(text, ctx, tm)
        assert mmi.machine_function("add").verify() is True
    assert_no_leaks()
