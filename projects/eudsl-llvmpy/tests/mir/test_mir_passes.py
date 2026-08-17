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

        assert f.machine_function.verify() is False
    assert_no_leaks()
