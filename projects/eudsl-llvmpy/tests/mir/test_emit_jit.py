#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Route B capstone: emit an object from hand-built selected MIR and JIT it.

emit_object runs the back half of codegen (register allocation, prologue/
epilogue, object emission) over already-selected MIR via -start-after=
finalize-isel, so no instruction selection runs. LLJIT.add_object then loads the
object so the DSL-built function can be called. The instructions are AArch64
(ADDWrr/COPY/RET_ReallyLR), so actually executing the result needs an AArch64
host; object emission itself is host-independent (cross ELF).
"""

import ctypes
import platform
from textwrap import dedent

import pytest

import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

# Object emission uses an AArch64 target (cross ELF), so needs the AArch64
# backend linked; JIT-executing additionally needs an AArch64 host (below).
pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)

_AARCH64_LINUX = "aarch64-unknown-linux-gnu"
_IS_AARCH64 = platform.machine() in ("arm64", "aarch64")


def _build_selected_add(mmi):
    """Hand-build a fully-selected AArch64 add(i32,i32)->i32 (see #589)."""
    mf = mmi.machine_function("add")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0, w1 = mf.physreg("W0"), mf.physreg("W1")
    entry.add_livein(w0)
    entry.add_livein(w1)
    v0, v1, v2 = (mf.create_vreg(gpr32) for _ in range(3))
    copy = mf.opcode("COPY")
    for dst, src in ((v0, w0), (v1, w1)):
        c = b.build_instr(copy)
        c.add_reg(dst, is_def=True)
        c.add_reg(src)
    add = b.build_instr(mf.opcode("ADDWrr"))
    add.add_reg(v2, is_def=True)
    add.add_reg(v0)
    add.add_reg(v1)
    ret_copy = b.build_instr(copy)
    ret_copy.add_reg(w0, is_def=True)
    ret_copy.add_reg(v2)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_emit_object_produces_a_relocatable_object():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object()
        assert len(obj) > 0
        assert obj[:4] == b"\x7fELF"  # a real ELF relocatable object
    assert_no_leaks()


def test_emit_object_twice_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object()
        with pytest.raises(RuntimeError, match="already emitted"):
            mmi.emit_object()
    assert_no_leaks()


def test_emit_object_requires_create_machine_function():
    src = dedent("""\
        define i32 @f(i32 %a) {
          ret i32 %a
        }
        """)
    with ir.Context() as ctx:
        mod = ir.parse_assembly(src, ctx, "m")
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.run_codegen_to_mir(mod, tm)  # no build-path MMIWrapperPass
        with pytest.raises(RuntimeError, match="create_machine_function"):
            mmi.emit_object()
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_hand_built_add():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object()

        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        assert add(-5, 5) == 0
        del j
    assert_no_leaks()
