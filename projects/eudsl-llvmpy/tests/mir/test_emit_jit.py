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


def _build_selected_add(mmi, declare_liveins=True):
    """Hand-build a fully-selected AArch64 add(i32,i32)->i32 MachineFunction:
    liveins $w0/$w1, two COPYs in, ADDWrr, a COPY to $w0, RET_ReallyLR. With
    declare_liveins=False the live-ins are omitted, so the MIR fails verify()."""
    mf = mmi.machine_function("add")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0, w1 = mf.physreg("W0"), mf.physreg("W1")
    if declare_liveins:
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
        # Host-independent guard that the object actually defines `add`: the
        # symbol name lives null-terminated in the ELF string table. (Executing
        # it is checked separately on AArch64 hosts.)
        assert b"add\x00" in obj
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


def test_mir_module_is_queryable_after_emit():
    """After emit_object the MirModule stays valid to query: to_mir() still
    prints the IR module. The emission pipeline appends FreeMachineFunctionPass,
    so the MachineFunctions themselves are gone -- machine_functions is empty."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object()
        assert "@add" in mmi.to_mir()
        assert list(mmi.machine_functions) == []
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


def test_emit_object_rejects_malformed_mir():
    """emit_object verifies first, so malformed hand-built MIR (here: physregs
    read without live-in declarations) raises instead of muddling through the
    emission pipeline to a garbage object or a fatal codegen abort."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi, declare_liveins=False)
        with pytest.raises(RuntimeError, match="failed verification"):
            mmi.emit_object()
    assert_no_leaks()


def test_add_object_rejects_non_object_bytes():
    """add_object parses eagerly, so bad bytes raise here rather than resurfacing
    as an opaque materialization error at a later lookup."""
    j = jit.LLJIT()
    with pytest.raises(RuntimeError):
        j.add_object(b"not an object file")
    del j
    assert_no_leaks()


def test_start_after_is_restored_after_emit():
    """emit_object mutates the process-global -start-after option and must
    restore it, so a subsequent run_codegen_to_mir still runs full instruction
    selection (rather than skipping to finalize-isel and producing empty MIR)."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object()
    assert_no_leaks()

    src = dedent("""\
        define i32 @f(i32 %a, i32 %b) {
          %s = add i32 %a, %b
          ret i32 %s
        }
        """)
    with ir.Context() as ctx:
        mod = ir.parse_assembly(src, ctx, "m")
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.run_codegen_to_mir(mod, tm)
        mf = mmi.machine_function("f")
        # Full ISel ran: the function has real selected instructions and
        # verifies. A leaked -start-after would skip ISel and leave it empty.
        assert mf.verify() is True
        assert list(mf.blocks[0].instructions)
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
