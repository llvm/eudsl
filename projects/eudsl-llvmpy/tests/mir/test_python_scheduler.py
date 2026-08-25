#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Drive the native trivial MachineScheduler strategy through codegen.

emit_object(scheduler="trivial") sets the process-global -misched option to the
strategy registered under that name, so the pre-RA MachineScheduler runs it
instead of the target's default while emitting the object. Scheduling is
semantics-preserving, so the emitted code alone cannot tell "trivial ran" from a
no-op; the strategy exposes a diagnostic pickNode counter, and the tests below
assert it is non-zero exactly when scheduler="trivial" is selected and zero
otherwise. A JIT-executed test additionally proves the result stays correct.
"""

import ctypes
import platform

import pytest

import llvm
from llvm import ir, jit, mir
from llvm.eudslllvm_ext import mir as _mir_ext
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
    """Hand-build a fully-selected AArch64 add(i32,i32)->i32 MachineFunction:
    liveins $w0/$w1, two COPYs in, ADDWrr, a COPY to $w0, RET_ReallyLR."""
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


def test_trivial_scheduler_is_registered():
    assert "trivial" in mir.registered_schedulers()
    assert_no_leaks()


def test_trivial_scheduler_runs_only_when_selected():
    """The trivial strategy's pickNode counter proves it drives the pre-RA
    MachineScheduler when scheduler="trivial", and is untouched otherwise --
    scheduling preserves semantics, so this counter is the only witness that the
    strategy (rather than the target default) actually ran."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_trivial_scheduler_pick_count()
        mmi.emit_object(scheduler="trivial")
        assert _mir_ext._trivial_scheduler_pick_count() > 0
    assert_no_leaks()

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_trivial_scheduler_pick_count()
        # The generic "converge" scheduler runs (the pipeline schedules), but it
        # is not our strategy, so the trivial counter must stay at zero.
        mmi.emit_object(scheduler="converge")
        assert _mir_ext._trivial_scheduler_pick_count() == 0
    assert_no_leaks()


def test_emit_object_with_trivial_scheduler_produces_object():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="trivial")
        assert obj[:4] == b"\x7fELF"
        assert b"add\x00" in obj
    assert_no_leaks()


def test_unknown_scheduler_name_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(RuntimeError, match="scheduler"):
            mmi.emit_object(scheduler="does-not-exist")
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_trivial_scheduled_add():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="trivial")

        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        del j
    assert_no_leaks()
