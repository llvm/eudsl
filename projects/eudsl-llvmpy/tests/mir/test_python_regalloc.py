#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Drive the native trivial RegAllocBase allocator through codegen.

emit_object(regalloc="eudsl-trivial") makes RegisterRegAlloc's default point at
the extension's PyRegAlloc pass (a RegAllocBase-derived MachineFunctionPass that
assigns the first non-interfering physreg per virtual register), so the back
half of codegen runs it instead of the target's default greedy/fast allocator
while emitting the object. Register allocation is semantics-preserving, so the
emitted code alone cannot tell "the trivial allocator ran" from a no-op; the
pass exposes a diagnostic selectOrSplit counter, and the tests below assert it
is non-zero exactly when regalloc="eudsl-trivial" is selected and zero otherwise
(the fail-if-no-op witness that setDefault actually takes effect). A
JIT-executed test additionally proves the allocated code stays correct.
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


def test_trivial_regalloc_runs_only_when_selected():
    """The trivial allocator's selectOrSplit counter proves it drives register
    allocation when regalloc="eudsl-trivial", and is untouched otherwise --
    allocation preserves semantics, so this counter is the only witness that the
    extension's allocator (rather than the target default) actually ran, i.e.
    that RegisterRegAlloc::setDefault took effect when the pipeline was built."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        mmi.emit_object(regalloc="eudsl-trivial")
        assert _mir_ext._regalloc_select_count() > 0
    assert_no_leaks()

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        # The target default allocator runs (the pipeline allocates registers),
        # but it is not our pass, so the trivial counter must stay at zero.
        mmi.emit_object()
        assert _mir_ext._regalloc_select_count() == 0
    assert_no_leaks()


def test_emit_object_with_trivial_regalloc_produces_object():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(regalloc="eudsl-trivial")
        assert obj[:4] == b"\x7fELF"
        assert b"add\x00" in obj
    assert_no_leaks()


def test_unknown_regalloc_name_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(RuntimeError, match="register allocator"):
            mmi.emit_object(regalloc="does-not-exist")
    assert_no_leaks()


# More GPR32 vregs held live at once than AArch64 has allocatable (~29), so the
# trivial allocator must take its spill branch. 40 comfortably exceeds it.
_HIGH_PRESSURE_N = 40


def _build_high_pressure(mmi):
    """Hand-build a selected AArch64 hot(i32 w0)->i32 that forces spilling.

    Define N distinct vregs in the entry block (a chain v0 = base+base,
    vi = v(i-1)+base, so vi = (i+2)*base -- distinct values, which MachineCSE
    cannot collapse), then branch to a second block that sums them all. Every vi
    is defined in the entry block and used in the successor, so all N are live
    across the CFG edge; block-local scheduling cannot shorten that cross-block
    liveness, so peak pressure is ~N simultaneously-live GPR32 vregs and the
    allocator must spill. The result is sum_{i=0..N-1} (i+2)*w0."""
    mf = mmi.machine_function("hot")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    tail = mf.create_block()
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    entry.add_livein(w0)
    copy = mf.opcode("COPY")
    addwrr = mf.opcode("ADDWrr")
    b.set_block(entry)
    base = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(base, is_def=True)
    c.add_reg(w0)
    vs = []
    prev = base
    for _ in range(_HIGH_PRESSURE_N):
        v = mf.create_vreg(gpr32)
        a = b.build_instr(addwrr)
        a.add_reg(v, is_def=True)
        a.add_reg(prev)
        a.add_reg(base)
        vs.append(v)
        prev = v
    br = b.build_instr(mf.opcode("B"))
    br.add_mbb(tail)
    entry.add_successor(tail)
    b.set_block(tail)
    acc = vs[0]
    for v in vs[1:]:
        nv = mf.create_vreg(gpr32)
        a = b.build_instr(addwrr)
        a.add_reg(nv, is_def=True)
        a.add_reg(acc)
        a.add_reg(v)
        acc = nv
    ret_copy = b.build_instr(copy)
    ret_copy.add_reg(w0, is_def=True)
    ret_copy.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def _high_pressure_expected(x):
    """hot(x) = sum_{i=0..N-1} (i+2)*x."""
    return x * sum(i + 2 for i in range(_HIGH_PRESSURE_N))


def test_trivial_regalloc_spills_under_high_pressure():
    """A high-register-pressure function forces the trivial allocator down its
    spill branch (more GPR32 vregs live across a CFG edge than allocatable regs).
    The spill counter proves the authored spill path -- not just the driver --
    ran, and the allocation still produces a well-formed ELF object."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "hot")
        _build_high_pressure(mmi)
        _mir_ext._reset_regalloc_spill_count()
        obj = mmi.emit_object(regalloc="eudsl-trivial")
        assert _mir_ext._regalloc_spill_count() > 0  # the spill path ran
        assert obj[:4] == b"\x7fELF"
        assert b"hot\x00" in obj
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_spilled_high_pressure():
    """Execute the spilled allocation to prove the spill/reload code the trivial
    allocator inserted is correct."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "hot")
        _build_high_pressure(mmi)
        _mir_ext._reset_regalloc_spill_count()
        obj = mmi.emit_object(regalloc="eudsl-trivial")
        assert _mir_ext._regalloc_spill_count() > 0

        j = jit.LLJIT()
        j.add_object(obj)
        hot = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(j.lookup("hot"))
        assert hot(3) == _high_pressure_expected(3)
        assert hot(5) == _high_pressure_expected(5)
        del j
    assert_no_leaks()


def test_regalloc_and_scheduler_are_independent():
    """regalloc is orthogonal to scheduler/pick: a caller may drive both the
    pre-RA scheduler and the allocator with the extension's strategies in one
    emission. Both counters fire, and the object is a well-formed ELF."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_trivial_scheduler_pick_count()
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(scheduler="trivial", regalloc="eudsl-trivial")
        assert _mir_ext._trivial_scheduler_pick_count() > 0
        assert _mir_ext._regalloc_select_count() > 0
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_trivial_regalloc_add():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(regalloc="eudsl-trivial")
        assert _mir_ext._regalloc_select_count() > 0  # our allocator drove RA

        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        del j
    assert_no_leaks()
