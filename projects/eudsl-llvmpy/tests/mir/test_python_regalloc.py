#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Drive the eudsl RegAllocBase allocator through codegen.

emit_object(regalloc="eudsl-python") makes RegisterRegAlloc's default point at
the extension's PyRegAlloc pass (a RegAllocBase-derived MachineFunctionPass that
assigns the first non-interfering physreg per virtual register), so the back
half of codegen runs it instead of the target's default greedy/fast allocator
while emitting the object. Register allocation is semantics-preserving, so the
emitted code alone cannot tell "the eudsl allocator ran" from a no-op; the
pass exposes a diagnostic selectOrSplit counter, and the tests below assert it
is non-zero exactly when regalloc="eudsl-python" is selected and zero otherwise
(the fail-if-no-op witness that setDefault actually takes effect). A
JIT-executed test additionally proves the allocated code stays correct.

emit_object(select=<callable>) instead routes the allocator's selectOrSplit
through a user Python callable: the callable receives the vreg's LiveInterval
and the legal (non-interfering) candidate physregs for that virtual register as
(live_interval, list[int] of physreg ids) and returns either an id from that set
(assign it) or None (spill). A callable that raises, or returns a value that is
neither None nor one of the presented candidate ids, has its Python exception
propagated out of emit_object (stashed and re-raised after codegen winds down),
rather than silently falling back. The callable appending to a list closed over
by the test is the witness that Python (not the target default) actually drove
selectOrSplit. select and regalloc name mutually exclusive ways of choosing the
allocator; select is independent of the scheduler pick/scheduler.
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


def test_eudsl_regalloc_runs_only_when_selected():
    """The eudsl allocator's selectOrSplit counter proves it drives register
    allocation when regalloc="eudsl-python", and is untouched otherwise --
    allocation preserves semantics, so this counter is the only witness that the
    extension's allocator (rather than the target default) actually ran, i.e.
    that RegisterRegAlloc::setDefault took effect when the pipeline was built."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        mmi.emit_object(regalloc="eudsl-python")
        assert _mir_ext._regalloc_select_count() > 0
    assert_no_leaks()

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        # The target default allocator runs (the pipeline allocates registers),
        # but it is not our pass, so the eudsl counter must stay at zero.
        mmi.emit_object()
        assert _mir_ext._regalloc_select_count() == 0
    assert_no_leaks()


def test_emit_object_with_eudsl_regalloc_produces_object():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(regalloc="eudsl-python")
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
# eudsl allocator must take its spill branch. 40 comfortably exceeds it.
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


def test_eudsl_regalloc_spills_under_high_pressure():
    """A high-register-pressure function forces the eudsl allocator down its
    spill branch (more GPR32 vregs live across a CFG edge than allocatable regs).
    The spill counter proves the authored spill path -- not just the driver --
    ran, and the allocation still produces a well-formed ELF object."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "hot")
        _build_high_pressure(mmi)
        _mir_ext._reset_regalloc_spill_count()
        obj = mmi.emit_object(regalloc="eudsl-python")
        assert _mir_ext._regalloc_spill_count() > 0  # the spill path ran
        assert obj[:4] == b"\x7fELF"
        assert b"hot\x00" in obj
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_spilled_high_pressure():
    """Execute the spilled allocation to prove the spill/reload code the eudsl
    allocator inserted is correct."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "hot")
        _build_high_pressure(mmi)
        _mir_ext._reset_regalloc_spill_count()
        obj = mmi.emit_object(regalloc="eudsl-python")
        assert _mir_ext._regalloc_spill_count() > 0

        j = jit.LLJIT()
        j.add_object(obj)
        hot = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(j.lookup("hot"))
        assert hot(3) == _high_pressure_expected(3)
        assert hot(5) == _high_pressure_expected(5)
        del j
    assert_no_leaks()


def test_regalloc_and_scheduler_are_independent():
    """regalloc is orthogonal to scheduler/pick: a caller may drive the pre-RA
    scheduler with a Python pick callback and the allocator with the eudsl
    allocator in one emission. The scheduler pick callback runs (its recording
    list is non-empty) and the allocator's select counter fires, proving a
    scheduler-side and an allocator-side option drive one emit independently; the
    object is a well-formed ELF."""
    picks = []

    def pick_cb(ready):
        picks.append(len(ready))
        return ready[0]

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(pick=pick_cb, regalloc="eudsl-python")
        assert picks  # the scheduler pick callback ran
        assert _mir_ext._regalloc_select_count() > 0
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


def test_python_select_callback_invoked_and_emits_object():
    """emit_object(select=cb) routes selectOrSplit through the callable: it
    receives the vreg's LiveInterval and the legal candidate physreg ids as a
    list[int], and returns the one to assign. The callable records into `picks`,
    so a non-empty `picks` witnesses that Python drove the allocator
    (semantics-preserving allocation leaves no other trace); the emitted object
    is a well-formed ELF."""
    picks = []

    def cb(live_interval, candidates):
        picks.append(list(candidates))
        return candidates[0]  # mimic the native first-free policy

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(select=cb)
        assert picks  # the callable really ran
        assert all(cands for cands in picks)  # legal candidates were presented
        assert all(isinstance(r, int) for cands in picks for r in cands)
        assert _mir_ext._regalloc_select_count() > 0
        assert obj[:4] == b"\x7fELF"
        assert b"add\x00" in obj
    assert_no_leaks()


def test_python_select_callback_receives_live_interval():
    """The LiveInterval marshalled to the callback exposes read-only accessors:
    the vreg id it covers (a virtual-register id, so nonzero once the
    virtual-register flag bit is set), its spill weight (a finite, non-negative
    float), and whether it is spillable (a bool)."""
    seen = []

    def cb(live_interval, candidates):
        seen.append(
            (live_interval.reg, live_interval.weight, live_interval.is_spillable)
        )
        return candidates[0]

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object(select=cb)
        assert seen  # the callable ran and read the interval
        for reg, weight, is_spillable in seen:
            assert isinstance(reg, int) and reg > 0  # a real vreg id
            assert isinstance(weight, float) and weight >= 0.0
            assert isinstance(is_spillable, bool)
    assert_no_leaks()


def test_python_select_callback_spills_via_none():
    """A callable that returns None when no candidate is free signals a spill,
    running the allocator's native spill path; when a candidate is free it
    returns the first, mimicking the native first-free policy. Under high
    register pressure this drives both the assign and the spill branch, and the
    object is a well-formed ELF."""
    selects = []

    def cb(live_interval, candidates):
        selects.append(len(candidates))
        return candidates[0] if candidates else None

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "hot")
        _build_high_pressure(mmi)
        _mir_ext._reset_regalloc_spill_count()
        obj = mmi.emit_object(select=cb)
        assert selects  # the callable ran
        assert 0 in selects  # at least once no candidate was free
        assert _mir_ext._regalloc_spill_count() > 0  # the None spill path ran
        assert obj[:4] == b"\x7fELF"
        assert b"hot\x00" in obj
    assert_no_leaks()


def test_python_select_callback_illegal_return_raises():
    """A callable that returns something that is neither None nor one of the
    presented candidate physreg ids is a misbehaving callback: selectOrSplit
    stashes a ValueError and re-raises it out of emit_object once the pipeline
    winds down, rather than silently falling back. The interpreter survives (a
    Python error, not a crash), and no Context/Module/callable leaks on the
    stash path."""
    picks = []

    def cb(live_interval, candidates):
        picks.append(list(candidates))
        return 999999  # not one of the presented candidate physreg ids

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="not one of the legal candidates"):
            mmi.emit_object(select=cb)
        assert picks  # the callable ran before its return was rejected
    assert_no_leaks()


def test_python_select_callback_raise_propagates():
    """A callable that raises surfaces as a Python exception out of emit_object
    (the exception is stashed and re-raised after codegen winds down, never
    thrown across LLVM's -fno-exceptions frames), and does not crash the
    interpreter. No Context/Module/callable leaks on the stash path."""
    picks = []

    def cb(live_interval, candidates):
        picks.append(len(candidates))
        raise ValueError("boom")

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="boom"):
            mmi.emit_object(select=cb)
        assert picks  # the callable ran and raised
    assert_no_leaks()


def test_select_and_pick_drive_one_emit():
    """select (Python allocator) is independent of pick (Python scheduler): a
    caller may drive both in one emission. Both callables run (each records into
    its own list) and the allocator counter fires; the object is a well-formed
    ELF."""
    picks = []
    selects = []

    def pick_cb(ready):
        picks.append(len(ready))
        return ready[0]

    def select_cb(live_interval, candidates):
        selects.append(list(candidates))
        return candidates[0]

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(pick=pick_cb, select=select_cb)
        assert picks and selects  # both Python callbacks ran
        assert _mir_ext._regalloc_select_count() > 0
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


def test_regalloc_and_select_are_mutually_exclusive():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="regalloc"):
            mmi.emit_object(regalloc="eudsl-python", select=lambda li, c: c[0])
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_python_selected_add():
    picks = []

    def cb(live_interval, candidates):
        picks.append(len(candidates))
        return candidates[0]

    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(select=cb)
        assert picks  # the python callback drove selectOrSplit

        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        del j
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_eudsl_regalloc_add():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(regalloc="eudsl-python")
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
