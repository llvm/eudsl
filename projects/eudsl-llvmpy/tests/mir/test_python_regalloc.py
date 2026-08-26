#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Drive the eudsl RegAllocBase allocator through codegen.

The allocator is a fixed C++ MachineFunctionPass (PyRegAlloc, RegAllocBase-
derived) registered in the RegisterRegAlloc registry, mirroring how LLVM's own
allocators are structured: the pass *is* the allocator, chosen by name. Three
ways drive it:

- emit_object(regalloc="eudsl-python") runs its native first-free/spill policy.
- emit_object(select=<callable>) routes selectOrSplit through a one-shot Python
  callable: it receives (live_interval, list[int] of legal candidate physreg
  ids) and returns an id to assign or None to spill.
- register_regalloc(name, cls) + emit_object(regalloc=name) instantiates cls
  afresh per MachineFunction and drives selectOrSplit through its
  select_or_split method; the class may also define priority(li) to order the
  allocation queue.

Register allocation is semantics-preserving, so the emitted code alone cannot
witness that our pass (rather than the target default) ran; the pass exposes
selectOrSplit / spill counters for that, and JIT-executed tests prove the
allocated code stays correct. A callable/method that raises, or returns an
illegal value, has its Python exception stashed and re-raised out of emit_object
after codegen winds down, never thrown across LLVM's -fno-exceptions frames.
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


# ---------------------------------------------------------------------------
# Native "eudsl-python" allocator (regalloc="eudsl-python")
# ---------------------------------------------------------------------------


def test_eudsl_regalloc_runs_only_when_selected():
    """The selectOrSplit counter proves the eudsl allocator drives register
    allocation when regalloc="eudsl-python", and is untouched otherwise --
    allocation preserves semantics, so this counter is the only witness that
    RegisterRegAlloc::setDefault took effect when the pipeline was built."""
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
        # The target default allocator runs, but it is not our pass, so the
        # eudsl counter must stay at zero.
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
    """Hand-build a selected AArch64 hot(i32 w0)->i32 that forces spilling: N
    distinct vregs defined in the entry block and summed in a successor, so all
    N are live across the CFG edge -- peak pressure exceeds the allocatable set.
    Result is sum_{i=0..N-1} (i+2)*w0."""
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
    spill branch. The spill counter proves the authored spill path -- not just
    the driver -- ran, and the allocation still produces a well-formed ELF."""
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
    """Execute the spilled allocation to prove the spill/reload code is
    correct."""
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


# ---------------------------------------------------------------------------
# One-shot select= callable
# ---------------------------------------------------------------------------


def test_python_select_callback_invoked_and_emits_object():
    """emit_object(select=cb) routes selectOrSplit through the callable: it
    receives the vreg's LiveInterval and the legal candidate physreg ids as a
    list[int], and returns the one to assign. A non-empty `picks` witnesses that
    Python drove the allocator; the emitted object is a well-formed ELF."""
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
    the vreg id it covers (nonzero once the virtual-register flag bit is set),
    its spill weight (a finite, non-negative float), and whether it is
    spillable (a bool)."""
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
    returns the first. Under high register pressure this drives both branches,
    and the object is a well-formed ELF."""
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
    presented candidate physreg ids is misbehaving: selectOrSplit stashes a
    ValueError and re-raises it out of emit_object once the pipeline winds down.
    The interpreter survives, and nothing leaks on the stash path."""
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
    (stashed and re-raised after codegen winds down, never thrown across LLVM's
    -fno-exceptions frames), and does not crash. Nothing leaks."""
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


# ---------------------------------------------------------------------------
# register_regalloc(name, cls) -- named, class-based allocators
# ---------------------------------------------------------------------------


class _FirstFreeAlloc:
    """The native first-free policy, expressed as a Python allocator class."""

    def select_or_split(self, live_interval, candidates):
        return candidates[0] if candidates else None


def test_register_regalloc_lists_and_replaces():
    """register_regalloc records the name (listed by registered_regallocs); a
    class missing select_or_split is rejected, and re-registering a name
    replaces the class without adding a duplicate registry entry."""

    class Missing:
        pass

    with pytest.raises(TypeError, match="select_or_split"):
        mir.register_regalloc("ra-missing", Missing)

    mir.register_regalloc("ra-listed", _FirstFreeAlloc)
    assert "ra-listed" in mir.registered_regallocs()
    before = mir.registered_regallocs().count("ra-listed")

    class Other(_FirstFreeAlloc):
        pass

    mir.register_regalloc("ra-listed", Other)  # replaces, no duplicate entry
    assert mir.registered_regallocs().count("ra-listed") == before


def test_register_regalloc_drives_selectorsplit():
    """emit_object(regalloc=name) runs a fresh instance's select_or_split; the
    counter proves it drove allocation and the object is a well-formed ELF."""
    mir.register_regalloc("ra-firstfree", _FirstFreeAlloc)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(regalloc="ra-firstfree")
        assert _mir_ext._regalloc_select_count() > 0
        assert obj[:4] == b"\x7fELF"
        assert b"add\x00" in obj
    assert_no_leaks()


def test_register_regalloc_fresh_instance_per_emission():
    """A fresh allocator instance is constructed for each emission (per
    MachineFunction), so the two emits below record two distinct instances."""
    seen_ids = []

    class Recording:
        def __init__(self):
            seen_ids.append(id(self))

        def select_or_split(self, live_interval, candidates):
            return candidates[0] if candidates else None

    mir.register_regalloc("ra-fresh", Recording)
    for _ in range(2):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "add")
            _build_selected_add(mmi)
            mmi.emit_object(regalloc="ra-fresh")
        assert_no_leaks()
    assert len(seen_ids) == 2
    assert seen_ids[0] != seen_ids[1]  # distinct instances


def test_register_regalloc_priority_orders_queue():
    """A class defining priority(li) supplies the allocation-queue key (instead
    of the default spill weight); priority runs for every enqueued interval and
    the object still emits as a well-formed ELF."""
    priorities = []

    class ByReg:
        def priority(self, live_interval):
            priorities.append(live_interval.reg)
            return float(live_interval.reg)  # order by vreg id

        def select_or_split(self, live_interval, candidates):
            return candidates[0] if candidates else None

    mir.register_regalloc("ra-priority", ByReg)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(regalloc="ra-priority")
        assert priorities  # priority() was consulted for the queue order
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


def test_register_regalloc_priority_raise_propagates():
    """A priority that raises is stashed and re-raised out of emit_object; the
    allocation still completes legally (weight fallback) before the re-raise, so
    the interpreter survives and nothing leaks."""

    class BadPriority:
        def priority(self, live_interval):
            raise ValueError("prio-boom")

        def select_or_split(self, live_interval, candidates):
            return candidates[0] if candidates else None

    mir.register_regalloc("ra-prio-boom", BadPriority)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="prio-boom"):
            mmi.emit_object(regalloc="ra-prio-boom")
    assert_no_leaks()


def test_register_regalloc_select_raise_propagates():
    """A select_or_split that raises surfaces out of emit_object."""

    class BadSelect:
        def select_or_split(self, live_interval, candidates):
            raise ValueError("sel-boom")

    mir.register_regalloc("ra-sel-boom", BadSelect)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="sel-boom"):
            mmi.emit_object(regalloc="ra-sel-boom")
    assert_no_leaks()


def test_register_regalloc_illegal_return_raises():
    """A select_or_split returning a non-candidate id raises ValueError."""

    class Illegal:
        def select_or_split(self, live_interval, candidates):
            return 999999

    mir.register_regalloc("ra-illegal", Illegal)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="not one of the legal candidates"):
            mmi.emit_object(regalloc="ra-illegal")
    assert_no_leaks()


def test_register_regalloc_init_raise_propagates():
    """A raising __init__ (the per-function instance construction) is stashed
    and re-raised out of emit_object rather than crashing."""

    class InitBoom:
        def __init__(self):
            raise RuntimeError("ctor-boom")

        def select_or_split(self, live_interval, candidates):
            return candidates[0] if candidates else None

    mir.register_regalloc("ra-init-boom", InitBoom)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(RuntimeError, match="ctor-boom"):
            mmi.emit_object(regalloc="ra-init-boom")
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_register_regalloc_add():
    mir.register_regalloc("ra-jit", _FirstFreeAlloc)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(regalloc="ra-jit")

        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        del j
    assert_no_leaks()


# ---------------------------------------------------------------------------
# Allocator is orthogonal to the scheduler
# ---------------------------------------------------------------------------


class _RecordingSched(mir.ReadyQueueStrategy):
    picks = []

    def pick(self, ready):
        _RecordingSched.picks.append(len(ready))
        return ready[0]


def test_regalloc_and_scheduler_are_independent():
    """A scheduler-side option (scheduler=) and an allocator-side option
    (regalloc=/select=) drive one emission independently: the scheduler runs (its
    recording list is non-empty) and the allocator counter fires, and the object
    is a well-formed ELF."""
    _RecordingSched.picks = []
    mir.register_scheduler("sched-indep", _RecordingSched)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(scheduler="sched-indep", regalloc="eudsl-python")
        assert _RecordingSched.picks  # the scheduler ran
        assert _mir_ext._regalloc_select_count() > 0  # the allocator ran
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


def test_scheduler_and_select_drive_one_emit():
    """select= (Python allocator) is independent of scheduler= (Python
    scheduler): both run in one emission and the object is a well-formed ELF."""
    _RecordingSched.picks = []
    selects = []

    def select_cb(live_interval, candidates):
        selects.append(list(candidates))
        return candidates[0]

    mir.register_scheduler("sched-with-select", _RecordingSched)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        _mir_ext._reset_regalloc_select_count()
        obj = mmi.emit_object(scheduler="sched-with-select", select=select_cb)
        assert _RecordingSched.picks and selects  # both Python callbacks ran
        assert _mir_ext._regalloc_select_count() > 0
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()
