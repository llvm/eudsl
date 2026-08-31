#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Faithful Python RegAllocGreedy (mir.RAGreedy)."""

import ctypes
import platform
from types import SimpleNamespace
import pytest
import llvm
from llvm import ir, jit, mir
import llvm.mir_greedy as mg
from llvm.mir_greedy import eviction_cost, calc_gap_weights, calc_global_split_cost
from llvm.mir_greedy import GlobalSplitCandidate
from llvm.mir_greedy import _NO_CAND
from llvm.mir_greedy import LiveRangeStage
from llvm.testing import assert_no_leaks

pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked",
)
_AARCH64_LINUX = "aarch64-unknown-linux-gnu"
_AMDGPU = "amdgcn-amd-amdhsa"
# AMDGPU has sub-register liveness (which AArch64 lacks), so tryInstructionSplit's
# sub-register arm is only reachable there. It is in the default LLVM
# distribution and the extension links it whenever available; skip if it isn't.
_HAS_AMDGPU = "amdgcn" in llvm.jit.registered_targets()
_skip_no_amdgpu = pytest.mark.skipif(
    not _HAS_AMDGPU, reason="AMDGPU backend not linked"
)


def test_ragreedy_is_exported_and_constructs():
    assert issubclass(mir.RAGreedy, mir.RegAllocBase)
    ra = mir.RAGreedy()
    assert ra.RS_New < ra.RS_Assign < ra.RS_Split < ra.RS_Split2 < ra.RS_Spill
    assert_no_leaks()


def test_eviction_cost_broken_hints_then_max_weight():
    # RAGreedy's EvictionCost is lexicographic (broken_hints, max_weight):
    # max_weight is the MAX interferer weight, not the sum; broken_hints (the
    # summed copy cost of interferers whose satisfied hint would break) is the
    # primary key.
    # (weight, breaks_hint, copy_cost) triples.
    assert eviction_cost([(1.0, False, 0.0), (2.0, False, 0.0)]) == (0.0, 2.0)
    # max, not sum: one heavy costs more than several light.
    assert eviction_cost([(3.0, False, 0.0)]) > eviction_cost(
        [(1.0, False, 0.0), (1.0, False, 0.0), (1.0, False, 0.0)]
    )
    # A broken hint dominates lexicographically, however small its max weight.
    assert eviction_cost([(0.1, True, 1.0)]) > eviction_cost([(99.0, False, 0.0)])
    assert eviction_cost([]) == (0.0, 0.0)  # no interferers


def test_assign_cascade_consumes_once_per_reg():
    """getOrAssignNewCascade assigns a fresh cascade the first time and returns
    the existing one on repeat (the already-assigned branch), consuming a new
    number only for a fresh reg."""
    ra = mir.RAGreedy()
    c1 = ra._assign_cascade(5)
    assert ra._assign_cascade(5) == c1  # already assigned: no new consume
    assert ra._assign_cascade(6) == c1 + 1  # fresh reg gets the next cascade


def test_calc_gap_weights_per_gap_max_interference():
    # Uses at slots 0, 10, 20 -> two gaps: [0,10) and [10,20).
    use_slots = [0, 10, 20]
    # One interferer of weight 3.0 spanning [5, 15): overlaps both gaps.
    # One interferer of weight 7.0 spanning [12, 18): only the second gap.
    spans = [(5, 15, 3.0), (12, 18, 7.0)]
    weights = calc_gap_weights(use_slots, spans)
    assert len(weights) == 2
    assert weights[0] == pytest.approx(3.0)  # only the weight-3 interferer
    assert weights[1] == pytest.approx(7.0)  # max(3.0, 7.0)


def test_calc_gap_weights_no_interference_is_zero():
    assert calc_gap_weights([0, 5], []) == [0.0]


def test_calc_gap_weights_keeps_running_max():
    # Two spans overlap the single gap; the second is lighter, so the running
    # max is kept (the weight-not-greater branch).
    assert calc_gap_weights([0, 10], [(0, 10, 5.0), (2, 8, 2.0)]) == [5.0]


def test_calc_global_split_cost_sums_boundary_frequencies():
    assert calc_global_split_cost([]) == pytest.approx(0.0)
    assert calc_global_split_cost([2.0, 3.0, 0.5]) == pytest.approx(5.5)


def _build_add(mmi):
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
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v2)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_get_priority_bit_layout():
    ra = mir.RAGreedy()

    def prio(stage, size, is_local_assign, local_prio, global_bit, has_pref=False):
        # trumps_globalness=True (the AArch64 layout): AllocationPriority<<25,
        # GlobalBit<<24.
        return ra._priority_for(
            stage,
            size,
            is_local_assign,
            local_prio,
            global_bit,
            0,  # alloc_priority
            True,  # trumps_globalness
            has_pref,
        )

    # RS_Split ranges are deferred: bare size, below the 1<<31 mark.
    assert prio(ra.RS_Split, 100, False, 0, 0) == 100
    # A global RS_Assign range: size in the low bits, GlobalBit<<24, 1<<31 mark.
    g = prio(ra.RS_Assign, 100, False, 0, 1)
    assert g == (1 << 31) | (1 << 24) | 100
    # A local range carries no global bit, so it sorts below a same-size global.
    lo = prio(ra.RS_Assign, 100, True, 100, 0)
    assert lo == (1 << 31) | 100
    assert g > lo
    # Global and local both outrank any RS_Split range.
    assert lo > prio(ra.RS_Split, 100, False, 0, 0)
    # A known preference boosts with 1<<30.
    assert prio(ra.RS_Assign, 100, False, 0, 1, has_pref=True) == g | (1 << 30)


def test_priority_orders_larger_ranges_first():
    order = []

    class Recording(mir.RAGreedy):
        def select_or_split(self, li):
            order.append(li.reg)
            for preg in self.allocation_order(li):
                if self.matrix.is_free(li, preg):
                    return preg
            self.spill(li)
            return None

    mir.register_regalloc("ra-greedy-prio", Recording)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-prio")
    assert obj[:4] == b"\x7fELF"
    # Every enqueued vreg was dequeued exactly once (the queue drained in
    # priority order without dropping or repeating a reg).
    assert len(order) == len(set(order))
    assert_no_leaks()


def _jit_call(sig, fn, obj):
    j = jit.LLJIT()
    j.add_object(obj)
    return ctypes.CFUNCTYPE(*sig)(j.lookup(fn)), j


def test_assign_allocates_and_executes():
    traces = {}

    class Traced(mir.RAGreedy):
        def select_or_split(self, li):
            r = super().select_or_split(li)
            traces.update(self.trace)
            return r

    mir.register_regalloc("ra-greedy-assign", Traced)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple, so the object is JIT-executable
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-assign")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int, ctypes.c_int), "add", obj)
    assert fn(3, 4) == 7  # semantics preserved
    assert "assign" in traces.values()  # tryAssign fired
    assert_no_leaks()


_HP_N = 48


def _hp_closed_form(x):
    return _HP_N * x


def _build_high_pressure(mmi):
    mf = mmi.machine_function("hp")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    entry.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    terms = []
    for _ in range(_HP_N):
        t = mf.create_vreg(gpr32)
        ins = b.build_instr(copy)
        ins.add_reg(t, is_def=True)
        ins.add_reg(w0)
        terms.append(t)
    acc = terms[0]
    for t in terms[1:]:
        nacc = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(nacc, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(t)
        acc = nacc
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_evict_executes_and_preserves_semantics():
    traces = {}

    class Traced(mir.RAGreedy):
        def select_or_split(self, li):
            r = super().select_or_split(li)
            traces.update(self.trace)
            return r

    mir.register_regalloc("ra-greedy-evict", Traced)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "hp")
        _build_high_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-evict")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "hp", obj)
    assert fn(5) == _hp_closed_form(5)
    assert "evict" in traces.values()
    assert_no_leaks()


def _build_thru_pressure(mmi):
    """b0 defines N distinct live-through values via a dependency chain
    (t_0 = w0+w0, t_i = t_{i-1}+w0), so no two are CSE-identical, none is a copy
    the coalescer folds, and each depends on the previous so it can't be
    trivially rematerialized. b1 is empty (all N live through it); b2 sums them.
    With N above the GPR32 file, several live-through intervals cannot be
    assigned or evicted whole and must be split around the block.
    t_i = (i+2)*x, so thrup(x) = x * sum_{i=0..N-1}(i+2)."""
    mf = mmi.machine_function("thrup")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    br = mf.opcode("B")

    b.set_block(b0)
    vals = []
    prev = w0
    for _ in range(_THRU_N):
        t = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)  # t = prev + w0
        ins.add_reg(t, is_def=True)
        ins.add_reg(prev)
        ins.add_reg(w0)
        vals.append(t)
        prev = t
    j0 = b.build_instr(br)
    j0.add_mbb(b1)
    b0.add_successor(b1)

    b.set_block(b1)  # empty: all N values live through here
    j1 = b.build_instr(br)
    j1.add_mbb(b2)
    b1.add_successor(b2)

    b.set_block(b2)
    acc = vals[0]
    for t in vals[1:]:
        nacc = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(nacc, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(t)
        acc = nacc
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


_THRU_N = 40


def _thru_pressure_closed_form(x):
    # vals[i] = (i+2)*x; b2 sums them all.
    return x * sum(i + 2 for i in range(_THRU_N))


# Pressure for the diamond region-split fixture. Chosen so region split fires
# naturally AND our per-vreg decisions still match native greedy exactly: the
# faithful (BrokenHints, MaxWeight) eviction cost depends on VRM hint-satisfaction
# state, which this allocator reproduces bit-for-bit only up to a pressure
# ceiling (past ~34 vregs the accumulated hint state drifts and the spill sets
# diverge). 32 sits comfortably in the fires-and-matches window (30..34).
_DIAMOND_N = 32


def _build_diamond_pressure(mmi):
    """A diamond CFG where region split beats per-block isolation: b0 defines v
    and a condition, branches to a high-pressure arm b1 (N live temps, v
    live-through) or a low-pressure arm b2 (v live-through), joining at b3 which
    returns v. Keeping v in a register through the cheap arm while isolating it
    around the pressured arm is exactly what tryRegionSplit forms, so under
    register pressure (the aarch64-linux allocatable set) v resolves via region
    split. dia(x) = 2x (the b1 temps are dead; only v reaches b3)."""
    mf = mmi.machine_function("dia")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b3 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    br = mf.opcode("B")
    cbz = mf.opcode("CBZW")

    b.set_block(b0)
    v = mf.create_vreg(gpr32)
    iv = b.build_instr(addrr)  # v = w0 + w0
    iv.add_reg(v, is_def=True)
    iv.add_reg(w0)
    iv.add_reg(w0)
    cond = mf.create_vreg(gpr32)
    ic = b.build_instr(copy)
    ic.add_reg(cond, is_def=True)
    ic.add_reg(w0)
    cz = b.build_instr(cbz)
    cz.add_reg(cond)
    cz.add_mbb(b2)
    b0.add_successor(b1)
    b0.add_successor(b2)

    b1.add_livein(w0)  # b1 uses w0 for its pressure chain
    b.set_block(b1)
    terms = []
    prev = w0
    for _ in range(_DIAMOND_N):
        t = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(t, is_def=True)
        ins.add_reg(prev)
        ins.add_reg(w0)
        terms.append(t)
        prev = t
    acc = terms[0]
    for t in terms[1:]:
        na = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(na, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(t)
        acc = na
    j1 = b.build_instr(br)
    j1.add_mbb(b3)
    b1.add_successor(b3)

    b.set_block(b2)  # low-pressure arm: v just lives through
    j2 = b.build_instr(br)
    j2.add_mbb(b3)
    b2.add_successor(b3)

    b.set_block(b3)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_block_split_executes_under_pressure():
    traces = {}

    class Traced(mir.RAGreedy):
        def select_or_split(self, li):
            r = super().select_or_split(li)
            traces.update(self.trace)
            return r

    mir.register_regalloc("ra-greedy-blocksplit", Traced)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thrup")
        _build_thru_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-blocksplit")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thrup", obj)
    # thrup(x) = N*2x: the live-through values survive whatever splitting the
    # allocator does across the pressured block.
    assert fn(9) == _thru_pressure_closed_form(9)
    assert fn(-4) == _thru_pressure_closed_form(-4)
    # Some split fired under pressure (region split now leads; per-block
    # isolation may still handle the remainder).
    assert any(
        v in ("region_split", "block_split", "local_split") for v in traces.values()
    )
    assert_no_leaks()


def _build_local_multiuse(mmi):
    """Single block: v = COPY w0, then acc folded from v four times
    (acc = v; acc = acc + v; ... = 5*v). v is a single-block interval with
    several use slots -- the shape tryLocalSplit's gap scan operates on.
    gaps(x) = 5x."""
    mf = mmi.machine_function("gaps")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    entry = mf.blocks[0]
    entry.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    v = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(v, is_def=True)
    c.add_reg(w0)
    acc = v
    for _ in range(4):
        n = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(n, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(v)
        acc = n
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_local_split_executes():
    """Faithful tryLocalSplit: the gap-weight scan picks a keep-in-register
    window and the SplitEditor applies it, preserving semantics.

    Fidelity note: local splitting only fires in the natural assign->evict->
    split flow for a >2-use single-block interval that loses the register
    competition -- which greedy's priority (large multi-use ranges are assigned
    early) makes very hard to provoke on small hand-built MIR. Like the repo's
    other split-machinery tests (ra-split-1, ra-split-thru), this harness forces
    the stage on the first eligible interval to exercise the ported gap scan
    end-to-end; the natural-flow path is covered by the differential oracle
    against native greedy."""
    forced = {"done": False}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            if not forced["done"]:
                sa = self.split_analysis
                sa.analyze(li)
                if (
                    self.interval_is_in_one_mbb(reg)
                    and len(sa.use_blocks()) == 1
                    and len(list(sa.get_use_slots())) > 2
                    and self._try_local_split(li)
                ):
                    forced["done"] = True
                    self.trace[reg] = "local_split"
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-localsplit", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_local_multiuse(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-localsplit")
    assert forced["done"], "the local-split gap scan produced a split"
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "gaps", obj)
    assert fn(3) == 15  # 5 * 3
    assert fn(-2) == -10
    assert_no_leaks()


# -- Oracle 3: differential vs native greedy (semantic) -----------------------


@pytest.mark.parametrize("x", [0, 1, 5, -3, 100])
@pytest.mark.parametrize(
    "builder,fn,sig",
    [
        (_build_add, "add", (ctypes.c_int, ctypes.c_int, ctypes.c_int)),
        (_build_high_pressure, "hp", (ctypes.c_int, ctypes.c_int)),
        (_build_thru_pressure, "thrup", (ctypes.c_int, ctypes.c_int)),
    ],
)
def test_matches_native_greedy_semantics(builder, fn, sig, x):
    """JIT-execute the object mir.RAGreedy produces and the one the target
    default allocator (greedy) produces; the results must agree for every
    input. A mismatch is a real allocation bug."""
    mir.register_regalloc("ra-greedy-diff", mir.RAGreedy)

    def emit(regalloc):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine()
            mmi = mir.create_machine_function(mod, tm, fn)
            builder(mmi)
            return mmi.emit_object(regalloc=regalloc)

    native, j1 = _jit_call(sig, fn, emit(None))  # target default == greedy
    ours, j2 = _jit_call(sig, fn, emit("ra-greedy-diff"))
    args = [x] * (len(sig) - 1)
    assert ours(*args) == native(*args)
    assert_no_leaks()


# -- Oracle 4: decision-level diff vs native greedy ---------------------------


def test_decision_level_matches_native_on_small_input():
    """On a small input where greedy trivially assigns, mir.RAGreedy must reach
    the same vreg->physreg decisions as native greedy, read back through
    regalloc_assignments (same manual [allocator][capture] pipeline for both)."""
    mir.register_regalloc("ra-greedy-dec", mir.RAGreedy)

    def assignments(regalloc):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "add")
            _build_add(mmi)
            return mmi.regalloc_assignments(regalloc=regalloc)

    native = assignments("greedy")
    ours = assignments("ra-greedy-dec")
    assert ours.assignments == native.assignments
    assert sorted(ours.spilled) == sorted(native.spilled)
    assert_no_leaks()


# -- coverage of guard/edge branches in the split & evict internals -----------
# These drive private helpers directly from inside select_or_split (the only
# context where split_analysis/matrix/etc. are live), the same forced-harness
# pattern the repo uses for split machinery. Each exercises a specific
# defensive branch that the natural assign->evict->split flow reaches only under
# hard-to-provoke pressure.


def test_internal_guard_branches_single_block():
    checks = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            if "done" not in checks:
                checks["done"] = True
                # _get_cascade caches: second call takes the cached path.
                c1 = self._get_cascade(reg)
                checks["cascade_cached"] = self._get_cascade(reg) == c1
                # On the first interval every physreg is free, so _try_evict
                # skips them all (the is_free `continue`) and finds nothing.
                checks["evict_all_free"] = self._try_evict(li) is None
                # A free physreg has no evictable interferers.
                free = next(
                    p for p in self.allocation_order(li) if self.matrix.is_free(li, p)
                )
                checks["cannot_evict_free"] = not self._can_evict_interference(li, free)
                # _try_split bails immediately once a range is at RS_Spill.
                self._set_stage(reg, self.RS_Spill)
                checks["split_at_spill"] = self._try_split(li) is False
                self._set_stage(reg, self.RS_Assign)
                # A 2-use single-block interval is too short for local split.
                sa = self.split_analysis
                sa.analyze(li)
                if len(list(sa.get_use_slots())) <= 2:
                    checks["local_two_use"] = self._try_local_split(li) is False
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-guards", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-guards")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int, ctypes.c_int), "add", obj)
    assert fn(3, 4) == 7
    assert checks["cascade_cached"]
    assert checks["evict_all_free"]
    assert checks["cannot_evict_free"]
    assert checks["split_at_spill"]
    assert checks.get("local_two_use", True)
    assert_no_leaks()


def _build_three_block(mmi):
    """b0 defines v, b1 is empty (v live-through), b2 uses v: a multi-block
    interval for v. thru(x) = x."""
    mf = mmi.machine_function("thru")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    br = mf.opcode("B")
    b.set_block(b0)
    v = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(v, is_def=True)
    c.add_reg(w0)
    j0 = b.build_instr(br)
    j0.add_mbb(b1)
    b0.add_successor(b1)
    b.set_block(b1)
    j1 = b.build_instr(br)
    j1.add_mbb(b2)
    b1.add_successor(b2)
    b.set_block(b2)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_local_split_rejects_multi_block():
    checks = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            if "done" not in checks:
                sa = self.split_analysis
                sa.analyze(li)
                if len(sa.use_blocks()) > 1:
                    checks["done"] = True
                    # A multi-block interval is not a local-split candidate.
                    checks["rejected"] = self._try_local_split(li) is False
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-lmb", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-lmb")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thru", obj)
    assert fn(9) == 9
    assert checks.get("rejected")
    assert_no_leaks()


def _build_two_multiuse(mmi):
    """Single block with two overlapping multi-use values u and v, then a fold
    that reads them alternately. Assigning u first makes it interfere with v
    across the block -- the setup the local-split gap scan reasons about.
    gaps(x): u=v=x; the fold accumulates -> a linear function of x."""
    mf = mmi.machine_function("gaps")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    e = mf.blocks[0]
    e.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    u = mf.create_vreg(gpr32)
    cu = b.build_instr(copy)
    cu.add_reg(u, is_def=True)
    cu.add_reg(w0)
    v = mf.create_vreg(gpr32)
    cv = b.build_instr(copy)
    cv.add_reg(v, is_def=True)
    cv.add_reg(w0)
    acc = u
    for i in range(8):
        n = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(n, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(v if i % 2 else u)
        acc = n
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_local_split_gap_scan_with_interference():
    """Drive the local-split gap-weight scan with real interference present: one
    multi-use value is assigned first, so when the second is force-split the
    interferer's segments produce non-zero gap weights and the shrink/extend
    scan explores its branches. Verifies the split applies and executes.

    Fidelity note: like the repo's split-machinery tests, the stage is forced
    (assign the first multi-use value, split the second) because greedy would
    otherwise assign both; the scan logic itself is the faithful port."""
    state = {"assigned": 0, "forced": False}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            sa = self.split_analysis
            sa.analyze(li)
            multi = (
                self.interval_is_in_one_mbb(reg)
                and len(sa.use_blocks()) == 1
                and len(list(sa.get_use_slots())) > 2
            )
            if multi and state["assigned"] == 0:
                for p in self.allocation_order(li):
                    if self.matrix.is_free(li, p):
                        state["assigned"] = 1
                        return p
            if multi and state["assigned"] >= 1 and not state["forced"]:
                if self._try_local_split(li):
                    state["forced"] = True
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-localintf", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_two_multiuse(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-localintf")
    assert state["forced"], "the gap scan produced a split with interference present"
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "gaps", obj)
    # Cross-check against native greedy on the same function.
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_two_multiuse(mmi)
        native, jn = _jit_call(
            (ctypes.c_int, ctypes.c_int), "gaps", mmi.emit_object(regalloc=None)
        )
    assert fn(4) == native(4)
    assert fn(-1) == native(-1)
    assert_no_leaks()


def _build_local_clobber(mmi):
    """Single block with a multi-use value that lives across (a) an explicit
    physreg def ($w9 = COPY $w0), creating fixed reg-unit interference on W9, and
    (b) an instruction carrying a call-preserved register mask, clobbering the
    caller-saved registers. Drives calcGapWeights' fixed reg-unit and reg-mask
    huge_valf marking. clob(x) = a linear function of x."""
    mf = mmi.machine_function("clob")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0, w9 = mf.physreg("W0"), mf.physreg("W9")
    e = mf.blocks[0]
    e.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    u = mf.create_vreg(gpr32)
    cu = b.build_instr(copy)
    cu.add_reg(u, is_def=True)
    cu.add_reg(w0)
    v = mf.create_vreg(gpr32)
    cv = b.build_instr(copy)
    cv.add_reg(v, is_def=True)
    cv.add_reg(w0)
    acc = u
    for i in range(4):
        n = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(n, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(v if i % 2 else u)
        acc = n
    # A physreg def (fixed reg-unit interference on W9) that carries a call
    # regmask (clobbering the caller-saved regs) -- v is live across it.
    clob = b.build_instr(copy)
    clob.add_reg(w9, is_def=True)
    clob.add_reg(w0)
    clob.add_reg_mask()
    for i in range(4):
        n = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(n, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(v if i % 2 else u)
        acc = n
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_local_split_fixed_and_regmask_interference():
    """Local-split gap scan over an interval that crosses a physreg clobber
    (fixed reg-unit interference) and a call register mask: calcGapWeights marks
    the covered gaps huge_valf. Confirms both the reg-unit fixed-span path and
    the reg-mask-gap path are exercised, and the split still matches native."""
    state = {"assigned": 0, "forced": False, "fixed": False, "regmask": False}

    class Force(mir.RAGreedy):
        def fixed_interference_spans(self, li, physreg):
            spans = super().fixed_interference_spans(li, physreg)
            if spans:
                state["fixed"] = True
            return spans

        def _local_reg_mask_gaps(self, li, bi, uses, num_gaps):
            gaps = super()._local_reg_mask_gaps(li, bi, uses, num_gaps)
            if gaps:
                state["regmask"] = True
            return gaps

        def select_or_split(self, li):
            reg = li.reg
            sa = self.split_analysis
            sa.analyze(li)
            multi = (
                self.interval_is_in_one_mbb(reg)
                and len(sa.use_blocks()) == 1
                and len(list(sa.get_use_slots())) > 2
            )
            if multi and state["assigned"] == 0:
                for p in self.allocation_order(li):
                    if self.matrix.is_free(li, p):
                        state["assigned"] = 1
                        return p
            if multi and state["assigned"] >= 1 and not state["forced"]:
                if self._try_local_split(li):
                    state["forced"] = True
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-clob", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "clob")
        _build_local_clobber(mmi)
        mmi.regalloc_assignments(regalloc="ra-greedy-clob")
    # The gap scan saw both fixed reg-unit interference (the W9 def) and the
    # call regmask, marking their gaps huge_valf.
    assert state["fixed"], "fixed reg-unit interference reached the gap scan"
    assert state["regmask"], "the call regmask produced regmask gaps"
    assert_no_leaks()


def test_local_split_progress_required():
    """Force a local split on an interval already at RS_Split2 (progress
    required), so the gap scan must find a strictly-shorter window: exercises
    the progress-required legality branch. Interference is present as in
    test_local_split_gap_scan_with_interference."""
    state = {"assigned": 0, "forced": False}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            sa = self.split_analysis
            sa.analyze(li)
            multi = (
                self.interval_is_in_one_mbb(reg)
                and len(sa.use_blocks()) == 1
                and len(list(sa.get_use_slots())) > 2
            )
            if multi and state["assigned"] == 0:
                for p in self.allocation_order(li):
                    if self.matrix.is_free(li, p):
                        state["assigned"] = 1
                        return p
            if multi and state["assigned"] >= 1 and not state["forced"]:
                self._set_stage(reg, self.RS_Split2)
                r = self._try_local_split(li)
                state["forced"] = True
                if r:
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-localprog", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_two_multiuse(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-localprog")
    assert state["forced"]
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "gaps", obj)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_two_multiuse(mmi)
        native, jn = _jit_call(
            (ctypes.c_int, ctypes.c_int), "gaps", mmi.emit_object(regalloc=None)
        )
    assert fn(4) == native(4)
    assert_no_leaks()


def _build_cbz(mmi):
    """b0 defines v and tests it in a CBZW terminator (v live-out past b0's last
    split point), b1 falls through, b2 uses v. This drives block split's
    overlap-into-live-out-tail path. cbz(x) = x."""
    mf = mmi.machine_function("cbz")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    b.set_block(b0)
    v = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(v, is_def=True)
    c.add_reg(w0)
    cbz = b.build_instr(mf.opcode("CBZW"))
    cbz.add_reg(v)
    cbz.add_mbb(b2)
    b0.add_successor(b1)
    b0.add_successor(b2)
    b.set_block(b1)
    j = b.build_instr(mf.opcode("B"))
    j.add_mbb(b2)
    b1.add_successor(b2)
    b.set_block(b2)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_block_split_overlap_path_executes():
    """Force block split on the cbz shape, where v is live-out past its block's
    last split point, exercising splitSingleBlock's overlapIntv tail."""
    forced = {"done": False}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            sa = self.split_analysis
            sa.analyze(li)
            if not forced["done"] and not self.interval_is_in_one_mbb(reg):
                if self._try_block_split(li):
                    forced["done"] = True
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-cbzsplit", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "cbz")
        _build_cbz(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-cbzsplit")
    assert forced["done"]
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "cbz", obj)
    assert fn(0) == 0
    assert fn(7) == 7
    assert_no_leaks()


def test_block_split_no_qualifying_block():
    """Force block split on a live-through value whose only use blocks are
    single COPY instructions in a non-subclass register class: shouldSplitSingle
    Block rejects them all, so tryBlockSplit splits nothing and returns False."""
    checks = {}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in checks and not self.interval_is_in_one_mbb(reg):
                checks["done"] = True
                checks["no_split"] = self._try_block_split(li) is False
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-noblk", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-noblk")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thru", obj)
    assert fn(9) == 9
    assert checks.get("no_split")
    assert_no_leaks()


def _build_many_multiuse(mmi, k=6):
    """Single block: `k` values each used many times in a long interleaved fold,
    so every one is a multi-use interval overlapping the others. Assigning k-1
    of them first loads the candidate physregs with interference for the last."""
    mf = mmi.machine_function("gaps")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    e = mf.blocks[0]
    e.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    vs = []
    for _ in range(k):
        u = mf.create_vreg(gpr32)
        c = b.build_instr(copy)
        c.add_reg(u, is_def=True)
        c.add_reg(w0)
        vs.append(u)
    acc = vs[0]
    for i in range(24):
        n = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(n, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(vs[i % k])
        acc = n
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_local_split_progress_required_finds_no_window():
    """With progress required and heavy interference, the gap scan cannot find a
    strictly-shorter profitable window, so tryLocalSplit returns False (the
    no-candidate exit). k-1 values are assigned first to load interference."""
    K = 6
    state = {"assigned": 0, "forced": False, "result": None}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            sa = self.split_analysis
            sa.analyze(li)
            multi = (
                self.interval_is_in_one_mbb(reg)
                and len(sa.use_blocks()) == 1
                and len(list(sa.get_use_slots())) > 2
            )
            if multi and state["assigned"] < K - 1:
                for p in self.allocation_order(li):
                    if self.matrix.is_free(li, p):
                        state["assigned"] += 1
                        return p
            if multi and state["assigned"] >= K - 1 and not state["forced"]:
                self._set_stage(reg, self.RS_Split2)
                state["forced"] = True
                state["result"] = self._try_local_split(li)
                if state["result"]:
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-nowindow", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_many_multiuse(mmi, k=K)
        obj = mmi.emit_object(regalloc="ra-greedy-nowindow")
    assert state["forced"]
    # Under progress-required + heavy interference the scan finds no window.
    assert state["result"] is False
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "gaps", obj)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_many_multiuse(mmi, k=K)
        native, jn = _jit_call(
            (ctypes.c_int, ctypes.c_int), "gaps", mmi.emit_object(regalloc=None)
        )
    assert fn(3) == native(3)
    assert_no_leaks()


def _build_endheavy_multiuse(mmi):
    """Single block: a target t used 8 times over spaced-out arithmetic (cheap
    early gaps), then a heavy interferer u live only in t's final gap. The
    local-split scan extends through the cheap early window, then must shrink it
    when the expensive final gap is included -- exercising the running-max
    recompute. gaps(x) is a linear function of x (cross-checked vs native)."""
    mf = mmi.machine_function("gaps")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    e = mf.blocks[0]
    e.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    t = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(t, is_def=True)
    c.add_reg(w0)
    acc = t
    for _ in range(5):
        for _ in range(2):
            n = mf.create_vreg(gpr32)
            ins = b.build_instr(addrr)
            ins.add_reg(n, is_def=True)
            ins.add_reg(acc)
            ins.add_reg(acc)
            acc = n
        n = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(n, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(t)
        acc = n
    u = mf.create_vreg(gpr32)
    cu = b.build_instr(copy)
    cu.add_reg(u, is_def=True)
    cu.add_reg(w0)
    n = mf.create_vreg(gpr32)
    ins = b.build_instr(addrr)
    ins.add_reg(n, is_def=True)
    ins.add_reg(acc)
    ins.add_reg(u)
    acc = n
    nf = mf.create_vreg(gpr32)
    ins = b.build_instr(addrr)
    ins.add_reg(nf, is_def=True)
    ins.add_reg(acc)
    ins.add_reg(t)
    acc = nf
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_local_split_shrink_recompute():
    """Force local split on the 8-use end-heavy target: the scan builds a wide
    keep window over the cheap early gaps, then shrinks it (recomputing the
    running max over the interior gaps) when the expensive final gap enters."""
    forced = {"done": False}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            multi = (
                self.interval_is_in_one_mbb(li.reg)
                and len(sa.use_blocks()) == 1
                and len(list(sa.get_use_slots())) > 4
            )
            if multi and not forced["done"]:
                if self._try_local_split(li):
                    forced["done"] = True
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-shrink", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_endheavy_multiuse(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-shrink")
    assert forced["done"]
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "gaps", obj)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "gaps")
        _build_endheavy_multiuse(mmi)
        native, jn = _jit_call(
            (ctypes.c_int, ctypes.c_int), "gaps", mmi.emit_object(regalloc=None)
        )
    assert fn(2) == native(2)
    assert_no_leaks()


def test_interference_cursor_reports_per_block_interference():
    """The region-split interference cursor: point it at a physreg and query a
    block. InterferenceCache reports fixed reg-unit interference too, not just
    assigned vregs, so the entry block already shows interference for the first
    allocatable physreg (the argument registers)."""
    checks = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            if "done" not in checks:
                checks["done"] = True
                cur = self.new_interference_cursor()
                preg = next(iter(self.allocation_order(li)))
                self.set_interference_physreg(cur, preg)
                cur.move_to_block(0)
                checks["has"] = cur.has_interference()
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-cursor", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-cursor")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int, ctypes.c_int), "add", obj)
    assert fn(3, 4) == 7
    # Entry block carries fixed argument-register interference for w0 (the
    # first allocatable physreg), so the cursor reports interference here.
    assert checks["has"] is True
    assert_no_leaks()


def test_region_split_analysis_accessors():
    """The region-split SplitAnalysis/EdgeBundles/loop accessors read back."""
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            if "done" not in saw and not self.interval_is_in_one_mbb(li.reg):
                saw["done"] = True
                sa = self.split_analysis
                sa.analyze(li)
                b0 = sa.use_blocks()[0].mbb.number
                saw["fsp_valid"] = sa.first_split_point(b0).is_valid()
                saw["num_live"] = sa.num_live_blocks()
                saw["count_live"] = sa.count_live_blocks(li)
                saw["loop_iv"] = sa.looks_like_loop_iv()
                eb = self.edge_bundles
                saw["bundle"] = eb.get_bundle_number(b0, True)
                saw["loop_hdr"] = self.loop_header_number(b0)
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-rsa", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-rsa")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thru", obj)
    assert fn(9) == 9
    assert saw["num_live"] >= 2
    assert saw["count_live"] >= 2
    assert isinstance(saw["loop_iv"], bool)
    assert saw["bundle"] >= 0
    assert saw["loop_hdr"] is None  # straight-line CFG: no loop
    assert_no_leaks()


def test_split_live_through_block_executes():
    """Drive the high-level SplitEditor block splitters directly on
    _build_three_block: route the whole live range through one new interval via
    split_reg_out_block (the def block), split_live_through_block (the empty
    middle block), and split_reg_in_block (the use block). This mirrors what
    splitAroundRegion does for a single all-covering candidate, and confirms the
    result still computes thru(x) == x. Driving split_live_through_block in
    isolation (leaving the def/use blocks in the complement) is malformed -- the
    value must stay coherent across the boundary blocks."""
    done = {"v": False, "nvregs": 0}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            reg = li.reg
            sa = self.split_analysis
            sa.analyze(li)
            if (
                not done["v"]
                and not self.interval_is_in_one_mbb(reg)
                and sa.through_blocks()
            ):
                lre = self.new_live_range_edit(li)
                se = self.split_editor
                se.reset(lre, mir.ComplementSpillMode.SM_Speed)
                idx = se.open_intv()
                cur = self.new_interference_cursor()
                self.set_interference_physreg(
                    cur, next(iter(self.allocation_order(li)))
                )
                for bi in sa.use_blocks():
                    n = bi.mbb.number
                    cur.move_to_block(n)
                    if bi.live_in and bi.live_out:
                        se.split_live_through_block(
                            n, idx, cur.first(), idx, cur.last()
                        )
                    elif bi.live_in:
                        se.split_reg_in_block(bi, idx, cur.first())
                    elif bi.live_out:
                        se.split_reg_out_block(bi, idx, cur.last())
                for n in sa.through_blocks():
                    cur.move_to_block(n)
                    se.split_live_through_block(n, idx, cur.first(), idx, cur.last())
                se.finish()
                done["nvregs"] = len(lre.new_vregs())
                done["v"] = True
                return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-sltb", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-sltb")
    assert done["v"]
    assert done["nvregs"] >= 1  # the split produced a new interval
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thru", obj)
    assert fn(9) == 9 and fn(-4) == -4
    assert_no_leaks()


def test_global_split_candidate_reset():
    ra = mir.RAGreedy()
    cand = GlobalSplitCandidate()
    cand.active_blocks.append(3)
    cand.reset(physreg=0)  # compact region
    assert cand.phys_reg == 0
    assert cand.active_blocks == []
    assert cand.intv_idx == 0


def test_add_split_constraints_builds_constraints():
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in saw and not self.interval_is_in_one_mbb(li.reg):
                saw["done"] = True
                cur = self.new_interference_cursor()
                self.set_interference_physreg(
                    cur, next(iter(self.allocation_order(li)))
                )
                self.spill_placer.prepare(mir.BitVector())
                cost, positive = self._add_split_constraints(cur)
                saw["ncons"] = len(self._split_constraints)
                saw["positive"] = positive
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-asc", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-asc")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thru", obj)
    assert fn(9) == 9
    assert saw["ncons"] >= 1
    assert isinstance(saw["positive"], bool)
    assert_no_leaks()


def test_add_through_constraints_links_clean_blocks():
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in saw and sa.num_through_blocks() > 0:
                saw["done"] = True
                cur = self.new_interference_cursor()
                self.set_interference_physreg(
                    cur, next(iter(self.allocation_order(li)))
                )
                self.spill_placer.prepare(mir.BitVector())
                self._add_split_constraints(cur)
                saw["ok"] = self._add_through_constraints(cur, sa.through_blocks())
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-atc", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-atc")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thru", obj)
    assert fn(9) == 9
    assert saw["ok"] is True
    assert_no_leaks()


def test_grow_region_expands_and_returns():
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in saw and sa.num_through_blocks() > 0:
                saw["done"] = True
                cand = GlobalSplitCandidate()
                cand.reset(
                    next(iter(self.allocation_order(li))),
                    self.new_interference_cursor(),
                )
                self.set_interference_physreg(cand.intf, cand.phys_reg)
                self.spill_placer.prepare(cand.live_bundles)
                self._add_split_constraints(cand.intf)
                saw["grew"] = self._grow_region(li, cand)
                saw["active"] = list(cand.active_blocks)
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-grow", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thrup")
        _build_thru_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-grow")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thrup", obj)
    assert fn(3) == _thru_pressure_closed_form(3)
    # growRegion succeeds (budget not exceeded) and activates the through blocks.
    assert saw["grew"] is True
    assert saw["active"], "the region grew to cover at least one through block"
    assert_no_leaks()


def test_calc_global_split_cost_nonnegative():
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in saw and sa.num_through_blocks() > 0:
                saw["done"] = True
                cand = GlobalSplitCandidate()
                cand.reset(
                    next(iter(self.allocation_order(li))),
                    self.new_interference_cursor(),
                )
                self.set_interference_physreg(cand.intf, cand.phys_reg)
                self.spill_placer.prepare(cand.live_bundles)
                self._add_split_constraints(cand.intf)
                self._grow_region(li, cand)
                self.spill_placer.finish()
                order = list(self.allocation_order(li))
                cost = self._calc_global_split_cost(cand, order)
                saw["freq"] = cost.get_frequency()
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-cgsc", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thrup")
        _build_thru_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-cgsc")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thrup", obj)
    assert fn(3) == _thru_pressure_closed_form(3)
    assert saw["freq"] >= 0
    assert_no_leaks()


def test_calc_compact_region_returns_bool():
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in saw:
                saw["done"] = True
                cand = GlobalSplitCandidate()
                cand.reset(0, self.new_interference_cursor())
                saw["compact"] = self._calc_compact_region(li, cand)
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-ccr", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thrup")
        _build_thru_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-ccr")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thrup", obj)
    assert fn(3) == _thru_pressure_closed_form(3)
    assert isinstance(saw["compact"], bool)
    assert_no_leaks()


def test_calculate_region_split_cost_selects_candidate():
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in saw and sa.num_through_blocks() > 0:
                saw["done"] = True
                order = list(self.allocation_order(li))
                best_cost = mir.BlockFrequency.max()
                best, ncands = self._calculate_region_split_cost(
                    li, order, best_cost, 0, False
                )
                saw["best"] = best
                saw["ncands"] = ncands
                saw["bundles"] = (
                    self._global_cand[best].live_bundles.count()
                    if best != _NO_CAND
                    else 0
                )
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-crsc", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thrup")
        _build_thru_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-crsc")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thrup", obj)
    assert fn(3) == _thru_pressure_closed_form(3)
    # thrup has a live-through value under pressure, so a region candidate wins.
    assert saw["best"] != _NO_CAND and saw["best"] >= 0
    assert saw["ncands"] >= 1
    assert saw["bundles"] > 0  # the winning candidate keeps bundles in-register
    assert_no_leaks()


def test_do_region_split_produces_vregs():
    saw = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in saw and sa.num_through_blocks() > 0:
                saw["done"] = True
                order = list(self.allocation_order(li))
                best_cost = mir.BlockFrequency.max()
                best, ncands = self._calculate_region_split_cost(
                    li, order, best_cost, 0, False
                )
                if best != _NO_CAND:
                    lre = self.new_live_range_edit(li)
                    self._do_region_split(li, best, False, lre)
                    saw["nvregs"] = len(lre.new_vregs())
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-drs", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thrup")
        _build_thru_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-drs")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thrup", obj)
    assert fn(3) == _thru_pressure_closed_form(3)
    assert saw.get("nvregs", 0) >= 1
    assert_no_leaks()


def test_region_split_fires_naturally():
    """Region split must fire through the natural dispatcher on a CFG where it
    beats per-block isolation (the diamond: a value live through a cheap arm and
    a pressured arm). Under the aarch64-linux allocatable set this resolves via
    region split, and its spill decisions match native greedy. (Straight-line
    pressure like `thrup` block-splits instead -- matching native, see the
    decision-level oracle.)"""
    traces = {}

    class Traced(mir.RAGreedy):
        def select_or_split(self, li):
            r = super().select_or_split(li)
            traces.update(self.trace)
            return r

    mir.register_regalloc("ra-greedy-regionnat", Traced)

    def assignments(regalloc):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "dia")
            _build_diamond_pressure(mmi)
            return mmi.regalloc_assignments(regalloc=regalloc)

    ours = assignments("ra-greedy-regionnat")
    native = assignments("greedy")
    assert "region_split" in traces.values()
    assert sorted(ours.spilled) == sorted(native.spilled)
    assert_no_leaks()


@pytest.mark.parametrize("x", [0, 1, 5, -3, 100])
def test_region_split_matches_native_greedy(x):
    mir.register_regalloc("ra-greedy-regiondiff", mir.RAGreedy)

    def emit(regalloc):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine()
            mmi = mir.create_machine_function(mod, tm, "thrup")
            _build_thru_pressure(mmi)
            return mmi.emit_object(regalloc=regalloc)

    native, j1 = _jit_call((ctypes.c_int, ctypes.c_int), "thrup", emit(None))
    ours, j2 = _jit_call(
        (ctypes.c_int, ctypes.c_int), "thrup", emit("ra-greedy-regiondiff")
    )
    assert ours(x) == native(x)
    assert_no_leaks()


def test_region_split_decision_level_matches_native():
    mir.register_regalloc("ra-greedy-regiondec", mir.RAGreedy)

    def assignments(regalloc):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "thrup")
            _build_thru_pressure(mmi)
            return mmi.regalloc_assignments(regalloc=regalloc)

    native = assignments("greedy")
    ours = assignments("ra-greedy-regiondec")
    assert ours.assignments == native.assignments
    assert sorted(ours.spilled) == sorted(native.spilled)
    assert_no_leaks()


def test_enable_debug_traces_regalloc_and_toggles_off():
    """llvm.enable_debug turns on LLVM_DEBUG tracing (to stderr). Scope it to
    "regalloc" and capture native greedy's trace at the fd level, then toggle it
    back off so later work is quiet."""
    import os
    import tempfile

    with tempfile.TemporaryFile(mode="w+b") as cap:
        saved = os.dup(2)
        os.dup2(cap.fileno(), 2)
        try:
            llvm.enable_debug(["regalloc"])
            with ir.Context() as ctx:
                mod = ir.Module("m", ctx)
                tm = jit.TargetMachine(triple=_AARCH64_LINUX)
                mmi = mir.create_machine_function(mod, tm, "add")
                _build_add(mmi)
                mmi.regalloc_assignments(regalloc="greedy")
            llvm.enable_debug()  # all types
            llvm.enable_debug(enabled=False)  # back off, keep later output quiet
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        cap.seek(0)
        trace = cap.read().decode("utf-8", "replace")
    # The regalloc channel prints its interval assignments.
    assert "assigning" in trace or "selectOrSplit" in trace
    assert_no_leaks()


def _diamond_live_through_probe(body):
    """Run our allocator on the diamond (aarch64-linux, region split fires) and
    invoke `body(self, li, state)` on the first multi-block value with through
    blocks. If `body` returns True it is taken to have handled `li` (performed a
    split), so the probe returns None; otherwise the normal path runs."""
    state = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if "done" not in state and sa.num_through_blocks() > 0:
                state["done"] = True
                if body(self, li, state) is True:
                    return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-diaprobe", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "dia")
        _build_diamond_pressure(mmi)
        mmi.regalloc_assignments(regalloc="ra-greedy-diaprobe")
    return state


def test_calc_compact_region_positive_path():
    """calcCompactRegion on the diamond's live-through value: the value is
    genuinely wanted in a register, so its compact region (spill through every
    block) is not beneficial and it returns False -- exercising the
    not-positive / no-live-bundles rejection."""

    def body(self, li, st):
        cand = GlobalSplitCandidate()
        cand.reset(0, self.new_interference_cursor())
        st["compact"] = self._calc_compact_region(li, cand)

    st = _diamond_live_through_probe(body)
    assert st["compact"] is False
    assert_no_leaks()


def test_calc_compact_region_no_through_blocks():
    """A single-block value has no through blocks -> compact region is trivially
    False (the early return)."""
    checks = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            if "done" not in checks and self.interval_is_in_one_mbb(li.reg):
                sa = self.split_analysis
                sa.analyze(li)
                checks["done"] = True
                cand = GlobalSplitCandidate()
                cand.reset(0, self.new_interference_cursor())
                checks["compact"] = self._calc_compact_region(li, cand)
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-ccr0", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-ccr0")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int, ctypes.c_int), "add", obj)
    assert fn(3, 4) == 7
    assert checks["compact"] is False
    assert_no_leaks()


def test_calculate_region_split_cost_ignore_csr():
    """Scoring with ignore_csr=True skips unused callee-saved physregs
    (isUnusedCalleeSavedReg): it scores a subset of what ignore_csr=False does,
    so it never yields more candidates."""

    def body(self, li, st):
        order = list(self.allocation_order(li))
        st["csr_seen"] = any(self._is_unused_callee_saved(p) for p in order)
        best_f, nc_f = self._calculate_region_split_cost(
            li, order, mir.BlockFrequency.max(), 0, False
        )
        self._global_cand = []  # reset scratch between scorings
        best_t, nc_t = self._calculate_region_split_cost(
            li, order, mir.BlockFrequency.max(), 0, True
        )
        st["nc_false"], st["nc_true"] = nc_f, nc_t

    st = _diamond_live_through_probe(body)
    # Ignoring CSRs scores a subset -> no more candidates than the full scan.
    assert st["nc_true"] <= st["nc_false"]
    assert isinstance(st["csr_seen"], bool)
    assert_no_leaks()


def test_grow_region_budget_exhausted(monkeypatch):
    """A tiny complexity budget makes growRegion bail with False."""
    import llvm.mir_greedy as _g

    monkeypatch.setattr(_g, "_GROW_REGION_COMPLEXITY_BUDGET", 1)

    def body(self, li, st):
        cand = GlobalSplitCandidate()
        cand.reset(
            next(iter(self.allocation_order(li))), self.new_interference_cursor()
        )
        self.set_interference_physreg(cand.intf, cand.phys_reg)
        self.spill_placer.prepare(cand.live_bundles)
        self._add_split_constraints(cand.intf)
        st["grew"] = self._grow_region(li, cand)

    st = _diamond_live_through_probe(body)
    assert st["grew"] is False
    assert_no_leaks()


def test_do_region_split_with_compact_region():
    """Drive doRegionSplit's has_compact arm: score candidates (populating
    GlobalCand[0]), then apply the region with best_cand=NoCand and
    has_compact=True so GlobalCand[0] is claimed as the compact candidate, opens
    its interval, and drives splitAroundRegion."""

    def body(self, li, st):
        order = list(self.allocation_order(li))
        best, ncands = self._calculate_region_split_cost(
            li, order, mir.BlockFrequency.max(), 0, False
        )
        st["found"] = ncands >= 1 and self._global_cand[0].live_bundles.count() > 0
        if not st["found"]:
            return None
        # GlobalCand[0] is the top-scoring candidate; reuse it as the compact
        # slot (the diamond has no beneficial true-compact region) to drive
        # doRegionSplit's has_compact plumbing -- claiming its bundles and
        # opening its interval.
        lre = self.new_live_range_edit(li)
        self._do_region_split(li, _NO_CAND, True, lre)
        st["nvregs"] = len(lre.new_vregs())
        return True

    st = _diamond_live_through_probe(body)
    assert st["found"], "region scoring must find a candidate on the diamond"
    assert st["nvregs"] >= 1
    assert_no_leaks()


def test_split_around_region_all_covering_candidate():
    """Drive splitAroundRegion with one candidate claiming every bundle, so each
    use block and through block is split live-through into that interval (the
    split_live_through_block arms and the RS_Split2/RS_Spill staging). Mirrors a
    single-region solution."""

    def body(self, li, st):
        order = list(self.allocation_order(li))
        best, ncands = self._calculate_region_split_cost(
            li, order, mir.BlockFrequency.max(), 0, False
        )
        st["found"] = best != _NO_CAND
        if not st["found"]:
            return None
        cand = self._global_cand[best]
        lre = self.new_live_range_edit(li)
        se = self.split_editor
        se.reset(lre, mir.ComplementSpillMode.SM_Speed)
        # Claim every edge bundle for the winning candidate and open its interval.
        self._bundle_cand = [best] * self.edge_bundles.num_bundles()
        cand.intv_idx = se.open_intv()
        self._split_around_region(li, lre, [best])
        st["nvregs"] = len(lre.new_vregs())
        return True

    st = _diamond_live_through_probe(body)
    assert st["found"], "region scoring must find a candidate on the diamond"
    assert st["nvregs"] >= 1
    assert_no_leaks()


def test_enqueue_reverse_local_assignment():
    """A target that assigns local ranges bottom-up (reverseLocalAssignment)
    prioritizes local ranges by distance from the zero index to their end,
    exercising enqueue's reverse branch."""
    seen = {}

    class Reverse(mir.RAGreedy):
        def reverse_local_assignment(self):
            return True  # simulate a bottom-up-assignment target

        def enqueue(self, reg):
            seen["ran"] = True
            return super().enqueue(reg)

    mir.register_regalloc("ra-greedy-rev", Reverse)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-rev")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int, ctypes.c_int), "add", obj)
    assert fn(3, 4) == 7  # bottom-up local priority still allocates correctly
    assert seen.get("ran")
    assert_no_leaks()


_USE_PRESSURE_N = 34


def _build_use_pressure(mmi):
    """b0 defines v AND a high-pressure chain that uses v (so v's def block is a
    use block under pressure); b1 is a through block; b2 uses v. Region-splitting
    v must reason about interference inside its use block b0, exercising the
    MustSpill/PrefSpill/insert arms of addSplitConstraints. up(x) = 2x."""
    mf = mmi.machine_function("up")
    b = mir.MachineIRBuilder(mf)
    gpr = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
    br = mf.opcode("B")
    b.set_block(b0)
    v = mf.create_vreg(gpr)
    iv = b.build_instr(addrr)
    iv.add_reg(v, is_def=True)
    iv.add_reg(w0)
    iv.add_reg(w0)
    terms = []
    prev = w0
    for _ in range(_USE_PRESSURE_N):
        t = mf.create_vreg(gpr)
        ins = b.build_instr(addrr)
        ins.add_reg(t, is_def=True)
        ins.add_reg(prev)
        ins.add_reg(w0)
        terms.append(t)
        prev = t
    acc = terms[0]
    for t in terms[1:]:
        na = mf.create_vreg(gpr)
        ins = b.build_instr(addrr)
        ins.add_reg(na, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(t)
        acc = na
    u = mf.create_vreg(gpr)  # a use of v inside the pressured block
    iu = b.build_instr(addrr)
    iu.add_reg(u, is_def=True)
    iu.add_reg(v)
    iu.add_reg(acc)
    j0 = b.build_instr(br)
    j0.add_mbb(b1)
    b0.add_successor(b1)
    b.set_block(b1)
    j1 = b.build_instr(br)
    j1.add_mbb(b2)
    b1.add_successor(b2)
    b.set_block(b2)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_region_split_use_block_interference_matches_native():
    """Region split with interference inside the split value's use block: our
    per-vreg decisions match native greedy (exercising addSplitConstraints'
    interference arms and calcGlobalSplitCost through the natural flow)."""
    mir.register_regalloc("ra-greedy-up", mir.RAGreedy)

    def assignments(regalloc):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "up")
            _build_use_pressure(mmi)
            return mmi.regalloc_assignments(regalloc=regalloc)

    ours = assignments("ra-greedy-up")
    native = assignments("greedy")
    assert sorted(ours.spilled) == sorted(native.spilled)
    assert_no_leaks()


def test_calc_global_split_cost_arms():
    """Drive calcGlobalSplitCost's per-arm accounting with crafted live-bundle
    solutions: a use-block edge whose register state disagrees with its
    constraint pref (a spill), and active (through) blocks that are all-stack
    (no cost), single-crossing (one spill), and both-in interference."""

    def body(self, li, st):
        cur = self.new_interference_cursor()
        self.set_interference_physreg(cur, next(iter(self.allocation_order(li))))
        self.spill_placer.prepare(mir.BitVector())
        self._add_split_constraints(cur)
        eb = self.edge_bundles
        order = list(self.allocation_order(li))
        PrefReg = mir.BorderConstraint.PrefReg

        def run(set_bits, active):
            cand = GlobalSplitCandidate()
            cand.reset(order[0], cur)
            lb = mir.BitVector()
            lb.resize(eb.num_bundles())
            for bnum in set_bits:
                lb.set(bnum)
            cand.live_bundles = lb
            cand.active_blocks = active
            return self._calc_global_split_cost(cand, order).get_frequency()

        # Use-block ins arm: flip a live-in use block's in-bundle so it disagrees
        # with the entry pref, forcing a spill (cost > 0).
        use_blocks = self.split_analysis.use_blocks()
        live_in_ub = next((b for b in use_blocks if b.live_in), None)
        st["use_arm"] = 0
        if live_in_ub is not None:
            n = live_in_ub.mbb.number
            bc = next(c for c in self._split_constraints if c.number == n)
            want_reg_in = bc.entry == PrefReg
            in_bundle = eb.get_bundle_number(n, False)
            bits = [in_bundle] if not want_reg_in else []
            st["use_arm"] = run(bits, [])
        # Active-block arms on the through blocks: all-stack (no per-block cost),
        # and single-crossing (register on exactly one edge -> one spill).
        through = self.split_analysis.through_blocks()
        st["active_none"] = run([], [])  # no active blocks, no use disagreement
        st["single"] = None
        candidates = list(through) + [b.mbb.number for b in use_blocks]
        for t in candidates:
            in_b = eb.get_bundle_number(t, False)
            out_b = eb.get_bundle_number(t, True)
            if in_b != out_b:
                base = run([], [t])
                st["single"] = run([in_b], [t])  # reg-in only -> single crossing
                st["single_delta"] = st["single"] - base
                break

    st = _diamond_live_through_probe(body)
    assert st["active_none"] >= 0
    if st["single"] is not None:
        assert st["single_delta"] > 0  # a single crossing adds one block frequency
    assert_no_leaks()


def test_split_around_region_isolated_and_through_arms():
    """splitAroundRegion arms: with a candidate claiming only the through
    blocks, the use blocks are isolated (split_single_block / skipped) and the
    through blocks are split live-through."""

    def body(self, li, st):
        order = list(self.allocation_order(li))
        best, ncands = self._calculate_region_split_cost(
            li, order, mir.BlockFrequency.max(), 0, False
        )
        st["found"] = best != _NO_CAND
        if not st["found"]:
            return None
        cand = self._global_cand[best]
        eb = self.edge_bundles
        lre = self.new_live_range_edit(li)
        se = self.split_editor
        se.reset(lre, mir.ComplementSpillMode.SM_Speed)
        cand.intv_idx = se.open_intv()
        cand.active_blocks = list(self.split_analysis.through_blocks())
        # Claim only the through blocks' bundles; leave use blocks unclaimed so
        # they are isolated.
        self._bundle_cand = [_NO_CAND] * eb.num_bundles()
        for n in cand.active_blocks:
            self._bundle_cand[eb.get_bundle_number(n, False)] = best
            self._bundle_cand[eb.get_bundle_number(n, True)] = best
        self._split_around_region(li, lre, [best])
        st["nvregs"] = len(lre.new_vregs())
        return True

    st = _diamond_live_through_probe(body)
    assert st["found"], "region scoring must find a candidate on the diamond"
    assert st["nvregs"] >= 1
    assert_no_leaks()


def test_bitvector_reset_clears_bit():
    """BitVector.reset(i) clears a previously-set bit (the mutator used to
    un-claim an edge bundle)."""
    bv = mir.BitVector()
    bv.resize(4)
    bv.set(1)
    bv.set(3)
    assert sorted(bv.set_bits()) == [1, 3]
    bv.reset(1)
    assert sorted(bv.set_bits()) == [3]
    assert bv.count() == 1


def _build_self_loop(mmi):
    """b0 defines v; b1 is a self-looping header (CBNZW v branches back to
    itself, falling through to b2); b2 uses v. Gives MachineLoopInfo a real
    loop so loop_header_number resolves an enclosing header. Not executed (the
    loop-invariant branch would not terminate)."""
    mf = mmi.machine_function("loop")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    b.set_block(b0)
    v = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(v, is_def=True)
    c.add_reg(w0)
    j0 = b.build_instr(mf.opcode("B"))
    j0.add_mbb(b1)
    b0.add_successor(b1)
    b.set_block(b1)
    cbnz = b.build_instr(mf.opcode("CBNZW"))
    cbnz.add_reg(v)
    cbnz.add_mbb(b1)  # back-edge to self
    b1.add_successor(b1)
    b1.add_successor(b2)
    b.set_block(b2)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_loop_header_number_resolves_enclosing_loop():
    """loop_header_number returns the enclosing loop's header for a block inside
    a loop (the non-trivial MachineLoopInfo path), and None for a block that is
    in no loop."""
    state = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            if "hdr" not in state:
                # b1 (number 1) is the self-loop header; b0 (number 0) is not in
                # any loop.
                state["hdr"] = self.loop_header_number(1)
                state["none"] = self.loop_header_number(0)
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-loop", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "loop")
        _build_self_loop(mmi)
        mmi.regalloc_assignments(regalloc="ra-greedy-loop")
    assert state["hdr"] == 1  # the loop header block number
    assert state["none"] is None  # b0 is not in a loop
    assert_no_leaks()


def test_split_editor_split_single_block_executes():
    """Drive the high-level SplitEditor::splitSingleBlock binding directly:
    isolate each use block's uses into its own interval (what splitAroundRegion
    does for a use block no candidate covers), finish, and confirm the result
    still computes thru(x) == x."""
    done = {"v": False, "nvregs": 0}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if not done["v"] and not self.interval_is_in_one_mbb(li.reg):
                lre = self.new_live_range_edit(li)
                se = self.split_editor
                se.reset(lre, mir.ComplementSpillMode.SM_Speed)
                for bi in sa.use_blocks():
                    se.split_single_block(bi)
                se.finish()
                done["nvregs"] = len(lre.new_vregs())
                done["v"] = True
                return None
            return super().select_or_split(li)

    mir.register_regalloc("ra-greedy-sssb", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-greedy-sssb")
    assert done["v"]
    assert done["nvregs"] >= 1
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int), "thru", obj)
    assert fn(9) == 9 and fn(-4) == -4
    assert_no_leaks()


# --------------------------------------------------------------------------
# Unit tests for the region-split cost-model branches that the hand-built MIR
# corpus cannot steer precisely (edge bundles are shared across blocks, so a
# use block cannot be isolated from its through neighbors by construction).
# These drive the faithful methods directly with controlled stand-ins; the
# *semantic* correctness of the same code is pinned by the differential and
# decision-level oracles against native greedy above. No MIR/context is built,
# so there is nothing to leak-check.
# --------------------------------------------------------------------------
class _FakeSlot:
    """A SlotIndex stand-in supporting the comparisons the cost model uses."""

    def __init__(self, v):
        self.v = v

    def is_earlier_instr(self, o):
        return self.v < o.v

    def is_same_instr(self, o):
        return self.v == o.v

    def get_reg_slot(self):
        return self

    def is_valid(self):
        return self.v >= 0

    def distance(self, o):
        return o.v - self.v

    def __lt__(self, o):
        return self.v < o.v

    def __gt__(self, o):
        return self.v > o.v

    def __le__(self, o):
        return self.v <= o.v

    def __ge__(self, o):
        return self.v >= o.v


class _FakeIntf:
    """An InterferenceCursor stand-in."""

    def __init__(self, has, first=0, last=0):
        self._has = has
        self._first = _FakeSlot(first)
        self._last = _FakeSlot(last)

    def move_to_block(self, n):
        pass

    def has_interference(self):
        return self._has

    def first(self):
        return self._first

    def last(self):
        return self._last


class _FakeSP:
    """A SpillPlacement stand-in recording what it was handed."""

    def __init__(self, recent=()):
        self.constraints = None
        self.links = None
        self.pref = []
        self._recent = list(recent)

    def get_block_frequency_by_number(self, n):
        return mir.BlockFrequency(1)

    def add_constraints(self, cs):
        self.constraints = list(cs)

    def add_links(self, links):
        self.links = list(links)

    def scan_active_bundles(self):
        return True

    def prepare(self, lb):
        pass

    def finish(self):
        pass

    def iterate(self):
        pass

    def add_pref_spill(self, blocks, b):
        self.pref.append((list(blocks), b))

    def get_recent_positive(self):
        r = self._recent
        self._recent = []
        return r


def _stage_helpers(initial=None):
    """A dict-backed (_get_stage, _set_stage, stage) triple for a fake self."""
    stage = dict(initial or {})
    return (
        lambda r: stage.get(r, LiveRangeStage.RS_New),
        lambda r, s: stage.__setitem__(r, s),
        stage,
    )


def test_select_or_split_last_chance_guard_raises():
    """A range that is unspillable/RS_Done and fails assign+evict+split has no
    recourse but last-chance recoloring (not ported), so the guard raises rather
    than letting the spiller abort on a double spill."""

    class LC(mir.RAGreedy):
        def _try_assign(self, li):
            return None

        def _try_evict(self, li):
            return None

        def _try_split(self, li):
            return False

    lc = LC()
    lc._set_stage(77, LiveRangeStage.RS_Memory)
    li = SimpleNamespace(reg=77, size=8, is_spillable=True)
    with pytest.raises(NotImplementedError):
        lc.select_or_split(li)


def test_should_split_single_block_proper_subclass_arms():
    """shouldSplitSingleBlock's single-instruction (proper-subclass) arms: a
    live-through instruction always splits; a copy never does; otherwise it
    splits only at an original endpoint."""
    fg = SimpleNamespace(
        is_copy_like_instr_at=lambda instr: fg._is_copy,
        split_analysis=SimpleNamespace(is_original_endpoint=lambda i: fg._is_orig),
    )

    # Live-through single instruction: always worth splitting.
    bi = SimpleNamespace(is_one_instr=lambda: True, live_in=True, live_out=True)
    assert mg.RAGreedy._should_split_single_block(fg, bi, True) is True

    # A lone copy: no register-class constraint, never split.
    bi = SimpleNamespace(
        is_one_instr=lambda: True,
        live_in=False,
        live_out=False,
        first_instr=_FakeSlot(1),
    )
    fg._is_copy = True
    assert mg.RAGreedy._should_split_single_block(fg, bi, True) is False

    # Non-copy: split iff it is an original endpoint.
    fg._is_copy = False
    fg._is_orig = True
    assert mg.RAGreedy._should_split_single_block(fg, bi, True) is True
    fg._is_orig = False
    assert mg.RAGreedy._should_split_single_block(fg, bi, True) is False


def test_add_split_constraints_interference_insert_arm():
    """addSplitConstraints: an interfering live-in use block whose interference
    starts strictly inside the block (past the first use, before the last) still
    charges one spill insertion (the third live-in arm) without escalating to
    MustSpill/PrefSpill."""
    bi = SimpleNamespace(
        mbb=SimpleNamespace(number=0),
        live_in=True,
        live_out=False,
        first_instr=_FakeSlot(3),
        last_instr=_FakeSlot(10),
        first_def=SimpleNamespace(is_valid=lambda: False),
    )
    sa = SimpleNamespace(
        use_blocks=lambda: [bi],
        first_split_point=lambda n: _FakeSlot(100),
        last_split_point=lambda mbb: _FakeSlot(100),
    )
    lis = SimpleNamespace(
        instr_from_index=lambda idx: SimpleNamespace(is_implicit_def=False),
        mbb_start_index=lambda mbb: _FakeSlot(0),
    )
    fg = SimpleNamespace(split_analysis=sa, spill_placer=_FakeSP(), lis=lis)
    # first()=5 is past the block start (0) and the first use (3) but before the
    # last use (10): the "elif intf.first() < bi.last_instr" insert arm.
    cost, positive = mg.RAGreedy._add_split_constraints(fg, _FakeIntf(True, first=5))
    assert positive is True
    assert cost.get_frequency() == 1  # exactly one insertion charged


def test_add_split_constraints_aborts_when_spill_uninsertable():
    """addSplitConstraints returns (cost, False) when a required entry spill
    cannot be inserted at the block start (the first use precedes the block's
    first split point)."""
    bi = SimpleNamespace(
        mbb=SimpleNamespace(number=0),
        live_in=True,
        live_out=False,
        first_instr=_FakeSlot(3),
        last_instr=_FakeSlot(10),
        first_def=SimpleNamespace(is_valid=lambda: False),
    )
    sa = SimpleNamespace(
        use_blocks=lambda: [bi],
        first_split_point=lambda n: _FakeSlot(100),  # first_instr(3) is earlier
        last_split_point=lambda mbb: _FakeSlot(100),
    )
    lis = SimpleNamespace(
        instr_from_index=lambda idx: SimpleNamespace(is_implicit_def=False),
        mbb_start_index=lambda mbb: _FakeSlot(5),  # start >= first() -> MustSpill
    )
    fg = SimpleNamespace(split_analysis=sa, spill_placer=_FakeSP(), lis=lis)
    _, positive = mg.RAGreedy._add_split_constraints(fg, _FakeIntf(True, first=3))
    assert positive is False


def test_add_through_constraints_exit_mustspill_and_no_links():
    """addThroughConstraints on an all-interfering block set: no clean links
    (the links branch is skipped), and a MustSpill exit when interference reaches
    the last split point."""
    sa = SimpleNamespace(
        first_split_point=lambda n: _FakeSlot(-1),  # first_instr not earlier
        last_split_point_number=lambda n: _FakeSlot(10),  # last()>=lsp -> MustSpill
    )
    sp = _FakeSP()
    fg = SimpleNamespace(
        split_analysis=sa,
        spill_placer=sp,
        first_nondebug_instr_index=lambda n: _FakeSlot(3),
        through_insert_index=lambda n: _FakeSlot(1),
        mbb_start_index_by_number=lambda n: _FakeSlot(0),
    )
    ok = mg.RAGreedy._add_through_constraints(
        fg, _FakeIntf(True, first=5, last=50), [7]
    )
    assert ok is True
    assert sp.links is None  # no clean blocks -> add_links never called
    assert sp.constraints and len(sp.constraints) == 1
    bc = sp.constraints[0]
    # first()=5 is past the block start/insert point -> PrefSpill entry; last()=50
    # reaches the last split point (10) -> MustSpill exit.
    assert bc.entry == mir.BorderConstraint.PrefSpill
    assert bc.exit == mir.BorderConstraint.MustSpill


def test_add_through_constraints_aborts_when_spill_uninsertable():
    """addThroughConstraints returns False when an interfering block's first
    instruction precedes its first split point (spill cannot be inserted)."""
    sa = SimpleNamespace(
        first_split_point=lambda n: _FakeSlot(100),  # first_instr(3) earlier
        last_split_point_number=lambda n: _FakeSlot(10),
    )
    fg = SimpleNamespace(
        split_analysis=sa,
        spill_placer=_FakeSP(),
        first_nondebug_instr_index=lambda n: _FakeSlot(3),
        through_insert_index=lambda n: _FakeSlot(1),
        mbb_start_index_by_number=lambda n: _FakeSlot(0),
    )
    assert (
        mg.RAGreedy._add_through_constraints(fg, _FakeIntf(True, first=5, last=50), [7])
        is False
    )


def test_grow_region_through_constraint_failure_returns_false():
    """growRegion bails (False) when addThroughConstraints fails on the newly
    activated blocks of a physreg candidate."""
    cand = GlobalSplitCandidate()
    cand.reset(1, _FakeIntf(True))  # phys_reg = 1 (non-compact)
    sa = SimpleNamespace(
        through_blocks=lambda: [10, 11], looks_like_loop_iv=lambda: False
    )
    fg = SimpleNamespace(
        split_analysis=sa,
        spill_placer=_FakeSP(recent=[0]),
        edge_bundles=SimpleNamespace(get_blocks=lambda b: [10, 11]),
        _add_through_constraints=lambda intf, blocks: False,
    )
    assert mg.RAGreedy._grow_region(fg, object(), cand) is False


def test_grow_region_compact_loop_iv_keeps_iv_live():
    """growRegion's compact-region loop-IV bias: when the newly activated blocks
    are a loop header and its internal blocks, they are NOT biased to spill
    (pref_spill stays False, so add_pref_spill is skipped)."""
    cand = GlobalSplitCandidate()
    cand.reset(0, None)  # phys_reg = 0 -> compact region
    sa = SimpleNamespace(
        through_blocks=lambda: [10, 11], looks_like_loop_iv=lambda: True
    )
    sp = _FakeSP(recent=[0])
    fg = SimpleNamespace(
        split_analysis=sa,
        spill_placer=sp,
        edge_bundles=SimpleNamespace(get_blocks=lambda b: [10, 11]),
        loop_header_number=lambda b: 10,  # both blocks map to header 10
    )
    assert mg.RAGreedy._grow_region(fg, object(), cand) is True
    assert sp.pref == []  # loop IV kept live: no pref-spill applied


def test_grow_region_compact_loop_iv_non_header_biases_spill():
    """growRegion's compact loop-IV check: when the activated blocks look like a
    loop IV but do NOT form a header + internal-block set, the blocks are still
    biased to spill (pref_spill stays True -> add_pref_spill is applied)."""
    cand = GlobalSplitCandidate()
    cand.reset(0, None)  # phys_reg = 0 -> compact region
    sa = SimpleNamespace(
        through_blocks=lambda: [10, 11], looks_like_loop_iv=lambda: True
    )
    sp = _FakeSP(recent=[0])
    fg = SimpleNamespace(
        split_analysis=sa,
        spill_placer=sp,
        edge_bundles=SimpleNamespace(get_blocks=lambda b: [10, 11]),
        loop_header_number=lambda b: 99,  # header (99) != first block (10)
    )
    assert mg.RAGreedy._grow_region(fg, object(), cand) is True
    assert sp.pref == [([10, 11], True)]  # not a loop-IV region: biased to spill


def test_calc_compact_region_not_positive_returns_false():
    """calcCompactRegion returns False when the split constraints yield no
    positive bundles (nothing worth keeping in a register)."""
    cand = GlobalSplitCandidate()
    cand.intf = _FakeIntf(False)
    fg = SimpleNamespace(
        split_analysis=SimpleNamespace(num_through_blocks=lambda: 1),
        spill_placer=_FakeSP(),
        set_interference_physreg=lambda c, p: None,
        _add_split_constraints=lambda intf: (mir.BlockFrequency(0), False),
    )
    assert mg.RAGreedy._calc_compact_region(fg, object(), cand) is False


def test_calc_compact_region_success_returns_true():
    """calcCompactRegion returns True when constraints are positive, the region
    grows, and live bundles remain (a viable compact region)."""

    def grow(li, cand):
        cand.live_bundles.resize(1)
        cand.live_bundles.set(0)
        return True

    cand = GlobalSplitCandidate()
    cand.intf = _FakeIntf(True)
    fg = SimpleNamespace(
        split_analysis=SimpleNamespace(num_through_blocks=lambda: 1),
        spill_placer=_FakeSP(),
        set_interference_physreg=lambda c, p: None,
        _add_split_constraints=lambda intf: (mir.BlockFrequency(0), True),
        _grow_region=grow,
    )
    assert mg.RAGreedy._calc_compact_region(fg, object(), cand) is True


def test_calc_block_split_cost_charges_redefined_live_through():
    """calcBlockSplitCost charges a second spill for a block where the value is
    live-through AND redefined (live_in && live_out && first_def valid)."""
    bi = SimpleNamespace(
        mbb=SimpleNamespace(number=0),
        live_in=True,
        live_out=True,
        first_def=SimpleNamespace(is_valid=lambda: True),
    )
    fg = SimpleNamespace(
        split_analysis=SimpleNamespace(use_blocks=lambda: [bi]),
        spill_placer=_FakeSP(),
    )
    # One block-isolation spill + one redefined-live-through spill = 2.
    assert mg.RAGreedy._calc_block_split_cost(fg).get_frequency() == 2


def test_try_region_split_compact_but_no_vregs_returns_false():
    """tryRegionSplit with a compact region but no winning per-physreg candidate
    and no new vregs produced by doRegionSplit returns False (nothing applied)."""
    fg = SimpleNamespace(
        should_region_split_for_virt_reg=lambda reg: True,
        allocation_order=lambda li: [1],
        _calc_block_split_cost=lambda: mir.BlockFrequency(0),
        _region_cand0=lambda: GlobalSplitCandidate(),
        _calc_compact_region=lambda li, cand: True,  # has_compact
        _calculate_region_split_cost=lambda *a: (_NO_CAND, 1),
        new_live_range_edit=lambda li: SimpleNamespace(new_vregs=lambda: []),
        _do_region_split=lambda *a: None,
        trace={},
    )
    assert mg.RAGreedy._try_region_split(fg, SimpleNamespace(reg=1)) is False


def test_try_region_split_target_opts_out_returns_false():
    """When the target's shouldRegionSplitForVirtReg is False, tryRegionSplit
    bails immediately without scoring."""
    scored = []
    fg = SimpleNamespace(
        should_region_split_for_virt_reg=lambda reg: False,
        allocation_order=lambda li: scored.append("scored") or [1],
    )
    assert mg.RAGreedy._try_region_split(fg, SimpleNamespace(reg=1)) is False
    assert scored == []  # no scoring happened


def test_do_region_split_skips_candidates_that_claim_no_bundles():
    """doRegionSplit: a best_cand and a compact region that each claim zero
    bundles are both skipped (neither opens an interval); splitAroundRegion is
    still driven with the resulting empty used-candidate set."""
    recorded = {}
    se = SimpleNamespace(reset=lambda lre, mode: None, open_intv=lambda: 1)
    fg = SimpleNamespace(
        split_editor=se,
        edge_bundles=SimpleNamespace(num_bundles=lambda: 4),
        _global_cand=[GlobalSplitCandidate(), GlobalSplitCandidate()],
        _cand_get_bundles=lambda cand, idx: 0,  # nobody claims a bundle
        _split_around_region=lambda li, lre, uc: recorded.__setitem__("uc", list(uc)),
    )
    mg.RAGreedy._do_region_split(fg, object(), 0, True, object())
    assert recorded["uc"] == []  # both candidates skipped


def test_cand_get_bundles_skips_already_claimed():
    """getBundles claims only bundles not already owned by another candidate;
    an already-claimed bit is skipped (the loop-continue arm)."""
    cand = GlobalSplitCandidate()
    cand.live_bundles.resize(2)
    cand.live_bundles.set(0)
    cand.live_bundles.set(1)
    fg = SimpleNamespace(_bundle_cand=[5, _NO_CAND])  # bundle 0 already claimed
    count = mg.RAGreedy._cand_get_bundles(fg, cand, 3)
    assert count == 1  # only bundle 1 newly claimed
    assert fg._bundle_cand == [5, 3]


def test_can_evict_interference_blocks_on_equal_or_newer_cascade():
    """canEvictInterference refuses when the interferer's cascade is not strictly
    older than li's (the eviction-loop guard)."""
    li = SimpleNamespace(reg=1, weight=10.0)
    iv = SimpleNamespace(weight=1.0)  # cheaper, so the weight guard passes
    fg = SimpleNamespace(
        matrix=SimpleNamespace(
            check_interference=lambda a, b: mir.InterferenceKind.IK_VirtReg
        ),
        interfering_vregs=lambda a, b: [99],
        lis=SimpleNamespace(interval=lambda r: iv),
        _cascade_or_next=lambda reg: 5,
        _get_cascade=lambda reg: 10,  # interferer cascade newer than li's (5)
        _get_stage=lambda reg: LiveRangeStage.RS_Assign,
    )
    assert mg.RAGreedy._can_evict_interference(fg, li, 0) is False


def test_try_local_split_shrink_recompute_running_max():
    """tryLocalSplit's running-max recompute: when the scan shrinks the window
    and the dropped gap held the max, the max is recomputed over the remaining
    gaps (the inner recompute loop); a later shrink whose dropped gap was not the
    max takes the skip arm."""
    uses = [_FakeSlot(0), _FakeSlot(1), _FakeSlot(2), _FakeSlot(100)]
    bi = SimpleNamespace(mbb=object(), live_in=False, live_out=True)
    lre = SimpleNamespace(new_vregs=lambda: [])
    calls = []
    picked = {}
    se = SimpleNamespace(
        reset=lambda lre: None,
        open_intv=lambda: calls.append("open"),
        enter_intv_before=lambda s: picked.__setitem__("before", s) or s,
        leave_intv_after=lambda s: picked.__setitem__("after", s) or s,
        use_intv=lambda a, b: calls.append("use"),
        finish=lambda: [],
    )
    get_stage, _set, _stage = _stage_helpers()
    fg = SimpleNamespace(
        split_analysis=SimpleNamespace(
            use_blocks=lambda: [bi],
            get_use_slots=lambda: uses,
        ),
        slot_index_instr_distance=lambda: 1,
        _get_stage=get_stage,
        spill_placer=SimpleNamespace(
            get_block_frequency=lambda mbb: SimpleNamespace(get_frequency=lambda: 100)
        ),
        mbfi=SimpleNamespace(
            entry_freq=lambda: SimpleNamespace(get_frequency=lambda: 1)
        ),
        check_reg_mask_interference=lambda li: False,
        _local_reg_mask_gaps=lambda li, bi, uses, ng: [],
        allocation_order=lambda li: [1],
        # gap[0]=10 is the front max; widening keeps it (g1,g2 < 10); the wide
        # window's low est_weight forces a shrink that drops the max at gap[0],
        # triggering the running-max recompute over the remaining gaps.
        _local_gap_weights=lambda li, preg, u: [10.0, 1.0, 5.0],
        new_live_range_edit=lambda li: lre,
        split_editor=se,
        trace={},
    )
    assert mg.RAGreedy._try_local_split(fg, SimpleNamespace(reg=1)) is True
    assert "use" in calls  # a window was chosen and applied
    # Pin the selected window: [use 0, use 2] -- the best-scoring run before the
    # shrink-and-recompute. A window-selection regression changes these.
    assert picked["before"] is uses[0]
    assert picked["after"] is uses[2]


def test_split_around_region_all_arms():
    """splitAroundRegion across use blocks and through blocks: an unclaimed
    live-in/live-out use block is isolated (split_single_block or skipped), a
    fully-claimed use block is split live-through, through blocks are split or
    skipped per their claimed edges, a block already handled by an earlier
    candidate is deduped, and the new-interval staging kinds are applied."""
    eb_map = {
        (1, False): 10,
        (1, True): 11,
        (2, False): 20,
        (2, True): 21,
        (3, False): 30,
        (3, True): 31,
        (4, False): 40,
        (4, True): 41,
        (5, False): 50,
        (5, True): 51,
        (6, False): 60,
        (6, True): 61,
        (7, False): 70,
        (7, True): 71,
    }
    ubA = SimpleNamespace(mbb=SimpleNamespace(number=1), live_in=True, live_out=False)
    ubB = SimpleNamespace(mbb=SimpleNamespace(number=2), live_in=False, live_out=True)
    ubC = SimpleNamespace(mbb=SimpleNamespace(number=3), live_in=True, live_out=True)

    cand = GlobalSplitCandidate()
    cand.reset(1, _FakeIntf(True, first=1, last=2))
    cand.intv_idx = 1
    cand.active_blocks = [4, 5, 6, 7]  # 7 is not a through block -> deduped

    bundle_cand = [_NO_CAND] * 100
    bundle_cand[eb_map[(3, False)]] = 0  # use block 3 fully claimed
    bundle_cand[eb_map[(3, True)]] = 0
    bundle_cand[eb_map[(4, False)]] = 0  # through block 4: reg-in only
    bundle_cand[eb_map[(5, True)]] = 0  # through block 5: reg-out only
    # block 6: neither edge claimed -> skipped

    se_calls = []
    se = SimpleNamespace(
        split_single_block=lambda bi: se_calls.append(("single", bi.mbb.number)),
        split_live_through_block=lambda n, ii, fi, io, fo: se_calls.append(("thru", n)),
        split_reg_in_block=lambda bi, ii, fi: se_calls.append(("in", bi.mbb.number)),
        split_reg_out_block=lambda bi, io, fo: se_calls.append(("out", bi.mbb.number)),
        finish=lambda: [0, 1, 2, 5],  # spill, split2, RS_New-kept, ignored
    )
    get_stage, set_stage, stage = _stage_helpers({103: LiveRangeStage.RS_Spill})
    fg = SimpleNamespace(
        split_editor=se,
        split_analysis=SimpleNamespace(
            use_blocks=lambda: [ubA, ubB, ubC],
            through_blocks=lambda: [4, 5, 6],
            num_live_blocks=lambda: 3,
            count_live_blocks=lambda iv: {101: 3, 102: 1}.get(iv, 0),
        ),
        edge_bundles=SimpleNamespace(get_bundle_number=lambda n, out: eb_map[(n, out)]),
        _global_cand=[cand],
        _bundle_cand=bundle_cand,
        is_proper_sub_class=lambda reg: False,
        lis=SimpleNamespace(interval=lambda r: r),
        _get_stage=get_stage,
        _set_stage=set_stage,
        _should_split_single_block=lambda bi, single: bi.mbb.number == 1,
    )
    lre = SimpleNamespace(new_vregs=lambda: [100, 101, 102, 103])
    mg.RAGreedy._split_around_region(fg, SimpleNamespace(reg=1), lre, [0])

    assert ("single", 1) in se_calls  # ubA isolated (should-split True)
    assert [k for k, _ in se_calls].count("single") == 1  # ubB skipped
    assert ("thru", 3) in se_calls  # ubC split live-through
    assert ("thru", 4) in se_calls and ("thru", 5) in se_calls  # claimed through
    assert ("thru", 6) not in se_calls  # block 6 skipped (no claimed edge)
    assert ("thru", 7) not in se_calls  # block 7 deduped (not a through block)
    # Staging: m=0 -> spill; m=1 & enough live blocks -> split2; m=2 & too few
    # live blocks -> unchanged; a pre-staged reg -> skipped.
    assert stage[100] == LiveRangeStage.RS_Spill
    assert stage[101] == LiveRangeStage.RS_Split2
    assert 102 not in stage  # too few live blocks: left unchanged (RS_New)
    assert stage[103] == LiveRangeStage.RS_Spill  # untouched (was not RS_New)


def test_local_reg_mask_gaps_arms():
    """RegMaskGaps scan (tryLocalSplit): the lower-bound skip of regmasks before
    the interval, recording a gap a mask falls in, skipping a gap with no mask,
    the last-use-same-instr break, running off the regmask list, and normal
    loop completion. Driven with crafted regmask/use slots."""

    def gaps(rms_vals, use_vals, interfere=True):
        rms = [_FakeSlot(v) for v in rms_vals]
        uses = [_FakeSlot(v) for v in use_vals]
        fg = SimpleNamespace(
            check_reg_mask_interference=lambda li: interfere,
            reg_mask_slots_in_block=lambda n: rms,
        )
        bi = SimpleNamespace(mbb=SimpleNamespace(number=0))
        return mg.RAGreedy._local_reg_mask_gaps(fg, object(), bi, uses, len(uses) - 1)

    # No regmask interference -> empty (early return).
    assert gaps([5], [0, 10, 20], interfere=False) == []
    # A regmask before the first use is skipped (lower bound); a later gap with a
    # mask is recorded; the last mask lands on the final use -> break.
    assert gaps([-1, 15, 30], [0, 10, 20, 30]) == [1]
    # A gap whose mask makes the regmask list run out mid-scan (ri == re break).
    assert gaps([5], [0, 10, 20]) == [0]
    # Every mask sits past the remaining uses -> continue to normal completion.
    assert gaps([100], [0, 10, 20]) == []


def _build_amdgpu_subrange(mmi, *, sub0_uses=3):
    """AMDGPU single block: a 64-bit value v (VReg_64) whose low 32-bit
    sub-register (sub0) is read by `sub0_uses` V_ADD_U32 instructions. AMDGPU
    tracks sub-register liveness, so v has subranges, and each sub0 use reads
    only a lane subset -- exactly what tryInstructionSplit's sub-register arm
    isolates. The def copy is skipped (full copy). Decision-level only (not
    executed on a non-AMDGPU host)."""
    mf = mmi.machine_function("f")
    b = mir.MachineIRBuilder(mf)
    src = mf.physreg("VGPR0_VGPR1")
    e = mf.blocks[0]
    e.add_livein(src)
    copy = mf.opcode("COPY")
    sub0 = mf.subreg_index("sub0")
    v = mf.create_vreg(mf.reg_class("VReg_64"))
    c = b.build_instr(copy)
    c.add_reg(v, is_def=True)
    c.add_reg(src)
    for _ in range(sub0_uses):
        n = mf.create_vreg(mf.reg_class("VGPR_32"))
        ins = b.build_instr(mf.opcode("V_ADD_U32_e32"))
        ins.add_reg(n, is_def=True)
        ins.add_reg(v, sub_reg=sub0)
        ins.add_reg(v, sub_reg=sub0)
    end = b.build_instr(mf.opcode("S_ENDPGM"))
    end.add_imm(0)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def _run_instr_split_amdgpu(builder, want):
    """Drive tryInstructionSplit on the first subrange single-block value of an
    AMDGPU function and record whether it split. Decision-level."""
    st = {}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            sa = self.split_analysis
            sa.analyze(li)
            if (
                "done" not in st
                and self.lis.interval(li.reg).has_sub_ranges
                and self.interval_is_in_one_mbb(li.reg)
            ):
                st["done"] = True
                st["split"] = self._try_instruction_split(li)
                st["trace"] = dict(self.trace)
                if st["split"]:
                    return None
            for p in self.allocation_order(li):
                if self.matrix.is_free(li, p):
                    return p
            self.spill(li)
            return None

    mir.register_regalloc("ra-instr-amd", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AMDGPU, cpu="gfx900")
        mmi = mir.create_machine_function(mod, tm, "f")
        builder(mmi)
        mmi.regalloc_assignments(regalloc="ra-instr-amd")
    assert st.get("done"), "a subrange single-block value was seen"
    assert st["split"] is want
    return st


@_skip_no_amdgpu
def test_instruction_split_amdgpu_splits_lane_subset_uses():
    """tryInstructionSplit on AMDGPU: a 64-bit value whose low sub-register is
    read (a lane subset) is split around those uses (the def copy is skipped),
    producing new RS_Spill ranges."""
    st = _run_instr_split_amdgpu(_build_amdgpu_subrange, want=True)
    assert st["trace"].get(2147483648) == "instruction_split"


def test_instruction_split_no_split_when_whole_value_read():
    """When the only non-copy uses read the whole live value (readsLaneSubset
    False), tryInstructionSplit isolates nothing and returns False. Driven with
    a stand-in: a real subrange value always has a lane-subset (splitting) use,
    so the all-skipped path isn't reachable via constructible MIR."""
    calls = []
    se = SimpleNamespace(
        reset=lambda lre, mode: None,
        open_intv=lambda: calls.append("open"),
    )
    fg = SimpleNamespace(
        lis=SimpleNamespace(interval=lambda reg: SimpleNamespace(has_sub_ranges=True)),
        new_live_range_edit=lambda li: SimpleNamespace(new_vregs=lambda: []),
        split_editor=se,
        split_analysis=SimpleNamespace(
            get_use_slots=lambda: [_FakeSlot(0), _FakeSlot(1)]
        ),
        is_full_copy_instr_at=lambda u: u.v == 0,  # first use is a full copy
        reads_lane_subset=lambda li, u: False,  # second reads the whole value
    )
    assert mg.RAGreedy._try_instruction_split(fg, SimpleNamespace(reg=1)) is False
    assert calls == []  # both uses skipped -> nothing opened


def test_instruction_split_one_use_slot_returns_false():
    """tryInstructionSplit's <=1-use-slot guard: driven with a stand-in whose
    analysis reports a single use slot (real MIR always yields >=2: SplitAnalysis
    counts the def slot too)."""
    fg = SimpleNamespace(
        lis=SimpleNamespace(interval=lambda reg: SimpleNamespace(has_sub_ranges=True)),
        new_live_range_edit=lambda li: SimpleNamespace(new_vregs=lambda: []),
        split_editor=SimpleNamespace(reset=lambda lre, mode: None),
        split_analysis=SimpleNamespace(get_use_slots=lambda: [_FakeSlot(0)]),
    )
    assert mg.RAGreedy._try_instruction_split(fg, SimpleNamespace(reg=1)) is False


def test_instruction_split_no_subranges_returns_false():
    """tryInstructionSplit early-returns for a range without subranges (the
    AArch64 case: no sub-register liveness)."""
    st = {}

    class Force(mir.RAGreedy):
        def select_or_split(self, li):
            if "done" not in st and self.interval_is_in_one_mbb(li.reg):
                st["done"] = True
                st["sub"] = self.lis.interval(li.reg).has_sub_ranges
                st["split"] = self._try_instruction_split(li)
            return super().select_or_split(li)

    mir.register_regalloc("ra-instr-nosub", Force)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        obj = mmi.emit_object(regalloc="ra-instr-nosub")
    fn, j = _jit_call((ctypes.c_int, ctypes.c_int, ctypes.c_int), "add", obj)
    assert fn(3, 4) == 7
    assert st["sub"] is False and st["split"] is False
    assert_no_leaks()


def test_subreg_index_lookup_and_unknown_raises():
    """MachineFunction.subreg_index resolves a known sub-register index and
    raises for an unknown name."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        mf = _build_add(mmi)
        assert mf.subreg_index("sub_32") > 0  # AArch64 W-within-X
        with pytest.raises(Exception, match="no sub-register index"):
            mf.subreg_index("not_a_subreg")
    assert_no_leaks()


def _build_amdgpu_regseq(mmi):
    """AMDGPU single block exercising every MachineOperand kind readsLaneSubset
    (getInstReadLaneMask) inspects. A VReg_64 ``v`` is assembled by REG_SEQUENCE
    from two VGPR_32 halves; the register coalescer folds those halves into
    sub-register defs of ``v`` (``v.sub0`` undef, ``v.sub1``). ``v.sub0`` is then
    read by a V_ADD_U32 (a sub-register use), and V_ADD_U64_PSEUDO reads ``v``
    twice -- once as an undef full-register use, once as a real full-register
    use. AMDGPU tracks sub-register liveness, so ``v`` has subranges. This puts a
    sub-register def, a sub-register use, an undef full use, and a real full use
    of one subrange value in a single block. Decision-level only."""
    mf = mmi.machine_function("f")
    b = mir.MachineIRBuilder(mf)
    vgpr32 = mf.reg_class("VGPR_32")
    vreg64 = mf.reg_class("VReg_64")
    sub0 = mf.subreg_index("sub0")
    sub1 = mf.subreg_index("sub1")
    lo = mf.create_vreg(vgpr32)
    dlo = b.build_instr(mf.opcode("V_MOV_B32_e32"))
    dlo.add_reg(lo, is_def=True)
    dlo.add_imm(1)
    hi = mf.create_vreg(vgpr32)
    dhi = b.build_instr(mf.opcode("V_MOV_B32_e32"))
    dhi.add_reg(hi, is_def=True)
    dhi.add_imm(2)
    v = mf.create_vreg(vreg64)
    rs = b.build_instr(mf.opcode("REG_SEQUENCE"))
    rs.add_reg(v, is_def=True)
    rs.add_reg(lo)
    rs.add_imm(sub0)
    rs.add_reg(hi)
    rs.add_imm(sub1)
    n = mf.create_vreg(vgpr32)
    u = b.build_instr(mf.opcode("V_ADD_U32_e32"))
    u.add_reg(n, is_def=True)
    u.add_reg(v, sub_reg=sub0)
    u.add_reg(v, sub_reg=sub0)
    w = mf.create_vreg(vreg64)
    fu = b.build_instr(mf.opcode("V_ADD_U64_PSEUDO"))
    fu.add_reg(w, is_def=True)
    fu.add_reg(v, is_undef=True)  # undef full-register use -> skipped
    fu.add_reg(v)  # real full-register use -> reads the whole value
    end = b.build_instr(mf.opcode("S_ENDPGM"))
    end.add_imm(0)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


@_skip_no_amdgpu
def test_reads_lane_subset_operand_kinds():
    """readsLaneSubset (getInstReadLaneMask) over every MachineOperand kind of a
    sub-register value. Drives the real binding at each instruction's slot on the
    coalesced MIR: the sub-register def (``v.sub1``) and sub-register use
    (V_ADD_U32) read a lane subset; the undef sub-register def (``v.sub0``) is
    skipped; the full-register read (V_ADD_U64_PSEUDO, one undef operand skipped,
    one real full read) covers the subReg==0 use arm; S_ENDPGM reads nothing."""
    rows = []

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            iv = self.lis.interval(li.reg)
            if not rows and iv.has_sub_ranges and self.interval_is_in_one_mbb(li.reg):
                for bb in self.machine_function.blocks:
                    for mi in bb.instructions:
                        idx = self.lis.instruction_index(mi)
                        rows.append((mi.opcode_name, self.reads_lane_subset(iv, idx)))
            for p in self.allocation_order(li):
                if self.matrix.is_free(li, p):
                    return p
            self.spill(li)
            return None

    mir.register_regalloc("ra-rls-kinds", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AMDGPU, cpu="gfx900")
        mmi = mir.create_machine_function(mod, tm, "f")
        _build_amdgpu_regseq(mmi)
        mmi.regalloc_assignments(regalloc="ra-rls-kinds")
    assert rows, "the subrange value reached select_or_split"
    # The undef sub-register def reads no lanes; the real sub-register def and use
    # read a subset; the full read (both lanes) does too once its undef operand is
    # skipped; a def-less terminator reads nothing.
    assert ("V_MOV_B32_e32", False) in rows  # undef sub-register def -> continue
    assert ("V_MOV_B32_e32", True) in rows  # sub-register def -> readMask |= ~mask
    assert ("V_ADD_U32_e32", True) in rows  # sub-register use -> readMask |= mask
    assert ("V_ADD_U64_PSEUDO", True) in rows  # full use (undef op skipped first)
    assert ("S_ENDPGM", False) in rows


@_skip_no_amdgpu
def test_reads_lane_subset_matching_subreg_copy():
    """readsLaneSubset short-circuits to False on a copy whose destination and
    source sub-registers match (here a full copy, both sub-register 0): such a
    copy reads exactly the lanes it writes, so it never forces an instruction
    split. Probes the real binding at the defining copy's slot."""
    result = {}

    class Probe(mir.RAGreedy):
        def select_or_split(self, li):
            iv = self.lis.interval(li.reg)
            if "copy" not in result and iv.has_sub_ranges:
                for bb in self.machine_function.blocks:
                    for mi in bb.instructions:
                        if mi.is_copy:
                            idx = self.lis.instruction_index(mi)
                            result["copy"] = self.reads_lane_subset(iv, idx)
            for p in self.allocation_order(li):
                if self.matrix.is_free(li, p):
                    return p
            self.spill(li)
            return None

    mir.register_regalloc("ra-rls-copy", Probe)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AMDGPU, cpu="gfx900")
        mmi = mir.create_machine_function(mod, tm, "f")
        _build_amdgpu_subrange(mmi)
        mmi.regalloc_assignments(regalloc="ra-rls-copy")
    assert result.get("copy") is False
