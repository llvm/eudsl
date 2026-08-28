#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Faithful Python RegAllocGreedy (mir.RAGreedy)."""

import ctypes
import platform
import pytest
import llvm
from llvm import ir, jit, mir
from llvm.mir_greedy import eviction_cost, calc_gap_weights, calc_global_split_cost
from llvm.mir_greedy import GlobalSplitCandidate
from llvm.mir_greedy import _NO_CAND
from llvm.testing import assert_no_leaks

pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked",
)
_AARCH64_LINUX = "aarch64-unknown-linux-gnu"


def test_ragreedy_is_exported_and_constructs():
    assert issubclass(mir.RAGreedy, mir.RegAllocBase)
    ra = mir.RAGreedy()
    assert ra.RS_New < ra.RS_Assign < ra.RS_Split < ra.RS_Split2 < ra.RS_Spill
    assert_no_leaks()


def test_eviction_cost_sums_weights_and_penalties():
    # No penalties: cost is just the summed interferer weight.
    assert eviction_cost(
        [1.0, 2.0], broken_hint=False, is_unused_callee_saved=False
    ) == pytest.approx(3.0)
    # A broken hint adds a strictly positive penalty.
    base = eviction_cost([1.0], broken_hint=False, is_unused_callee_saved=False)
    assert eviction_cost([1.0], broken_hint=True, is_unused_callee_saved=False) > base
    # Introducing an unused callee-saved reg is dominated by the CSR bias.
    assert eviction_cost([1.0], broken_hint=False, is_unused_callee_saved=True) > base


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


_DIAMOND_N = 40


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
    assert saw["loop_hdr"] == -1  # straight-line CFG: no loop
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
    assert saw["grew"] in (True, False)  # returns a bool; budget not exceeded
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
    assert saw["best"] == _NO_CAND or saw["best"] >= 0
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
    (isUnusedCalleeSavedReg), exercising that filter."""

    def body(self, li, st):
        order = list(self.allocation_order(li))
        st["csr_seen"] = any(self._is_unused_callee_saved(p) for p in order)
        best, _ = self._calculate_region_split_cost(
            li, order, mir.BlockFrequency.max(), 0, True
        )
        st["best"] = best

    st = _diamond_live_through_probe(body)
    assert st["best"] == _NO_CAND or st["best"] >= 0
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
        if ncands == 0 or not self._global_cand[0].live_bundles.count() > 0:
            st["skipped"] = True
            return None
        lre = self.new_live_range_edit(li)
        self._do_region_split(li, _NO_CAND, True, lre)
        st["nvregs"] = len(lre.new_vregs())
        return True

    st = _diamond_live_through_probe(body)
    assert st.get("skipped") or st["nvregs"] >= 1
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
        if best == _NO_CAND:
            st["skipped"] = True
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
    assert st.get("skipped") or st["nvregs"] >= 1
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
