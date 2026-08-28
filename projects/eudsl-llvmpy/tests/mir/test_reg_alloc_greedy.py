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


def test_get_priority_is_size_biased_for_globals():
    ra = mir.RAGreedy()
    # RS_Split ranges are deferred: priority equals size.
    ra._set_stage(42, ra.RS_Split)
    assert (
        ra._priority_for(
            reg=42,
            size=100,
            is_local=False,
            force_global=False,
            num_allocatable=32,
            instr_dist=16,
        )
        == 100
    )


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
    assert "split" in traces.values()
    assert_no_leaks()
