#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ILP register allocators: pure-helper unit tests + end-to-end + comparison."""

import pytest
import llvm
from llvm import mir_ilp_model as model
from llvm import ir, jit, mir
from llvm.mir_ilp_base import RAILPBase, ILPSolution, ILPStats
from llvm.testing import assert_no_leaks

# The ILP allocators need OR-Tools (the optional `ilp` extra); skip the whole
# module where it is not installed rather than erroring at import.
pytest.importorskip("ortools", reason="ortools not installed (eudsl-llvmpy[ilp])")

_AARCH64_LINUX = "aarch64-unknown-linux-gnu"

aarch64 = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked",
)


def _build_add(mmi):
    """Pressure-free add: two argument copies + an ADD + a return copy."""
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


_HP_N = 48


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


def test_require_ortools_returns_cp_model():
    cp = model._require_ortools()
    assert hasattr(cp, "CpModel")


def test_scale_weight_is_positive_int():
    assert model.scale_weight(0.0) >= 1
    assert isinstance(model.scale_weight(1.5), int)
    assert model.scale_weight(2.0) > model.scale_weight(1.0)


def test_build_interference_overlap_and_disjoint():
    # v1 [0,4) overlaps v2 [2,6); v3 [6,8) is disjoint from both.
    intervals = {1: [(0, 4)], 2: [(2, 6)], 3: [(6, 8)]}
    edges = model.build_interference(intervals)
    assert frozenset((1, 2)) in edges
    assert frozenset((1, 3)) not in edges
    assert frozenset((2, 3)) not in edges  # [2,6) and [6,8) touch but half-open


def test_build_interference_multi_segment_holes():
    # v1 is live [0,2) and [8,10) (a hole); v2 [3,7) fits in the hole -> no edge.
    intervals = {1: [(0, 2), (8, 10)], 2: [(3, 7)]}
    assert model.build_interference(intervals) == set()


def test_compact_time_axis():
    intervals = {1: [(0, 4)], 2: [(4, 10)]}
    mapping, n = model.compact_time_axis(intervals)
    assert n == 3  # points {0, 4, 10}
    assert mapping[0] == 0 and mapping[4] == 1 and mapping[10] == 2


def test_candidate_pregs_filters_forbidden():
    assert model.candidate_pregs([10, 11, 12], {11}) == [10, 12]


def test_single_class_k():
    assert model.single_class_k({1: 32, 2: 32}) == 32
    assert model.single_class_k({1: 32, 2: 16}) is None
    assert model.single_class_k({}) is None


def test_ilp_stats_gap():
    assert ILPStats(status="INFEASIBLE").gap is None
    assert ILPStats(status="OPTIMAL", objective=0.0).gap == 0.0
    s = ILPStats(status="FEASIBLE", objective=10.0, best_bound=8.0)
    assert abs(s.gap - 0.2) < 1e-9


def _mk_problem(**overrides):
    from llvm.mir_ilp_base import ILPProblem

    base = dict(
        vregs=[1, 2, 3],
        intervals={1: [(0, 6)], 2: [(0, 6)], 3: [(0, 6)]},
        order={1: [10, 11], 2: [10, 11], 3: [10, 11]},
        forbidden={1: set(), 2: set(), 3: set()},
        weight={1: 5, 2: 5, 3: 5},
        hints={1: 0, 2: 0, 3: 0},
        num_regs={1: 2, 2: 2, 3: 2},
        spillable={1: True, 2: True, 3: True},
    )
    base.update(overrides)
    return ILPProblem(**base)


def test_packing_solve_standalone_multiclass_raises():
    with pytest.raises(RuntimeError, match="single register class"):
        mir.RAILPPacking()._solve(_mk_problem(num_regs={1: 8, 2: 16, 3: 8}))


def test_packing_solve_standalone_spill_and_degenerate_segment():
    # One spill forced; vreg 3 also carries a zero-length segment (skipped).
    sol = mir.RAILPPacking()._solve(
        _mk_problem(
            intervals={1: [(0, 6)], 2: [(0, 6)], 3: [(0, 6), (6, 6)]},
        )
    )
    assert sol.stats.status in ("OPTIMAL", "FEASIBLE")
    assert len(sol.spilled) == 1


def test_alloc_result_weighted_spill_cost():
    from llvm import mir_ilp_compare as compare

    r = compare.AllocResult(
        name="x",
        valid=True,
        spills=[2, 3],
        weight={1: 10, 2: 5, 3: 7},
        copies_remaining=1,
        wall_time_s=0.1,
        gap=0.0,
        error=None,
    )
    assert r.weighted_spill_cost == 12  # 5 + 7
    assert r.num_spills == 2


def test_format_table_contains_rows_and_header():
    from llvm import mir_ilp_compare as compare

    rows = [
        compare.AllocResult("greedy", True, [], {}, 0, None, None, None),
        compare.AllocResult("ilp-pack", True, [1], {1: 4}, 0, 0.2, 0.0, None),
        compare.AllocResult(
            "ilp-pack2", False, [], {}, 0, None, None, "register-fitting only"
        ),
    ]
    text = compare.format_table("add", rows)
    assert "add" in text
    assert "greedy" in text
    assert "ilp-pack" in text
    assert "gap" in text.lower()
    assert "hard-fail" in text or "register-fitting" in text


def _greedy_color(prob, exclude=frozenset()):
    """Valid first-fit coloring over interference, skipping `exclude` vregs."""
    keep = [v for v in prob.vregs if v not in exclude]
    adj = {v: set() for v in keep}
    for edge in model.build_interference({v: prob.intervals[v] for v in keep}):
        a, b = tuple(edge)
        adj[a].add(b)
        adj[b].add(a)
    asg = {}
    for v in keep:
        used = {asg[n] for n in adj[v] if n in asg}
        for p in model.candidate_pregs(prob.order[v], prob.forbidden[v]):
            if p not in used:
                asg[v] = p
                break
    return asg


class _StubValidColoring(RAILPBase):
    """Produces a valid coloring in _solve, exercising the cached-return path."""

    def _solve(self, prob):
        return ILPSolution(
            assignment=_greedy_color(prob), spilled=set(), stats=ILPStats("OPTIMAL")
        )


class _StubSpillUnspillable(RAILPBase):
    """Spills a vreg that is not spillable -> base must raise (never abort)."""

    def _solve(self, prob):
        target = min(v for v in prob.vregs if not prob.spillable[v])
        return ILPSolution(_greedy_color(prob, {target}), {target}, ILPStats("OPTIMAL"))


@aarch64
def test_base_hard_fails_spilling_unspillable():
    mir.register_regalloc("ilp-stub-unspill", _StubSpillUnspillable)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        with pytest.raises(RuntimeError, match="not spillable"):
            mmi.regalloc_assignments(regalloc="ilp-stub-unspill")


class _StubMissingAssignment(RAILPBase):
    """Omits one vreg's decision entirely -> base must raise 'no assignment'."""

    def _solve(self, prob):
        target = max(prob.vregs)
        return ILPSolution(_greedy_color(prob, {target}), set(), ILPStats("OPTIMAL"))


@aarch64
def test_base_hard_fails_on_missing_decision():
    mir.register_regalloc("ilp-stub-missing", _StubMissingAssignment)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        with pytest.raises(RuntimeError, match="no assignment or spill"):
            mmi.regalloc_assignments(regalloc="ilp-stub-missing")


@aarch64
def test_base_returns_cached_valid_assignment():
    mir.register_regalloc("ilp-stub-valid", _StubValidColoring)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        result = mmi.regalloc_assignments(regalloc="ilp-stub-valid")
        assignments = dict(result.assignments)
        spilled = list(result.spilled)
    v0, v1, v2 = sorted(assignments)
    assert assignments[v0] != assignments[v1]  # simultaneously live -> distinct
    assert spilled == []
    assert_no_leaks()


class _StubSameReg(RAILPBase):
    """Assigns every vreg the same physreg (a candidate legal for all). Two
    simultaneously-live vregs then collide, so the base must hard-fail on the
    "not free" branch rather than silently repair it."""

    def _solve(self, prob):
        common = set(prob.order[prob.vregs[0]])
        for v in prob.vregs[1:]:
            common &= set(prob.order[v])
        reg = min(common)
        asg = {v: reg for v in prob.vregs}
        return ILPSolution(assignment=asg, spilled=set(), stats=ILPStats("OPTIMAL"))


@aarch64
def test_base_hard_fails_on_infeasible_assignment():
    mir.register_regalloc("ilp-stub-bad", _StubSameReg)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        with pytest.raises(RuntimeError, match="not free"):
            mmi.regalloc_assignments(regalloc="ilp-stub-bad")


class _StubSpillOne(RAILPBase):
    """Spills one spillable vreg and colors the rest (greedy over interference),
    exercising the spill-routing + reload-vreg first-free path."""

    def _solve(self, prob):
        target = min(v for v in prob.vregs if prob.spillable[v])
        edges = model.build_interference(prob.intervals)
        adj = {v: set() for v in prob.vregs}
        for edge in edges:
            a, b = tuple(edge)
            adj[a].add(b)
            adj[b].add(a)
        asg = {}
        for v in prob.vregs:
            if v == target:
                continue
            used = {asg[n] for n in adj[v] if n in asg}
            for p in model.candidate_pregs(prob.order[v], prob.forbidden[v]):
                if p not in used:
                    asg[v] = p
                    break
        return ILPSolution(assignment=asg, spilled={target}, stats=ILPStats("OPTIMAL"))


@aarch64
def test_base_stub_routes_spill():
    mir.register_regalloc("ilp-stub-spill", _StubSpillOne)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        result = mmi.regalloc_assignments(regalloc="ilp-stub-spill")
        spilled = list(result.spilled)
    assert len(spilled) >= 1
    assert_no_leaks()


@aarch64
def test_ilp_packing_pressure_free_valid():
    mir.register_regalloc("ilp-pack-t", mir.RAILPPacking)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        result = mmi.regalloc_assignments(regalloc="ilp-pack-t")
        assignments = dict(result.assignments)
        spilled = list(result.spilled)
    v0, v1, v2 = sorted(assignments)
    assert assignments[v0] != assignments[v1]
    assert spilled == []
    assert_no_leaks()


@aarch64
def test_ilp_packing_high_pressure_hard_fails():
    # Whole-interval spill decisions ignore reload pressure and are not reliably
    # realizable, so RAILPPacking refuses to spill: it hard-fails cleanly (never
    # crashes) when a function needs spilling.
    mir.register_regalloc("ilp-pack-hp", mir.RAILPPacking)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "hp")
        _build_high_pressure(mmi)
        with pytest.raises(RuntimeError, match="register-fitting"):
            mmi.regalloc_assignments(regalloc="ilp-pack-hp")


@aarch64
def test_comparison_low_pressure_packing_valid_and_optimal():
    mir.register_regalloc("cmp-basic", mir.BasicRegAlloc)
    mir.register_regalloc("cmp-pack", mir.RAILPPacking)
    for alloc in ["greedy", "cmp-basic", "cmp-pack"]:
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "add")
            _build_add(mmi)
            result = mmi.regalloc_assignments(regalloc=alloc)
            assignments = dict(result.assignments)
            spilled = list(result.spilled)
        assert spilled == [], f"{alloc} spilled on a pressure-free function"
        v0, v1, _ = sorted(assignments)
        assert assignments[v0] != assignments[v1], f"{alloc} gave a bad coloring"
    # RAILPPacking proves optimality (gap 0) on this trivial function.
    stats = RAILPBase.last_stats["RAILPPacking"]
    assert stats.status in ("OPTIMAL", "FEASIBLE")
    assert stats.gap == 0.0
    assert_no_leaks()


class _StubCapturePoints(RAILPBase):
    """Captures ``_points_in_register`` for every vreg alongside the problem's
    intervals, then colors validly so the run completes. Lets a test assert the
    helper's points land in the same coordinate space as ``ILPProblem.intervals``.
    """

    captured_points = {}
    captured_intervals = {}

    def _solve(self, prob):
        _StubCapturePoints.captured_points = {
            v: self._points_in_register(self.lis.interval(v)) for v in prob.vregs
        }
        _StubCapturePoints.captured_intervals = dict(prob.intervals)
        return ILPSolution(
            assignment=_greedy_color(prob), spilled=set(), stats=ILPStats("OPTIMAL")
        )


@aarch64
def test_points_in_register_aligns_with_intervals():
    mir.register_regalloc("ilp-stub-points", _StubCapturePoints)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        mmi.regalloc_assignments(regalloc="ilp-stub-points")
    points = _StubCapturePoints.captured_points
    intervals = _StubCapturePoints.captured_intervals
    assert set(points) == set(intervals)
    for v, pts in points.items():
        assert pts, f"vreg {v} has no must-be-in-register point"
        assert all(isinstance(p, int) for p in pts)
        # The def point is the live range's start (same coordinate space), and
        # every must-reg point falls within the vreg's live span.
        start = min(s for s, _ in intervals[v])
        end = max(e for _, e in intervals[v])
        assert min(pts) == start
        assert all(start <= p <= end for p in pts)
    assert_no_leaks()
