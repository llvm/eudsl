#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""RAILPDecomp: SSA spill-then-color decomposition on CP-SAT (report section 5.9).

Phase 1 (ILP): choose a minimum-weight set of vregs to spill so that at every
program point the number of values that must occupy a register is at most k.
Crucially this is the per-program-point (Appel-George) model, not whole-interval
spilling: a spilled value still needs a register *at each of its def/use points*
(the store source / reload), so those points count toward pressure even when the
value is spilled. Only live-*through* points of a spilled value drop to zero.
This is what makes the spill set realizable -- reloads always fit.

Phase 2 (polynomial): color the survivors greedily in definition (dominance)
order. With per-point pressure <= k on a chordal (SSA) interference graph a
valid coloring exists; a first-fit failure leaves the vreg unassigned and the
base hard-fails, surfacing the gap rather than masking it.

Scoped to a single register class (like RAILPPacking): multi-class raises.
"""

import time

from . import mir
from .mir_ilp_assign import stats_from_solver, make_solver
from .mir_ilp_base import RAILPBase, ILPSolution, ILPStats
from .mir_ilp_model import (
    build_interference,
    candidate_pregs,
    single_class_k,
    _require_ortools,
)


def interval_live_at(segments, point):
    """True if `point` falls in a half-open [start, end) segment (live-through)."""
    return any(s <= point < e for s, e in segments)


class RAILPDecomp(RAILPBase):
    def _solve(self, prob):
        k = single_class_k(prob.num_regs)
        if k is None:
            raise RuntimeError(
                "RAILPDecomp supports a single register class only; this "
                "function mixes classes"
            )

        cp = _require_ortools()
        model = cp.CpModel()
        zero = self.zero_slot_index()

        # Points where each vreg must be in a register (defs + uses). A spilled
        # value still needs a register at these points (store source / reload).
        must_reg = {}
        for v in prob.vregs:
            li = self.lis.interval(v)
            pts = {zero.distance(li.get_val_num_info(i).def_index)
                   for i in range(li.num_val_nums)}
            self.split_analysis.analyze(li)
            pts |= {zero.distance(s) for s in self.split_analysis.get_use_slots()}
            must_reg[v] = pts

        spill = {v: model.new_bool_var(f"spill_{v}")
                 for v in prob.vregs if prob.spillable[v]}

        # Per-point pressure: fixed cost (must-be-in-register values + unspillable
        # live-through) plus optional live-through spillable values.
        for p in sorted({pt for v in prob.vregs for pt in must_reg[v]}):
            fixed = 0
            optional = []
            for v in prob.vregs:
                if p in must_reg[v]:
                    fixed += 1
                elif interval_live_at(prob.intervals[v], p):
                    if v in spill:
                        optional.append(v)
                    else:
                        fixed += 1
            if fixed + len(optional) > k:
                model.add(fixed + sum(1 - spill[v] for v in optional) <= k)

        model.minimize(sum(prob.weight[v] * spill[v] for v in spill))

        solver = make_solver(cp, self.time_limit_s)
        t0 = time.perf_counter()
        status = solver.solve(model)
        wall = time.perf_counter() - t0
        stats = stats_from_solver(cp, solver, status, wall)

        if status not in (cp.OPTIMAL, cp.FEASIBLE):
            return ILPSolution(assignment={}, spilled=set(), stats=stats)

        spilled = {v for v in spill if solver.value(spill[v])}
        assignment = self._color_survivors(prob, spilled)
        return ILPSolution(assignment=assignment, spilled=spilled, stats=stats)

    @staticmethod
    def _color_survivors(prob, spilled):
        """First-fit color the non-spilled vregs in definition order (a perfect
        elimination order for straight-line SSA), respecting interference."""
        survivors = [v for v in prob.vregs if v not in spilled]
        adjacency = {v: set() for v in survivors}
        segs = {v: prob.intervals[v] for v in survivors}
        for edge in build_interference(segs):
            a, b = tuple(edge)
            adjacency[a].add(b)
            adjacency[b].add(a)
        by_def = sorted(survivors, key=lambda v: min(s for s, _ in prob.intervals[v]))
        assignment = {}
        for v in by_def:
            used = {assignment[n] for n in adjacency[v] if n in assignment}
            for preg in candidate_pregs(prob.order[v], prob.forbidden[v]):
                if preg not in used:
                    assignment[v] = preg
                    break
        return assignment


mir.RAILPDecomp = RAILPDecomp
