#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""RAILPDecomp: SSA spill-then-color decomposition on CP-SAT (report section 5.9).

Phase 1 (ILP): choose a minimum-weight set of vregs to spill so that at every
program point the number of values that must occupy a register is at most k.
This is the per-program-point (Appel-George) model, not whole-interval spilling:
a spilled value still needs a register *at each of its def/use points* (the store
source / reload), so those points count toward pressure even when the value is
spilled; only live-*through* points of a spilled value drop to zero. Pressure is
checked at every def/use point AND every live-range segment boundary, so a peak
inside a pass-through (holed) segment -- a point that is neither a def nor a use
of any value -- is not missed. This is what makes the spill set realizable.

Phase 2 (polynomial): color the survivors first-fit in a perfect elimination
order (maximum cardinality search). On the chordal (SSA) interference graph,
per-point pressure <= k guarantees a k-coloring exists and greedy in this order
finds it; a first-fit failure leaves the vreg unassigned and the base hard-fails,
surfacing the gap rather than masking it.

Scoped to a single register class (like RAILPPacking): multi-class raises.
"""

import time

from . import mir
from .mir_ilp_base import (
    RAILPBase,
    ILPSolution,
    stats_from_solver,
    make_solver,
    build_interference,
    candidate_pregs,
    single_class_k,
    _require_ortools,
)


def interval_live_at(segments, point):
    """True if `point` falls in a half-open [start, end) segment (live-through)."""
    return any(s <= point < e for s, e in segments)


def pressure_constraints(intervals, must_reg, spillable, k):
    """Yield ``(point, fixed, optional)`` for each program point where more than
    ``k`` values may need a register at once.

    * ``fixed``    -- values that occupy a register there no matter what: those
                      with a def/use at the point (``must_reg``; a spilled value
                      still needs a register for its store source / reload) plus
                      unspillable values live *through* the point.
    * ``optional`` -- spillable values live through the point; each frees its
                      register only if spilled.

    Points checked are every def/use point AND every live-range segment
    boundary, so a pressure peak inside a pass-through (holed) segment -- a point
    that is neither a def nor a use of any value -- is not missed. ``intervals``
    and ``must_reg`` share one coordinate space.
    """
    points = {pt for v in intervals for pt in must_reg[v]}
    points |= {s for v in intervals for s, _e in intervals[v]}
    for p in sorted(points):
        fixed = 0
        optional = []
        for v in intervals:
            if p in must_reg[v]:
                fixed += 1
            elif interval_live_at(intervals[v], p):
                if spillable[v]:
                    optional.append(v)
                else:
                    fixed += 1
        if fixed + len(optional) > k:
            yield p, fixed, optional


def perfect_elimination_order(adjacency):
    """A maximum-cardinality-search ordering of the interference graph.

    An SSA interference graph is chordal; greedy coloring in this order uses
    exactly max-clique-many colors, so with per-point pressure <= k the
    survivors are k-colorable. Ties are broken by vreg id for determinism.
    """
    weight = {v: 0 for v in adjacency}
    unnumbered = set(adjacency)
    order = []
    while unnumbered:
        v = max(unnumbered, key=lambda u: (weight[u], u))
        order.append(v)
        unnumbered.remove(v)
        for n in adjacency[v]:
            if n in unnumbered:
                weight[n] += 1
    return order


class RAILPDecomp(RAILPBase):
    def _solve(self, prob):
        r"""CP-SAT model (SSA spill-then-color decomposition, report section 5.9):

        Phase 1 (ILP) -- choose the spill set.

            variables   spill[v] in {0, 1}   for each spillable vreg v

            pressure    at each program point p (every def/use and every
                        live-range segment boundary), with
                          F(p) = #{v : p is a def/use of v}
                               + #{unspillable v live-through p}
                          O(p) = {spillable v live-through p}
                        require   F(p) + sum_{v in O(p)} (1 - spill[v]) <= k
                        (a spilled value still needs a register at its def/use,
                         so it counts in F there; only live-through drops to 0)

            objective   minimize  sum_v  w_v * spill[v]

        Phase 2 (polynomial) -- first-fit color the survivors in a perfect
        elimination order. On the chordal SSA interference graph, pressure <= k
        guarantees a k-coloring and greedy in this order finds it; a coloring
        failure leaves a vreg unassigned and the base hard-fails.
        """
        k = single_class_k(prob.reg_class_id, prob.num_regs)
        if k is None:
            raise RuntimeError(
                "RAILPDecomp supports a single register class only; this "
                "function mixes classes"
            )

        cp = _require_ortools()
        model = cp.CpModel()

        # Points where each vreg must be in a register (defs + uses). A spilled
        # value still needs a register at these points (store source / reload).
        # `_points_in_register` returns them in the same coordinate space as
        # prob.intervals, so the per-point pressure counts below line up.
        must_reg = {
            v: self._points_in_register(self.lis.interval(v)) for v in prob.vregs
        }

        spill = {
            v: model.new_bool_var(f"spill_{v}") for v in prob.vregs if prob.spillable[v]
        }

        for _p, fixed, optional in pressure_constraints(
            prob.intervals, must_reg, prob.spillable, k
        ):
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
        """First-fit color the non-spilled vregs in a perfect elimination order,
        respecting interference. On a chordal (SSA) interference graph with
        per-point pressure <= k this uses <= k colors; a first-fit failure leaves
        the vreg unassigned and the base hard-fails."""
        survivors = [v for v in prob.vregs if v not in spilled]
        adjacency = {v: set() for v in survivors}
        segs = {v: prob.intervals[v] for v in survivors}
        for edge in build_interference(segs):
            a, b = tuple(edge)
            adjacency[a].add(b)
            adjacency[b].add(a)
        assignment = {}
        for v in perfect_elimination_order(adjacency):
            used = {assignment[n] for n in adjacency[v] if n in assignment}
            for preg in candidate_pregs(prob.order[v], prob.forbidden[v]):
                if preg not in used:
                    assignment[v] = preg
                    break
        return assignment


mir.RAILPDecomp = RAILPDecomp
