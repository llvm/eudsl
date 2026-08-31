#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""RAILPAssign: Goodwin-Wilken 0-1 ILP register allocation on CP-SAT.

Binary x[v,p] (vreg v -> physreg p) plus, for spillable vregs, spill[v]; exactly
one holds per vreg. Interference edges forbid two overlapping vregs sharing a
physreg. Objective minimizes weighted spill cost, minus a unit bonus for
honoring copy hints (coalescing). The canonical ILP from the literature (report
section 2.2); maps directly onto eudsl's allocation_order / matrix / live-
interval segments. Unspillable vregs get no spill variable, so they are always
force-assigned (the base never spills them).
"""

import time

from . import mir
from .mir_ilp_base import RAILPBase, ILPSolution, ILPStats
from .mir_ilp_model import (
    HINT_BONUS,
    build_interference,
    candidate_pregs,
    _require_ortools,
)


def stats_from_solver(cp, solver, status, wall):
    """Build an ILPStats from a finished CP-SAT solve (shared by all models)."""
    name = {cp.OPTIMAL: "OPTIMAL", cp.FEASIBLE: "FEASIBLE",
            cp.INFEASIBLE: "INFEASIBLE"}.get(status, "UNKNOWN")
    solved = name in ("OPTIMAL", "FEASIBLE")
    return ILPStats(
        status=name,
        objective=solver.objective_value if solved else 0.0,
        best_bound=solver.best_objective_bound if solved else 0.0,
        wall_time_s=wall,
    )


def make_solver(cp, time_limit_s):
    """A CP-SAT solver configured for reproducible, deterministic search.

    A fixed seed and a single worker make allocation reproducible across runs
    (the study compares solutions, and reproducible compiler output matters);
    the problems here are small enough that single-threaded search is fine.
    """
    solver = cp.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.random_seed = 0
    solver.parameters.num_workers = 1
    return solver


class RAILPAssign(RAILPBase):
    # Whole-interval spill decisions ignore reload register pressure and are not
    # reliably realizable; hard-fail cleanly when a function needs spilling.
    realizes_spills = False

    def _solve(self, prob):
        cp = _require_ortools()
        model = cp.CpModel()

        cands = {v: candidate_pregs(prob.order[v], prob.forbidden[v])
                 for v in prob.vregs}
        x = {}
        spill = {}
        for v in prob.vregs:
            literals = []
            for p in cands[v]:
                var = model.new_bool_var(f"x_{v}_{p}")
                x[(v, p)] = var
                literals.append(var)
            if prob.spillable[v]:
                spill[v] = model.new_bool_var(f"spill_{v}")
                literals.append(spill[v])
            # Exactly one of {assigned physreg, spill}. For an unspillable vreg
            # this forces a physreg; an empty literal list (no legal candidate)
            # is infeasible and the solve fails -- surfaced as a hard error.
            model.add_exactly_one(literals)

        for edge in build_interference(prob.intervals):
            u, w = tuple(edge)
            for p in set(cands[u]) & set(cands[w]):
                model.add(x[(u, p)] + x[(w, p)] <= 1)

        terms = [prob.weight[v] * spill[v] for v in prob.vregs if v in spill]
        for v in prob.vregs:
            hint = prob.hints[v]
            if hint and (v, hint) in x:
                terms.append(-HINT_BONUS * x[(v, hint)])
        model.minimize(sum(terms))

        solver = make_solver(cp, self.time_limit_s)
        t0 = time.perf_counter()
        status = solver.solve(model)
        wall = time.perf_counter() - t0
        stats = stats_from_solver(cp, solver, status, wall)

        assignment, spilled = {}, set()
        if status in (cp.OPTIMAL, cp.FEASIBLE):
            for v in prob.vregs:
                if v in spill and solver.value(spill[v]):
                    spilled.add(v)
                    continue
                for p in cands[v]:
                    if solver.value(x[(v, p)]):
                        assignment[v] = p
                        break
        return ILPSolution(assignment=assignment, spilled=spilled, stats=stats)


mir.RAILPAssign = RAILPAssign
