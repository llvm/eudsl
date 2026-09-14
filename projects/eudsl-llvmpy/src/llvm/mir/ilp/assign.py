#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""RAILPAssign: Goodwin-Wilken 0-1 ILP register allocation on CP-SAT.

Binary x[v,p] (vreg v -> physreg p) plus, for spillable vregs, spill[v]; exactly
one holds per vreg. Interference edges forbid two overlapping vregs sharing a
physreg. The objective is lexicographic: first minimize weighted spill cost,
then, among minimum-spill solutions, honor as many copy hints as possible
(coalescing). This is realized in a single objective by scaling the spill terms
above the maximum achievable hint reward, so a hint can never force or prevent a
spill. The canonical ILP from the literature (report section 2.2); maps directly
onto eudsl's allocation_order / matrix / live-interval segments. Unspillable
vregs get no spill variable, so they are always force-assigned (the base never
spills them).
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
    _require_ortools,
)

# Objective reward for honoring one copy hint (coalescing). A pure tie-breaker:
# _solve scales the spill terms above the maximum total hint reward, so no
# combination of honored hints can ever outweigh a spill.
HINT_BONUS = 1


class RAILPAssign(RAILPBase):
    # Whole-interval spill decisions ignore reload register pressure and are not
    # reliably realizable; hard-fail cleanly when a function needs spilling.
    realizes_spills = False

    def _solve(self, prob):
        r"""CP-SAT model (Goodwin-Wilken 0-1 assignment):

            variables   x[v, p] in {0, 1}   for each vreg v, legal candidate p
                        spill[v] in {0, 1}   for each spillable vreg v

            assignment  for each v:  sum_p x[v, p]  (+ spill[v] if spillable) = 1
                        (exactly-one; an unspillable v with no candidate makes
                         the model infeasible -> hard error)

            interference for each pair (u, w) of overlapping live ranges and each
                        shared candidate p:   x[u, p] + x[w, p] <= 1

            objective   minimize  S * sum_v w_v * spill[v]              (spills)
                                 - HINT_BONUS * sum_v x[v, hint(v)]     (coalesce)
                        with  S = HINT_BONUS * |{v : hint(v) legal}| + 1

        S scales the spill cost strictly above the maximum achievable hint
        reward, so the objective is lexicographic: minimize spills first, then
        (among minimum-spill solutions) honor as many copy hints as possible.
        """
        cp = _require_ortools()
        model = cp.CpModel()

        cands = {
            v: candidate_pregs(prob.order[v], prob.forbidden[v]) for v in prob.vregs
        }
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

        # Lexicographic objective: spilling always dominates, coalescing only
        # breaks ties. Each honored copy hint is worth HINT_BONUS, and hints
        # stack across vregs, so scale every spill term by a factor strictly
        # greater than the maximum achievable total hint reward. Then adding any
        # spill (cost >= spill_scale * 1) outweighs all hints combined, so a hint
        # can never buy a spill -- it only chooses among equal-spill solutions.
        hinted = [
            x[(v, prob.hints[v])]
            for v in prob.vregs
            if prob.hints[v] and (v, prob.hints[v]) in x
        ]
        spill_scale = HINT_BONUS * len(hinted) + 1
        terms = [
            spill_scale * prob.weight[v] * spill[v] for v in prob.vregs if v in spill
        ]
        terms += [-HINT_BONUS * h for h in hinted]
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
