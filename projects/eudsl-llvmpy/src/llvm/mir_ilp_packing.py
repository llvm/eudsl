#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""RAILPPacking: 2D no-overlap rectangle-packing register allocation on CP-SAT.

A port of claude-compiler's integrated_solver model. Each vreg gets an integer
register variable whose domain is {compact physreg indices} + {its own private
memory slot}; a value in the memory region means spilled. For each live segment
the vreg contributes a rectangle (time interval x a width-1 register interval),
and add_no_overlap_2d forbids two vregs sharing a register over overlapping
time. Objective: minimize weighted spills.

Scoped to single-register-class functions: the flat register axis cannot model
AArch64 aliasing / differing widths (report section 4.3). Multi-class functions
raise -- there is no greedy fallback to defer to. Each spilled vreg gets its own
memory slot so spills never falsely interfere. Unspillable vregs get no memory
value in their domain, forcing a physreg assignment.
"""

import time

from . import mir
from .mir_ilp_base import (
    RAILPBase,
    ILPSolution,
    stats_from_solver,
    make_solver,
    candidate_pregs,
    compact_time_axis,
    single_class_k,
    _require_ortools,
)


class RAILPPacking(RAILPBase):
    # Whole-interval spill decisions ignore reload register pressure and are not
    # reliably realizable; hard-fail cleanly when a function needs spilling.
    realizes_spills = False

    def _solve(self, prob):
        r"""CP-SAT model (single register class, k = #allocatable regs):

            variables   r_v in D_v  for each vreg v
                        D_v = {0..P-1}                 (dense physreg indices)
                              U {P + i}  if v spillable (private memory slot)
                        where P = |union of legal physregs across all vregs|

            rectangles  for each live segment [s, e) of v, one 2D box
                        x-extent [t(s), t(e))   (compacted time axis)
                        y-extent [r_v, r_v + 1) (width 1, at v's register)

            constraint  no_overlap_2d(all boxes)
                        => two vregs live at the same time cannot share a
                           register index; distinct memory slots (P + i) never
                           collide, so spilled values never falsely interfere

            spill flag  b_v = 1  <=>  r_v >= P   (v placed in memory)
            objective   minimize  sum_v  w_v * b_v   (weighted spill cost)

        Unspillable vregs get no memory value in D_v, forcing r_v < P.
        """
        if single_class_k(prob.reg_class_id, prob.num_regs) is None:
            raise RuntimeError(
                "RAILPPacking supports a single register class only; this "
                "function mixes classes (the flat register axis cannot model "
                "aliasing)"
            )

        cp = _require_ortools()
        model = cp.CpModel()

        cands = {
            v: candidate_pregs(prob.order[v], prob.forbidden[v]) for v in prob.vregs
        }
        # Register axis: union of legal physregs across vregs -> dense indices.
        pregs = sorted({p for v in prob.vregs for p in cands[v]})
        preg_index = {p: i for i, p in enumerate(pregs)}
        n_pregs = len(pregs)
        time_map, _ = compact_time_axis(prob.intervals)

        reg = {}
        x_ivs, y_ivs = [], []
        for i, v in enumerate(prob.vregs):
            allowed = [preg_index[p] for p in cands[v]]
            mem_id = n_pregs + i  # private memory slot -> spills never conflict
            values = allowed + [mem_id] if prob.spillable[v] else allowed
            reg[v] = model.new_int_var_from_domain(
                cp.Domain.from_values(values), f"reg_{v}"
            )
            for start, end in prob.intervals[v]:
                s, e = time_map[start], time_map[end]
                if e <= s:
                    continue
                x_ivs.append(model.new_fixed_size_interval_var(s, e - s, f"x_{v}_{s}"))
                y_ivs.append(model.new_fixed_size_interval_var(reg[v], 1, f"y_{v}_{s}"))
        model.add_no_overlap_2d(x_ivs, y_ivs)

        terms = []
        for i, v in enumerate(prob.vregs):
            if not prob.spillable[v]:
                continue
            is_mem = model.new_bool_var(f"is_mem_{v}")
            model.add(reg[v] >= n_pregs).only_enforce_if(is_mem)
            model.add(reg[v] < n_pregs).only_enforce_if(~is_mem)
            terms.append(prob.weight[v] * is_mem)
        model.minimize(sum(terms))

        solver = make_solver(cp, self.time_limit_s)
        t0 = time.perf_counter()
        status = solver.solve(model)
        wall = time.perf_counter() - t0
        stats = stats_from_solver(cp, solver, status, wall)

        assignment, spilled = {}, set()
        if status in (cp.OPTIMAL, cp.FEASIBLE):
            for v in prob.vregs:
                idx = solver.value(reg[v])
                if idx >= n_pregs:
                    spilled.add(v)
                else:
                    assignment[v] = pregs[idx]
        return ILPSolution(assignment=assignment, spilled=spilled, stats=stats)


mir.RAILPPacking = RAILPPacking
