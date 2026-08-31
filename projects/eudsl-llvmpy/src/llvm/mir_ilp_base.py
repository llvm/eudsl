#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Shared base for the ILP register allocators.

The LLVM driver drives allocation one live interval at a time
(``select_or_split``), but an ILP wants the whole function at once. RAILPBase
bridges the two: it collects every seeded vreg (LLVM enqueues them all before
the first dequeue), solves one global model on the first ``dequeue`` via the
subclass ``_solve``, then answers each ``select_or_split`` from the cached
solution -- always re-checking the live-register matrix and falling back to
first-free-or-spill. That fallback makes the result a valid allocation even
when a model's interference encoding is incomplete (aliasing) or a vreg is a
post-solve spill product not present in the solution.
"""

from dataclasses import dataclass, field

from . import mir
from .mir_ilp_model import scale_weight


@dataclass
class ILPProblem:
    """Everything a ``_solve`` needs, extracted from the framework once."""

    vregs: list                 # list[int] vreg ids, in seed order
    intervals: dict             # vreg -> list[(start:int, end:int)] half-open
    order: dict                 # vreg -> list[preg] legal candidates (class order)
    forbidden: dict             # vreg -> set[preg] fixed/regmask-interfered pregs
    weight: dict                # vreg -> int scaled spill weight
    hints: dict                 # vreg -> preg copy hint (0 if none)
    num_regs: dict              # vreg -> int allocatable regs in its class
    spillable: dict             # vreg -> bool (LLVM li.is_spillable)


@dataclass
class ILPStats:
    status: str = "unsolved"
    objective: float = 0.0
    best_bound: float = 0.0
    wall_time_s: float = 0.0

    @property
    def gap(self):
        """Relative optimality gap in [0, 1], or None if not solved.

        0.0 means proven optimal (bound met the incumbent objective)."""
        if self.status not in ("OPTIMAL", "FEASIBLE"):
            return None
        if self.objective == 0.0:
            return 0.0
        return max(0.0, (self.objective - self.best_bound) / abs(self.objective))


@dataclass
class ILPSolution:
    assignment: dict            # vreg -> preg
    spilled: set                # set[int] vregs to spill
    stats: ILPStats = field(default_factory=ILPStats)


class RAILPBase(mir.RegAllocBase):
    # CP-SAT wall-clock budget per solve; on timeout the incumbent is used and
    # the optimality gap is reported (may be > 0).
    time_limit_s = 10.0

    # Whether this allocator realizes spills through the framework's inline
    # spiller. Whole-interval models (assign, packing) set this False: their
    # minimum-spill solutions ignore reload register pressure and are not
    # reliably realizable, so they hard-fail cleanly when a function needs
    # spilling (register-fitting functions only). The per-point decomposition
    # model, which accounts for reloads, sets this True.
    realizes_spills = True

    def __init__(self):
        super().__init__()
        self._pending = []
        self._solved = False
        self._solution = {}
        self._spill = set()
        self._problem_vregs = set()
        self.solve_stats = ILPStats()

    # -- queue: collect seeds, no priority ---------------------------------
    def enqueue(self, reg):
        self._pending.append(reg)

    def dequeue(self):
        if not self._solved:
            self._solve_all()
        if not self._pending:
            return None
        return self._pending.pop(0)

    # -- one global solve on first dequeue ---------------------------------
    def _solve_all(self):
        self._solved = True
        if not self._pending:
            return
        self._problem_vregs = set(self._pending)
        problem = self._build_problem(self._pending)
        solution = self._solve(problem)
        if solution.spilled and not self.realizes_spills:
            raise RuntimeError(
                f"{type(self).__name__} does not realize spills, but the ILP "
                f"requires spilling {len(solution.spilled)} vreg(s); this "
                f"allocator supports register-fitting functions only"
            )
        self._solution = solution.assignment
        self._spill = set(solution.spilled)
        self.solve_stats = solution.stats

    def _build_problem(self, vregs):
        intervals, order, forbidden = {}, {}, {}
        weight, hints, num_regs, spillable = {}, {}, {}, {}
        zero = self.zero_slot_index()
        free = mir.InterferenceKind.IK_Free
        virt = mir.InterferenceKind.IK_VirtReg
        for reg in vregs:
            li = self.lis.interval(reg)
            intervals[reg] = [
                (zero.distance(s.start), zero.distance(s.end)) for s in li.segments()
            ]
            cls = self.reg_class(reg)
            num_regs[reg] = self.num_allocatable_regs(cls)
            allowed, forb = [], set()
            seen = set()
            for preg in self.allocation_order(li):
                # allocation_order can list a physreg more than once (LLVM
                # front-loads copy-hint regs), which would create duplicate x
                # variables; keep the first occurrence only.
                if preg in seen:
                    continue
                seen.add(preg)
                kind = self.matrix.check_interference(li, preg)
                if kind == free or kind == virt:
                    allowed.append(preg)
                else:  # IK_RegUnit / IK_RegMask: a physical/clobber conflict
                    forb.add(preg)
            order[reg] = allowed
            forbidden[reg] = forb
            weight[reg] = scale_weight(li.weight)
            hints[reg] = self.simple_hint(reg)
            spillable[reg] = li.is_spillable
        return ILPProblem(
            vregs=list(vregs), intervals=intervals, order=order,
            forbidden=forbidden, weight=weight, hints=hints, num_regs=num_regs,
            spillable=spillable,
        )

    # -- per-interval answer: cached solution, no ILP-masking fallback -----
    def select_or_split(self, li):
        reg = li.reg
        if reg not in self._problem_vregs:
            # A reload/def vreg minted by self.spill after the solve: the ILP
            # never saw it. Color it first-free. A spill product cannot itself
            # be spilled (LLVM's InlineSpiller aborts on that), so if no
            # register is free here the model under-spilled -- reload register
            # pressure exceeds capacity at this use point. Fail loudly.
            for cand in self.allocation_order(li):
                if self.matrix.is_free(li, cand):
                    return cand
            raise RuntimeError(
                f"reload vreg {reg} has no free register; the ILP model "
                f"under-spilled (whole-interval spilling ignores reload "
                f"register pressure at use points)"
            )
        # An original vreg the ILP solved: it must have a valid decision. No
        # greedy fallback -- a missing or infeasible assignment is a model bug
        # and must fail loudly rather than be silently repaired (which would
        # make the ILP-vs-greedy comparison meaningless).
        if reg in self._spill:
            self._spill_or_fail(li)
            return None
        preg = self._solution.get(reg)
        if preg is None:
            raise RuntimeError(
                f"ILP produced no assignment or spill decision for vreg {reg}"
            )
        if not self.matrix.is_free(li, preg):
            raise RuntimeError(
                f"ILP assigned vreg {reg} -> physreg {preg}, but the "
                f"live-register matrix reports it is not free (model "
                f"interference bug)"
            )
        return preg

    def _spill_or_fail(self, li):
        # LLVM aborts if asked to spill an unspillable interval (weight == inf).
        # A well-formed model never spills one; if it did, fail loudly.
        if not li.is_spillable:
            raise RuntimeError(
                f"ILP chose to spill vreg {li.reg}, but it is not spillable"
            )
        self.spill(li)

    def _solve(self, problem):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError
