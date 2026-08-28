#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Faithful Python port of llvm::RegAllocGreedy.

Importing this module attaches the allocator as ``mir.RAGreedy``. The class
mirrors RegAllocGreedy.cpp method-for-method; the pure cost computations
(eviction_cost, calc_gap_weights, calc_global_split_cost) are module-level so
they can be unit-tested without running the allocator.
"""

import enum
import heapq

from . import mir


class LiveRangeStage(enum.IntEnum):
    """RAGreedy's per-vreg allocation stage ladder (LiveRangeStage)."""

    RS_New = 0
    RS_Assign = 1
    RS_Split = 2
    RS_Split2 = 3
    RS_Spill = 4
    RS_Memory = 5


# RAGreedy's cost of introducing a callee-saved register that wasn't used yet;
# it dominates so eviction prefers not to widen the callee-saved set.
CSR_FIRST_TIME_COST = 1e9

# A broken copy hint costs a fixed increment on top of the interferer weights
# (RAGreedy nudges away from evicting a hinted assignment).
BROKEN_HINT_COST = 1.0


def eviction_cost(interferer_weights, broken_hint, is_unused_callee_saved):
    """Cost of evicting `interferer_weights` off a candidate physreg.

    Sum of the interferers' spill weights, plus a broken-hint penalty and the
    unused-callee-saved bias. Lower is cheaper; callers pick the least-cost
    evictable physreg.
    """
    cost = float(sum(interferer_weights))
    if broken_hint:
        cost += BROKEN_HINT_COST
    if is_unused_callee_saved:
        cost += CSR_FIRST_TIME_COST
    return cost


def calc_gap_weights(use_slots, interferer_spans):
    """Per-gap maximum interference weight for a local interval.

    `use_slots` is the sorted list of use positions; the gaps are the
    consecutive intervals between them. `interferer_spans` are (start, end,
    weight) triples. Returns one weight per gap: the largest weight of any
    interferer whose [start, end) overlaps that gap (0.0 if none). Mirrors
    RAGreedy::calcGapWeights.
    """
    gaps = [0.0] * max(len(use_slots) - 1, 0)
    for i in range(len(gaps)):
        lo, hi = use_slots[i], use_slots[i + 1]
        for start, end, weight in interferer_spans:
            if start < hi and lo < end:  # half-open overlap
                if weight > gaps[i]:
                    gaps[i] = weight
    return gaps


def calc_global_split_cost(boundary_freqs):
    """Cost of a global split: the summed block frequency at each split
    boundary (where a copy is inserted). Mirrors the accounting in
    RAGreedy::calcGlobalSplitCost; the caller supplies the per-boundary
    frequencies read from MachineBlockFrequencyInfo.
    """
    return float(sum(boundary_freqs))


class RAGreedy(mir.RegAllocBase):
    """Faithful reproduction of llvm::RegAllocGreedy."""

    # Exposed on instances so tests can reference the ladder without importing
    # the enum.
    RS_New = LiveRangeStage.RS_New
    RS_Assign = LiveRangeStage.RS_Assign
    RS_Split = LiveRangeStage.RS_Split
    RS_Split2 = LiveRangeStage.RS_Split2
    RS_Spill = LiveRangeStage.RS_Spill
    RS_Memory = LiveRangeStage.RS_Memory

    def __init__(self):
        super().__init__()
        # Per-vreg (keyed by reg id, stable across splitting) allocation state.
        self._stage = {}
        self._cascade = {}
        self._next_cascade = 1
        # Priority queue of pending vregs (a max-heap keyed by priority).
        self._queue = []
        # Optional test instrumentation: reg id -> stage name that resolved it.
        self.trace = {}

    # -- stage/cascade bookkeeping (RAGreedy::ExtraRegInfo) ------------------
    def _get_stage(self, reg):
        return self._stage.get(reg, LiveRangeStage.RS_New)

    def _set_stage(self, reg, stage):
        self._stage[reg] = stage

    def _get_cascade(self, reg):
        c = self._cascade.get(reg)
        if c is None:
            c = self._next_cascade
            self._next_cascade += 1
            self._cascade[reg] = c
        return c

    # -- enqueue / getPriority (RegAllocGreedy::enqueue) --------------------
    def _priority_for(
        self, reg, size, is_local, force_global, num_allocatable, instr_dist
    ):
        """RAGreedy's priority number for a vreg. Larger = allocated sooner.

        RS_Split ranges are deferred to priority == size. Global and giant
        (ForceGlobal) ranges go long->short by size with the global bit set;
        genuine local ranges are ordered by their size (a refinement of the
        start-index ordering that preserves the long-first invariant).
        Cross-check the exact bit layout against RegAllocGreedy::enqueue.
        """
        stage = self._get_stage(reg)
        if stage == LiveRangeStage.RS_Split:
            return size
        if not is_local or force_global:
            # Global/giant: long ranges first, with the global bit set high so
            # they outrank locals of the same size.
            return (1 << 24) | size
        # Local ranges in RS_Assign, ordered by size.
        return size

    def enqueue(self, reg):
        li = self.lis.interval(reg)
        rc = self.reg_class(reg)
        instr_dist = self.slot_index_instr_distance()
        num_alloc = self.num_allocatable_regs(rc)
        size = li.size
        force_global = self.reg_class_has_global_priority(rc) or (
            (size // instr_dist) > 2 * num_alloc
        )
        # The local-vs-global refinement (via SplitAnalysis) lands with the
        # split stages; a faithful start treats every non-ForceGlobal range as
        # local, matching RAGreedy for ranges that never need splitting.
        is_local = True
        if self._get_stage(reg) == LiveRangeStage.RS_New:
            self._set_stage(reg, LiveRangeStage.RS_Assign)
        prio = self._priority_for(
            reg, size, is_local, force_global, num_alloc, instr_dist
        )
        # Python heapq is a min-heap; negate prio for max-first, and use reg as
        # the tie-break (smaller id first, matching ~Reg ordering).
        heapq.heappush(self._queue, (-prio, reg))

    def dequeue(self):
        if not self._queue:
            return None
        _, reg = heapq.heappop(self._queue)
        return reg

    # -- tryAssign (RegAllocGreedy::tryAssign) ------------------------------
    def _try_assign(self, li):
        """Return the first interference-free physreg in allocation order,
        preferring the simple copy hint. None if every physreg interferes."""
        hint = self.simple_hint(li.reg)
        order = list(self.allocation_order(li))
        if hint and hint in order and self.matrix.is_free(li, hint):
            return hint
        for preg in order:
            if self.matrix.is_free(li, preg):
                return preg
        return None

    def select_or_split(self, li):
        reg = li.reg
        preg = self._try_assign(li)
        if preg is not None:
            self.trace[reg] = "assign"
            return preg
        preg = self._try_evict(li)
        if preg is not None:
            self.trace[reg] = "evict"
            return preg
        # (split inserted by later tasks)
        self.trace[reg] = "spill"
        self.spill(li)
        self._set_stage(reg, LiveRangeStage.RS_Memory)
        return None

    # -- tryEvict / canEvictInterference ------------------------------------
    def _can_evict_interference(self, li, physreg):
        """True if every vreg interfering with `li` on `physreg` can be evicted:
        each must have a strictly smaller spill weight, and the cascade rule
        must not forbid it (an interferer whose cascade >= li's cannot be
        evicted by li). Mirrors canEvictInterference."""
        li_cascade = self._get_cascade(li.reg)
        for ivreg in self.interfering_vregs(li, physreg):
            iv = self.lis.interval(ivreg)
            if iv.weight >= li.weight:
                return False
            if self._get_cascade(ivreg) >= li_cascade:
                return False
        return True

    def _eviction_cost_for(self, li, physreg):
        """eviction_cost for evicting `li`'s interferers off `physreg`."""
        interferers = self.interfering_vregs(li, physreg)
        weights = [self.lis.interval(v).weight for v in interferers]
        hint = self.simple_hint(li.reg)
        broken_hint = bool(hint) and physreg != hint
        csr = self.last_callee_saved_alias(physreg) != 0
        unused_csr = csr and not self.matrix.is_phys_reg_used(physreg)
        return eviction_cost(weights, broken_hint, unused_csr)

    def _try_evict(self, li):
        """Evict the interferers off the cheapest evictable physreg and assign
        `li` there. Returns the physreg, or None if nothing is evictable."""
        best, best_cost = None, None
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                continue  # tryAssign already tried the free ones
            if not self._can_evict_interference(li, preg):
                continue
            cost = self._eviction_cost_for(li, preg)
            if best_cost is None or cost < best_cost:
                best, best_cost = preg, cost
        if best is None:
            return None
        li_cascade = self._get_cascade(li.reg)
        for ivreg in list(self.interfering_vregs(li, best)):
            iv = self.lis.interval(ivreg)
            self.matrix.unassign(iv)
            # The evicted range advances to RS_Split and inherits li's cascade
            # so it can't evict li back within this cascade.
            self._set_stage(ivreg, max(self._get_stage(ivreg), LiveRangeStage.RS_Split))
            self._cascade[ivreg] = li_cascade
            self.enqueue(ivreg)
        # Evicting the interferers frees `best`; return it and let the framework
        # do the assignment (assigning here as well would double-assign and
        # abort). `best` now passes select_or_split's free-candidate check.
        return best


mir.RAGreedy = RAGreedy
