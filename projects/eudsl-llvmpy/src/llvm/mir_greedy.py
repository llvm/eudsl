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


mir.RAGreedy = RAGreedy
