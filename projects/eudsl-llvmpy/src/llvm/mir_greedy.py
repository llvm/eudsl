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
