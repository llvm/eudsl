#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure, binding-free helpers shared by the ILP register allocators.

Kept free of any `ortools` import at module scope so the base package imports
without the optional dependency; the CP-SAT module is fetched lazily via
``_require_ortools``. The functions here operate on plain Python data (ints,
lists, dicts) so they can be unit-tested without constructing a MachineFunction.
"""

# Spill weights are floats; CP-SAT objectives are integer. Scale by this factor
# so the weighted-spill objective dominates the unit coalescing-hint bonus (a
# hint can only ever break ties, never force or prevent a spill).
_WEIGHT_SCALE = 1000

# LLVM marks must-not-spill intervals with an infinite (HUGE_VALF) spill weight.
# CP-SAT needs a finite integer coefficient, so clamp to a value large enough to
# dominate any realistic sum of ordinary weights yet safely within int range.
_MAX_WEIGHT = 1_000_000_000

# Reward (in scaled objective units) for assigning a vreg to its copy hint.
# Strictly less than the smallest scaled spill weight so it never buys a spill.
HINT_BONUS = 1


def _require_ortools():
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "ortools is required for the ILP register allocators; install "
            "eudsl-llvmpy[ilp] or `pip install ortools>=9.0`"
        ) from e
    return cp_model


def scale_weight(weight):
    """Scale a float spill weight to a positive integer objective coefficient.

    Infinite / huge weights (LLVM's must-not-spill marker) clamp to _MAX_WEIGHT.
    """
    if weight == float("inf") or weight * _WEIGHT_SCALE >= _MAX_WEIGHT:
        return _MAX_WEIGHT
    return max(1, round(weight * _WEIGHT_SCALE))


def _segments_overlap(segs_a, segs_b):
    """True if any half-open [start, end) segment of A overlaps one of B."""
    for s1, e1 in segs_a:
        for s2, e2 in segs_b:
            if s1 < e2 and s2 < e1:
                return True
    return False


def build_interference(intervals):
    """Pairwise interference edges from live-range overlap.

    `intervals` maps vreg id -> list of (start, end) half-open integer segments.
    Returns a set of ``frozenset({u, v})`` for every pair whose ranges overlap.
    """
    vregs = sorted(intervals)
    edges = set()
    for i, u in enumerate(vregs):
        for v in vregs[i + 1:]:
            if _segments_overlap(intervals[u], intervals[v]):
                edges.add(frozenset((u, v)))
    return edges


def compact_time_axis(intervals):
    """Map the sorted set of all segment endpoints to contiguous ints.

    Returns ``(mapping, n_points)`` where `mapping` sends each original endpoint
    to its index in the sorted order. Used to give the packing model a dense
    time axis instead of raw (possibly large, sparse) slot distances.
    """
    points = sorted({p for segs in intervals.values() for seg in segs for p in seg})
    mapping = {p: i for i, p in enumerate(points)}
    return mapping, len(points)


def candidate_pregs(order, forbidden):
    """Allocation-order physregs minus the matrix-forbidden ones."""
    return [p for p in order if p not in forbidden]


def single_class_k(num_regs):
    """If every vreg shares one register-class size, return k; else None.

    `num_regs` maps vreg id -> allocatable-register count of its class. The
    packing and decomposition models are scoped to single-class functions.
    """
    ks = set(num_regs.values())
    return next(iter(ks)) if len(ks) == 1 else None
