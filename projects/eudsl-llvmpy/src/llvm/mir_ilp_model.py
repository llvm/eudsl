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
    """Scale a float spill weight to a positive integer objective coefficient."""
    return max(1, round(weight * _WEIGHT_SCALE))
