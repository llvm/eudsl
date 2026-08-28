#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Faithful Python RegAllocGreedy (mir.RAGreedy)."""

import ctypes
import platform
import pytest
import llvm
from llvm import ir, jit, mir
from llvm.mir_greedy import eviction_cost, calc_gap_weights, calc_global_split_cost
from llvm.testing import assert_no_leaks

pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked",
)
_AARCH64_LINUX = "aarch64-unknown-linux-gnu"


def test_ragreedy_is_exported_and_constructs():
    assert issubclass(mir.RAGreedy, mir.RegAllocBase)
    ra = mir.RAGreedy()
    assert ra.RS_New < ra.RS_Assign < ra.RS_Split < ra.RS_Split2 < ra.RS_Spill
    assert_no_leaks()


def test_eviction_cost_sums_weights_and_penalties():
    # No penalties: cost is just the summed interferer weight.
    assert eviction_cost(
        [1.0, 2.0], broken_hint=False, is_unused_callee_saved=False
    ) == pytest.approx(3.0)
    # A broken hint adds a strictly positive penalty.
    base = eviction_cost([1.0], broken_hint=False, is_unused_callee_saved=False)
    assert eviction_cost([1.0], broken_hint=True, is_unused_callee_saved=False) > base
    # Introducing an unused callee-saved reg is dominated by the CSR bias.
    assert eviction_cost([1.0], broken_hint=False, is_unused_callee_saved=True) > base


def test_calc_gap_weights_per_gap_max_interference():
    # Uses at slots 0, 10, 20 -> two gaps: [0,10) and [10,20).
    use_slots = [0, 10, 20]
    # One interferer of weight 3.0 spanning [5, 15): overlaps both gaps.
    # One interferer of weight 7.0 spanning [12, 18): only the second gap.
    spans = [(5, 15, 3.0), (12, 18, 7.0)]
    weights = calc_gap_weights(use_slots, spans)
    assert len(weights) == 2
    assert weights[0] == pytest.approx(3.0)  # only the weight-3 interferer
    assert weights[1] == pytest.approx(7.0)  # max(3.0, 7.0)


def test_calc_gap_weights_no_interference_is_zero():
    assert calc_gap_weights([0, 5], []) == [0.0]


def test_calc_global_split_cost_sums_boundary_frequencies():
    assert calc_global_split_cost([]) == pytest.approx(0.0)
    assert calc_global_split_cost([2.0, 3.0, 0.5]) == pytest.approx(5.5)
