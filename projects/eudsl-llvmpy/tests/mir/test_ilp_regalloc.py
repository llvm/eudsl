#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ILP register allocators: pure-helper unit tests + end-to-end + comparison."""

import pytest
import llvm
from llvm import mir_ilp_model as model


def test_require_ortools_returns_cp_model():
    cp = model._require_ortools()
    assert hasattr(cp, "CpModel")


def test_scale_weight_is_positive_int():
    assert model.scale_weight(0.0) >= 1
    assert isinstance(model.scale_weight(1.5), int)
    assert model.scale_weight(2.0) > model.scale_weight(1.0)


def test_build_interference_overlap_and_disjoint():
    # v1 [0,4) overlaps v2 [2,6); v3 [6,8) is disjoint from both.
    intervals = {1: [(0, 4)], 2: [(2, 6)], 3: [(6, 8)]}
    edges = model.build_interference(intervals)
    assert frozenset((1, 2)) in edges
    assert frozenset((1, 3)) not in edges
    assert frozenset((2, 3)) not in edges  # [2,6) and [6,8) touch but half-open


def test_build_interference_multi_segment_holes():
    # v1 is live [0,2) and [8,10) (a hole); v2 [3,7) fits in the hole -> no edge.
    intervals = {1: [(0, 2), (8, 10)], 2: [(3, 7)]}
    assert model.build_interference(intervals) == set()


def test_compact_time_axis():
    intervals = {1: [(0, 4)], 2: [(4, 10)]}
    mapping, n = model.compact_time_axis(intervals)
    assert n == 3  # points {0, 4, 10}
    assert mapping[0] == 0 and mapping[4] == 1 and mapping[10] == 2


def test_candidate_pregs_filters_forbidden():
    assert model.candidate_pregs([10, 11, 12], {11}) == [10, 12]


def test_single_class_k():
    assert model.single_class_k({1: 32, 2: 32}) == 32
    assert model.single_class_k({1: 32, 2: 16}) is None
    assert model.single_class_k({}) is None
