#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Pure-Python unit tests for the convenience strategy classes in
llvm.mir_strategies.

These exercise the Python-side logic directly (no codegen), so they run
everywhere -- including x86-only builds where the AArch64 backend is not linked
and the emit/scheduler/regalloc tests skip. mir.ReadyQueueStrategy is a plain
ready-queue helper whose methods do not touch the codegen pipeline, so it can be
driven with placeholder nodes.
"""

from llvm import mir
from llvm.testing import assert_no_leaks


def test_ready_queue_strategy_default_pick_is_fifo():
    s = mir.ReadyQueueStrategy()
    s.initialize(None)  # dag unused; just resets the ready queue

    policy = s.get_policy()
    assert policy.only_top_down
    assert not policy.should_track_pressure

    a, b = object(), object()
    s.release_top_node(a)
    s.release_top_node(b)
    s.release_bottom_node(a)  # top-down strategy ignores bottom releases
    s.sched_node(a, True)  # no-op hook

    # Default pick() takes the first ready node; pick_node returns (node, is_top)
    # and removes it, then None once drained.
    assert s.pick_node() == (a, True)
    assert s.pick_node() == (b, True)
    assert s.pick_node() is None
    assert_no_leaks()


def test_ready_queue_strategy_pick_override():
    class LastReady(mir.ReadyQueueStrategy):
        def pick(self, ready):
            return ready[-1]

    s = LastReady()
    s.initialize(None)
    a, b, c = object(), object(), object()
    for su in (a, b, c):
        s.release_top_node(su)
    assert s.pick_node() == (c, True)
    assert s.pick_node() == (b, True)
    assert s.pick_node() == (a, True)
    assert s.pick_node() is None
    assert_no_leaks()
