#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Python-subclassable MachineScheduler strategy.

mir.MachineSchedStrategy binds llvm::MachineSchedStrategy so Python can subclass
it and override the scheduling virtuals. register_scheduler adds it to the
MachineScheduler registry under a name; emit_object(scheduler="name") runs it as
the pre-RA scheduler. Scheduling is semantics-preserving, so a strategy
recording into a test-visible object is the witness that Python drove it; a
JIT-executed test proves the result stays correct.
"""

from llvm import mir

import pytest


def test_machine_sched_policy_fields_roundtrip():
    p = mir.MachineSchedPolicy()
    assert p.only_top_down is False
    assert p.should_track_pressure is False
    p.only_top_down = True
    p.only_bottom_up = True
    p.should_track_pressure = False
    p.should_track_lane_masks = False
    assert p.only_top_down and p.only_bottom_up


class _TopDownFirstReady(mir.MachineSchedStrategy):
    """Minimal top-down strategy: schedule ready nodes in first-ready order."""

    def initialize(self, dag):
        self.q = []

    def get_policy(self):
        p = mir.MachineSchedPolicy()
        p.only_top_down = True
        p.should_track_pressure = False
        return p

    def release_top_node(self, su):
        self.q.append(su)

    def release_bottom_node(self, su):
        pass

    def pick_node(self):
        return self.q.pop(0), True

    def sched_node(self, su, is_top):
        pass


def test_register_scheduler_appears_in_registry():
    mir.register_scheduler("t4-appears", _TopDownFirstReady)
    assert "t4-appears" in mir.registered_schedulers()


def test_register_scheduler_missing_method_raises():
    class Incomplete(mir.MachineSchedStrategy):
        def initialize(self, dag):
            pass

        def get_policy(self):
            return mir.MachineSchedPolicy()

        def sched_node(self, su, is_top):
            pass

        def release_top_node(self, su):
            pass

        def release_bottom_node(self, su):
            pass

        # pick_node intentionally missing

    with pytest.raises(TypeError, match="pick_node"):
        mir.register_scheduler("t4-incomplete", Incomplete)

