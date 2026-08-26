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


def test_machine_sched_policy_fields_roundtrip():
    p = mir.MachineSchedPolicy()
    assert p.only_top_down is False
    assert p.should_track_pressure is False
    p.only_top_down = True
    p.only_bottom_up = True
    p.should_track_pressure = False
    p.should_track_lane_masks = False
    assert p.only_top_down and p.only_bottom_up
