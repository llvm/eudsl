#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Convenience MachineSchedStrategy subclasses (pure Python).

Importing this module attaches them to the `mir` submodule, so they read as
`mir.ReadyQueueStrategy`.
"""

from . import mir

# name -> registered MachineSchedStrategy subclass; populated by
# mir.register_scheduler (C++) and read back by it via
# llvm.mir_strategies._scheduler_classes. Python owns this so classes are
# released at interpreter teardown rather than pinned in a C++ static.
_scheduler_classes = {}


class ReadyQueueStrategy(mir.MachineSchedStrategy):
    """Top-down strategy that maintains the ready queue for you.

    Subclass and override ``pick(ready)`` to choose one node from the currently
    ready list; the default picks the first (native first-ready). Power users
    that need the full lifecycle (custom priority, deferral, bottom-up) subclass
    mir.MachineSchedStrategy directly.
    """

    def initialize(self, dag):
        self._q = []

    def get_policy(self):
        p = mir.MachineSchedPolicy()
        p.only_top_down = True
        p.should_track_pressure = False
        return p

    def release_top_node(self, su):
        self._q.append(su)

    def release_bottom_node(self, su):
        pass

    def sched_node(self, su, is_top):
        pass

    def pick_node(self):
        if not self._q:
            return None
        chosen = self.pick(self._q)
        self._q.remove(chosen)
        return chosen, True

    def pick(self, ready):
        return ready[0]


mir.ReadyQueueStrategy = ReadyQueueStrategy
