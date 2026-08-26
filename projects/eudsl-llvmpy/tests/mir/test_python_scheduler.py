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

import ctypes
import platform

import pytest

import llvm
from llvm import ir, jit
from llvm.testing import assert_no_leaks

# Object emission uses an AArch64 target (cross ELF), so needs the AArch64
# backend linked; JIT-executing additionally needs an AArch64 host.
pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)
_AARCH64_LINUX = "aarch64-unknown-linux-gnu"
_IS_AARCH64 = platform.machine() in ("arm64", "aarch64")


def _build_selected_add(mmi):
    """Hand-build a fully-selected AArch64 add(i32,i32)->i32 MachineFunction."""
    mf = mmi.machine_function("add")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0, w1 = mf.physreg("W0"), mf.physreg("W1")
    entry.add_livein(w0)
    entry.add_livein(w1)
    v0, v1, v2 = (mf.create_vreg(gpr32) for _ in range(3))
    copy = mf.opcode("COPY")
    for dst, src in ((v0, w0), (v1, w1)):
        c = b.build_instr(copy)
        c.add_reg(dst, is_def=True)
        c.add_reg(src)
    add = b.build_instr(mf.opcode("ADDWrr"))
    add.add_reg(v2, is_def=True)
    add.add_reg(v0)
    add.add_reg(v1)
    ret_copy = b.build_instr(copy)
    ret_copy.add_reg(w0, is_def=True)
    ret_copy.add_reg(v2)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


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
        if not self.q:
            return None
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



def test_unknown_scheduler_name_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(RuntimeError, match="scheduler"):
            mmi.emit_object(scheduler="does-not-exist")
    assert_no_leaks()


def test_registered_strategy_emits_object():
    mir.register_scheduler("t5-firstready", _TopDownFirstReady)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="t5-firstready")
        assert obj[:4] == b"\x7fELF"
        assert b"add\x00" in obj
    assert_no_leaks()
