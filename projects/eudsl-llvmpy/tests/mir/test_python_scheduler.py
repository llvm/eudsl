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
        with pytest.raises(RuntimeError, match="unknown scheduler"):
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


def test_python_strategy_actually_drives_scheduling():
    """A strategy recording into a list witnesses that Python drove
    initialize/pick (semantics-preserving scheduling leaves no other trace)."""
    trace = []

    class Recording(_TopDownFirstReady):
        def initialize(self, dag):
            super().initialize(dag)
            trace.append("init")

        def pick_node(self):
            trace.append("pick")
            return super().pick_node()

    mir.register_scheduler("t6-recording", Recording)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="t6-recording")
        assert "init" in trace and "pick" in trace
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


def test_two_strategies_coexist():
    ran = {"a": 0, "b": 0}

    class A(_TopDownFirstReady):
        def pick_node(self):
            ran["a"] += 1
            return super().pick_node()

    class B(_TopDownFirstReady):
        def pick_node(self):
            ran["b"] += 1
            return super().pick_node()

    mir.register_scheduler("t6-a", A)
    mir.register_scheduler("t6-b", B)
    assert "t6-a" in mir.registered_schedulers()
    assert "t6-b" in mir.registered_schedulers()
    for name, key in (("t6-a", "a"), ("t6-b", "b")):
        other = "b" if key == "a" else "a"
        before_other = ran[other]
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "add")
            _build_selected_add(mmi)
            mmi.emit_object(scheduler=name)
        assert ran[key] > 0  # this run's strategy ran
        assert ran[other] == before_other  # the other did not
    assert_no_leaks()


def test_fresh_instance_per_function():
    """The registry ctor constructs a fresh strategy instance per
    MachineFunction, not a reused singleton. Two emissions (each with one
    MachineFunction) must construct two *distinct* instances."""
    instances = []

    class Counting(_TopDownFirstReady):
        def __init__(self):
            super().__init__()
            instances.append(self)

    mir.register_scheduler("t6-perfunc", Counting)
    for _ in range(2):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "add")
            _build_selected_add(mmi)
            mmi.emit_object(scheduler="t6-perfunc")
    assert len(instances) == 2  # one instance per MachineFunction emitted
    assert instances[0] is not instances[1]  # fresh, not a cached singleton
    assert_no_leaks()


def test_pick_node_raise_propagates():
    """A pick_node that raises surfaces as a Python exception out of emit_object
    (stashed and re-raised after codegen winds down, never thrown across LLVM's
    -fno-exceptions frames); the interpreter survives and nothing leaks."""

    class Boom(_TopDownFirstReady):
        def pick_node(self):
            raise ValueError("boom")

    mir.register_scheduler("t6-boom", Boom)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="boom"):
            mmi.emit_object(scheduler="t6-boom")
    assert_no_leaks()


def test_pick_node_bad_return_propagates():
    """pick_node returning a non-(SUnit, bool) surfaces as a Python error, not a
    crash: the trampoline's cast fails, is stashed, and re-raised."""

    class BadReturn(_TopDownFirstReady):
        def pick_node(self):
            return 123  # not (SUnit, bool) and not None

    mir.register_scheduler("t6-badret", BadReturn)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(RuntimeError, match="bad_cast"):
            mmi.emit_object(scheduler="t6-badret")
    assert_no_leaks()


def test_initialize_raise_propagates():
    """Every override is a stash site: a raise from initialize (not just
    pick_node) is stashed and re-raised out of emit_object."""

    class InitBoom(_TopDownFirstReady):
        def initialize(self, dag):
            raise ValueError("init-boom")

    mir.register_scheduler("t6-initboom", InitBoom)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="init-boom"):
            mmi.emit_object(scheduler="t6-initboom")
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_python_scheduled_add():
    mir.register_scheduler("t6-jit", _TopDownFirstReady)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="t6-jit")
        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        del j
    assert_no_leaks()


def test_ready_queue_strategy_helper_pick():
    """mir.ReadyQueueStrategy maintains the ready queue; the subclass only
    overrides pick(ready)."""
    picks = []

    class LastReady(mir.ReadyQueueStrategy):
        def pick(self, ready):
            picks.append(len(ready))
            return ready[-1]

    mir.register_scheduler("t7-lastready", LastReady)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="t7-lastready")
        assert picks  # the helper's pick() hook ran
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


class _BottomUpStrategy(mir.MachineSchedStrategy):
    """Bottom-up scheduling: consume nodes released bottom-ready and pick them
    with is_top=False. Impossible with the top-down ReadyQueueStrategy helper --
    exercises get_policy(only_bottom_up), release_bottom_node, and the
    is_top=False pick path."""

    def initialize(self, dag):
        self.q = []

    def get_policy(self):
        p = mir.MachineSchedPolicy()
        p.only_bottom_up = True
        p.should_track_pressure = False
        return p

    def release_top_node(self, su):
        pass

    def release_bottom_node(self, su):
        self.q.append(su)

    def pick_node(self):
        if not self.q:
            return None
        return self.q.pop(0), False  # is_top_node = False

    def sched_node(self, su, is_top):
        pass


def test_bottom_up_strategy_drives_scheduling():
    """Witness that Python drove a *bottom-up* schedule: the recording subclass
    sees nodes via release_bottom_node and returns them with is_top=False (a path
    the top-down ReadyQueueStrategy helper cannot reach)."""
    released_bottom = []
    picks = []

    class Recording(_BottomUpStrategy):
        def release_bottom_node(self, su):
            released_bottom.append(su.node_num)
            super().release_bottom_node(su)

        def pick_node(self):
            choice = super().pick_node()
            if choice is not None:
                picks.append(choice[1])  # is_top_node
            return choice

    mir.register_scheduler("t8-bottomup", Recording)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="t8-bottomup")
        assert released_bottom  # nodes arrived via the bottom-ready path
        assert picks and all(is_top is False for is_top in picks)  # is_top=False
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_bottom_up_scheduled_add():
    mir.register_scheduler("t8-bottomup-jit", _BottomUpStrategy)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="t8-bottomup-jit")
        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(7, 8) == 15  # bottom-up schedule still correct
        del j
    assert_no_leaks()


def test_sunit_accessors_readable():
    """The SUnit read accessors return sensible values for ready nodes."""
    seen = []

    class ReadsFields(_TopDownFirstReady):
        def pick_node(self):
            if not self.q:
                return None
            su = self.q[0]
            seen.append((su.node_num, su.is_top_ready, su.is_bottom_ready, su.instr))
            return self.q.pop(0), True

    mir.register_scheduler("t9-fields", ReadsFields)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object(scheduler="t9-fields")
    assert seen
    node_num, is_top, is_bottom, instr = seen[0]
    assert node_num >= 0
    assert is_top  # a top-ready node has no unscheduled predecessors
    assert isinstance(is_bottom, bool)
    assert instr is not None  # every ready SUnit wraps a MachineInstr
    assert_no_leaks()


@pytest.mark.parametrize(
    "method",
    [
        "get_policy",
        "release_top_node",
        "release_bottom_node",
        "sched_node",
        "register_roots",
        "enter_mbb",
        "leave_mbb",
    ],
)
def test_override_raise_propagates(method):
    """A raise from any forwarded override is stashed and re-raised out of
    emit_object (every override is a -fno-exceptions stash site)."""

    def boom(self, *args):
        raise ValueError("boom-" + method)

    Strat = type("Strat", (_TopDownFirstReady,), {method: boom})
    mir.register_scheduler("t9-" + method, Strat)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="boom-" + method):
            mmi.emit_object(scheduler="t9-" + method)
    assert_no_leaks()


def test_bottom_up_pick_raise_propagates():
    """A bottom-up strategy whose pick_node raises: the exception is stashed and
    re-raised, and the single shadow ready-set drains (returning still-ready
    nodes with is_top from isTopReady()) so the pipeline winds down cleanly."""

    class BottomBoom(_BottomUpStrategy):
        def pick_node(self):
            raise ValueError("bottom-boom")

    mir.register_scheduler("t9-bottomboom", BottomBoom)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="bottom-boom"):
            mmi.emit_object(scheduler="t9-bottomboom")
    assert_no_leaks()


def test_reregister_replaces_scheduler():
    """Registering an existing name replaces its class (the registry node for
    that name is reused)."""
    picks = []

    class First(_TopDownFirstReady):
        def pick_node(self):
            picks.append("first")
            return super().pick_node()

    class Second(_TopDownFirstReady):
        def pick_node(self):
            picks.append("second")
            return super().pick_node()

    mir.register_scheduler("t9-dup", First)
    mir.register_scheduler("t9-dup", Second)  # replaces First
    assert mir.registered_schedulers().count("t9-dup") == 1
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object(scheduler="t9-dup")
    assert "second" in picks and "first" not in picks
    assert_no_leaks()


def test_register_non_subclass_raises():
    """A class with the right methods but not subclassing MachineSchedStrategy is
    rejected (it would otherwise reach nb::inst_ptr on a non-bound object)."""

    class NotAStrategy:  # duck-typed, but not a mir.MachineSchedStrategy
        def initialize(self, dag): ...
        def get_policy(self): ...
        def pick_node(self): ...
        def sched_node(self, su, is_top): ...
        def release_top_node(self, su): ...
        def release_bottom_node(self, su): ...

    with pytest.raises(TypeError, match="subclass"):
        mir.register_scheduler("t10-notsub", NotAStrategy)


def test_raising_init_propagates():
    """A subclass whose __init__ raises surfaces as a Python exception out of
    emit_object rather than crashing across LLVM's -fno-exceptions frames (the
    strategy-construction path is a stash site too)."""

    class InitBoom(_TopDownFirstReady):
        def __init__(self):
            raise ValueError("ctor-boom")

    mir.register_scheduler("t10-ctorboom", InitBoom)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(ValueError, match="ctor-boom"):
            mmi.emit_object(scheduler="t10-ctorboom")
    assert_no_leaks()


def test_get_policy_wrong_type_propagates():
    """get_policy returning a non-MachineSchedPolicy fails the trampoline cast;
    the error is stashed and re-raised out of emit_object."""

    class BadPolicy(_TopDownFirstReady):
        def get_policy(self):
            return 123  # not a MachineSchedPolicy

    mir.register_scheduler("t10-badpolicy", BadPolicy)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(RuntimeError, match="bad_cast"):
            mmi.emit_object(scheduler="t10-badpolicy")
    assert_no_leaks()


def test_pressure_tracking_emits_object():
    """A strategy with should_track_pressure=True exercises the
    shouldTrackPressure() true-branch through codegen (register-pressure
    tracking) and still emits a well-formed object."""

    class Pressure(_TopDownFirstReady):
        def get_policy(self):
            p = mir.MachineSchedPolicy()
            p.only_top_down = True
            p.should_track_pressure = True
            return p

    mir.register_scheduler("t10-pressure", Pressure)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(scheduler="t10-pressure")
        assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


def test_optional_lifecycle_hooks_invoked():
    """The optional hooks (register_roots, enter_mbb, leave_mbb) are forwarded to
    a subclass that defines them: register_roots fires once the initial ready set
    is released, enter_mbb/leave_mbb bracket the block, and enter_mbb receives the
    MachineBasicBlock."""
    events = []

    class Hooked(_TopDownFirstReady):
        def enter_mbb(self, mbb):
            events.append(("enter", mbb.name))

        def register_roots(self):
            events.append(("roots", None))

        def leave_mbb(self):
            events.append(("leave", None))

    mir.register_scheduler("t11-hooks", Hooked)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object(scheduler="t11-hooks")
    kinds = [k for k, _ in events]
    assert "enter" in kinds and "roots" in kinds and "leave" in kinds
    assert kinds.index("enter") < kinds.index("roots") < kinds.index("leave")
    assert isinstance(dict(events).get("enter"), str)  # enter_mbb got an MBB name
    assert_no_leaks()
