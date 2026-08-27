#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Python-subclassable RegAllocBase register allocator.

mir.RegAllocBase binds llvm::RegAllocBase (via a trampoline) so Python can
subclass it and implement an arbitrary allocator. register_regalloc adds it
under a name; emit_object(regalloc="name") drives the codegen allocator slot
with it. Allocation is semantics-preserving, so an allocator recording into a
test-visible object witnesses that Python drove it; JIT-executed tests prove the
result stays correct.
"""

import ctypes, platform
import pytest
import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

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


# result(x) = sum of N copies of x = N*x, over N vregs kept live simultaneously
# (all defined before the reduction consumes them) so allocation is forced past
# the GPR32 register file and must spill.
_HP_N = 48


def _hp_closed_form(x):
    return _HP_N * x


def _build_high_pressure(mmi):
    """Hand-build a single-block hp(i32)->i32 with high register pressure."""
    mf = mmi.machine_function("hp")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    entry.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")

    # N independent copies of the input, all live until the reduction consumes
    # them (distinct vregs, so nothing coalesces them away).
    terms = []
    for _ in range(_HP_N):
        t = mf.create_vreg(gpr32)
        ins = b.build_instr(copy)
        ins.add_reg(t, is_def=True)
        ins.add_reg(w0)
        terms.append(t)

    acc = terms[0]
    for t in terms[1:]:
        nacc = mf.create_vreg(gpr32)
        ins = b.build_instr(addrr)
        ins.add_reg(nacc, is_def=True)
        ins.add_reg(acc)
        ins.add_reg(t)
        acc = nacc

    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(acc)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def _emit(name, cls, builder=_build_selected_add, fn="add"):
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, fn)
        builder(mmi)
        return mmi.emit_object(regalloc=name)


def _jit_call(sig, fn, obj):
    j = jit.LLJIT()
    j.add_object(obj)
    return ctypes.CFUNCTYPE(*sig)(j.lookup(fn)), j


# -- registration contract ---------------------------------------------------


def test_register_regalloc_requires_regallocbase_subclass():
    class NotAnAllocator:
        def select_or_split(self, li): ...

    with pytest.raises(TypeError, match="RegAllocBase"):
        mir.register_regalloc("ra-bad", NotAnAllocator)


def test_register_regalloc_requires_select_or_split():
    class NoSelect(mir.RegAllocBase):
        pass

    with pytest.raises(TypeError, match="select_or_split"):
        mir.register_regalloc("ra-nosel", NoSelect)


def test_registered_regalloc_appears_in_registry():
    mir.register_regalloc("ra-listed", mir.BasicRegAlloc)
    assert "ra-listed" in mir.registered_regallocs()


def test_unknown_regalloc_name_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        with pytest.raises(RuntimeError, match="unknown regalloc"):
            mmi.emit_object(regalloc="ra-does-not-exist")
    assert_no_leaks()


# -- behavioral ---------------------------------------------------------------


def test_basic_regalloc_emits_object():
    mir.register_regalloc("ra-basic", mir.BasicRegAlloc)
    obj = _emit("ra-basic", mir.BasicRegAlloc)
    assert obj[:4] == b"\x7fELF"
    assert b"add\x00" in obj
    assert_no_leaks()


@pytest.mark.skipif(not _IS_AARCH64, reason="executing hand-built AArch64 MIR")
def test_basic_regalloc_executes():
    mir.register_regalloc("ra-basic-x", mir.BasicRegAlloc)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object(regalloc="ra-basic-x")
        add, j = _jit_call(
            (ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), "add", obj
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        del j
    assert_no_leaks()


def test_custom_select_or_split_drives_allocation():
    """An allocator recording each (vreg, physreg) choice witnesses that Python
    drove select_or_split."""
    seen = []

    class Recording(mir.RegAllocBase):
        def select_or_split(self, li):
            for preg in self.allocation_order(li):
                if self.matrix.is_free(li, preg):
                    seen.append((li.reg, preg))
                    return preg
            self.spill(li)
            return None

    mir.register_regalloc("ra-recording", Recording)
    obj = _emit("ra-recording", Recording)
    assert obj[:4] == b"\x7fELF"
    assert seen, "select_or_split ran"
    assert_no_leaks()


def test_query_accessors_and_manual_assignment():
    """Exercise the query/mutate surface: the analysis accessors, the interval
    fields, and check_interference/assign/unassign driven from Python."""
    saw = {}

    class Manual(mir.RegAllocBase):
        def select_or_split(self, li):
            saw["lis"] = self.lis is not None
            saw["vrm"] = self.vrm is not None
            saw["mf"] = self.machine_function is not None
            saw["weight"] = li.weight
            saw["spillable"] = li.is_spillable
            for preg in self.allocation_order(li):
                if (
                    self.matrix.check_interference(li, preg)
                    == mir.InterferenceKind.IK_Free
                ):
                    self.matrix.assign(li, preg)
                    self.matrix.unassign(li)
                    self.matrix.assign(li, preg)
                    return None
            self.spill(li)
            return None

    mir.register_regalloc("ra-manual", Manual)
    obj = _emit("ra-manual", Manual)
    assert obj[:4] == b"\x7fELF"
    assert saw["lis"] and saw["vrm"] and saw["mf"]
    assert isinstance(saw["weight"], float)
    assert saw["spillable"] is True
    assert_no_leaks()


def test_custom_enqueue_dequeue_queue():
    """A Python-side FIFO queue via enqueue/dequeue drives the assignment order
    instead of the native spill-weight queue."""

    class ListQueue(mir.BasicRegAlloc):
        def __init__(self):
            super().__init__()
            self.q = []

        def enqueue(self, li):
            self.q.append(li)

        def dequeue(self):
            if not self.q:
                return None
            return self.q.pop(0)

    mir.register_regalloc("ra-listq", ListQueue)
    obj = _emit("ra-listq", ListQueue)
    assert obj[:4] == b"\x7fELF"
    assert_no_leaks()


def test_post_optimization_forwarded():
    ran = []

    class WithPost(mir.BasicRegAlloc):
        def post_optimization(self):
            ran.append(True)

    mir.register_regalloc("ra-post", WithPost)
    obj = _emit("ra-post", WithPost)
    assert obj[:4] == b"\x7fELF"
    assert ran, "post_optimization ran"
    assert_no_leaks()


def test_fresh_instance_per_emission():
    instances = []

    class Recorder(mir.BasicRegAlloc):
        def __init__(self):
            super().__init__()
            instances.append(self)

    mir.register_regalloc("ra-fresh", Recorder)
    _emit("ra-fresh", Recorder)
    _emit("ra-fresh", Recorder)
    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert_no_leaks()


# -- spilling -----------------------------------------------------------------


def test_high_pressure_forces_spill():
    """A function with more simultaneously-live vregs than the register file
    forces BasicRegAlloc down the spill path; the emitted object stays valid."""
    spilled = []

    class Spilling(mir.RegAllocBase):
        def select_or_split(self, li):
            for preg in self.allocation_order(li):
                if self.matrix.is_free(li, preg):
                    return preg
            spilled.append(li.reg)
            self.spill(li)
            return None

    mir.register_regalloc("ra-hp", Spilling)
    obj = _emit("ra-hp", Spilling, builder=_build_high_pressure, fn="hp")
    assert obj[:4] == b"\x7fELF"
    assert spilled, "spill path exercised"
    assert_no_leaks()


@pytest.mark.skipif(not _IS_AARCH64, reason="executing hand-built AArch64 MIR")
def test_high_pressure_executes():
    class Spilling(mir.RegAllocBase):
        def select_or_split(self, li):
            for preg in self.allocation_order(li):
                if self.matrix.is_free(li, preg):
                    return preg
            self.spill(li)
            return None

    mir.register_regalloc("ra-hp-x", Spilling)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "hp")
        _build_high_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-hp-x")
        hp, j = _jit_call((ctypes.c_int32, ctypes.c_int32), "hp", obj)
        assert hp(3) == _hp_closed_form(3)
        assert hp(0) == _hp_closed_form(0)
        del j
    assert_no_leaks()


# -- SlotIndex / LiveIntervals accessors --------------------------------------


def test_slot_index_and_live_intervals_accessors():
    """Read the program-point surface from inside select_or_split: block
    start/end indices are valid and ordered, the interval is present, and the
    SlotIndex value methods round-trip."""
    saw = {}

    class Reader(mir.RegAllocBase):
        def select_or_split(self, li):
            if not saw:
                mbb = self.machine_function.blocks[0]
                start = self.lis.mbb_start_index(mbb)
                end = self.lis.mbb_end_index(mbb)
                mi = list(mbb.instructions)[0]
                idx = self.lis.instruction_index(mi)
                saw["ordered"] = start < end
                saw["eq"] = start == start
                saw["valid"] = start.is_valid()
                saw["idx_in_block"] = (start < idx) and (idx < end)
                saw["reg_slot"] = idx.get_reg_slot().is_valid()
                saw["base"] = idx.get_base_index().is_valid()
                saw["boundary"] = idx.get_boundary_index().is_valid()
                saw["next"] = idx.get_next_index().is_valid()
                saw["repr"] = repr(start)
                saw["has"] = self.lis.has_interval(li.reg)
                saw["interval_reg"] = self.lis.interval(li.reg).reg
            for preg in self.allocation_order(li):
                if self.matrix.is_free(li, preg):
                    return preg
            self.spill(li)
            return None

    mir.register_regalloc("ra-slots", Reader)
    obj = _emit("ra-slots", Reader)
    assert obj[:4] == b"\x7fELF"
    assert saw["ordered"] and saw["eq"] and saw["valid"]
    assert saw["idx_in_block"]
    assert saw["reg_slot"] and saw["base"] and saw["boundary"] and saw["next"]
    assert "SlotIndex" in saw["repr"]
    assert saw["has"]
    assert_no_leaks()


# -- exception matrix ---------------------------------------------------------


def _expect_raise(name, cls, match, builder=_build_selected_add, fn="add"):
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, fn)
        builder(mmi)
        with pytest.raises(Exception, match=match):
            mmi.emit_object(regalloc=name)
    assert_no_leaks()


def test_select_or_split_raise_propagates():
    class Boom(mir.RegAllocBase):
        def select_or_split(self, li):
            raise ValueError("select boom")

    mir.register_regalloc("ra-sel-boom", Boom)
    _expect_raise("ra-sel-boom", Boom, "select boom")


def test_select_or_split_bad_return_propagates():
    class BadReturn(mir.RegAllocBase):
        def select_or_split(self, li):
            return "not an int"

    mir.register_regalloc("ra-sel-bad", BadReturn)
    _expect_raise("ra-sel-bad", BadReturn, "bad_cast")


def test_enqueue_raise_propagates():
    class Boom(mir.BasicRegAlloc):
        def enqueue(self, li):
            raise ValueError("enqueue boom")

    mir.register_regalloc("ra-enq-boom", Boom)
    _expect_raise("ra-enq-boom", Boom, "enqueue boom")


def test_dequeue_raise_propagates():
    class Boom(mir.BasicRegAlloc):
        def dequeue(self):
            raise ValueError("dequeue boom")

    mir.register_regalloc("ra-deq-boom", Boom)
    _expect_raise("ra-deq-boom", Boom, "dequeue boom")


def test_init_raise_propagates():
    class Boom(mir.RegAllocBase):
        def __init__(self):
            super().__init__()
            raise ValueError("init boom")

        def select_or_split(self, li):
            return None

    mir.register_regalloc("ra-init-boom", Boom)
    _expect_raise("ra-init-boom", Boom, "init boom")


def test_post_optimization_raise_propagates():
    class Boom(mir.BasicRegAlloc):
        def post_optimization(self):
            raise ValueError("post boom")

    mir.register_regalloc("ra-post-boom", Boom)
    _expect_raise("ra-post-boom", Boom, "post boom")


def test_native_fallback_spills_after_init_raise_under_pressure():
    """When __init__ raises under high pressure, the C++ NativeRegAlloc fallback
    still allocates (down its own spill path), so the MIR is valid and the
    exception re-raises rather than aborting the rewriter on unallocated vregs."""

    class Boom(mir.RegAllocBase):
        def __init__(self):
            super().__init__()
            raise ValueError("hp init boom")

        def select_or_split(self, li):
            return None

    mir.register_regalloc("ra-hp-init", Boom)
    _expect_raise(
        "ra-hp-init", Boom, "hp init boom", builder=_build_high_pressure, fn="hp"
    )
