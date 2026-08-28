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
        add, j = _jit_call((ctypes.c_int32, ctypes.c_int32, ctypes.c_int32), "add", obj)
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


_listq_dequeued = []


def test_custom_enqueue_dequeue_queue():
    """A Python-side FIFO queue via enqueue/dequeue drives the assignment order
    instead of the native spill-weight queue. enqueue/dequeue traffic in
    register ids, not LiveInterval objects, which splitting would invalidate."""
    _listq_dequeued.clear()

    class ListQueue(mir.BasicRegAlloc):
        def __init__(self):
            super().__init__()
            self.q = []

        def enqueue(self, reg):
            self.q.append(reg)

        def dequeue(self):
            if not self.q:
                return None
            reg = self.q.pop(0)
            _listq_dequeued.append(reg)
            return reg

    mir.register_regalloc("ra-listq", ListQueue)
    obj = _emit("ra-listq", ListQueue)
    assert obj[:4] == b"\x7fELF"
    # Witness that the Python dequeue (not the native drain) drove allocation:
    # native dequeue would leave this list empty.
    assert _listq_dequeued, "the Python queue drove dequeue"
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


# -- eviction (interference query + VirtRegMap) -------------------------------


_evict_log = []
# reg ids handed to select_or_split, in order; a reg that is evicted, re-enqueued
# and re-processed appears more than once.
_evict_selected = []


class _EvictingAllocator(mir.RegAllocBase):
    """Greedy-style eviction: take a free physreg if one exists, else evict the
    interfering (no-higher-weight) virtual registers off a candidate physreg,
    re-enqueueing them onto the Python queue. Each register is evicted at most
    once, which bounds the cascade so allocation terminates (a re-enqueued reg
    that finds no free/evictable candidate spills)."""

    def __init__(self):
        super().__init__()
        self.q = []
        self.evicted_once = set()

    def enqueue(self, reg):
        self.q.append(reg)

    def dequeue(self):
        return self.q.pop(0) if self.q else None

    def select_or_split(self, li):
        _evict_selected.append(li.reg)
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        for preg in self.allocation_order(li):
            intf = self.interfering_vregs(li, preg)
            if (
                intf
                and all(r not in self.evicted_once for r in intf)
                and all(self.vrm.has_phys(r) for r in intf)
                and all(self.lis.interval(r).weight <= li.weight for r in intf)
            ):
                for r in intf:
                    _evict_log.append((r, self.vrm.get_phys(r)))
                    self.evicted_once.add(r)
                    self.matrix.unassign(self.lis.interval(r))
                    self.q.append(r)
                return preg
        self.spill(li)
        return None


def test_eviction_identifies_and_unassigns_interferer():
    """Under forced interference the allocator enumerates the interfering vreg
    on an occupied physreg (via interfering_vregs), reads its current physreg
    (get_phys/has_phys), and evicts it -- exercising the eviction surface. The
    re-enqueue path is witnessed: an evicted reg is later re-processed."""
    _evict_log.clear()
    _evict_selected.clear()
    mir.register_regalloc("ra-evict", _EvictingAllocator)
    obj = _emit("ra-evict", _EvictingAllocator, builder=_build_high_pressure, fn="hp")
    assert obj[:4] == b"\x7fELF"
    assert _evict_log, "an interferer was identified and evicted"
    # Witness re-enqueue -> re-processing: at least one evicted reg was handed
    # back to select_or_split a second time (it can't be re-processed unless the
    # eviction actually re-enqueued it).
    evicted = {r for r, _ in _evict_log}
    assert any(
        _evict_selected.count(r) >= 2 for r in evicted
    ), "an evicted reg was re-enqueued and re-processed"
    assert_no_leaks()


@pytest.mark.skipif(not _IS_AARCH64, reason="executing hand-built AArch64 MIR")
def test_eviction_executes():
    _evict_log.clear()
    mir.register_regalloc("ra-evict-x", _EvictingAllocator)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "hp")
        _build_high_pressure(mmi)
        obj = mmi.emit_object(regalloc="ra-evict-x")
        hp, j = _jit_call((ctypes.c_int32, ctypes.c_int32), "hp", obj)
        assert hp(3) == _hp_closed_form(3)
        assert hp(7) == _hp_closed_form(7)
        del j
    assert _evict_log, "eviction path exercised during the executed emission"
    assert_no_leaks()


_alias_seen = {}


class _AliasChecker(mir.RegAllocBase):
    """Assigns GPR32 vregs (which land in W registers) and, once one is
    assigned, queries interfering_vregs against each enclosing X (GPR64)
    super-register. Finding the W-assigned vreg via an X query proves the query
    walks every reg unit and reports aliasing/subregister interference -- it is
    not a Python-side physreg->vreg shadow (which would only know about the W it
    assigned, never the containing X)."""

    def __init__(self):
        super().__init__()
        self.assigned = []

    def select_or_split(self, li):
        if "hit" not in _alias_seen and self.assigned:
            mf = self.machine_function
            for r, wphys in self.assigned:
                if not self.lis.has_interval(r):
                    continue
                for xi in range(29):  # X0..X28 (X29/X30/X31 have special names)
                    try:
                        x = mf.physreg(f"X{xi}").id
                    except KeyError:
                        continue
                    intf = self.interfering_vregs(li, x)
                    if r in intf and x != wphys:
                        # No duplicate ids even though a super-register query can
                        # surface the same vreg across several reg units.
                        assert len(intf) == len(set(intf))
                        _alias_seen["hit"] = (r, wphys, xi)
                        break
                if "hit" in _alias_seen:
                    break
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                self.matrix.assign(li, preg)
                self.assigned.append((li.reg, preg))
                return None
        self.spill(li)
        return None


def test_interfering_vregs_reports_subregister_aliasing():
    """A vreg assigned to a W register is reported when interfering_vregs is
    queried against the enclosing X super-register -- exercising the every-reg-
    unit / alias-correct behavior on a genuine sub/super-register pair (a shadow
    map keyed on the assigned physreg could not find it)."""
    _alias_seen.clear()
    mir.register_regalloc("ra-alias", _AliasChecker)
    obj = _emit("ra-alias", _AliasChecker, builder=_build_high_pressure, fn="hp")
    assert obj[:4] == b"\x7fELF"
    assert "hit" in _alias_seen, (
        "a GPR32 vreg assigned to a W register was found by querying its "
        "enclosing X super-register (interfering_vregs walks every reg unit)"
    )
    assert_no_leaks()


# -- MachineBlockFrequencyInfo ------------------------------------------------


def test_block_frequency_info_accessors():
    """Read block frequencies from inside select_or_split: the entry block's
    relative frequency is 1.0 and its BlockFrequency equals entry_freq, and a
    conditionally-reached block is strictly less frequent than the entry (both
    as a ratio and as a raw BlockFrequency)."""
    saw = {}

    class FreqReader(mir.RegAllocBase):
        def select_or_split(self, li):
            if not saw:
                mbfi = self.mbfi
                mf = self.machine_function
                entry, b1 = mf.blocks[0], mf.blocks[1]
                saw["entry_rel"] = mbfi.block_freq_relative_to_entry_block(entry)
                saw["b1_rel"] = mbfi.block_freq_relative_to_entry_block(b1)
                saw["entry_freq"] = mbfi.entry_freq()
                saw["entry_block_freq"] = mbfi.block_freq(entry)
                saw["b1_block_freq"] = mbfi.block_freq(b1)
                # Exercise the BlockFrequency arithmetic/limit surface.
                saw["sum"] = mbfi.block_freq(entry) + mbfi.block_freq(b1)
                saw["diff"] = mbfi.block_freq(entry) - mbfi.block_freq(b1)
                saw["max"] = mir.BlockFrequency.max()
                saw["repr"] = repr(mbfi.block_freq(entry))
            for preg in self.allocation_order(li):
                if self.matrix.is_free(li, preg):
                    return preg
            self.spill(li)
            return None

    mir.register_regalloc("ra-mbfi", FreqReader)
    obj = _emit("ra-mbfi", FreqReader, builder=_build_cbz, fn="cbz")
    assert obj[:4] == b"\x7fELF"
    assert saw["entry_rel"] == 1.0
    assert saw["entry_freq"].get_frequency() > 0
    # The entry block's BlockFrequency equals the relative denominator
    # (exercises BlockFrequency.__eq__).
    assert saw["entry_block_freq"] == saw["entry_freq"]
    # b1 is only reached on the not-taken branch, so it cannot exceed entry.
    assert 0.0 < saw["b1_rel"] <= saw["entry_rel"]
    # block_freq must actually depend on its mbb argument: the conditionally
    # reached b1 is strictly less frequent than the entry (BlockFrequency.__lt__).
    assert saw["b1_block_freq"] < saw["entry_block_freq"]
    assert saw["b1_block_freq"] != saw["entry_block_freq"]
    assert saw["entry_block_freq"] > saw["b1_block_freq"]
    assert saw["entry_block_freq"] >= saw["b1_block_freq"]
    assert saw["b1_block_freq"] <= saw["entry_block_freq"]
    # BlockFrequency arithmetic and saturation limit.
    ef = saw["entry_block_freq"].get_frequency()
    bf = saw["b1_block_freq"].get_frequency()
    assert saw["sum"].get_frequency() == ef + bf
    assert saw["diff"].get_frequency() == ef - bf
    assert saw["max"].get_frequency() > ef
    assert saw["repr"] == f"BlockFrequency({ef})"
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
                saw["ne"] = start == end  # distinct points compare unequal
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
    assert saw["ne"] is False, "distinct SlotIndexes compare unequal"
    assert saw["idx_in_block"]
    assert saw["reg_slot"] and saw["base"] and saw["boundary"] and saw["next"]
    assert "SlotIndex" in saw["repr"]
    assert saw["has"]
    assert_no_leaks()


def _build_three_block(mmi):
    """b0 defines v, b1 is empty (v live-through), b2 uses v: gives SplitAnalysis
    one through block and two use blocks for v's interval."""
    mf = mmi.machine_function("thru")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    br = mf.opcode("B")

    b.set_block(b0)
    v = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(v, is_def=True)
    c.add_reg(w0)
    j0 = b.build_instr(br)
    j0.add_mbb(b1)
    b0.add_successor(b1)

    b.set_block(b1)
    j1 = b.build_instr(br)
    j1.add_mbb(b2)
    b1.add_successor(b2)

    b.set_block(b2)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


# -- SplitAnalysis ------------------------------------------------------------


def test_split_analysis_use_and_through_blocks():
    saw = {}

    class Analyzer(mir.RegAllocBase):
        def select_or_split(self, li):
            if "use_blocks" not in saw:
                sa = self.split_analysis
                sa.analyze(li)
                blocks = sa.use_blocks()
                saw["use_blocks"] = len(blocks)
                bi = blocks[0]
                saw["mbb_num"] = bi.mbb.number
                saw["first_valid"] = bi.first_instr.is_valid()
                saw["last_valid"] = bi.last_instr.is_valid()
                # The def block reports a valid first_def; the use-only block
                # reports an invalid one.
                saw["any_first_def"] = any(b.first_def.is_valid() for b in blocks)
                saw["live_in"] = bi.live_in
                saw["live_out"] = bi.live_out
                saw["num_through"] = sa.num_through_blocks()
                saw["through"] = sa.through_blocks()
            for preg in self.allocation_order(li):
                if self.matrix.is_free(li, preg):
                    return preg
            self.spill(li)
            return None

    mir.register_regalloc("ra-split-analysis", Analyzer)
    obj = _emit("ra-split-analysis", Analyzer, builder=_build_three_block, fn="thru")
    assert obj[:4] == b"\x7fELF"
    assert saw["use_blocks"] >= 1
    assert saw["first_valid"] and saw["last_valid"]
    assert saw["any_first_def"], "the def block reports a valid first_def"
    assert saw["num_through"] == 1
    assert len(saw["through"]) == 1
    assert isinstance(saw["live_in"], bool) and isinstance(saw["live_out"], bool)
    assert_no_leaks()


# -- SplitEditor region splitting (the success bar) ---------------------------


def _build_cbz(mmi):
    """b0 defines v and uses it in a CBZW terminator (so v is live-out and its
    last use sits at b0's last split point), b1 falls through, b2 uses v. This
    is the shape that drives SplitEditor's overlap path."""
    mf = mmi.machine_function("cbz")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    b0 = mf.blocks[0]
    b1 = mf.create_block()
    b2 = mf.create_block()
    b0.add_livein(w0)
    copy = mf.opcode("COPY")
    b.set_block(b0)
    v = mf.create_vreg(gpr32)
    c = b.build_instr(copy)
    c.add_reg(v, is_def=True)
    c.add_reg(w0)
    cbz = b.build_instr(mf.opcode("CBZW"))
    cbz.add_reg(v)
    cbz.add_mbb(b2)
    b0.add_successor(b1)
    b0.add_successor(b2)
    b.set_block(b1)
    j = b.build_instr(mf.opcode("B"))
    j.add_mbb(b2)
    b1.add_successor(b2)
    b.set_block(b2)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


class _SingleBlockSplit(mir.RegAllocBase):
    """Split around the last use block (enter_intv_before + leave_intv_after)."""

    split = False

    def select_or_split(self, li):
        if not type(self).split:
            sa = self.split_analysis
            sa.analyze(li)
            cand = [bi for bi in sa.use_blocks() if not bi.live_out]
            if cand and sa.num_through_blocks() > 0:
                bi = cand[-1]
                se = self.split_editor
                se.reset(self.new_live_range_edit(li))
                se.open_intv()
                start = se.enter_intv_before(bi.first_instr)
                stop = se.leave_intv_after(bi.last_instr)
                se.use_intv(start, stop)
                se.finish()
                type(self).split = True
                return None
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)
        return None


def test_region_split_single_block_emits():
    _SingleBlockSplit.split = False
    mir.register_regalloc("ra-split-1", _SingleBlockSplit)
    obj = _emit("ra-split-1", _SingleBlockSplit, builder=_build_three_block, fn="thru")
    assert obj[:4] == b"\x7fELF"
    assert _SingleBlockSplit.split, "the split path ran"
    assert_no_leaks()


@pytest.mark.skipif(not _IS_AARCH64, reason="executing hand-built AArch64 MIR")
def test_region_split_single_block_executes():
    _SingleBlockSplit.split = False
    mir.register_regalloc("ra-split-1x", _SingleBlockSplit)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-split-1x")
        thru, j = _jit_call((ctypes.c_int32, ctypes.c_int32), "thru", obj)
        assert thru(5) == 5 and thru(-3) == -3
        del j
    assert_no_leaks()


class _ThroughBlockSplit(mir.RegAllocBase):
    """Keep the value in a register across the through block using the
    block-level primitives (select_intv/enter_intv_at_end/use_intv_mbb/
    leave_intv_at_top), with the SM_Size complement mode."""

    split = False
    new_vregs = 0

    def select_or_split(self, li):
        cls = type(self)
        if not cls.split:
            sa = self.split_analysis
            sa.analyze(li)
            if sa.num_through_blocks() > 0:
                mf = self.machine_function
                b0, b1, b2 = mf.blocks[0], mf.blocks[1], mf.blocks[2]
                lre = self.new_live_range_edit(li)
                se = self.split_editor
                se.reset(lre, mir.ComplementSpillMode.SM_Size)
                idx = se.open_intv()
                se.select_intv(idx)
                se.enter_intv_at_end(b0)
                se.use_intv_mbb(b1)
                se.leave_intv_at_top(b2)
                se.finish()
                cls.new_vregs = len(lre.new_vregs())
                cls.split = True
                return None
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)
        return None


def test_region_split_through_block_emits():
    _ThroughBlockSplit.split = False
    mir.register_regalloc("ra-split-thru", _ThroughBlockSplit)
    obj = _emit(
        "ra-split-thru", _ThroughBlockSplit, builder=_build_three_block, fn="thru"
    )
    assert obj[:4] == b"\x7fELF"
    assert _ThroughBlockSplit.split
    assert _ThroughBlockSplit.new_vregs > 0
    assert_no_leaks()


@pytest.mark.skipif(not _IS_AARCH64, reason="executing hand-built AArch64 MIR")
def test_region_split_through_block_executes():
    _ThroughBlockSplit.split = False
    mir.register_regalloc("ra-split-thru-x", _ThroughBlockSplit)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-split-thru-x")
        thru, j = _jit_call((ctypes.c_int32, ctypes.c_int32), "thru", obj)
        assert thru(11) == 11
        del j
    assert_no_leaks()


class _EnterAfterSplit(mir.RegAllocBase):
    """Split out of the def block with enter_intv_after + leave_intv_before."""

    split = False

    def select_or_split(self, li):
        cls = type(self)
        if not cls.split:
            sa = self.split_analysis
            sa.analyze(li)
            defblk = [bi for bi in sa.use_blocks() if bi.first_def.is_valid()]
            if defblk and sa.num_through_blocks() > 0:
                bi = defblk[0]
                se = self.split_editor
                se.reset(self.new_live_range_edit(li))
                se.open_intv()
                start = se.enter_intv_after(bi.first_def)
                stop = se.leave_intv_before(sa.last_split_point(bi.mbb))
                se.use_intv(start, stop)
                se.finish()
                cls.split = True
                return None
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)
        return None


def test_split_enter_after_leave_before_emits():
    _EnterAfterSplit.split = False
    mir.register_regalloc("ra-split-after", _EnterAfterSplit)
    obj = _emit(
        "ra-split-after", _EnterAfterSplit, builder=_build_three_block, fn="thru"
    )
    assert obj[:4] == b"\x7fELF"
    assert _EnterAfterSplit.split
    assert_no_leaks()


@pytest.mark.skipif(not _IS_AARCH64, reason="executing hand-built AArch64 MIR")
def test_split_enter_after_leave_before_executes():
    _EnterAfterSplit.split = False
    mir.register_regalloc("ra-split-after-x", _EnterAfterSplit)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "thru")
        _build_three_block(mmi)
        obj = mmi.emit_object(regalloc="ra-split-after-x")
        thru, j = _jit_call((ctypes.c_int32, ctypes.c_int32), "thru", obj)
        assert thru(9) == 9 and thru(-4) == -4
        del j
    assert_no_leaks()


class _OverlapSplit(mir.RegAllocBase):
    """Split the def block whose last use is its terminator, driving the
    overlap_intv path (last use at the last split point, value live-out)."""

    split = False

    def select_or_split(self, li):
        cls = type(self)
        if not cls.split:
            sa = self.split_analysis
            sa.analyze(li)
            b0 = self.machine_function.blocks[0]
            cand = [
                bi
                for bi in sa.use_blocks()
                if bi.mbb.number == b0.number and bi.live_out
            ]
            if cand:
                bi = cand[0]
                lsp = sa.last_split_point(b0)
                se = self.split_editor
                se.reset(self.new_live_range_edit(li))
                se.open_intv()
                start = bi.first_instr if bi.first_instr < lsp else lsp
                seg_start = se.enter_intv_before(start)
                seg_stop = se.leave_intv_before(lsp)
                se.use_intv(seg_start, seg_stop)
                se.overlap_intv(seg_stop, bi.last_instr)
                se.finish()
                cls.split = True
                return None
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)
        return None


def test_split_overlap_intv_emits():
    _OverlapSplit.split = False
    mir.register_regalloc("ra-split-overlap", _OverlapSplit)
    obj = _emit("ra-split-overlap", _OverlapSplit, builder=_build_cbz, fn="cbz")
    assert obj[:4] == b"\x7fELF"
    assert _OverlapSplit.split
    assert_no_leaks()


class _SplitRecordsRemoval(_ThroughBlockSplit):
    """The through-block region split leaves the complement empty in a block, so
    allocatePhysRegs drops that interval; an about_to_remove_interval override
    witnesses the hook firing."""

    removed = 0

    def about_to_remove_interval(self, li):
        type(self).removed += 1


def test_about_to_remove_interval_forwarded_during_split():
    _SplitRecordsRemoval.split = False
    _SplitRecordsRemoval.removed = 0
    mir.register_regalloc("ra-removal", _SplitRecordsRemoval)
    obj = _emit(
        "ra-removal",
        _SplitRecordsRemoval,
        builder=_build_three_block,
        fn="thru",
    )
    assert obj[:4] == b"\x7fELF"
    assert _SplitRecordsRemoval.removed > 0, "about_to_remove_interval fired"
    assert_no_leaks()


@pytest.mark.skipif(not _IS_AARCH64, reason="executing hand-built AArch64 MIR")
def test_split_overlap_intv_executes():
    _OverlapSplit.split = False
    mir.register_regalloc("ra-split-overlap-x", _OverlapSplit)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()
        mmi = mir.create_machine_function(mod, tm, "cbz")
        _build_cbz(mmi)
        obj = mmi.emit_object(regalloc="ra-split-overlap-x")
        cbz, j = _jit_call((ctypes.c_int32, ctypes.c_int32), "cbz", obj)
        assert cbz(5) == 5 and cbz(0) == 0
        del j
    assert_no_leaks()


_preassign = {}


class _PreAssignAllocator(mir.RegAllocBase):
    """Records enqueued register ids (the native queue still drives dequeue) and,
    on its first call, assigns a *different* still-queued interval out of order.
    When the native queue later pops that already-assigned reg it is skipped --
    the stale-entry guard in the default dequeue -- so it never reaches
    select_or_split."""

    def __init__(self):
        super().__init__()
        self.queued = []

    def enqueue(self, reg):
        self.queued.append(reg)

    def select_or_split(self, li):
        _preassign.setdefault("selected", []).append(li.reg)
        if "assigned" not in _preassign:
            for r in self.queued:
                if r == li.reg or not self.lis.has_interval(r):
                    continue
                other = self.lis.interval(r)
                for preg in self.allocation_order(other):
                    if self.matrix.is_free(other, preg):
                        self.matrix.assign(other, preg)
                        _preassign["assigned"] = r
                        break
                if "assigned" in _preassign:
                    break
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)
        return None


def test_default_dequeue_skips_already_assigned():
    _preassign.clear()
    mir.register_regalloc("ra-preassign", _PreAssignAllocator)
    obj = _emit("ra-preassign", _PreAssignAllocator)
    assert obj[:4] == b"\x7fELF"
    # The out-of-order assignment happened, and the native dequeue skipped that
    # already-assigned reg: it is never handed to select_or_split.
    assert "assigned" in _preassign, "pre-assigned an out-of-order interval"
    assert (
        _preassign["assigned"] not in _preassign["selected"]
    ), "the already-assigned reg was skipped by the default dequeue"
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


def test_about_to_remove_interval_raise_propagates():
    class Boom(_ThroughBlockSplit):
        def about_to_remove_interval(self, li):
            raise ValueError("atr boom")

    Boom.split = False
    mir.register_regalloc("ra-atr-boom", Boom)
    _expect_raise(
        "ra-atr-boom", Boom, "atr boom", builder=_build_three_block, fn="thru"
    )


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


# -- misuse guards (raise instead of segfault/abort) --------------------------


def test_spill_outside_select_or_split_raises():
    """self.spill is only valid inside select_or_split; calling it from another
    callback raises rather than dereferencing the null split context."""

    class Boom(mir.BasicRegAlloc):
        def enqueue(self, reg):
            self.spill(self.lis.interval(reg))

    mir.register_regalloc("ra-spill-misuse", Boom)
    _expect_raise("ra-spill-misuse", Boom, "only valid inside select_or_split")


def test_new_live_range_edit_outside_select_or_split_raises():
    class Boom(mir.BasicRegAlloc):
        def enqueue(self, reg):
            self.new_live_range_edit(self.lis.interval(reg))

    mir.register_regalloc("ra-lre-misuse", Boom)
    _expect_raise("ra-lre-misuse", Boom, "only valid inside select_or_split")


def test_select_or_split_bad_physreg_raises():
    """Returning a physreg that is not a free candidate raises rather than
    aborting in Matrix::assign. The high-pressure vregs are all simultaneously
    live, so handing a later one the register the first already got is an
    occupied, interfering candidate."""

    class ReturnsOccupied(mir.RegAllocBase):
        def __init__(self):
            super().__init__()
            self.first = None

        def select_or_split(self, li):
            if self.first is None:
                for preg in self.allocation_order(li):
                    if self.matrix.is_free(li, preg):
                        self.first = preg
                        return preg
            return self.first  # taken by an interfering vreg -> not free

    mir.register_regalloc("ra-bad-phys", ReturnsOccupied)
    _expect_raise(
        "ra-bad-phys",
        ReturnsOccupied,
        "not a free candidate",
        builder=_build_high_pressure,
        fn="hp",
    )


_pyskip = {}


class _StaleQueueAllocator(mir.BasicRegAlloc):
    """Drives dequeue from a Python queue but assigns one still-queued interval
    out of order, so when dequeue later yields that reg id it is already
    assigned and the trampoline skips it (the Python-path stale-entry guard)."""

    def __init__(self):
        super().__init__()
        self.q = []

    def enqueue(self, reg):
        self.q.append(reg)

    def dequeue(self):
        return self.q.pop(0) if self.q else None

    def select_or_split(self, li):
        if "assigned" not in _pyskip:
            for r in self.q:
                if not self.lis.has_interval(r):
                    continue
                other = self.lis.interval(r)
                for preg in self.allocation_order(other):
                    if self.matrix.is_free(other, preg):
                        self.matrix.assign(other, preg)
                        _pyskip["assigned"] = r
                        break
                if "assigned" in _pyskip:
                    break
        _pyskip.setdefault("selected", []).append(li.reg)
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)
        return None


def test_python_dequeue_skips_already_assigned():
    _pyskip.clear()
    mir.register_regalloc("ra-pyskip", _StaleQueueAllocator)
    obj = _emit("ra-pyskip", _StaleQueueAllocator)
    assert obj[:4] == b"\x7fELF"
    assert "assigned" in _pyskip, "pre-assigned an out-of-order interval"
    assert (
        _pyskip["assigned"] not in _pyskip["selected"]
    ), "the already-assigned reg id from Python dequeue was skipped"
    assert_no_leaks()
