#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""regalloc_assignments: read back any allocator's post-RA vreg->physreg map."""

import platform
import pytest
import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked",
)
_AARCH64_LINUX = "aarch64-unknown-linux-gnu"

_ADD_IR = """\
define i32 @add(i32 %a, i32 %b) {
entry:
  %s = add i32 %a, %b
  ret i32 %s
}
"""


def _build_add(mmi, declare_liveins=True):
    mf = mmi.machine_function("add")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0, w1 = mf.physreg("W0"), mf.physreg("W1")
    # With TracksLiveness set, omitting the live-in declarations makes the MIR
    # fail verify() (a physreg is used without being declared live-in).
    if declare_liveins:
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
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v2)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


# High register pressure (48 simultaneously-live vregs) forces greedy to spill,
# so the post-RA map has a non-empty `spilled` set.
_HP_N = 48


def _build_high_pressure(mmi):
    mf = mmi.machine_function("hp")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    entry.add_livein(w0)
    copy = mf.opcode("COPY")
    addrr = mf.opcode("ADDWrr")
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


def test_regalloc_assignments_native_greedy():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        mf = _build_add(mmi)
        w0, w1 = mf.physreg("W0").id, mf.physreg("W1").id
        result = mmi.regalloc_assignments(regalloc="greedy")
        assignments = dict(result.assignments)
        spilled = list(result.spilled)
    # Every virtual register present at RA is accounted for: assigned or spilled.
    assert isinstance(result.assignments, dict)
    assert isinstance(result.spilled, list)
    # No coalescing runs here, so all three vregs (the two argument copies and
    # the add result) survive to RA and are assigned; none spill. Pin the actual
    # physregs: the argument copies land in their live-in registers (W0, W1) and
    # the result reuses W0. vreg ids are assigned in creation order (v0, v1, v2).
    v0, v1, v2 = sorted(assignments)
    assert assignments[v0] == w0
    assert assignments[v1] == w1
    assert assignments[v2] == w0
    assert assignments[v0] != assignments[v1]  # v0, v1 are simultaneously live
    assert spilled == []
    assert_no_leaks()


def test_regalloc_assignments_greedy_matches_basic():
    """On this pressure-free function greedy and basic must reach the same
    coloring; a decision-level oracle that the capture reports real per-vreg
    decisions, not allocator-specific noise."""

    def assignments(alloc):
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_AARCH64_LINUX)
            mmi = mir.create_machine_function(mod, tm, "add")
            _build_add(mmi)
            return dict(mmi.regalloc_assignments(regalloc=alloc).assignments)

    mir.register_regalloc("ra-basic-oracle", mir.BasicRegAlloc)
    assert assignments("greedy") == assignments("ra-basic-oracle")
    assert_no_leaks()


def test_regalloc_assignments_python_allocator():
    mir.register_regalloc("ra-basic-cap", mir.BasicRegAlloc)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        result = mmi.regalloc_assignments(regalloc="ra-basic-cap")
    assert result.assignments and result.spilled == []
    assert_no_leaks()


def test_regalloc_assignments_unknown_name_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        with pytest.raises(Exception, match="unknown regalloc"):
            mmi.regalloc_assignments(regalloc="ra-nope")
    assert_no_leaks()


def test_regalloc_assignments_spills_under_pressure():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "hp")
        _build_high_pressure(mmi)
        result = mmi.regalloc_assignments(regalloc="greedy")
    # 48 simultaneously-live vregs exceed the GPR32 file, so some spill: the
    # spilled set is non-empty and disjoint from the assigned set.
    assert result.spilled
    assert not (set(result.spilled) & set(result.assignments))
    assert_no_leaks()


def test_regalloc_assignments_one_shot():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        mmi.regalloc_assignments(regalloc="greedy")
        with pytest.raises(Exception, match="already emitted"):
            mmi.regalloc_assignments(regalloc="greedy")
    assert_no_leaks()


def test_regalloc_assignments_result_outlives_module():
    """The capture pass owns its result, so the returned snapshot is a
    standalone value -- still readable after the context (and the captured MIR
    it was read from) are torn down. Guards against the pass borrowing a pointer
    that dangles once the one-shot pipeline is parked."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        result = mmi.regalloc_assignments(regalloc="greedy")
    assert dict(result.assignments)  # readable after teardown
    assert result.spilled == []
    assert_no_leaks()


def test_regalloc_assignments_requires_build_path():
    with ir.Context() as ctx:
        mod = ir.parse_assembly(_ADD_IR, ctx, "m")
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.run_codegen_to_mir(mod, tm)
        with pytest.raises(Exception, match="requires a module built"):
            mmi.regalloc_assignments(regalloc="greedy")
    assert_no_leaks()


def test_regalloc_assignments_malformed_mir_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi, declare_liveins=False)
        with pytest.raises(Exception, match="failed verification"):
            mmi.regalloc_assignments(regalloc="greedy")
    assert_no_leaks()


def test_regalloc_assignments_python_allocator_exception_propagates():
    class Raising(mir.RegAllocBase):
        def select_or_split(self, li):
            raise RuntimeError("boom from select_or_split")

    mir.register_regalloc("ra-raise-cap", Raising)
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        with pytest.raises(RuntimeError, match="boom from select_or_split"):
            mmi.regalloc_assignments(regalloc="ra-raise-cap")
    assert_no_leaks()


def test_regalloc_assignments_copies_remaining_all_coalesced():
    # Every COPY in _build_add is coalesceable (src and dst land in the same
    # physreg), so none survive.
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_add(mmi)
        result = mmi.regalloc_assignments(regalloc="greedy")
        copies = result.copies_remaining
    assert copies == 0
    assert_no_leaks()


def _build_cross_reg_move(mmi):
    """An argument arrives in W1 and is returned in W0, forcing a surviving COPY
    (its source and destination land in different physical registers)."""
    mf = mmi.machine_function("mv")
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0, w1 = mf.physreg("W0"), mf.physreg("W1")
    entry.add_livein(w1)
    v0 = mf.create_vreg(gpr32)
    copy = mf.opcode("COPY")
    c = b.build_instr(copy)
    c.add_reg(v0, is_def=True)
    c.add_reg(w1)
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v0)
    b.build_instr(mf.opcode("RET_ReallyLR")).add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_regalloc_assignments_copies_remaining_counts_surviving_move():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "mv")
        _build_cross_reg_move(mmi)
        result = mmi.regalloc_assignments(regalloc="greedy")
        copies = result.copies_remaining
    # The W1 -> W0 move cannot be coalesced away: one COPY survives.
    assert copies >= 1
    assert_no_leaks()
