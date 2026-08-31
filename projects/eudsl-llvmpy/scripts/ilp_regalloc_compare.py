#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the ILP-vs-greedy register-allocation comparison over curated fixtures.

Usage (AArch64 backend + ortools required):
    LLVM_BINDIR=... python scripts/ilp_regalloc_compare.py

Prints, per fixture, a table comparing each allocator's spills, weighted spill
cost, solve time, and optimality gap. Allocators that hard-fail (the whole-
interval models on functions that need spilling) are shown as ``hard-fail``.
"""

import llvm
from llvm import ir, jit, mir
from llvm.mir_ilp_base import RAILPBase
from llvm.mir_ilp_compare import AllocResult, format_table

_TRIPLE = "aarch64-unknown-linux-gnu"


def _low_pressure(mmi, name):
    """Pressure-free: two argument copies + an ADD + a return copy."""
    mf = mmi.machine_function(name)
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
    rc = b.build_instr(copy)
    rc.add_reg(w0, is_def=True)
    rc.add_reg(v2)
    b.build_instr(mf.opcode("RET_ReallyLR")).add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def _high_pressure(mmi, name, n=48):
    """n values live simultaneously, then chain-added: forces spilling."""
    mf = mmi.machine_function(name)
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0 = mf.physreg("W0")
    entry.add_livein(w0)
    copy, addrr = mf.opcode("COPY"), mf.opcode("ADDWrr")
    terms = []
    for _ in range(n):
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
    b.build_instr(mf.opcode("RET_ReallyLR")).add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


_FIXTURES = [("low_pressure", _low_pressure), ("high_pressure", _high_pressure)]
# (display name, registered regalloc name, ILP?) -- ILP allocators report gap/time.
_ALLOCS = [
    ("greedy", "greedy", False),
    ("basic", "cmp-basic", False),
    ("ilp-assign", "cmp-assign", True),
    ("ilp-packing", "cmp-pack", True),
    ("ilp-decomp", "cmp-decomp", True),
]
_ILP_CLASS = {"cmp-assign": "RAILPAssign", "cmp-pack": "RAILPPacking",
              "cmp-decomp": "RAILPDecomp"}


def _run_one(fixture, fname, name, regalloc, is_ilp):
    try:
        with ir.Context() as ctx:
            mod = ir.Module("m", ctx)
            tm = jit.TargetMachine(triple=_TRIPLE)
            mmi = mir.create_machine_function(mod, tm, fname)
            fixture(mmi, fname)
            result = mmi.regalloc_assignments(regalloc=regalloc)
            assignments = dict(result.assignments)
            spilled = list(result.spilled)
    except RuntimeError as e:
        return AllocResult(name=name, valid=False, spills=[], weight={},
                           copies_remaining=0, wall_time_s=None, gap=None,
                           error=str(e).split(";")[0][:40])
    stats = RAILPBase.last_stats.get(_ILP_CLASS[regalloc]) if is_ilp else None
    return AllocResult(
        name=name,
        valid=all(isinstance(p, int) for p in assignments.values()),
        spills=spilled,
        weight={v: 1 for v in spilled},  # uniform (copies/weights are informational)
        copies_remaining=0,              # stubbed pending an instruction-walk binding
        wall_time_s=stats.wall_time_s if stats else None,
        gap=stats.gap if stats else None,
    )


def main():
    if "aarch64" not in llvm.jit.registered_targets():
        print("AArch64 backend not linked; nothing to compare.")
        return
    mir.register_regalloc("cmp-basic", mir.BasicRegAlloc)
    mir.register_regalloc("cmp-assign", mir.RAILPAssign)
    mir.register_regalloc("cmp-pack", mir.RAILPPacking)
    mir.register_regalloc("cmp-decomp", mir.RAILPDecomp)
    for fixture_name, fixture in _FIXTURES:
        results = [_run_one(fixture, fixture_name, name, regalloc, is_ilp)
                   for name, regalloc, is_ilp in _ALLOCS]
        print(format_table(fixture_name, results))
        print()


if __name__ == "__main__":
    main()
