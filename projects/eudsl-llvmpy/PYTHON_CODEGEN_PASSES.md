<!--
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Python-driven codegen passes

`llvm.mir` lets a Python callable drive two of the back-half codegen passes that
`MirModule.emit_object()` runs over already-selected Machine IR: the pre-RA
**MachineScheduler** and the **register allocator**. This is the codegen sibling
of the IR-level Python-pass feature (see "Python-driven passes" in the
[README](README.md)): there a callable is the *body* of an IR pass; here a
callable makes the per-node / per-interval *decision* inside a native codegen
pass while the surrounding pass machinery stays in C++.

Both features are selected through keyword arguments to `emit_object`, which
otherwise runs register allocation and object emission over the hand-built MIR
(via `-start-after=finalize-isel`, so no instruction selection). See "Building
target MIR and lowering to native code" in the README for how to build the input
MachineFunction and JIT-execute the emitted object.

```python
emit_object(scheduler=None, pick=None, regalloc=None, select=None) -> bytes
```

- A **scheduler-side** option (`scheduler` *or* `pick`) chooses the pre-RA
  MachineScheduler. `scheduler` and `pick` are mutually exclusive.
- An **allocator-side** option (`regalloc` *or* `select`) chooses the register
  allocator. `regalloc` and `select` are mutually exclusive.
- The two sides are independent: one scheduler-side option may be combined with
  one allocator-side option in a single `emit_object` call.
- With none of the four given, `emit_object` uses the target's default scheduler
  and allocator.

## Scheduler API

### Select a registered strategy by name: `scheduler=`

`emit_object(scheduler="<name>")` sets the process-global `-misched` option to
the MachineScheduler strategy registered under that name, so the pre-RA
MachineScheduler runs it instead of the target's default. An unknown name raises
`RuntimeError`.

`mir.registered_schedulers()` returns the list of registered strategy names.
This extension registers its own:

- `"python"` — a top-down strategy whose `pickNode` delegates the choice to a
  Python callable (selected via `pick=`, below).

Names registered by LLVM itself (e.g. `"converge"`) are in the list too and are
equally selectable by name.

### Drive `pickNode` from Python: `pick=`

`emit_object(pick=<callable>)` selects the `"python"` strategy and routes each
`pickNode` choice through the callable.

**Contract.** The callable receives one argument: the ready nodes as a
`list[SUnit]`. It must **return one of those `SUnit`s** — the node to schedule
next (matched back by pointer identity). `SUnit` exposes these read-only fields:

| field            | type          | meaning                                                              |
| ---------------- | ------------- | -------------------------------------------------------------------- |
| `node_num`       | `int`         | Entry number of this node in the DAG's node vector.                  |
| `is_top_ready`   | `bool`        | All predecessors scheduled (ready for top-down scheduling).          |
| `is_bottom_ready`| `bool`        | All successors scheduled (ready for bottom-up scheduling).           |
| `instr`          | `MachineInstr`| The representative `MachineInstr` this scheduling unit wraps.        |

The strategy is top-down, so the presented nodes report `is_top_ready`.
Selecting `scheduler="python"` *without* a `pick` callback is legal: `pickNode`
keeps the native first-ready choice.

## Register-allocator API

### Select the eudsl register allocator by name: `regalloc=`

`emit_object(regalloc="eudsl-python")` points `RegisterRegAlloc`'s process-
global default at this extension's eudsl register allocator (a `RegAllocBase`-
derived pass whose `selectOrSplit` assigns the first non-interfering physreg per
virtual register and spills only when none is free), so the codegen pipeline
allocates with it instead of the target default. An unknown name raises
`RuntimeError`.

### Drive `selectOrSplit` from Python: `select=`

`emit_object(select=<callable>)` selects the `"eudsl-python"` allocator and
routes each `selectOrSplit` decision through the callable.

**Contract.** The callable receives `(live_interval, candidates)`:

- `live_interval` is a `LiveInterval` — the live range of the one virtual
  register being assigned — with these read-only fields:

  | field          | type    | meaning                                                                 |
  | -------------- | ------- | ----------------------------------------------------------------------- |
  | `reg`          | `int`   | Id of the virtual register this interval covers.                        |
  | `weight`       | `float` | The spill weight; higher means costlier to spill.                       |
  | `is_spillable` | `bool`  | Whether this interval may be spilled (a finite spill weight).           |

- `candidates` is a `list[int]` of the legal (non-interfering) candidate physreg
  ids for this virtual register, in allocation order.

The callable must return either **a candidate id from that list** (assign it) or
**`None`** (spill the virtual register).

## Decision-callback contract and errors

A decision callback (`pick` or `select`) must return a legal choice. Two failure
modes both surface as a Python exception raised out of `emit_object` — never a
crash across LLVM's `-fno-exceptions` frames:

- **The callback raises.** The original exception propagates out of
  `emit_object` (it is stashed and re-raised after the unskippable codegen
  pipeline winds down). A `raise ValueError("boom")` in the callback surfaces as
  that same `ValueError("boom")`.
- **The callback returns an illegal value** — for `pick`, a value that is not one
  of the presented ready `SUnit`s; for `select`, a value that is neither `None`
  nor one of the presented candidate ids. This raises `ValueError`
  (`"...not one of the ready nodes"` for the scheduler, `"...not one of the legal
  candidates"` for the allocator).

Passing both members of a mutually-exclusive pair (`scheduler`+`pick`, or
`regalloc`+`select`) raises `ValueError` before any codegen runs.

### Diagnostic counters

Because scheduling and register allocation are semantics-preserving, the emitted
code alone cannot distinguish "the extension's pass ran" from the target default
having run. The extension exposes diagnostic counters (in
`llvm.eudslllvm_ext.mir`) that the test suite uses as witnesses:

- `_regalloc_select_count()` / `_reset_regalloc_select_count()`
- `_regalloc_spill_count()` / `_reset_regalloc_spill_count()`

## Caveats

- **Codegen is GIL-serialized.** Selection uses process-global LLVM options
  (`-misched`, `-start-after`) and `RegisterRegAlloc::setDefault`, set and
  restored around the run under the GIL with no additional lock. Do not run
  concurrent, nested, or free-threaded codegen.
- **Python callbacks are slow.** `pick` fires once per `pickNode` and `select`
  once per `selectOrSplit`, each acquiring the GIL and marshalling to Python.
  This feature is for expressiveness and experimentation, not production
  codegen.
- **PostRA scheduling is not covered.** Only the pre-RA MachineScheduler is
  overridable (there is no `-misched=`-style override for the post-RA
  scheduler).

## Build note

The eudsl register allocator is built on LLVM's `RegAllocBase` driver, which
needs two private CodeGen headers — `RegAllocBase.h` and `AllocationOrder.h`. These live
under `llvm/lib/CodeGen/` upstream and are **not** installed into the public
`include/` directory the prebuilt LLVM distribution ships, so verbatim copies are
vendored into `src/MIR/`. The `bump_llvm.yml` workflow re-copies them from
`third_party/llvm-project` on every LLVM bump, keeping them in sync.

## Runnable examples

Both examples build the same fully-selected AArch64 `add(i32, i32) -> i32`
MachineFunction, then JIT-execute the emitted object with `ctypes`. They use a
host `TargetMachine` so the object is loadable in-process, and so require an
AArch64 host with the AArch64 backend linked (see the `skipif` guards in the
tests they are drawn from).

```python
import ctypes

from llvm import ir, jit, mir


def build_selected_add(mmi):
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
```

### A Python scheduler (`pick`)

```python
def pick(ready):
    # `ready` is a list[SUnit]; return one of them. Mimic the native
    # first-ready policy.
    return ready[0]

with ir.Context() as ctx:
    mod = ir.Module("m", ctx)
    tm = jit.TargetMachine()  # host triple -> object loadable in-process
    mmi = mir.create_machine_function(mod, tm, "add")
    build_selected_add(mmi)
    obj = mmi.emit_object(pick=pick)

    j = jit.LLJIT()
    j.add_object(obj)
    add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
        j.lookup("add")
    )
    assert add(2, 3) == 5
    assert add(40, 2) == 42
```

### A Python register allocator (`select`)

```python
def select(live_interval, candidates):
    # `candidates` is a list[int] of legal physreg ids; return one to assign,
    # or None to spill. Mimic the native first-free policy.
    return candidates[0] if candidates else None

with ir.Context() as ctx:
    mod = ir.Module("m", ctx)
    tm = jit.TargetMachine()  # host triple -> object loadable in-process
    mmi = mir.create_machine_function(mod, tm, "add")
    build_selected_add(mmi)
    obj = mmi.emit_object(select=select)

    j = jit.LLJIT()
    j.add_object(obj)
    add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
        j.lookup("add")
    )
    assert add(2, 3) == 5
    assert add(40, 2) == 42
```

The scheduler example is drawn from `test_jit_executes_python_scheduled_add`
(`tests/mir/test_python_scheduler.py`) and the allocator example from
`test_jit_executes_python_selected_add` (`tests/mir/test_python_regalloc.py`);
`build_selected_add` is the tests' `_build_selected_add` helper verbatim.
</content>
</invoke>
