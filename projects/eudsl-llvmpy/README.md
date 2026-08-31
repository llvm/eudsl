<!--
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# eudsl-llvmpy

Python bindings for LLVM IR, plus a small DSL for authoring IR from Python.

There are two layers. The lower one is a hand-written [nanobind](https://github.com/wjakob/nanobind)
wrapper over the LLVM **C++** API, so the Python class hierarchy mirrors
`llvm::` (a `Value` really is the base of `Instruction`, `Constant`, and the
rest). The upper one is a DSL: decorate a Python function, write ordinary
arithmetic and `if`/`while`/`for`, and get back a compiled `llvm::Function`
whose control flow is lowered to basic blocks and phi nodes.

The DSL sits on top of the bindings. You can use the bindings alone.

A third surface, `llvm.mir`, reaches one level lower — LLVM's post-instruction-
selection **Machine IR** — with the same split: hand-written bindings over the
`MachineFunction`/`MachineInstr` object model plus a `@machine_function` DSL. See
[Machine IR (MIR)](#machine-ir-mir).

## Package layout

The API is organized into submodules, MLIR-style; nothing is re-exported at the
package top level:

- `llvm.ir` — `Context`, `Module`, `Function`, `BasicBlock`, `IRBuilder`,
  `InsertPoint`, the `Value` hierarchy, constants (`const_int`, `const_fp`,
  `const_bool`), `parse_assembly`, metadata, and the `CmpPredicate` /
  `AtomicOrdering` / `AtomicRMWBinOp` enums.
- `llvm.types` — the `Type` factories (`i1`…`i64`, `f16`/`f32`/`f64`, `ptr`,
  `void`, `function`, `struct`, `array`, `vector`) and `TypeID`.
- `llvm.instructions` — free-function instruction emitters.
- `llvm.passmanager` — `run_passes`, `run_default_pipeline`, and Python-driven
  passes (`run_python_pass_on_module`, `run_python_pass_on_function`,
  `register_python_pass`).
- `llvm.jit` — `LLJIT`, `TargetMachine`, `link_into`, `host_triple`,
  `registered_targets`.
- `llvm.intrinsics` — intrinsic declarations by name.
- `llvm.dsl` — the `@function` decorator, `range_`, and the control-flow
  machinery.
- `llvm.mir` — LLVM **Machine IR**: the `MachineFunction`/`MachineBasicBlock`/
  `MachineInstr` object model, `LLT`, `MachineIRBuilder`, and the
  `@machine_function` build DSL. See [Machine IR (MIR)](#machine-ir-mir).

## Object layer

Comparable in reach to `llvmlite`, with its own API rather than a compatible
shim. Bound surface:

- **Context and Module** with real ownership. A `llvm.ir.Context` is a context
  manager; leaving the block frees the underlying `LLVMContext`, and touching
  anything that outlived it raises instead of segfaulting.
  `Context._get_live_count()` reports live objects for leak checks.
- **The `Type` hierarchy** (`llvm.types`): integers, floats, pointers, structs
  (named and literal), arrays, vectors, functions. Values downcast to their
  concrete Python class automatically.
- **The `Value` hierarchy** down to the instruction classes worth traversing:
  `PHINode`, `CallInst`, `GetElementPtrInst`, `AllocaInst`, `LoadInst`,
  `StoreInst`, the atomics, and the branch and compare instructions.
  `Argument`, `BasicBlock`, `Function`, `GlobalVariable`, and the `Constant`
  subclasses are all here too.
- **`IRBuilder`** with an MLIR-style insertion-point model (see below).
- **Attributes, linkage, visibility, calling convention.**
- **Metadata**: `MDString`, `MDNode`, named module metadata.
- **Errors as exceptions.** A bad parse raises `ParseError`, a failed
  `verify()` raises `VerifyError`, and any `llvm::Expected`/`llvm::Error`
  failure raises `RuntimeError` carrying the LLVM message.
- **Verify, and bitcode read/write.**
- **Optimization** through `llvm.passmanager.run_passes(module, "instcombine,gvn")`,
  which runs a new-pass-manager pipeline parsed from the string and raises on an
  unknown pass.
- **`TargetMachine`** (`llvm.jit`), with assembly and object emission;
  `host_triple()` and `registered_targets()`.
- **Module linking** through `llvm.jit.link_into(dest, src)`, which raises on
  conflicting symbols.
- **ORC `LLJIT`**: add a module, look up a symbol, get an address you can call
  through `ctypes`.
- **Intrinsics** by name. `llvm.intrinsics.sqrt(module, [f64])` resolves the
  intrinsic ID, uses the given overload types, and inserts the declaration, all
  against LLVM's own tables. The primitives `lookup_intrinsic_id`,
  `intrinsic_is_overloaded`, and `get_intrinsic_declaration` are there too.

```python
import llvm

with llvm.ir.Context() as ctx:
    mod = llvm.ir.parse_assembly(
        "define i32 @f(i32 %x) {\n"
        "entry:\n"
        "  %s = add i32 %x, 1\n"
        "  ret i32 %s\n"
        "}\n",
        ctx, "m",
    )
    mod.verify()
    add = mod.get_function("f").basic_blocks[0].instructions[0]
    print(type(add).__name__)  # BinaryOperator
```

### Building IR: insertion points and instruction emitters

An `IRBuilder` carries an insertion point. `with InsertPoint(...):` positions it
and restores the previous point on exit, mirroring MLIR's `ir.InsertionPoint`:
`InsertPoint(block)` appends at the end of a block, `InsertPoint(instruction)`
inserts before an instruction, and there are `InsertPoint.at_block_begin`,
`at_block_terminator`, and `after` factories. Entering `with builder:` (or
passing `builder=` to `InsertPoint`) makes a builder the *contextual* builder,
so `current_builder()` and `current_function()` resolve without threading the
builder through every call.

`llvm.instructions` provides a free function for every `IRBuilder` emitter
(`add`, `load`, `gep`, `icmp`, `select`, `switch_`, `atomic_rmw`, `call`, …).
Each takes an optional keyword-only `builder`; when omitted it uses the
contextual builder. Plain Python numbers are materialized into constants of the
inferred type (a sibling operand's type, the function's return type for `ret`,
or the callee's parameter type for `call`), so you can write `add(x, 1)` or
`ret(0)` directly.

```python
import llvm
from llvm import ir, types
from llvm import instructions as I

with ir.Context() as ctx:
    i32 = types.i32()
    mod = ir.Module("m", ctx)
    fn = ir.Function.create(types.function(i32, [i32]), "inc", mod)
    b = ir.IRBuilder(ctx)
    with ir.InsertPoint(fn.append_basic_block("entry"), builder=b):
        I.ret(I.add(fn.arg(0), 1, "s"))   # `1` becomes an i32 constant
    print(str(mod))
```

### Python-driven passes

Besides running LLVM's own passes from a string, the *body* of a pass can be a
Python callable. Two one-shot entry points run a callable directly:
`run_python_pass_on_module(module, callback)` runs it as a module pass, and
`run_python_pass_on_function(module, callback)` runs it once per defined function
(the callback receives the `Function`). To compose a Python pass with builtins in
a `run_passes` pipeline, register it by name: `register_python_pass(name,
callback, on=PassKind.MODULE)` names a module pass used directly (e.g.
`"my-pass,instcombine"`), and `on=PassKind.FUNCTION` names a function pass invoked
inside a `function(...)` pipeline (e.g. `"function(my-pass)"`). Pick a name that
does not collide with a builtin pass.

Return a truthy value if the callback mutated the IR (so analyses are
invalidated), `None`/falsy otherwise — reporting "unchanged" after a mutation
leaves stale analyses, so return `True` when unsure. An exception raised in the
callback propagates out of the call and leaves the module usable. The pass runs
synchronously on the calling thread with the GIL held. These passes operate on
LLVM IR (module and function scope).

```python
def rename(m):
    m.get_function("f").name = "g"
    return True  # mutated the IR

llvm.passmanager.run_python_pass_on_module(mod, rename)
```

## DSL layer

Decorate a function with `@llvm.dsl.function(module=...)`. Type annotations are
LLVM types. Arithmetic and comparison operators emit the matching instruction,
dispatched on the operand type: `+` becomes `add` for integers and `fadd` for
floats, `<` becomes `icmp slt` or `fcmp olt`. A Python `int` or `float` operand
coerces to a constant of the other side's type. This matches `ArithValue` in
`eudsl-python-extras` so moving between the MLIR DSL and this one costs nothing.

```python
import ctypes
import llvm
from llvm import ir, types, jit
from llvm.dsl import function, range_
from llvm.ast.canonicalize import canonicalize
from llvm.dsl.cf import LLVMCanonicalizer

ctx = ir.Context()
mod = ir.Module("m", ctx)
i32 = types.i32(ctx)

@function(module=mod)
def inc(x: i32) -> i32:
    return x + 1

@function(module=mod)
@canonicalize(using=LLVMCanonicalizer())
def total(n: i32) -> i32:
    acc = ir.const_int(i32, 0)
    for i in range_(0, n):
        acc = acc + i
        yield acc          # loop-carried value -> phi at the header
    return acc

j = jit.LLJIT()
j.add_module(mod)
f = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(j.lookup("total"))
assert f(5) == 0 + 1 + 2 + 3 + 4
```

Control flow uses a `yield` protocol borrowed from `eudsl-python-extras`' `scf`
dialect. Functions that use `if`/`elif`/`else`/`while`/`for` add
`@canonicalize(using=LLVMCanonicalizer())` beneath `@function`; the canonicalizer
rewrites those statements into basic blocks, and any value you `yield` out of a
region becomes a phi node at the join or loop header. `if` produces a branch and
a join-block phi. `while` and `for` produce a header/body/exit structure with
loop-carried phis for whatever the body yields. `for i in range_(lo, hi, step)`
gives you an induction variable.

`@function` also covers declarations (an empty `...` body), definitions, calls
between decorated functions (via `__call__`), multiple return values, function
attributes, linkage, calling convention, and varargs.

```python
@function(module=mod)
@canonicalize(using=LLVMCanonicalizer())
def pick(c: types.i1(ctx), a: i32, b: i32) -> i32:
    if c:
        r = yield a + 1
    else:
        r = yield b
    return r
```

### Custom value casters

Like MLIR's `register_value_caster`, you can register a Python subclass to wrap
values of a given type kind. The DSL uses this to make integer and float values
come back as `ArithValue` (the class carrying the operator overloads).

```python
from llvm import ir
from llvm.types import TypeID
from llvm.dsl.casters import register_value_caster

@register_value_caster(TypeID.Integer)
class MyInt(ir.Value):
    ...
```

## Machine IR (MIR)

`llvm.mir` binds LLVM's **Machine IR** — the target-level representation the
backend uses after instruction selection — and adds a build DSL on top, mirroring
the IR layering above. It covers three uses: **inspect** MIR the compiler
produces, **build** MIR from Python (generic GlobalISel `G_*` or target-specific),
and **lower** built MIR to native code.

The object model is bound as its own hierarchy (MIR is not part of the `Value`
hierarchy): `MachineFunction`, `MachineBasicBlock`, `MachineInstr`,
`MachineOperand`, `Register`, plus `LLT` (the generic low-level type) and
`MachineFunctionProperty`. `MirModule` owns the `MachineFunction`s and
keeps everything they reference alive. A `Register` carries the
`MachineFunction` that minted it (a physical register carries none), so a vreg
passed into a *different* function's builder is rejected — its numeric id would
otherwise silently alias a same-typed register in that function.

> The examples below use AArch64 opcodes/registers/triples, so they need the
> AArch64 backend linked (`EUDSL_LLVMPY_TARGETS`); on other hosts
> `mf.opcode("ADDWrr")` raises `KeyError`. JIT-*executing* the Route B object
> (the last example's `add(2, 3)`) additionally needs an AArch64 host — as the
> source tests' `skipif` guards encode.

### Inspecting MIR from the compiler

`run_codegen_to_mir` runs instruction selection on an IR module and hands back the
`MirModule` owning the resulting `MachineFunction`s. Pass
`global_isel=True` for the GlobalISel pipeline instead of SelectionDAG.
`to_mir()` prints the `.mir` textual form and `parse_mir` reads it back;
`MachineFunction.verify()` runs the machine verifier.

```python
from llvm import ir, jit, mir

with ir.Context() as ctx:
    mod = ir.parse_assembly(
        "define i32 @add(i32 %a, i32 %b) {\n"
        "  %s = add i32 %a, %b\n"
        "  ret i32 %s\n"
        "}\n",
        ctx, "m",
    )
    mmi = mir.run_codegen_to_mir(mod, jit.TargetMachine(triple="aarch64-unknown-linux-gnu"))
    mf = mmi.machine_function("add")
    print([i.opcode_name for i in mf.blocks[0].instructions])  # COPY, COPY, ADDWrr, COPY, RET_ReallyLR
    print(mf.verify())                                          # True
```

### Building generic MIR: `@machine_function`

`@machine_function` mirrors the IR `@function` DSL, one level lower. Parameters
are annotated with an `LLT`; each arrives as a `MachineValue` over a fresh generic
virtual register, and `+ - *` and comparisons emit `G_ADD`/`G_SUB`/`G_MUL`/
`G_ICMP` through a contextual `MachineIRBuilder`. Python ints coerce to
`G_CONSTANT`s. `if`/`else` and `for`/`while` lower to `MachineBasicBlock`s and
`G_PHI` nodes, reusing the same `@canonicalize` yield-protocol as the IR DSL — only
the runtime differs (`MIRCanonicalizer`). The canonicalizer injects the loop
builtins (`range_`, `while_`, …) into the decorated function, so they need no
import.

```python
from llvm import ir, jit, mir
from llvm.dsl import machine_function
from llvm.dsl.machine_cf import MIRCanonicalizer
from llvm.ast.canonicalize import canonicalize

with ir.Context() as ctx:
    mod = ir.Module("m", ctx)
    tm = jit.TargetMachine(triple="aarch64-unknown-linux-gnu")
    s32 = mir.LLT.scalar(32)

    @machine_function(module=mod, target=tm)
    @canonicalize(using=MIRCanonicalizer())
    def total(n: s32, acc: s32):
        for i in range_(0, n):
            acc = acc + i
            yield acc          # loop-carried value -> G_PHI at the header
        return acc

    print("G_PHI" in total.to_mir())   # True
```

For lower-level construction, `MachineIRBuilder` exposes the typed `build_*`
helpers (`build_constant`, `build_add`, `build_br`, `build_brcond`, `build_phi`,
…) and a generic `build_instr(opcode)` for any opcode; `MachineFunction.opcode`
looks a target opcode up by mnemonic (the generated opcode enums are not in
installed LLVM headers, so lookup is by name via `TargetInstrInfo`). For
control flow, `branch(dest)` and `cond_branch(cond, true_block, false_block)`
fold the terminator emission and the CFG successor edge(s) into one call — MIR
tracks the successor list and the branch operand separately, so these keep the
two in sync (`cond_branch` returns the fall-through `G_BR` so its target can be
repointed).

### Building target MIR and lowering to native code

Because GlobalISel's fallback re-runs SelectionDAG *from the IR* — which
hand-built MIR does not have — the robust way to reach native code is to build
already-selected **target** MIR directly and run only the back half of codegen.
`reg_class`/`physreg` resolve register classes and physical registers by name,
`create_vreg` makes class-constrained vregs, `add_reg` appends operands with the
full flag set (def/use, implicit, kill, …), and `set_property` marks the function.
(`build(opcode, dsts, srcs)` is a typed one-shot alternative to
`build_instr` + `add_reg`: each dst is an `LLT` to mint a fresh vreg for or a
`Register` to define, each src is a `Register` use.)
`emit_object()` then runs register allocation and emission (via
`-start-after=finalize-isel`, so no instruction selection), and `LLJIT.add_object`
loads the result:

```python
import ctypes
from llvm import ir, jit, mir

with ir.Context() as ctx:
    mod = ir.Module("m", ctx)
    tm = jit.TargetMachine()                       # host triple, so the object loads in-process
    mmi = mir.create_machine_function(mod, tm, "add")
    mf = mmi.machine_function("add")
    b = mir.MachineIRBuilder(mf)
    gpr32 = mf.reg_class("GPR32")
    w0, w1 = mf.physreg("W0"), mf.physreg("W1")
    mf.blocks[0].add_livein(w0); mf.blocks[0].add_livein(w1)
    v0, v1, v2 = (mf.create_vreg(gpr32) for _ in range(3))
    for dst, src in ((v0, w0), (v1, w1)):
        b.build(mf.opcode("COPY"), [dst], [src])   # dst = COPY src
    b.build(mf.opcode("ADDWrr"), [v2], [v0, v1])   # v2 = ADDWrr v0, v1
    b.build(mf.opcode("COPY"), [w0], [v2])         # $w0 = COPY v2
    # RET_ReallyLR takes an *implicit* use of $w0, which build's plain-use srcs
    # can't express, so drop to build_instr + add_reg for that one operand.
    r = b.build_instr(mf.opcode("RET_ReallyLR")); r.add_reg(w0, implicit=True)
    for p in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, p))

    j = jit.LLJIT(); j.add_object(mmi.emit_object())
    add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(j.lookup("add"))
    assert add(2, 3) == 5
```

### Python-driven instruction scheduling

`mir.MachineSchedStrategy` is the pre-RA MachineScheduler strategy interface,
subclassable from Python. Override `initialize(dag)`, `get_policy()`,
`release_top_node(su)` / `release_bottom_node(su)` (LLVM hands you ready
`SUnit`s; you maintain your own ready set), `pick_node()` → `(SUnit,
is_top_node)` or `None` when nothing is ready, and `sched_node(su, is_top)`.
Register a subclass under a name and select it per emission:

```python
class MySched(mir.MachineSchedStrategy):
    def initialize(self, dag): self.q = []
    def get_policy(self):
        p = mir.MachineSchedPolicy(); p.only_top_down = True
        p.should_track_pressure = False; return p
    def release_top_node(self, su): self.q.append(su)
    def release_bottom_node(self, su): pass
    def pick_node(self): return (self.q.pop(0), True) if self.q else None
    def sched_node(self, su, is_top): pass

mir.register_scheduler("mysched", MySched)
obj = mmi.emit_object(scheduler="mysched")   # runs MySched as the pre-RA scheduler
```

For the common case, subclass `mir.ReadyQueueStrategy` and override only
`pick(ready)`.

### Python-driven register allocation

`mir.RegAllocBase` is `llvm::RegAllocBase` — the register-allocation driver and
interface — subclassable from Python. The one required override is
`select_or_split(li)`: for each unassigned live interval, return a physical
register id, or return `None` after handling it yourself (spill or split, which
append new virtual registers that get re-enqueued). The allocator queries and
mutates state through `self`: `allocation_order(li)` (physregs to try, in target
order), `interfering_vregs(li, physreg)` (ids of vregs whose ranges interfere
with `li` on that physreg -- assigned to it or to an aliasing physreg; the
eviction candidates),
`register_cost(physreg)` (per-use cost, the CostPerUseLimit heuristic),
`self.matrix` (`is_free`/`check_interference`/`assign`/`unassign`),
`self.lis` (LiveIntervals: `instruction_index`, `mbb_start_index`,
`mbb_end_index`, `has_interval`, `interval`), `self.vrm` (VirtRegMap:
`has_phys`/`get_phys`, to read an interferer's current physreg before evicting
it; and, for the eviction cost model, `self.reg_allocation_hints(reg)` (a
`(type, [ids])` pair) / `self.simple_hint(reg)` (broken-hint accounting),
`self.matrix.is_phys_reg_used`,
and `self.last_callee_saved_alias(physreg)`),
`self.machine_function`, and `self.mbfi` (MachineBlockFrequencyInfo:
`block_freq`, `block_freq_relative_to_entry_block`, `entry_freq`, for
frequency-weighted spill/split cost models). The `li` passed in is a
`LiveInterval` exposing `reg`, `weight`, `is_spillable`, its own extent
(`begin_index`/`end_index`/`size`; `SlotIndex.distance` measures raw slot-space
distance and `SlotIndex.get_approx_instr_distance` measures instruction-space
gaps), its value numbers (`get_vni_at`, `num_val_nums`, `get_val_num_info`), and
its `segments()` (each a `[start, end)` with a `valno`, for reconstructing
per-block/per-gap interference). For priority/pressure heuristics,
`self.reg_class(reg)` (a `TargetRegisterClass` with `id`),
`self.num_allocatable_regs(reg_class)`,
`self.is_trivially_rematerializable(mi)`, and
`self.calculate_spill_weight_and_hint(reg)` (recompute a split product's
weight). Optional overrides: `enqueue(reg)`/`dequeue()` (drive
the assignment order; both traffic in register ids -- stable across splitting,
unlike interval objects -- and `dequeue` returns a reg id or `None`; the default
is a spill-weight priority queue), `post_optimization()`, and
`about_to_remove_interval(li)`.

Register a subclass under a name and select it per emission. For the trivial
first-free-or-spill allocator, use the built-in `mir.BasicRegAlloc`:

```python
class FirstFree(mir.RegAllocBase):
    def select_or_split(self, li):
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)        # append spill-code vregs; they get re-enqueued
        return None

mir.register_regalloc("first-free", FirstFree)
obj = mmi.emit_object(regalloc="first-free")   # drives FirstFree instead of greedy
```

For RAGreedy-style live-range splitting, `self.split_analysis` (a
`SplitAnalysis`: `analyze(li)`, `use_blocks()`, `num_through_blocks()`,
`through_blocks()`, `get_use_slots()`, `last_split_point(mbb)`) plans the split and
`self.split_editor` (a `SplitEditor`: `reset`, `open_intv`, `enter_intv_*`,
`use_intv`/`use_intv_mbb`, `leave_intv_*`, `overlap_intv`, `finish`) applies it,
writing into `self.new_live_range_edit(li)`:

```python
class RegionSplit(mir.RegAllocBase):
    def select_or_split(self, li):
        sa = self.split_analysis
        sa.analyze(li)
        if sa.num_through_blocks() > 0:                 # live across a block: split it out
            se = self.split_editor
            se.reset(self.new_live_range_edit(li))
            se.open_intv()
            bi = [b for b in sa.use_blocks() if not b.live_out][-1]
            start = se.enter_intv_before(bi.first_instr)
            se.use_intv(start, se.leave_intv_after(bi.last_instr))
            se.finish()                                 # new vregs re-enqueued
            return None
        for preg in self.allocation_order(li):
            if self.matrix.is_free(li, preg):
                return preg
        self.spill(li)
        return None
```

For global (region) splitting, `self.edge_bundles` (an `EdgeBundles`:
`get_bundle(mbb, out)`, `num_bundles()`, `get_blocks(bundle)`) and
`self.spill_placer` (a `SpillPlacement` Hopfield network: `prepare(bit_vector)`,
`add_constraints([BlockConstraint])`, `add_pref_spill`, `add_links`,
`scan_active_bundles`, `iterate`, `get_recent_positive`, `finish`,
`get_block_frequency`) expose the machinery RAGreedy's `splitAroundRegion`
drives. Build `mir.BlockConstraint`s (with `mir.BorderConstraint` entry/exit
prefs) from the live-through use blocks, iterate the network, and read the
in-register edge bundles out of the `mir.BitVector` passed to `prepare` to
choose split boundaries.

For rematerialization (recomputing a value at its use instead of keeping it
live), a value's def is reached through its value number: `li.get_vni_at(idx)` /
`li.get_val_num_info(i)` return a `VNInfo` (`def_index`, `is_phi_def`,
`is_unused`), and `self.lis.instr_from_index(vni.def_index)` gives the defining
instruction. Build a `mir.LiveRangeEdit.Remat(vni)` with its `orig_mi` set, then
`lre.rematerialize_at(mbb, before, dest_reg, remat)` clones the def into a fresh
`lre.create()` vreg before a use; redirect the use with
`use_mi.substitute_register(old, new)`, compute the clone's interval with
`self.lis.compute_interval(new)`, and drop the now-dead original with
`self.lis.shrink_to_uses(old)` feeding `lre.eliminate_dead_defs(dead)`. (Note the
built-in spiller already rematerializes trivially-rematerializable defs inside
`self.spill()`; this surface is for driving remat yourself.)

A fresh allocator instance is constructed per `MachineFunction`. Because this
build has assertions enabled, an invalid split aborts with a diagnostic rather
than emitting bad code, and an exception raised in any override propagates out
of `emit_object`.

### ILP register allocators (optional, requires OR-Tools)

Integer-linear-programming register allocators, solved with Google OR-Tools
CP-SAT, are available when the `ilp` extra is installed
(`pip install eudsl-llvmpy[ilp]`). Each subclasses a shared `RAILPBase` that
collects every seeded vreg, solves one global model on the first `dequeue`, and
answers each `select_or_split` from the cached solution. The solution is
verified against the live-register matrix: a missing or infeasible ILP decision
is a **hard error** (no silent greedy fallback), so a model bug surfaces rather
than being masked. Register assignments are read back with `regalloc_assignments`
like any allocator.

- `mir.RAILPPacking` — 2D no-overlap rectangle packing (time × register), one
  rectangle per live segment; the register variable's domain includes a private
  memory slot so a value in the memory region means spilled. Single register
  class only (the flat register axis cannot model aliasing).

Whole-interval spill decisions ignore reload register pressure and are not
reliably realizable, so `RAILPPacking` is scoped to register-fitting functions
and hard-fails cleanly when a function needs spilling.

## Limitations

- **`break`, `continue`, and early `return` inside DSL control flow are not
  supported.** They would need edge duplication and predecessor bookkeeping the
  phi-based yield-protocol lowering does not do; the canonicalizer detects them
  and raises `NotImplementedError` rather than emitting wrong IR.
- **`if`/`while`/`for` otherwise compose freely** — an `if` inside a loop, a
  loop inside an `if` branch, nested loops, etc. — with one caveat: **do not
  reassign a variable in one `if`/`elif` branch and read it in a sibling
  branch.** Both branches are traced in the same Python frame, so the
  reassignment leaks across; `verify()` then rejects the IR. Yield the value out
  of the region instead (`r = yield x`) or use a distinct name per branch.
- **The fatal-error path is best-effort.** LLVM's fatal error handler cannot
  return, so where a bad argument would trip it, the bindings validate up front
  and raise. A genuine LLVM fatal error still aborts the process after a warning.
- **No raw C-API escape hatch.** The `llvm-c` surface is not exposed. If
  something isn't bound, it isn't reachable from Python; the fix is to bind it.
- **Scope of the binding.** `llvm/IR` is not bound exhaustively. The rule is:
  what the DSL needs, what `llvmlite.binding` exposes, and everything reachable
  by traversal from a `Module`. Out of scope for now: DIBuilder/DebugInfo,
  remarks, the disassembler, object-file inspection, funclet-based exception
  handling, and ORC customization points that need Python callbacks.
- **Targets.** The build links the host target by default (whatever
  `LLVM_NATIVE_ARCH` is), plus AMDGPU whenever the LLVM being built against
  provides it (it is in the default distribution) — AMDGPU is the one target
  with sub-register liveness, which `mir.RAGreedy`'s `tryInstructionSplit` test
  needs. Extra targets stay reachable through the `EUDSL_LLVMPY_TARGETS` CMake
  option. Emitting AMDGCN assembly never needed an AMD device; only running it
  did.

## Build

Needs an LLVM install (headers and libraries) and a C++ compiler. Point
`CMAKE_PREFIX_PATH` at the install, then:

```bash
pip install -e . --no-build-isolation
```

CMake options, passed through `--config-settings=cmake.define.<NAME>=<VALUE>`:

- `EUDSL_LLVMPY_TARGETS` (default: the host target, `LLVM_NATIVE_ARCH`) selects
  which LLVM target backends to link and initialize; AMDGPU is added on top
  whenever the linked LLVM provides it.
- `EUDSL_LLVMPY_ENABLE_COVERAGE` (default `OFF`) instruments the extension for
  `llvm-cov`.

## Tests and coverage

```bash
pytest tests
```

Python coverage is gated in `pyproject.toml` (`--cov=llvm`, branch coverage,
fail under 99%). C++ coverage runs through `scripts/cpp_coverage.sh`, which
builds the extension instrumented, runs the suite under `LLVM_PROFILE_FILE`,
merges the profile, and enforces a `src/IR` line-coverage threshold with
`scripts/check_coverage.py`:

```bash
COVERAGE_THRESHOLD=100 bash scripts/cpp_coverage.sh
```

Both gates run in CI.
