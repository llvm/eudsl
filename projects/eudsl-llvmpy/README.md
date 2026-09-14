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

### Building IR: insertion points and instruction emitters

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
coerces to a constant of the other side's type. Control flow uses a `yield` protocol borrowed from MLIR's `scf`
dialect.

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

Functions that use `if`/`elif`/`else`/`while`/`for` add
`@canonicalize(using=LLVMCanonicalizer())` beneath `@function`; the canonicalizer
rewrites those statements into basic blocks, and any value you `yield` out of a
region becomes a phi node at the join or loop header. `if` produces a branch and
a join-block phi. `while` and `for` produce a header/body/exit structure with
loop-carried phis for whatever the body yields. `for i in range_(lo, hi, step)`
gives you an induction variable.

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

## Machine IR (MIR)

`llvm.mir` binds LLVM's **Machine IR** — the target-level representation the
backend uses after instruction selection — and adds a build DSL on top, mirroring
the IR layering above. It covers three uses: **inspect** MIR the compiler
produces, **build** MIR from Python (generic GlobalISel `G_*` or target-specific),
and **lower** built MIR to native code.

### Building generic MIR: `@machine_function`

`@machine_function` mirrors the IR `@function` DSL, one level lower. Parameters
are annotated with an `LLT`; each arrives as a `MachineValue` over a fresh generic
virtual register, and `+ - *` and comparisons emit `G_ADD`/`G_SUB`/`G_MUL`/
`G_ICMP` through a contextual `MachineIRBuilder`. Python ints coerce to
`G_CONSTANT`s. `if`/`else` and `for`/`while` lower to `MachineBasicBlock`s and
`G_PHI` nodes, reusing the same `@canonicalize` yield-protocol as the IR DSL — only
the runtime differs (`MIRCanonicalizer`).

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
