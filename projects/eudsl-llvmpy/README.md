<!--
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# eudsl-llvmpy

Python bindings for LLVM IR, plus a small DSL for authoring IR from Python. Both
ship in the top-level `llvm` package.

There are two layers. The lower one is a hand-written [nanobind](https://github.com/wjakob/nanobind)
wrapper over the LLVM **C++** API, so the Python class hierarchy mirrors
`llvm::` (a `Value` really is the base of `Instruction`, `Constant`, and the
rest). The upper one is a DSL: decorate a Python function, write ordinary
arithmetic and `if`/`while`/`for`, and get back a compiled `llvm::Function`
whose control flow is lowered to basic blocks and phi nodes.

The DSL sits on top of the bindings. You can use the bindings alone.

## Object layer

Comparable in reach to `llvmlite`, with its own API rather than a compatible
shim. Bound surface:

- **Context and Module** with real ownership. A `Context` is a context manager;
  leaving the block frees the underlying `LLVMContext`, and touching anything
  that outlived it raises instead of segfaulting. `Context._get_live_count()`
  reports live objects for leak checks.
- **The `Type` hierarchy**: integers, floats, pointers, structs (named and
  literal), arrays, vectors, functions. Values downcast to their concrete
  Python class automatically.
- **The `Value` hierarchy** down to the instruction classes worth traversing:
  `PHINode`, `CallInst`, `GetElementPtrInst`, `AllocaInst`, `LoadInst`,
  `StoreInst`, and the branch and compare instructions. `Argument`,
  `BasicBlock`, `Function`, `GlobalVariable`, and the `Constant` subclasses are
  all here too.
- **`IRBuilder`** with a `with builder.at_end_of(block):` insertion-point
  context manager.
- **Attributes, linkage, visibility, calling convention.**
- **Metadata**: `MDString`, `MDNode`, named module metadata.
- **Errors as exceptions.** A bad parse raises `ParseError`, a failed
  `verify()` raises `VerifyError`, and any `llvm::Expected`/`llvm::Error`
  failure raises `RuntimeError` carrying the LLVM message.
- **Verify, and bitcode read/write.**
- **Optimization** through `llvm.run_passes(module, "instcombine,gvn")`, which
  runs a new-pass-manager pipeline parsed from the string and raises on an
  unknown pass.
- **`Target`, `TargetMachine`, `DataLayout`**, with assembly and object
  emission.
- **Module linking** through `llvm.link_into(dest, src)`, which raises on
  conflicting symbols.
- **ORC `LLJIT`**: add a module, look up a symbol, get an address you can call
  through `ctypes`.
- **Intrinsics** by name. Import `llvm.intrinsics`, then
  `llvm.intrinsics.sqrt(module, [f64])` resolves the intrinsic ID, uses the
  given overload types, and inserts the declaration, all against LLVM's own
  tables. The primitives `lookup_intrinsic_id`, `intrinsic_is_overloaded`, and
  `get_intrinsic_declaration` are there too.

Because `llvm::Value` and `llvm::Type` have no vtables, nanobind's RTTI-based
downcasting can't apply. The bindings use `nanobind::detail::type_hook` keyed on
`Value.def`/`Instruction.def` and `Type::TypeID` to pick the right Python class
for a returned pointer.

```python
import llvm

with llvm.Context() as ctx:
    mod = llvm.parse_assembly(
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

## DSL layer

Decorate a function with `@llvm.function`. Type annotations are LLVM types.
Arithmetic and comparison operators emit the matching instruction, dispatched on
the operand type: `+` becomes `add` for integers and `fadd` for floats, `<`
becomes `icmp slt` or `fcmp olt`. A Python `int` or `float` operand coerces to a
constant of the other side's type. This matches `ArithValue` in
`eudsl-python-extras` so moving between the MLIR DSL and this one costs nothing.

```python
import ctypes
import llvm
from llvm.dsl.cf import range_

ctx = llvm.Context()
mod = llvm.Module("m", ctx)
i32 = llvm.i32(ctx)

@llvm.function(module=mod)
def inc(x: i32) -> i32:
    return x + 1

@llvm.function(module=mod)
def total(n: i32) -> i32:
    acc = llvm.const_int(i32, 0)
    for i in range_(0, n):
        acc = acc + i
        yield acc          # loop-carried value -> phi at the header
    return acc

jit = llvm.LLJIT()
jit.add_module(mod)
f = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total"))
assert f(5) == 0 + 1 + 2 + 3 + 4
```

Control flow uses a `yield` protocol borrowed from `eudsl-python-extras`' `scf`
dialect. An AST canonicalizer rewrites `if`/`elif`/`else`/`while`/`for` into
basic blocks, and any value you `yield` out of a region becomes a phi node at
the join or loop header. `if` produces a branch and a join-block phi. `while`
and `for` produce a header/body/exit structure with loop-carried phis for
whatever the body yields. `for i in range_(lo, hi, step)` gives you an induction
variable.

`@llvm.function` also covers declarations (an empty `...` body), definitions,
calls between decorated functions (via `__call__`), multiple return values,
function attributes, linkage, calling convention, and varargs.

```python
@llvm.function(module=mod)
def pick(c: llvm.i1, a: i32, b: i32) -> i32:
    if c:
        r = yield a + 1
    else:
        r = yield b
    return r
```

### Custom value casters

Like MLIR's `register_value_caster`, you can register a Python subclass to wrap
values of a given type kind. The DSL uses this to make integer and float values
come back as `ArithValue` (the class carrying the operator overloads). The
registry lives in Python (`llvm.dsl.casters`), so nothing holds Python
references at interpreter shutdown; C++ exposes only a stateless
`_wrap_value_as` primitive.

```python
from llvm.dsl.casters import register_value_caster
from llvm.eudslllvm_ext import TypeID

@register_value_caster(TypeID.Integer)
class MyInt(llvm.Value):
    ...
```

## Limitations

- **`break`, `continue`, and early `return` inside DSL control flow are not
  supported.** The loop transforms lift a body into a nested function that the
  `if`/`else` transformer does not revisit, so a bare `break` or `return` there
  would emit wrong IR silently. The transformer detects these and raises
  `NotImplementedError` with a message rather than miscompiling.
- **Control flow nested inside a loop body is rejected** for the same reason.
  An `if` at the top level of a function works; an `if` inside a `while` or
  `for` body raises `NotImplementedError`. This is the piece most worth
  revisiting.
- **The fatal-error path is best-effort.** LLVM's fatal error handler cannot
  return, so where a bad argument would trip it, the bindings validate up front
  and raise. A genuine LLVM fatal error still aborts the process after a warning.
- **No raw C-API escape hatch.** The `llvm-c` surface and `from_capsule` are
  not exposed. If something isn't bound, it isn't reachable from Python; the fix
  is to bind it.
- **Scope of the binding.** `llvm/IR` is not bound exhaustively. The rule is:
  what the DSL needs, what `llvmlite.binding` exposes, and everything reachable
  by traversal from a `Module`. Out of scope for now: DIBuilder/DebugInfo,
  remarks, the disassembler, object-file inspection, and ORC customization
  points that need Python callbacks.
- **Targets.** The build links host targets only by default (AArch64 and X86).
  AMDGPU and NVPTX stay reachable through the `EUDSL_LLVMPY_TARGETS` CMake
  option. Emitting AMDGCN assembly never needed an AMD device; only running it
  did.

## Build

Needs an LLVM install (headers and libraries) and a C++ compiler. Point
`CMAKE_PREFIX_PATH` at the install, then:

```bash
pip install -e . --no-build-isolation
```

CMake options, passed through `--config-settings=cmake.define.<NAME>=<VALUE>`:

- `EUDSL_LLVMPY_TARGETS` (default `AArch64;X86`) selects which LLVM target
  backends to link and initialize.
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
