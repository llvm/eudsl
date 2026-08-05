# eudsl-llvmpy: LLVM IR bindings and DSL frontend

Date: 2026-08-04

## Goal

Turn `eudsl-llvmpy` into two things at once:

1. A full LLVM IR manipulation library, comparable in capability to `llvmlite`
   (build, parse, traverse, mutate, optimize, target-compile, JIT).
2. A DSL frontend for authoring LLVM IR from Python, comparable to what
   `eudsl-python-extras` provides for MLIR (operator overloading, Python
   control flow lowered to blocks and phi nodes, decorated functions).

The DSL layer sits on the binding layer. Both are delivered from the same
`llvm` package.

## Current state

`projects/eudsl-llvmpy` today is:

- `eudsl-llvmpy-generate.py` (440 lines) which runs `litgen` over the LLVM
  **C** headers (`llvm-c/*.h`) and emits ~1181 free functions into
  `eudslllvm_ext`, plus a 6039-line `amdgcn.py` scraped from `Intrinsics.td`
  via `eudsl-tblgen`.
- Thin Python wrappers: `types_.py` (147), `instructions.py` (727),
  `function.py` (147), `context.py` (62), `util.py` (42).
- 4 tests in `tests/test_bindings.py`.

Handles are bare one-field structs (`struct LLVMValueRef { void* ptr; }`) with
no hierarchy, no methods, no ownership, and no iteration. `dispose_*` is
manual, so a stale `ModuleRef` is a segfault rather than an exception.

## Decisions

These were settled during design and are not open questions.

| Decision | Choice |
|---|---|
| llvmlite compatibility | Feature parity, own API. No import-compatible shim. |
| Binding layer | Hand-written nanobind over the LLVM **C++** API. |
| litgen / C API layer | Deleted entirely. No raw escape hatch retained. |
| Cutover sequencing | litgen deleted in commit 1, together with a minimal working replacement. |
| SSA construction | Explicit `yield` in DSL bodies lowered to phi nodes, mirroring `eudsl-python-extras`' `scf` machinery. |
| AST canonicalizer | Vendored copy into `eudsl-llvmpy`; the two copies may diverge. |
| Existing free-function API | Replaced. `instructions.py` and `types_.py` are deleted. |
| Import path | Stays top-level `llvm`. |
| `from_capsule` | Dropped. Not needed. |
| AMDGPU / NVPTX | Out of scope. Host targets only, behind a CMake option. |
| `break` / `continue` / early `return` | Deferred. Transformer raises `NotImplementedError`. |

## Architecture

Two layers, both shipped in the `llvm` package.

### Binding layer (C++, `src/ir/`)

Hand-written nanobind bindings over `llvm::` C++ classes. The Python class
hierarchy mirrors LLVM's C++ inheritance:

```cpp
nb::class_<llvm::Value>(m, "Value");
nb::class_<llvm::User, llvm::Value>(m, "User");
nb::class_<llvm::Instruction, llvm::User>(m, "Instruction");
nb::class_<llvm::GlobalValue, llvm::Constant>(m, "GlobalValue");
nb::class_<llvm::Function, llvm::GlobalObject>(m, "Function");
```

`BasicBlock` is a `Value` subclass here, matching LLVM, unlike the C API where
`LLVMBasicBlockRef` and `LLVMValueRef` are unrelated handles.

#### Downcasting

`llvm::Value` and `llvm::Type` are **not polymorphic** — they have no vtables.
`Value.h` states the destructor is deliberately non-virtual. nanobind's
automatic RTTI-based downcasting therefore cannot work.

nanobind provides `type_hook<T>::get(ptr)` (declared in `nb_cast.h`) for
exactly this case: a hook that selects the Python type from a non-polymorphic
C++ pointer.

```cpp
template <> struct nanobind::detail::type_hook<llvm::Value> {
  static const std::type_info *get(llvm::Value *v) {
    if (!v) return &typeid(llvm::Value);
    unsigned id = v->getValueID();
    if (id >= llvm::Value::InstructionVal)
      return instructionTypeInfo(id - llvm::Value::InstructionVal);
    switch (id) {
      case llvm::Value::FunctionVal: return &typeid(llvm::Function);
      case llvm::Value::ArgumentVal: return &typeid(llvm::Argument);
      // ... from Value.def
    }
    return &typeid(llvm::Value);
  }
};
```

Two details that must be handled:

- For instructions, `getValueID()` returns `InstructionVal + opcode`, not a
  plain enum member. Instruction dispatch is a second switch keyed on the
  opcode, sourced from `Instruction.def`.
- The 72 `HANDLE_*` entries in `Value.def` and the opcodes in
  `Instruction.def` are expanded by including those `.def` files with local
  macro definitions, so an LLVM bump that adds a value kind is a compile-time
  event rather than a silent fallback to the base class.

`llvm::Type` gets the same treatment keyed on `Type::getTypeID()`.

### DSL layer (Python, `llvm/dsl/` and `llvm/ast/`)

- `llvm/ast/` — vendored copy of `eudsl-python-extras`' `canonicalize.py`,
  `util.py`, `py_type.py` (~590 lines). These are pure `ast`/`dis`/`opcode`
  with no MLIR imports, so the lift is clean.
- `llvm/dsl/values.py` — arithmetic and comparison dunders attached to the
  bound `Value` classes.
- `llvm/dsl/cf.py` — AST transformers lowering `if`/`elif`/`else`/`while`/`for`
  to basic blocks and phi nodes.
- `llvm/dsl/func.py` — the `@function` decorator.
- `llvm/intrinsics.py` — `__getattr__` shim over `Intrinsic::lookupIntrinsicID`.

## Ownership and lifetime

Ownership becomes C++-native rather than hand-rolled.

- Owning: `std::unique_ptr<LLVMContext>`, `std::unique_ptr<Module>`,
  `IRBuilder<>`, `TargetMachine`, `LLJIT`. Exposed with `__enter__`/`__exit__`.
- Non-owning: `Value*`, `Type*`, `BasicBlock*`, `Use*` returned with
  `rv_policy::reference_internal` plus `nb::keep_alive`, tying the reference to
  its owning context or module.
- Transfer: `LLJIT::addIRModule` consumes the module. The Python wrapper marks
  the source moved-from, so later use raises rather than crashes.
- `Context._get_live_count()` is exposed so tests can assert no leaks, matching
  the convention in `mlir/test/python/ir/*.py`.

## Error handling

The LLVM in use is built `LLVM_ENABLE_RTTI=OFF`, `LLVM_ENABLE_EH=OFF`. Our
extension compiles with `-fexceptions -frtti` regardless: `typeid()` requires
RTTI in the translation unit that calls it even for non-polymorphic types, but
that's a property of our own `.cpp` files, not of the LLVM objects we link
against. `type_hook` never calls `typeid()` on a base class through a vtable,
so the two settings don't conflict. Two consequences of the EH/exceptions
split:

**Exceptions must not unwind through LLVM frames.** This is fine for ordinary
calls, where the exception is thrown in our own binding code. It constrains
callbacks: diagnostic handlers, pass instrumentation, and any ORC callback must
catch at the C++/Python boundary and convert to a return value or a stored
error, never let a Python exception propagate into an LLVM frame. Every bound
callback must do this explicitly.

**`report_fatal_error` aborts the process.** Bad data layout strings, bad
target lookups, and similar kill the interpreter with no Python traceback. We
install a fatal error handler that reports the message in a Python-visible
form, but the handler cannot return. Where practical, arguments are validated
before the call. This is a genuine limitation, not something the design fully
solves.

Ordinary errors become exceptions:

- `llvm::Expected<T>` and `llvm::Error` unwrap; failure raises.
- `parseIRFile` / `parseAssembly` diagnostics raise `ParseError` carrying the
  `SMDiagnostic` message.
- `verifyModule` failure raises `VerifyError` carrying the message.

## Binding scope

`llvm/IR` is too large to bind exhaustively, and with no escape hatch, gaps are
costly. The inclusion rule:

> Bind what the DSL layer needs, plus what `llvmlite.binding` exposes, plus
> everything reachable by traversal from a `Module`.

The third clause is what makes the object graph navigable rather than a set of
disconnected entry points.

**In scope:** Context; Module; the Type hierarchy; the Value hierarchy down to
instruction classes carrying interesting accessors (PHINode, CallBase,
GetElementPtrInst, AllocaInst, LoadInst, StoreInst, BranchInst, SwitchInst);
Argument; BasicBlock; Constant subclasses; GlobalVariable; Attribute and
AttributeList; Metadata, MDNode, named metadata; IRBuilder; PassBuilder and the
new pass manager; Target, TargetMachine, DataLayout; Linker; ORC LLJIT;
intrinsic lookup and declaration.

**Out of scope for this spec:** the DIBuilder / DebugInfo surface; remarks; the
disassembler; object file inspection; ORC customization points requiring Python
callbacks.

## Intrinsics

With AMDGPU dropped, intrinsics need no TableGen scraping. Four APIs in
`llvm/IR/Intrinsics.h` cover it (all verified present):

- `Intrinsic::lookupIntrinsicID(StringRef)`
- `Intrinsic::isOverloaded(ID)`
- `Intrinsic::getType(LLVMContext&, ID, ArrayRef<Type*>)`
- `Intrinsic::getOrInsertDeclaration(Module*, ID, ArrayRef<Type*>)`

`llvm/intrinsics.py` becomes a `__getattr__` shim: `llvm.intrinsics.sqrt(x)`
maps to `llvm.sqrt`, resolves the ID, infers overload types from the argument
values, and emits the call. Overload resolution happens in C++ against LLVM's
own tables.

**This removes `eudsl-tblgen` as a build dependency of `eudsl-llvmpy`.**
`amdgcn.py` (6039 lines) and `generate_amdgcn_intrinsics` are deleted. If
static autocompletion is wanted later, a generated `.pyi` covers it without
returning TableGen to the build.

## Targets

A CMake option `EUDSL_LLVMPY_TARGETS` defaults to host only (AArch64 and X86).
The AMDGPU and NVPTX target libraries currently linked unconditionally are
dropped from the default build, cutting build time and binary size.

Note that emitting AMDGCN assembly never required an AMD device; only executing
it did. The option keeps that path reachable by flag rather than by patch.

## DSL semantics

### Values

Arithmetic and comparison dunders attach to the bound `Value` classes.
Dispatch is on `getType()`:

- `a + b` emits `add` for integers, `fadd` for floats.
- `a < b` emits `icmp slt` or `fcmp olt`.
- Python `int` / `float` operands coerce to constants of the other operand's
  type.

This mirrors `ArithValue` in `mlir/extras/dialects/arith.py` deliberately, so
moving between the MLIR DSL and the LLVM DSL costs nothing.

### Control flow

Bodies use the `yield` protocol from `eudsl-python-extras`' `scf` dialect. The
AST canonicalizer rewrites `if`/`elif`/`else`/`while`/`for`, and values yielded
from a region become phi nodes at the join block.

`break`, `continue`, and early `return` inside DSL control flow are not
supported in this spec. The transformer detects them and raises
`NotImplementedError` with a clear message, rather than silently emitting wrong
IR.

### Functions

`@function` handles declarations (empty body), definitions, calls via
`__call__`, multiple returns, function attributes, linkage, calling convention,
and varargs.

## Testing

Three tiers. IR text alone will not catch a wrong phi node, so the third tier
is what actually validates control flow.

1. **pytest units** per module, styled after `mlir/test/python/ir/*.py`, each
   ending with `gc.collect()` and an assertion that
   `Context._get_live_count() == 0`.
2. **FileCheck over emitted IR**, using a vendored adaptation of
   `eudsl-python-extras`' `filecheck_with_comments` fixture.
3. **Execution tests**: JIT the function via LLJIT, call it through ctypes,
   assert the numeric result. Host-target-only means these run on developer
   machines and in CI rather than being aspirational.

## Work breakdown

Each item is one atomic, independently testable commit.

**Accepted tradeoff:** deleting litgen first means the package has *less*
capability than it does today, from commit 1 until roughly commit 21 (where
JIT and intrinsics land and the new layer overtakes the old). Commit 1 deletes
litgen and lands a minimal working replacement in the same commit, and every
commit from there is green. The regression is in capability, not in build
health.

### Phase 0 — scaffolding

1. Delete litgen, `eudslllvm_ext`, `instructions.py`, `types_.py`,
   `amdgcn.py`, `eudsl-llvmpy-generate.py`; drop the litgen and eudsl-tblgen
   build deps; add a hand-written nanobind extension binding `LLVMContext` and
   `Module` with parse and print.

   The four existing tests are handled as: `test_smoke` ported (parse and
   print survive); `test_symbol_collision` ported (it only checks that
   importing `eudsl_tblgen` alongside does not clash, which still applies);
   `test_from_capsule` deleted (`from_capsule` is dropped); `test_builder`
   deleted here and its replacement reintroduced at item 21, since it asserts
   on `amdgcn` intrinsic output.
2. CMake `EUDSL_LLVMPY_TARGETS` option, host-only default.

### Phase A1 — object layer

3. Full Context and Module ownership: RAII, `__enter__`/`__exit__`,
   moved-from tracking, `_get_live_count()`. (Item 1 binds these classes
   minimally; this item makes their lifetime correct.)
4. `Type` base and primitive types; `__str__`, `__eq__`, `__hash__`.
5. Derived types: Integer, Pointer, Struct, Array, Vector, Function.
6. `type_hook` for `Type`, dispatching on `Type::TypeID`.
7. `Value` base: name, type, users, `__str__`, `__eq__`, `__hash__`.
8. `type_hook` for `Value`, generated from `Value.def` and `Instruction.def`,
   including the `InstructionVal + opcode` case.
9. `Function`, `Argument`, `BasicBlock`, and traversal iterators.
10. `Instruction` subclasses with accessors; `PHINode` incoming values.
11. `Constant` subclasses and constant construction.
12. `IRBuilder` with an insertion-point context manager.
13. Attributes, linkage, visibility, calling convention.
14. Metadata, MDNode, named metadata.
15. Error handling: `Expected`/`Error` unwrap, `ParseError`, `VerifyError`,
    fatal error handler.

### Phase A2 — compile and run

16. `verifyModule`; bitcode read and write.
17. `PassBuilder` and pipeline execution.
18. `Target`, `TargetMachine`, `DataLayout`, assembly and object emission.
19. `Linker`.
20. ORC `LLJIT`: add module, lookup, ctypes-callable addresses, execution
    tests.
21. `Intrinsic` lookup and declaration; the `llvm.intrinsics` `__getattr__`
    module.

### Phase B1 — typed values

22. Arithmetic dunders with integer/float dispatch and constant coercion.
23. Comparison dunders to `icmp`/`fcmp`.
24. Pointer, GEP, load, store sugar including `__getitem__`/`__setitem__`.
25. Aggregate construction and indexing.

### Phase B2 — control flow

26. Vendor the AST canonicalizer with its tests; no behavior change.
27. `if`/`else` lowering to blocks and phi nodes.
28. `elif` canonicalization.
29. `while` loops.
30. `for` / `range_` with loop-carried values.
31. `break` / `continue` / early `return` detection raising
    `NotImplementedError`.

### Phase B3 — program structure

32. `@function` rework: declarations, calls, `__call__`, multiple returns.
33. Function attributes, linkage, calling convention, varargs.
34. Globals, constant initializers, address spaces.

## Deferred

- `break`, `continue`, early `return` in DSL control flow.
- DebugInfo / DIBuilder bindings.
- AMDGPU and NVPTX target support (reachable via `EUDSL_LLVMPY_TARGETS`).
- Generated `.pyi` stubs for intrinsic autocompletion.
- An `llvmlite`-compatible import shim.
