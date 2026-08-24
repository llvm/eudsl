# eudsl-llvmpy conventions

Guidance for this project: the hand-written nanobind bindings over the LLVM
C++ API, plus the DSL frontend built on them. These apply to every PR in the
stack, so check new code against them before pushing.

## Mirror the LLVM C++ API

Bind names as LLVM spells them. Do not rename or merge a method or field
because it reads as semantically equivalent. A `Module` has a
`module_identifier` (`getModuleIdentifier`/`setModuleIdentifier`) and a
separate `source_filename` (`getSourceFileName`/`setSourceFileName`). Expose
both. Do not collapse them into one `name`, and do not set the source filename
as a side effect of setting the identifier. If a rename genuinely seems
worthwhile, raise it rather than doing it silently.

## Bind the full API; do not hardcode assumptions

Expose the LLVM API's parameters to the user rather than fixing them to a
default in C++. For example, `Function.create` takes a `linkage` argument
(defaulting to `ExternalLinkage`) instead of hardcoding the linkage; likewise
for calling convention, visibility, address space, and similar. If a binding
would silently bake in one choice from a set the LLVM API offers, surface that
choice as an argument.

## Prefer specific nanobind types over `nb::object`/`nb::handle`

In binding signatures, stored state, and return types, use the most specific
type nanobind can bind rather than a generic `nb::object`/`nb::handle`. If a
parameter is really an `llvm::MachineIRBuilder`, bind it as
`llvm::MachineIRBuilder *`, not `nb::object` — nanobind casts the argument, and
pointer equality gives a precise identity check. When you need to hand the same
Python object back (e.g. a context manager's `__enter__`, or a `current_*()`
accessor), return the pointer with `nb::rv_policy::reference`: nanobind's
instance registry maps it back to the same Python object, so `is` identity
holds. Reach for `nb::object`/`nb::handle` only when the value is genuinely an
arbitrary Python object with no more specific type — e.g. the ignored
`exc_type`/`exc_value`/`traceback` parameters of `__exit__`.

## No forward-reference comments

Do not write comments that reference future work, later PRs, or task numbers
("added in Task 10", "activates once X lands"). They go stale as the stack is
rebased and squash-merged. Describe what the code does now; if something is
intentionally a stub, say so without pointing at a future commit.

## C++ coding style (LLVM)

Follow the LLVM coding standards. A conditional takes braces whenever its
condition spans multiple lines or its body spans multiple lines:

```cpp
if (!mod) {
  throw std::runtime_error(
      "module has been consumed and can no longer be used");
}
```

A single-line body under a single-line condition may omit braces.

## Python imports

Every import goes at module top. No function-local imports. When a test exists
to show that two extensions coexist, import both at module scope and assert on
them in the body.

## Tests: leak checking

`llvm.testing.assert_no_leaks()` already runs `gc.collect()` before it checks
the live context count. Do not call `gc.collect()` right before it, and prefer
`assert_no_leaks()` over a hand-written `gc.collect()` followed by
`Context._get_live_count() == 0`. Keep an explicit `gc.collect()` only where a
following assertion reads a live count directly, such as checking that a module
holds its context at one after the context handle is dropped.

## C++ coverage

`scripts/cpp_coverage.sh` builds the extension instrumented, runs the suite, and
enforces **100%** line and function coverage over `src/IR` and `src/MIR`. CI runs
this **per PR**, so every new binding/line must be exercised by a test in the
same PR (not a later one) — or marked `// LCOV_EXCL_LINE` (/ `LCOV_EXCL_START` /
`LCOV_EXCL_STOP`) for a genuinely unreachable line, as `Machine.cpp` does.

Running it locally: the profile format the compiler bakes into the `.so` must
match the `llvm-profdata`/`llvm-cov` that read it. The bundled `mlir_wheel`
tools are a newer LLVM than the system compiler that builds the `.so`, so their
`llvm-profdata` rejects the system `profraw` ("raw profile version mismatch").
Point the script at the matching system tools instead:

```
LLVM_PROFDATA="$(xcrun -f llvm-profdata)" LLVM_COV="$(xcrun -f llvm-cov)" \
  COVERAGE_THRESHOLD=100 bash scripts/cpp_coverage.sh
```
