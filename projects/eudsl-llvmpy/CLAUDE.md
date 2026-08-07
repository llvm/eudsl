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
