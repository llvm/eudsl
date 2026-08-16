# eudsl-llvmpy: LLVM IR bindings and DSL frontend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `eudsl-llvmpy`'s litgen-generated LLVM **C** API bindings with hand-written nanobind bindings over the LLVM **C++** API, then layer a Python DSL frontend (operator overloading, Python control flow lowered to basic blocks and phi nodes, `@function` decorator) on top.

**Architecture:** Two layers in one `llvm` package. The binding layer is hand-written nanobind C++ in `src/IR/*.cpp`, mirroring LLVM's C++ class hierarchy, with downcasting done through `nanobind::detail::type_hook` because `llvm::Value` and `llvm::Type` have no vtables. The DSL layer is pure Python in `llvm/dsl/` and `llvm/ast/`, reusing the AST canonicalizer machinery from `eudsl-python-extras` (vendored) and mirroring its `scf` yield protocol, except regions become real basic blocks and yielded values become phi nodes.

**Tech Stack:** LLVM 24.0.0git C++ API (`llvm/IR`, `llvm/AsmParser`, `llvm/Passes`, `llvm/Target`, `llvm/Linker`, `llvm/ExecutionEngine/Orc`), nanobind 2.12.0, scikit-build-core 0.10.7, CMake ≥ 3.29, pytest, FileCheck.

## Global Constraints

- **Import path stays top-level `llvm`.** `import llvm`, not `import eudsl.llvm`.
- **No C API anywhere.** Nothing in `src/` may `#include <llvm-c/...>`. litgen, `eudsl-llvmpy-generate.py`, and `eudslllvm_ext`'s generated sources are deleted in Task 1 and never come back. No raw escape hatch is retained.
- **No `eudsl-tblgen` dependency.** Removed from `pyproject.toml` build requires and from `CMakeLists.txt` in Task 1.
- **C++17** (`set(CMAKE_CXX_STANDARD 17)`, already in `CMakeLists.txt`).
- **Our translation units compile `-fexceptions -frtti`** even though LLVM is built `LLVM_ENABLE_RTTI=OFF LLVM_ENABLE_EH=OFF`. `typeid()` requires RTTI in the TU that *calls* it; that is a property of our `.cpp` files, not of the LLVM objects we link. The flags are already in `nanobind_options` in `CMakeLists.txt` — do not remove them.
- **A Python exception must never unwind into an LLVM frame.** Every bound callback (diagnostic handlers, pass instrumentation, ORC callbacks) catches at the boundary and converts to a return value or a stored error.
- **Every `type_info` returned by a `type_hook` must name a nanobind-registered class.** If it doesn't, nanobind raises "Unable to convert function return value to a Python type" at runtime. This is why Tasks 6, 8 and 10 register *every* class their tables can name, driven by the same `.def` X-macro that builds the table.
- **Host targets only by default.** `EUDSL_LLVMPY_TARGETS` defaults to `AArch64;X86`. AMDGPU and NVPTX are removed from the default link line.
- **Naming:** C++ classes keep LLVM's names (`Value`, `PHINode`, `GetElementPtrInst`). Methods and properties are `snake_case` (`get_type` → `.type`, `getIncomingValue` → `.incoming_value(i)`). Getters with no arguments become `def_prop_ro`.
- **Every pytest test ends with `gc.collect()` then `assert llvm.Context._get_live_count() == 0`**, following the convention in `llvm-project/mlir/test/python/ir/*.py`. The helper for this is written in Task 3.
- **Deferred, out of scope for this plan:** `break`/`continue`/early `return` inside DSL control flow (Task 31 makes them raise `NotImplementedError`), DebugInfo/DIBuilder, remarks, the disassembler, object-file inspection, ORC customization points needing Python callbacks, generated `.pyi` intrinsic stubs, an llvmlite-compatible import shim.

## Environment

Commands in this plan assume:

```bash
export EUDSL=/Users/mlevental/dev_projects/eudsl
export PY=/Users/mlevental/miniconda3/envs/eudsl/bin/python   # Python 3.12.13, nanobind 2.12.0
export CMAKE_PREFIX_PATH=/Users/mlevental/dev_projects/llvm-project/cmake-build-debug  # LLVM 24.0.0git
```

Build (editable, in-place — `pyproject.toml` sets `editable.mode = "inplace"`, so the `.so` lands in `src/llvm/`):

```bash
cd $EUDSL/projects/eudsl-llvmpy
$PY -m pip install -e . --no-build-isolation -v
```

Test:

```bash
cd $EUDSL/projects/eudsl-llvmpy
$PY -m pytest tests -v
```

`FileCheck` must be on `PATH` or in `$(python -c 'import sys;print(sys.prefix)')/bin` — `projects/mlir-native-tools` ships it, and `llvm-project/cmake-build-debug/bin/FileCheck` also works.

## Rebuild discipline

Every C++ task ends with a rebuild before the test run. The build is incremental but `editable.rebuild = false`, so **an editable install does not pick up C++ edits automatically** — you must re-run the `pip install -e .` line above (or `cmake --build build/<tag>` directly) after touching any `.cpp`/`.h`. Each C++ task's "run the test" step therefore reads: rebuild, then pytest.

---

## Delivery: two stacked draft PRs via `gh stack`

The work ships as **two stacked pull requests**, one per phase, managed with the
`gh stack` extension (github/gh-stack). Each PR keeps its own per-task commits.
Phase B stacks on Phase A because it depends on the bindings Phase A lands.

- **PR 1 — Phase A, the object / binding layer (Tasks 1–21).** Bottom of the
  stack, based on `main`.
- **PR 2 — Phase B, the DSL frontend (Tasks 22–34).** Stacked on top of the
  Phase A branch.

**Commits stay per-task.** Every task ends with its own `git commit` (failing
test → implement → suite green → commit). `gh stack` handles branch topology and
PR creation; the two phase boundaries are the only extra steps.

**Stack setup and submission:**

```bash
# Before Task 1 — start the stack; creates the branch off main.
cd $EUDSL && git checkout main && git pull \
  && gh stack init users/makslevental/eudsl-llvmpy-object-layer

# ... Tasks 1–21, one commit each ...

# End of Task 21 — push Phase A and open PR 1 as a draft.
#   --auto skips the editor and creates new PRs as drafts (run plain
#   `gh stack submit` instead to set the title/description interactively).
gh stack submit --auto

# Before Task 22 — add the Phase B branch on top of the stack.
cd $EUDSL && gh stack add users/makslevental/eudsl-llvmpy-dsl

# ... Tasks 22–34, one commit each ...

# End of Task 34 — push Phase B, open PR 2 as a draft, link the stack.
gh stack submit --auto
```

`gh stack submit --auto` opens PRs as drafts. When Phase A merges, run
`gh stack sync` (and `gh stack rebase`) to retarget PR 2 onto `main`. Keep the
PRs draft until each phase's full suite is green.

---

## File Structure

### C++ binding layer — `projects/eudsl-llvmpy/src/IR/`

| File | Responsibility |
|---|---|
| `Common.h` | `nb` aliases, `toString(const T&)` via `raw_string_ostream`, `unwrap(Expected<T>&&)` / `unwrap(Error&&)` throwing helpers. Included by every other file. |
| `Ownership.h` / `Ownership.cpp` | `eudsl::Context` and `eudsl::Module` owning holders. Live-instance counting. Moved-from tracking. |
| `Kinds.h` / `Kinds.cpp` | `type_hook<llvm::Value>`, `type_hook<llvm::Type>`, and the `.def`-driven `valueTypeInfo` / `typeTypeInfo` tables. |
| `Errors.h` / `Errors.cpp` | `ParseError`, `VerifyError` Python exception registration; `report_fatal_error` handler. |
| `Context.cpp` | `populate_context(nb::module_&)` — `Context`, `Module`, parse, print, verify, bitcode. |
| `Types.cpp` | `populate_types(nb::module_&)` — the whole `llvm::Type` hierarchy. |
| `Values.cpp` | `populate_values(nb::module_&)` — `Value`, `User`, `Use`, `Argument`, `BasicBlock`, `Function`, `GlobalVariable`, traversal iterators. |
| `Instructions.cpp` | `populate_instructions(nb::module_&)` — `Instruction` and every opcode class. |
| `Constants.cpp` | `populate_constants(nb::module_&)` — `Constant` hierarchy and constructors. |
| `Builder.cpp` | `populate_builder(nb::module_&)` — `IRBuilder` and its insertion-point context manager. |
| `Attributes.cpp` | `populate_attributes(nb::module_&)` — `Attribute`, `AttributeList`, linkage, visibility, calling convention enums. |
| `Metadata.cpp` | `populate_metadata(nb::module_&)` — `Metadata`, `MDNode`, `MDString`, named metadata. |
| `Passes.cpp` | `populate_passes(nb::module_&)` — `PassBuilder`, `run_passes`. |
| `Target.cpp` | `populate_target(nb::module_&)` — `Target`, `TargetMachine`, `DataLayout`, asm/object emission. |
| `Linker.cpp` | `populate_linker(nb::module_&)` — `link_modules`. |
| `JIT.cpp` | `populate_jit(nb::module_&)` — ORC `LLJIT`. |
| `Intrinsics.cpp` | `populate_intrinsics(nb::module_&)` — `Intrinsic` lookup, type inference, declaration. |
| `../eudslllvm_ext.cpp` | `NB_MODULE(eudslllvm_ext, m)` calling every `populate_*` in dependency order. Replaces the current file. |

### Python layer — `projects/eudsl-llvmpy/src/llvm/`

| File | Responsibility |
|---|---|
| `__init__.py` | `from .eudslllvm_ext import *`, then install DSL dunders and submodule re-exports. |
| `ast/__init__.py`, `ast/canonicalize.py`, `ast/util.py`, `ast/py_type.py` | Vendored from `eudsl-python-extras` (`mlir/extras/ast/`), MLIR imports stripped. |
| `dsl/values.py` | Arithmetic/comparison/GEP dunders attached to the bound `Value` classes. |
| `dsl/cf.py` | `if_ctx_manager`, `else_ctx_manager`, `while_`, `range_`, `yield_`, and the AST transformers that rewrite Python control flow into them. |
| `dsl/func.py` | The `@function` decorator. |
| `intrinsics.py` | `__getattr__` shim over `Intrinsic::lookupIntrinsicID`. |
| `testing.py` | `filecheck_with_comments` and the `live_count_zero` helper, vendored/adapted from `eudsl-python-extras`' `mlir/extras/testing/testing.py`. |

### Tests — `projects/eudsl-llvmpy/tests/`

One file per binding module: `test_context.py`, `test_types.py`, `test_values.py`, `test_instructions.py`, `test_constants.py`, `test_builder.py`, `test_attributes.py`, `test_metadata.py`, `test_errors.py`, `test_verify_bitcode.py`, `test_passes.py`, `test_target.py`, `test_linker.py`, `test_jit.py`, `test_intrinsics.py`, plus DSL tests `test_ast.py`, `test_dsl_values.py`, `test_dsl_cf.py`, `test_dsl_func.py`, `test_globals.py`. `test_bindings.py` keeps the two surviving smoke tests.

---

## Phase 0 — scaffolding

### Task 1: Delete litgen, land a minimal working replacement

This is the cutover. The package ends this task with *less* capability than it started with (parse and print only) but a green build and green tests. Capability is regained through Task 21.

**Files:**
- Delete: `src/eudslllvm_ext.cpp` (rewritten), `src/types.h`, `eudsl-llvmpy-generate.py`, `src/llvm/instructions.py`, `src/llvm/types_.py`, `src/llvm/function.py`, `src/llvm/context.py`, `src/llvm/util.py`, `src/llvm/amdgcn.py`, `src/llvm/eudslllvm_ext.pyi`
- Create: `src/IR/Common.h`, `src/IR/Ownership.h`, `src/IR/Ownership.cpp`, `src/IR/Context.cpp`, `src/eudslllvm_ext.cpp`
- Modify: `CMakeLists.txt`, `pyproject.toml`, `src/llvm/__init__.py`, `tests/test_bindings.py`

**Interfaces:**
- Produces: `llvm.Context()` (context manager), `llvm.Context._get_live_count() -> int`, `llvm.Module(name, ctx)`, `llvm.Module.name -> str`, `llvm.Module.__str__() -> str`, `llvm.parse_assembly(ir: str, ctx: Context, name: str = "<string>") -> Module`.

- [ ] **Step 1: Write the failing test**

Replace `tests/test_bindings.py` entirely:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import gc
from textwrap import dedent

import llvm


def test_symbol_collision():
    # eudsl-tblgen is a separate extension in a different nanobind domain;
    # importing both must not clash.
    import eudsl_tblgen  # noqa: F401

    import llvm  # noqa: F401


def test_smoke():
    src = dedent(
        """\
        declare i32 @foo()
        declare i32 @bar()
        define i32 @entry(i32 %argc) {
        entry:
          %and = and i32 %argc, 1
          %tobool = icmp eq i32 %and, 0
          br i1 %tobool, label %if.end, label %if.then
        if.then:
          %call = tail call i32 @foo()
          br label %return
        if.end:
          %call1 = tail call i32 @bar()
          br label %return
        return:
          %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.end ]
          ret i32 %retval.0
        }
        """
    )
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(src, ctx, "test_smoke")
        assert mod.name == "test_smoke"
        printed = str(mod)
        assert "define i32 @entry(i32 %argc)" in printed
        assert "phi i32" in printed
        del mod
    gc.collect()
    assert llvm.Context._get_live_count() == 0
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_bindings.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'Context'` (or an import error from the stale `.so`).

- [ ] **Step 3: Delete the litgen world**

```bash
cd $EUDSL/projects/eudsl-llvmpy
git rm eudsl-llvmpy-generate.py src/types.h \
       src/llvm/instructions.py src/llvm/types_.py src/llvm/function.py \
       src/llvm/context.py src/llvm/util.py src/llvm/amdgcn.py \
       src/llvm/eudslllvm_ext.pyi
rm -rf src/llvm/__pycache__ src/llvm/eudslllvm_ext.abi3.so build
mkdir -p src/ir
```

- [ ] **Step 4: Write `src/IR/Common.h`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

#include <nanobind/nanobind.h>

#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace eudsl {

/// Render anything with a `print(raw_ostream&)` method to a std::string.
template <typename T> std::string toString(const T &t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  t.print(os);
  return s;
}

/// Unwrap an llvm::Expected, raising RuntimeError on failure. Callers that
/// need a more specific Python exception catch and re-raise.
template <typename T> T unwrap(llvm::Expected<T> &&e) {
  if (!e)
    throw std::runtime_error(llvm::toString(e.takeError()));
  return std::move(*e);
}

/// Unwrap an llvm::Error, raising RuntimeError on failure.
inline void unwrap(llvm::Error &&e) {
  if (e)
    throw std::runtime_error(llvm::toString(std::move(e)));
}

} // namespace eudsl
```

- [ ] **Step 5: Write `src/IR/Ownership.h`**

`llvm::LLVMContext` and `llvm::Module` are wrapped in owning holders rather than bound directly, for three reasons: the live-instance count needs constructor/destructor hooks; the module must be able to report "I was consumed by the JIT" instead of segfaulting; and Python-visible lifetimes then have one obvious owner each.

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <cstdint>
#include <memory>
#include <string>

namespace eudsl {

/// Owns an llvm::LLVMContext. Counts live instances so tests can assert that
/// nothing leaked, mirroring mlir/test/python/ir/*.py.
class Context {
public:
  Context();
  ~Context();
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  llvm::LLVMContext &get() const { return *ctx; }

  static int64_t liveCount();

private:
  std::unique_ptr<llvm::LLVMContext> ctx;
};

/// Owns an llvm::Module. `get()` throws once the module has been handed to the
/// JIT, so a stale reference is a Python exception rather than a segfault.
class Module {
public:
  Module(const std::string &name, Context &ctx);
  Module(std::unique_ptr<llvm::Module> mod, Context &ctx);

  llvm::Module &get() const;
  /// Relinquish ownership. Every later get() throws.
  std::unique_ptr<llvm::Module> take();

  Context &context() const { return *owner; }

private:
  std::unique_ptr<llvm::Module> mod;
  Context *owner;
};

} // namespace eudsl
```

- [ ] **Step 6: Write `src/IR/Ownership.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Ownership.h"

#include <stdexcept>

namespace eudsl {

static int64_t gLiveContexts = 0;

Context::Context() : ctx(std::make_unique<llvm::LLVMContext>()) {
  ++gLiveContexts;
}

Context::~Context() { --gLiveContexts; }

int64_t Context::liveCount() { return gLiveContexts; }

Module::Module(const std::string &name, Context &ctx)
    : mod(std::make_unique<llvm::Module>(name, ctx.get())), owner(&ctx) {}

Module::Module(std::unique_ptr<llvm::Module> m, Context &ctx)
    : mod(std::move(m)), owner(&ctx) {}

llvm::Module &Module::get() const {
  if (!mod)
    throw std::runtime_error(
        "module has been consumed (moved into the JIT) and can no longer be "
        "used");
  return *mod;
}

std::unique_ptr<llvm::Module> Module::take() {
  get(); // throws if already consumed
  return std::move(mod);
}

} // namespace eudsl
```

- [ ] **Step 7: Write `src/IR/Context.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/Support/SourceMgr.h>

void populate_context(nb::module_ &m) {
  nb::class_<eudsl::Context>(m, "Context")
      .def(nb::init<>())
      .def("__enter__", [](eudsl::Context &self) -> eudsl::Context & { return self; },
           nb::rv_policy::reference_internal)
      .def("__exit__",
           [](eudsl::Context &, nb::handle, nb::handle, nb::handle) {})
      .def_static("_get_live_count", &eudsl::Context::liveCount);

  nb::class_<eudsl::Module>(m, "Module")
      .def(nb::init<const std::string &, eudsl::Context &>(), "name"_a,
           "context"_a, nb::keep_alive<1, 3>())
      .def_prop_ro("name",
                   [](eudsl::Module &self) {
                     return self.get().getModuleIdentifier();
                   })
      .def_prop_ro("context",
                   [](eudsl::Module &self) -> eudsl::Context & {
                     return self.context();
                   },
                   nb::rv_policy::reference_internal)
      .def("__str__",
           [](eudsl::Module &self) { return eudsl::toString(self.get()); });

  m.def(
      "parse_assembly",
      [](const std::string &ir, eudsl::Context &ctx, const std::string &name) {
        llvm::SMDiagnostic err;
        std::unique_ptr<llvm::Module> mod =
            llvm::parseAssemblyString(ir, err, ctx.get());
        if (!mod) {
          std::string msg;
          llvm::raw_string_ostream os(msg);
          err.print(name.c_str(), os);
          throw std::runtime_error(msg);
        }
        mod->setModuleIdentifier(name);
        mod->setSourceFileName(name);
        return new eudsl::Module(std::move(mod), ctx);
      },
      "ir"_a, "context"_a, "name"_a = "<string>", nb::keep_alive<0, 2>(),
      "Parse LLVM textual IR into a new Module.");
}
```

`nb::keep_alive<0, 2>()` on `parse_assembly` keeps the context (argument 2) alive as long as the returned module lives. `nb::keep_alive<1, 3>()` on `Module.__init__` does the same for the constructor (self is 1, context is 3).

- [ ] **Step 8: Rewrite `src/eudslllvm_ext.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void populate_context(nb::module_ &m);

NB_MODULE(eudslllvm_ext, m) {
  m.doc() = "Hand-written nanobind bindings over the LLVM C++ IR API.";
  populate_context(m);
}
```

- [ ] **Step 9: Rewrite `CMakeLists.txt`**

Replace everything from the `if (EUDSLLLVM_STANDALONE_BUILD)` generate block down to `target_link_libraries` with the block below. Keep the header comment, `cmake_minimum_required`, `project`/`enable_language`, `find_package(Python ...)`, `find_package(LLVM ...)`, `include_directories(${LLVM_INCLUDE_DIRS})`, `add_definitions(${LLVM_DEFINITIONS})`, the nanobind `find_package`, and `nanobind_options` verbatim. Delete the `if(NOT TARGET LLVMSupport)` guard, the whole generate/`_gen_src` machinery, and the `_eudsl_tblgen_path` block.

```cmake
set(EUDSLLLVM_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
include_directories(${EUDSLLLVM_SRC_DIR})

nanobind_add_module(eudslllvm_ext
  NB_STATIC STABLE_ABI
  NB_DOMAIN eudslllvm
  src/eudslllvm_ext.cpp
  src/IR/Ownership.cpp
  src/IR/Context.cpp
)

set(eudslllvm_ext_libs
  LLVMCore
  LLVMAsmParser
  LLVMSupport
)
target_link_libraries(eudslllvm_ext PRIVATE ${eudslllvm_ext_libs})
```

Leave the `set_target_properties`, `target_compile_options`, stubgen, and `install` blocks at the bottom of the file as they are. Later tasks append source files to `nanobind_add_module` and libraries to `eudslllvm_ext_libs`.

- [ ] **Step 10: Drop the build deps in `pyproject.toml`**

Replace the `[build-system]` requires list with:

```toml
requires = [
    "nanobind==2.12.0",
    "ninja",
    "scikit-build-core==0.10.7",
    "typing_extensions>=4.12.2",
]
```

(`eudsl-tblgen` and the `litgen` git URL are gone. The pin moves from 2.4.0 to the 2.12.0 in the dev environment, which is where `type_hook` was verified present.)

- [ ] **Step 11: Simplify `src/llvm/__init__.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

from .eudslllvm_ext import *
from .eudslllvm_ext import __doc__
```

- [ ] **Step 12: Build and run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests/test_bindings.py -v
```

Expected: both tests PASS.

- [ ] **Step 13: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Replace litgen C API bindings with hand-written C++ nanobind bindings for Context and Module"
```

---

### Task 2: `EUDSL_LLVMPY_TARGETS` CMake option, host-only default

**Files:**
- Modify: `CMakeLists.txt`
- Test: `tests/test_target_option.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `llvm.registered_targets() -> list[str]`, the CMake cache variable `EUDSL_LLVMPY_TARGETS` (semicolon-separated LLVM target names, default `AArch64;X86`).

- [ ] **Step 1: Write the failing test**

`tests/test_target_option.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm


def test_only_host_targets_are_linked():
    targets = llvm.registered_targets()
    # Host targets are present.
    assert any(t in targets for t in ("AArch64", "X86"))
    # The GPU backends were dropped from the default build.
    assert "AMDGPU" not in targets
    assert "NVPTX" not in targets
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_target_option.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'registered_targets'`.

- [ ] **Step 3: Add the option and the target libraries to `CMakeLists.txt`**

Insert after the `find_package(LLVM ...)` / `include_directories(${LLVM_INCLUDE_DIRS})` block:

```cmake
set(EUDSL_LLVMPY_TARGETS "AArch64;X86" CACHE STRING
    "LLVM targets to link into eudslllvm_ext (semicolon-separated)")
message(STATUS "EUDSL_LLVMPY_TARGETS: ${EUDSL_LLVMPY_TARGETS}")

set(EUDSLLLVM_TARGET_LIBS)
set(EUDSLLLVM_TARGET_INITS)
foreach(_tgt IN LISTS EUDSL_LLVMPY_TARGETS)
  if(NOT TARGET LLVM${_tgt}Info)
    message(FATAL_ERROR
        "EUDSL_LLVMPY_TARGETS names ${_tgt} but LLVM${_tgt}Info does not exist; "
        "the LLVM being built against was configured without that target")
  endif()
  list(APPEND EUDSLLLVM_TARGET_LIBS
      LLVM${_tgt}Info LLVM${_tgt}Desc LLVM${_tgt}CodeGen
      LLVM${_tgt}AsmParser LLVM${_tgt}Disassembler)
  string(APPEND EUDSLLLVM_TARGET_INITS
      "  LLVMInitialize${_tgt}Target();\n"
      "  LLVMInitialize${_tgt}TargetInfo();\n"
      "  LLVMInitialize${_tgt}TargetMC();\n"
      "  LLVMInitialize${_tgt}AsmParser();\n"
      "  LLVMInitialize${_tgt}AsmPrinter();\n")
endforeach()
configure_file("${CMAKE_CURRENT_LIST_DIR}/src/IR/TargetInit.h.in"
               "${EUDSLLLVM_BINARY_DIR}/ir/TargetInit.h" @ONLY)
include_directories("${EUDSLLLVM_BINARY_DIR}")
```

Note: `LLVMInitializeXTarget` and friends are C++ functions declared by `llvm/Support/TargetSelect.h` in namespace scope (they are *not* the C API — `TargetSelect.h` is a C++ header in `llvm/Support`). Add `${EUDSLLLVM_TARGET_LIBS}` to `eudslllvm_ext_libs`, plus `LLVMTarget LLVMMC LLVMMCParser LLVMCodeGen LLVMTargetParser`.

- [ ] **Step 4: Create the generated-header template `src/IR/TargetInit.h.in`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generated from TargetInit.h.in by CMake. Do not edit.

#pragma once

#include <llvm/Support/TargetSelect.h>

namespace eudsl {

/// Initialize exactly the targets named by EUDSL_LLVMPY_TARGETS.
inline void initializeTargets() {
@EUDSLLLVM_TARGET_INITS@
}

} // namespace eudsl
```

- [ ] **Step 5: Expose `registered_targets()` from `src/IR/Context.cpp`**

Add these includes at the top of `Context.cpp`:

```cpp
#include "IR/TargetInit.h"

#include <llvm/MC/TargetRegistry.h>
```

and this at the end of `populate_context`:

```cpp
  eudsl::initializeTargets();

  m.def(
      "registered_targets",
      []() {
        std::vector<std::string> names;
        for (const llvm::Target &t : llvm::TargetRegistry::targets())
          names.emplace_back(t.getName());
        return names;
      },
      "Names of the LLVM targets linked into this extension.");
```

Add `#include <nanobind/stl/string.h>` and `#include <nanobind/stl/vector.h>` to `Common.h` so `std::string` and `std::vector<std::string>` convert.

- [ ] **Step 6: Rebuild and run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS. Confirm the build log no longer mentions `LLVMAMDGPUCodeGen`.

- [ ] **Step 7: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add EUDSL_LLVMPY_TARGETS with a host-only default"
```

---

## Phase A1 — object layer

### Task 3: Context and Module lifetime, `_get_live_count`, testing helper

Task 1 bound these classes minimally. This task makes their lifetime correct and observable, and lands the shared test helper every later task uses.

**Files:**
- Modify: `src/IR/Ownership.h`, `src/IR/Ownership.cpp`, `src/IR/Context.cpp`
- Create: `src/llvm/testing.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `eudsl::Context`, `eudsl::Module` from Task 1.
- Produces: `llvm.Module._is_consumed -> bool`, `llvm.Module._take()` (test-only ownership sink), `llvm.Context.__exit__` invalidating the context, `llvm.testing.assert_no_leaks()`.

- [ ] **Step 1: Write the failing tests**

`tests/test_context.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import gc

import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_context_is_counted():
    assert llvm.Context._get_live_count() == 0
    ctx = llvm.Context()
    assert llvm.Context._get_live_count() == 1
    del ctx
    gc.collect()
    assert llvm.Context._get_live_count() == 0


def test_nested_contexts_are_counted():
    with llvm.Context() as a, llvm.Context() as b:
        assert a is not b
        assert llvm.Context._get_live_count() == 2
    gc.collect()
    assert_no_leaks()


def test_module_keeps_context_alive():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    del ctx
    gc.collect()
    # The module's keep_alive kept the context object alive, so this is safe.
    assert llvm.Context._get_live_count() == 1
    assert mod.name == "m"
    del mod
    gc.collect()
    assert_no_leaks()


def test_module_rename():
    with llvm.Context() as ctx:
        mod = llvm.Module("before", ctx)
        mod.name = "after"
        assert mod.name == "after"
        assert "ModuleID = 'after'" in str(mod)
        del mod
    gc.collect()
    assert_no_leaks()


def test_consumed_module_raises_instead_of_crashing():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        assert mod._is_consumed is False
        mod._take()
        assert mod._is_consumed is True
        with pytest.raises(RuntimeError, match="has been consumed"):
            _ = mod.name
        with pytest.raises(RuntimeError, match="has been consumed"):
            str(mod)
        del mod
    gc.collect()
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_context.py -v
```

Expected: `ModuleNotFoundError: No module named 'llvm.testing'`.

- [ ] **Step 3: Write `src/llvm/testing.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test helpers. Not imported by the llvm package itself."""

import gc

from . import Context


def assert_no_leaks():
    """Assert every Context has been destroyed.

    Mirrors the convention in llvm-project/mlir/test/python/ir/*.py: a test
    that constructs IR must leave no live context behind.
    """
    gc.collect()
    live = Context._get_live_count()
    assert live == 0, f"{live} Context object(s) still alive"
```

- [ ] **Step 4: Add `_is_consumed` and a settable `name` to `src/IR/Ownership.h`**

Add to `class Module`, in the public section:

```cpp
  bool isConsumed() const { return mod == nullptr; }
```

- [ ] **Step 5: Extend the `Module` bindings in `src/IR/Context.cpp`**

Replace the `def_prop_ro("name", ...)` line with a read/write property and add the consumed-state members:

```cpp
      .def_prop_rw(
          "name",
          [](eudsl::Module &self) { return self.get().getModuleIdentifier(); },
          [](eudsl::Module &self, const std::string &name) {
            self.get().setModuleIdentifier(name);
            self.get().setSourceFileName(name);
          })
      .def_prop_ro("_is_consumed", &eudsl::Module::isConsumed)
      .def("_take",
           [](eudsl::Module &self) {
             // Test-only ownership sink: drops the module on the floor so
             // tests can observe the consumed state without a JIT.
             std::unique_ptr<llvm::Module> owned = self.take();
             owned.reset();
           })
```

- [ ] **Step 6: Rebuild and run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Make Context and Module lifetimes correct and observable"
```

---

### Task 4: `Type` base and primitive types

**Files:**
- Create: `src/IR/Types.cpp`
- Modify: `CMakeLists.txt` (add `src/IR/Types.cpp`), `src/eudslllvm_ext.cpp`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: `llvm.Context`.
- Produces: `llvm.Type` with `.is_void`, `.is_integer`, `.is_floating_point`, `.is_pointer`, `.is_sized`, `.__str__`, `.__eq__`, `.__hash__`; free functions `llvm.void_t(ctx)`, `llvm.i1(ctx)`, `llvm.i8(ctx)`, `llvm.i16(ctx)`, `llvm.i32(ctx)`, `llvm.i64(ctx)`, `llvm.f16(ctx)`, `llvm.f32(ctx)`, `llvm.f64(ctx)`, `llvm.label_t(ctx)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_types.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_primitive_types_print():
    with llvm.Context() as ctx:
        assert str(llvm.void_t(ctx)) == "void"
        assert str(llvm.i1(ctx)) == "i1"
        assert str(llvm.i32(ctx)) == "i32"
        assert str(llvm.f32(ctx)) == "float"
        assert str(llvm.f64(ctx)) == "double"
        assert str(llvm.f16(ctx)) == "half"
    assert_no_leaks()


def test_type_predicates():
    with llvm.Context() as ctx:
        assert llvm.void_t(ctx).is_void
        assert not llvm.void_t(ctx).is_sized
        assert llvm.i32(ctx).is_integer
        assert not llvm.i32(ctx).is_floating_point
        assert llvm.f64(ctx).is_floating_point
        assert llvm.i32(ctx).is_sized
    assert_no_leaks()


def test_types_are_uniqued_and_hashable():
    with llvm.Context() as a, llvm.Context() as b:
        assert llvm.i32(a) == llvm.i32(a)
        assert llvm.i32(a) != llvm.i64(a)
        # Types are interned per context, so two contexts give distinct types.
        assert llvm.i32(a) != llvm.i32(b)
        assert len({llvm.i32(a), llvm.i32(a), llvm.i64(a)}) == 2
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_types.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'void_t'`.

- [ ] **Step 3: Write `src/IR/Types.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Type.h>

void populate_types(nb::module_ &m) {
  nb::class_<llvm::Type>(m, "Type")
      .def_prop_ro("is_void", &llvm::Type::isVoidTy)
      .def_prop_ro("is_label", &llvm::Type::isLabelTy)
      .def_prop_ro("is_integer",
                   [](llvm::Type &self) { return self.isIntegerTy(); })
      .def_prop_ro("is_floating_point", &llvm::Type::isFloatingPointTy)
      .def_prop_ro("is_pointer", &llvm::Type::isPointerTy)
      .def_prop_ro("is_sized", [](llvm::Type &self) { return self.isSized(); })
      .def("__str__", [](llvm::Type &self) { return eudsl::toString(self); })
      .def("__eq__",
           [](llvm::Type &self, nb::handle other) {
             llvm::Type *o;
             if (!nb::try_cast<llvm::Type *>(other, o))
               return false;
             return &self == o;
           })
      .def("__hash__", [](llvm::Type &self) {
        return static_cast<Py_ssize_t>(
            reinterpret_cast<std::uintptr_t>(&self));
      });

  // Primitive type factories. Each takes the owning context and returns an
  // interned Type*, non-owning, kept alive by the context.
#define EUDSL_PRIMITIVE_TYPE(pyName, getter)                                   \
  m.def(                                                                       \
      pyName,                                                                  \
      [](eudsl::Context &ctx) -> llvm::Type * {                                \
        return llvm::Type::getter(ctx.get());                                  \
      },                                                                       \
      "context"_a, nb::rv_policy::reference_internal)

  EUDSL_PRIMITIVE_TYPE("void_t", getVoidTy);
  EUDSL_PRIMITIVE_TYPE("label_t", getLabelTy);
  EUDSL_PRIMITIVE_TYPE("i1", getInt1Ty);
  EUDSL_PRIMITIVE_TYPE("i8", getInt8Ty);
  EUDSL_PRIMITIVE_TYPE("i16", getInt16Ty);
  EUDSL_PRIMITIVE_TYPE("i32", getInt32Ty);
  EUDSL_PRIMITIVE_TYPE("i64", getInt64Ty);
  EUDSL_PRIMITIVE_TYPE("f16", getHalfTy);
  EUDSL_PRIMITIVE_TYPE("f32", getFloatTy);
  EUDSL_PRIMITIVE_TYPE("f64", getDoubleTy);
#undef EUDSL_PRIMITIVE_TYPE
}
```

Add `#include <cstdint>` to `Common.h` for `uintptr_t`.

- [ ] **Step 4: Register `populate_types` in `src/eudslllvm_ext.cpp`**

```cpp
void populate_context(nb::module_ &m);
void populate_types(nb::module_ &m);

NB_MODULE(eudslllvm_ext, m) {
  m.doc() = "Hand-written nanobind bindings over the LLVM C++ IR API.";
  populate_context(m);
  populate_types(m);
}
```

Add `src/IR/Types.cpp` to `nanobind_add_module` in `CMakeLists.txt`.

- [ ] **Step 5: Rebuild and run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind llvm::Type and the primitive types"
```

---

### Task 5: Derived types — Integer, Pointer, Struct, Array, Vector, Function

**Files:**
- Modify: `src/IR/Types.cpp`
- Test: `tests/test_types.py` (append)

**Interfaces:**
- Consumes: `llvm.Type`.
- Produces: `llvm.IntegerType` (`.bit_width`), `llvm.PointerType` (`.address_space`), `llvm.StructType` (`.name`, `.num_elements`, `.element_type(i)`, `.is_packed`, `.is_opaque`, `.set_body(elts, packed=False)`), `llvm.ArrayType` (`.num_elements`, `.element_type`), `llvm.VectorType` (`.min_num_elements`, `.element_type`, `.is_scalable`), `llvm.FunctionType` (`.return_type`, `.num_params`, `.param_type(i)`, `.params`, `.is_var_arg`); factories `llvm.int_t(ctx, bits)`, `llvm.ptr_t(ctx, addrspace=0)`, `llvm.struct_t(ctx, elts, packed=False)`, `llvm.named_struct_t(ctx, name)`, `llvm.array_t(elt, n)`, `llvm.vector_t(elt, n, scalable=False)`, `llvm.function_t(ret, params, var_arg=False)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_types.py`:

```python
def test_integer_and_pointer_types():
    with llvm.Context() as ctx:
        i7 = llvm.int_t(ctx, 7)
        assert str(i7) == "i7"
        assert i7.bit_width == 7
        p = llvm.ptr_t(ctx)
        assert str(p) == "ptr"
        assert p.address_space == 0
        p3 = llvm.ptr_t(ctx, 3)
        assert str(p3) == "ptr addrspace(3)"
        assert p3.address_space == 3
        assert p != p3
    assert_no_leaks()


def test_array_and_vector_types():
    with llvm.Context() as ctx:
        a = llvm.array_t(llvm.i32(ctx), 4)
        assert str(a) == "[4 x i32]"
        assert a.num_elements == 4
        assert a.element_type == llvm.i32(ctx)
        v = llvm.vector_t(llvm.f32(ctx), 8)
        assert str(v) == "<8 x float>"
        assert v.min_num_elements == 8
        assert not v.is_scalable
        sv = llvm.vector_t(llvm.f32(ctx), 8, scalable=True)
        assert str(sv) == "<vscale x 8 x float>"
        assert sv.is_scalable
    assert_no_leaks()


def test_literal_and_named_struct_types():
    with llvm.Context() as ctx:
        s = llvm.struct_t(ctx, [llvm.i32(ctx), llvm.f64(ctx)])
        assert str(s) == "{ i32, double }"
        assert s.num_elements == 2
        assert s.element_type(1) == llvm.f64(ctx)
        assert not s.is_packed

        packed = llvm.struct_t(ctx, [llvm.i8(ctx), llvm.i32(ctx)], packed=True)
        assert str(packed) == "<{ i8, i32 }>"
        assert packed.is_packed

        named = llvm.named_struct_t(ctx, "Pair")
        assert named.name == "Pair"
        assert named.is_opaque
        named.set_body([llvm.i32(ctx), llvm.i32(ctx)])
        assert not named.is_opaque
        assert named.num_elements == 2
        assert str(named) == "%Pair"
    assert_no_leaks()


def test_function_types():
    with llvm.Context() as ctx:
        ft = llvm.function_t(llvm.i32(ctx), [llvm.i32(ctx), llvm.f32(ctx)])
        assert str(ft) == "i32 (i32, float)"
        assert ft.return_type == llvm.i32(ctx)
        assert ft.num_params == 2
        assert ft.param_type(1) == llvm.f32(ctx)
        assert ft.params == [llvm.i32(ctx), llvm.f32(ctx)]
        assert not ft.is_var_arg

        va = llvm.function_t(llvm.void_t(ctx), [llvm.ptr_t(ctx)], var_arg=True)
        assert str(va) == "void (ptr, ...)"
        assert va.is_var_arg
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_types.py -v -k "integer or array or struct or function_types"
```

Expected: `AttributeError: module 'llvm' has no attribute 'int_t'`.

- [ ] **Step 3: Add the derived types to `src/IR/Types.cpp`**

Add `#include <llvm/IR/DerivedTypes.h>` at the top, and this at the end of `populate_types`:

```cpp
  nb::class_<llvm::IntegerType, llvm::Type>(m, "IntegerType")
      .def_prop_ro("bit_width", &llvm::IntegerType::getBitWidth);

  nb::class_<llvm::PointerType, llvm::Type>(m, "PointerType")
      .def_prop_ro("address_space", &llvm::PointerType::getAddressSpace);

  nb::class_<llvm::StructType, llvm::Type>(m, "StructType")
      .def_prop_ro("name",
                   [](llvm::StructType &self) -> std::optional<std::string> {
                     if (!self.hasName())
                       return std::nullopt;
                     return self.getName().str();
                   })
      .def_prop_ro("num_elements", &llvm::StructType::getNumElements)
      .def("element_type", &llvm::StructType::getElementType, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("is_packed", &llvm::StructType::isPacked)
      .def_prop_ro("is_opaque", &llvm::StructType::isOpaque)
      .def(
          "set_body",
          [](llvm::StructType &self, std::vector<llvm::Type *> elts,
             bool packed) { self.setBody(elts, packed); },
          "element_types"_a, "packed"_a = false);

  nb::class_<llvm::ArrayType, llvm::Type>(m, "ArrayType")
      .def_prop_ro("num_elements", &llvm::ArrayType::getNumElements)
      .def_prop_ro("element_type", &llvm::ArrayType::getElementType,
                   nb::rv_policy::reference_internal);

  nb::class_<llvm::VectorType, llvm::Type>(m, "VectorType")
      .def_prop_ro("min_num_elements",
                   [](llvm::VectorType &self) {
                     return self.getElementCount().getKnownMinValue();
                   })
      .def_prop_ro("is_scalable",
                   [](llvm::VectorType &self) {
                     return self.getElementCount().isScalable();
                   })
      .def_prop_ro("element_type", &llvm::VectorType::getElementType,
                   nb::rv_policy::reference_internal);

  nb::class_<llvm::FunctionType, llvm::Type>(m, "FunctionType")
      .def_prop_ro("return_type", &llvm::FunctionType::getReturnType,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("num_params", &llvm::FunctionType::getNumParams)
      .def("param_type", &llvm::FunctionType::getParamType, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("params",
                   [](llvm::FunctionType &self) {
                     return std::vector<llvm::Type *>(self.param_begin(),
                                                      self.param_end());
                   })
      .def_prop_ro("is_var_arg", &llvm::FunctionType::isVarArg);

  m.def(
      "int_t",
      [](eudsl::Context &ctx, unsigned bits) -> llvm::Type * {
        return llvm::IntegerType::get(ctx.get(), bits);
      },
      "context"_a, "bits"_a, nb::rv_policy::reference_internal);
  m.def(
      "ptr_t",
      [](eudsl::Context &ctx, unsigned addressSpace) -> llvm::Type * {
        return llvm::PointerType::get(ctx.get(), addressSpace);
      },
      "context"_a, "address_space"_a = 0, nb::rv_policy::reference_internal);
  m.def(
      "struct_t",
      [](eudsl::Context &ctx, std::vector<llvm::Type *> elts,
         bool packed) -> llvm::Type * {
        return llvm::StructType::get(ctx.get(), elts, packed);
      },
      "context"_a, "element_types"_a, "packed"_a = false,
      nb::rv_policy::reference_internal);
  m.def(
      "named_struct_t",
      [](eudsl::Context &ctx, const std::string &name) -> llvm::Type * {
        return llvm::StructType::create(ctx.get(), name);
      },
      "context"_a, "name"_a, nb::rv_policy::reference_internal);
  m.def(
      "array_t",
      [](llvm::Type *elt, uint64_t n) -> llvm::Type * {
        return llvm::ArrayType::get(elt, n);
      },
      "element_type"_a, "num_elements"_a, nb::rv_policy::reference_internal,
      nb::keep_alive<0, 1>());
  m.def(
      "vector_t",
      [](llvm::Type *elt, unsigned n, bool scalable) -> llvm::Type * {
        return llvm::VectorType::get(elt, n, scalable);
      },
      "element_type"_a, "num_elements"_a, "scalable"_a = false,
      nb::rv_policy::reference_internal, nb::keep_alive<0, 1>());
  m.def(
      "function_t",
      [](llvm::Type *ret, std::vector<llvm::Type *> params,
         bool varArg) -> llvm::Type * {
        return llvm::FunctionType::get(ret, params, varArg);
      },
      "return_type"_a, "params"_a, "var_arg"_a = false,
      nb::rv_policy::reference_internal, nb::keep_alive<0, 1>());
```

Add `#include <nanobind/stl/optional.h>` to `Common.h`.

Note the factories return `llvm::Type *` rather than the concrete subclass. Task 6 installs the `type_hook` that makes the returned Python object the concrete class anyway, which is why these tests only check `str()` and the base predicates until then. `.bit_width` etc. do not work until Task 6 lands, so **the four tests above are written to be run after Task 6** — run them at the end of Task 6, and in this task run only `test_primitive_types_print`, `test_type_predicates`, `test_types_are_uniqued_and_hashable`.

- [ ] **Step 4: Rebuild and run the pre-hook subset**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v -k "not integer_and_pointer and not array_and_vector and not struct_types and not function_types"
```

Expected: all selected tests PASS. The four new tests fail with `AttributeError: 'Type' object has no attribute 'bit_width'` — that is Task 6's job.

- [ ] **Step 5: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind the derived llvm::Type classes and their factories"
```

---

### Task 6: `type_hook` for `Type`, dispatching on `Type::TypeID`

`llvm::Type` has no vtable, so nanobind's RTTI-based downcasting cannot work. `nanobind::detail::type_hook<T>::get(ptr)` exists for exactly this case: it picks the Python type from a non-polymorphic C++ pointer.

**Files:**
- Create: `src/IR/Kinds.h`, `src/IR/Kinds.cpp`
- Modify: `src/IR/Types.cpp` (include `Kinds.h`), `CMakeLists.txt`
- Test: `tests/test_types.py` (append)

**Interfaces:**
- Consumes: the type classes from Task 5.
- Produces: `eudsl::typeTypeInfo(llvm::Type::TypeID) -> const std::type_info *` and `nanobind::detail::type_hook<llvm::Type>`. Every `llvm::Type *` crossing into Python from now on arrives as its concrete Python class.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_types.py`:

```python
def test_types_downcast_to_concrete_classes():
    with llvm.Context() as ctx:
        assert type(llvm.i32(ctx)).__name__ == "IntegerType"
        assert type(llvm.ptr_t(ctx)).__name__ == "PointerType"
        assert type(llvm.array_t(llvm.i32(ctx), 2)).__name__ == "ArrayType"
        assert type(llvm.vector_t(llvm.i32(ctx), 2)).__name__ == "VectorType"
        assert type(llvm.struct_t(ctx, [llvm.i32(ctx)])).__name__ == "StructType"
        assert (
            type(llvm.function_t(llvm.void_t(ctx), [])).__name__ == "FunctionType"
        )
        # Types with no concrete subclass stay Type.
        assert type(llvm.void_t(ctx)).__name__ == "Type"
        assert type(llvm.f64(ctx)).__name__ == "Type"
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_types.py::test_types_downcast_to_concrete_classes -v
```

Expected: FAIL — `assert 'Type' == 'IntegerType'`.

- [ ] **Step 3: Write `src/IR/Kinds.h`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Downcasting support. llvm::Value and llvm::Type are deliberately
// non-polymorphic (Value.h documents the non-virtual destructor), so
// nanobind's RTTI-based downcast cannot see through a base pointer.
// nanobind::detail::type_hook is the documented hook for this case.
//
// INVARIANT: every std::type_info returned here must name a class registered
// with nanobind, otherwise the conversion raises "Unable to convert function
// return value to a Python type". The tables and the registrations are both
// driven by llvm/IR/Value.def and llvm/IR/Instruction.def so they cannot
// drift apart.

#pragma once

#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <nanobind/nanobind.h>

#include <typeinfo>

namespace eudsl {
const std::type_info *typeTypeInfo(llvm::Type::TypeID id);
const std::type_info *valueTypeInfo(unsigned valueID);
} // namespace eudsl

template <> struct nanobind::detail::type_hook<llvm::Type> {
  static const std::type_info *get(llvm::Type *t) {
    return t ? eudsl::typeTypeInfo(t->getTypeID()) : &typeid(llvm::Type);
  }
};

template <> struct nanobind::detail::type_hook<llvm::Value> {
  static const std::type_info *get(llvm::Value *v) {
    return v ? eudsl::valueTypeInfo(v->getValueID()) : &typeid(llvm::Value);
  }
};
```

- [ ] **Step 4: Write the `Type` half of `src/IR/Kinds.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Kinds.h"

#include <llvm/IR/DerivedTypes.h>

namespace eudsl {

const std::type_info *typeTypeInfo(llvm::Type::TypeID id) {
  switch (id) {
  case llvm::Type::IntegerTyID:
    return &typeid(llvm::IntegerType);
  case llvm::Type::FunctionTyID:
    return &typeid(llvm::FunctionType);
  case llvm::Type::PointerTyID:
    return &typeid(llvm::PointerType);
  case llvm::Type::StructTyID:
    return &typeid(llvm::StructType);
  case llvm::Type::ArrayTyID:
    return &typeid(llvm::ArrayType);
  case llvm::Type::FixedVectorTyID:
  case llvm::Type::ScalableVectorTyID:
    return &typeid(llvm::VectorType);
  default:
    // Void, Label, Metadata, Token, the float kinds, TypedPointer and
    // TargetExtType have no bound subclass; they stay llvm::Type.
    return &typeid(llvm::Type);
  }
}

} // namespace eudsl
```

- [ ] **Step 5: Include the hook everywhere a `Type*` or `Value*` is returned**

`type_hook` must be visible in every TU that converts an `llvm::Type *` or `llvm::Value *`, or that TU silently keeps the base-class behaviour. Add `#include "IR/Kinds.h"` to `Common.h` — that guarantees it, since every binding file includes `Common.h`. Add `src/IR/Kinds.cpp` to `nanobind_add_module` in `CMakeLists.txt`.

- [ ] **Step 6: Rebuild and run the full type test file**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests/test_types.py -v
```

Expected: all of `tests/test_types.py` PASSES, including the four tests deferred from Task 5.

- [ ] **Step 7: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add a type_hook downcasting llvm::Type on TypeID"
```

---

### Task 7: `Value` base and `User`

**Files:**
- Create: `src/IR/Values.cpp`
- Modify: `CMakeLists.txt` (add `src/IR/Values.cpp`), `src/eudslllvm_ext.cpp`
- Test: `tests/test_values.py`

**Interfaces:**
- Consumes: `llvm.Type`, `llvm.parse_assembly`.
- Produces: `llvm.Value` with `.name` (read/write), `.type`, `.users`, `.num_uses`, `.replace_all_uses_with(v)`, `.__str__`, `.__eq__`, `.__hash__`; `llvm.User` (subclass) with `.num_operands`, `.operand(i)`, `.operands`.

- [ ] **Step 1: Write the failing tests**

`tests/test_values.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @f(i32 %x, i32 %y) {
    entry:
      %sum = add i32 %x, %y
      ret i32 %sum
    }
    """
)


def _first_function(mod):
    # llvm.functions() lands in Task 9; until then reach through the module text
    # is not enough, so this helper is defined once Task 9 exists. For Task 7 we
    # test Value via a global's operands instead (see below).
    raise NotImplementedError


def test_value_name_type_str_via_global():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            "@g = global i32 7\n", ctx, "m"
        )
        # A GlobalVariable is a Value; Task 9/11 give richer access, but the
        # base Value surface is exercisable now through module text round-trip.
        assert "@g = global i32 7" in str(mod)
        del mod
    assert_no_leaks()
```

Note: `Value`'s accessors only become reachable once there is a way to *get* a `Value` from Python — that arrives with `functions()`/traversal in Task 9. So Task 7 registers the classes and the base surface, and the substantive `Value` tests live in Task 9. This test only asserts the module still round-trips, confirming `populate_values` did not break the module.

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_values.py -v
```

Expected: FAIL at import of a not-yet-added symbol, or PASS trivially — if it passes, still proceed (the class registrations are the deliverable and are verified by Task 9's tests).

- [ ] **Step 3: Write `src/IR/Values.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/IR/Value.h>
#include <llvm/IR/User.h>

#include <vector>

void populate_values(nb::module_ &m) {
  nb::class_<llvm::Value>(m, "Value")
      .def_prop_rw(
          "name", [](llvm::Value &self) { return self.getName().str(); },
          [](llvm::Value &self, const std::string &n) { self.setName(n); })
      .def_prop_ro("type", &llvm::Value::getType,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("num_uses",
                   [](llvm::Value &self) { return self.getNumUses(); })
      .def_prop_ro("users",
                   [](llvm::Value &self) {
                     return std::vector<llvm::User *>(self.user_begin(),
                                                      self.user_end());
                   })
      .def("replace_all_uses_with", &llvm::Value::replaceAllUsesWith,
           "value"_a)
      .def("__str__", [](llvm::Value &self) { return eudsl::toString(self); })
      .def("__eq__",
           [](llvm::Value &self, nb::handle other) {
             llvm::Value *o;
             if (!nb::try_cast<llvm::Value *>(other, o))
               return false;
             return &self == o;
           })
      .def("__hash__", [](llvm::Value &self) {
        return static_cast<Py_ssize_t>(
            reinterpret_cast<std::uintptr_t>(&self));
      });

  nb::class_<llvm::User, llvm::Value>(m, "User")
      .def_prop_ro("num_operands", &llvm::User::getNumOperands)
      .def("operand", &llvm::User::getOperand, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("operands", [](llvm::User &self) {
        return std::vector<llvm::Value *>(self.op_begin(), self.op_end());
      });
}
```

`llvm::User::op_begin()` yields `Use*`; dereferencing a `Use` converts to `Value*`, and `std::vector<Value*>(op_begin(), op_end())` uses `Use`'s `operator Value*`. Confirm the build; if the iterator does not implicitly convert, replace with an index loop over `getNumOperands()`/`getOperand(i)`.

- [ ] **Step 4: Register `populate_values` in `src/eudslllvm_ext.cpp`** (after `populate_types`), add `src/IR/Values.cpp` to `nanobind_add_module`.

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind the llvm::Value base and llvm::User"
```

---

### Task 8: `type_hook` for `Value`, driven by `Value.def` and `Instruction.def`

`getValueID()` returns `InstructionVal + opcode` for instructions and a plain `ValueTy` enumerator otherwise. The hook dispatches accordingly. Both tables are generated by including the `.def` files with local macro definitions, and each returned `type_info` is guarded by `pick()` so a not-yet-registered class falls back to base `Value` — this keeps every commit green while later tasks register the leaves.

**Files:**
- Modify: `src/IR/Kinds.cpp`, `src/IR/Values.cpp` (register the structural spine)
- Test: covered by Task 9's tests (no `Value*` is obtainable from Python until Task 9).

**Interfaces:**
- Consumes: `type_hook<llvm::Value>` declaration from Task 6's `Kinds.h`.
- Produces: `eudsl::valueTypeInfo(unsigned) -> const std::type_info *`, and `eudsl::pick(const std::type_info *) -> const std::type_info *`.

- [ ] **Step 1: Add `pick()` and `valueTypeInfo` to `src/IR/Kinds.cpp`**

Add includes and code:

```cpp
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>

#include <nanobind/nanobind.h>

namespace eudsl {

// Return `concrete` if a nanobind class is registered for it, else fall back to
// the base Value. Keeps intermediate commits green while leaf classes are
// still being registered, and makes an LLVM version bump that adds an
// unbound kind degrade to Value rather than raise.
const std::type_info *pick(const std::type_info *concrete,
                           const std::type_info *base) {
  return nanobind::detail::nb_type_lookup(concrete) ? concrete : base;
}

const std::type_info *valueTypeInfo(unsigned id) {
  const std::type_info *base = &typeid(llvm::Value);

  if (id >= llvm::Value::InstructionVal) {
    switch (id - llvm::Value::InstructionVal) {
#define HANDLE_INST(num, opcode, Class)                                        \
  case num:                                                                    \
    return pick(&typeid(llvm::Class), base);
#include "llvm/IR/Instruction.def"
    default:
      return &typeid(llvm::Instruction);
    }
  }

  switch (id) {
#define HANDLE_VALUE(Name)                                                     \
  case llvm::Value::Name##Val:                                                 \
    return pick(&typeid(llvm::Name), base);
#include "llvm/IR/Value.def"
  default:
    return base;
  }
}

} // namespace eudsl
```

Declare `pick` in `Kinds.h`:

```cpp
const std::type_info *pick(const std::type_info *concrete,
                           const std::type_info *base);
```

The `HANDLE_INST`/`HANDLE_VALUE` includes expand a `case` per opcode and per value kind. Because `Value.def` routes `HANDLE_GLOBAL_VALUE`, `HANDLE_CONSTANT`, `HANDLE_MEMORY_VALUE`, `HANDLE_INSTRUCTION`, etc. through `HANDLE_VALUE` when only `HANDLE_VALUE` is defined, every enumerator gets a case. `MemoryUse`/`MemoryDef`/`MemoryPhi` are not real IR `Value` subclasses reachable here (they belong to MemorySSA) but their enumerators exist; `pick()` sends them to base since they are never registered. The `HANDLE_INSTRUCTION(Instruction)` marker and any opcode gap default to `llvm::Instruction`.

- [ ] **Step 2: Register the structural spine in `src/IR/Values.cpp`**

Insert these *between* `User` and the end of `populate_values`, so the inheritance chain the hook can name exists as registered classes (methods are added by later tasks reopening them):

```cpp
  // Structural spine of the Value hierarchy. Leaf method bindings are added by
  // Tasks 9-11 which reopen these classes via nb::type<T>(). Registering them
  // here lets the Value type_hook name them without raising.
  nb::class_<llvm::Constant, llvm::User>(m, "Constant");
  nb::class_<llvm::GlobalValue, llvm::Constant>(m, "GlobalValue");
  nb::class_<llvm::GlobalObject, llvm::GlobalValue>(m, "GlobalObject");
  nb::class_<llvm::Instruction, llvm::User>(m, "Instruction");
```

Add `#include <llvm/IR/Constant.h>`, `#include <llvm/IR/GlobalValue.h>`, `#include <llvm/IR/GlobalObject.h>`, `#include <llvm/IR/Instruction.h>`, `#include <llvm/IR/Constants.h>` to `Values.cpp`.

- [ ] **Step 3: Add the reopen helper to `src/IR/Common.h`**

```cpp
/// Fetch an already-registered nanobind class so a later translation unit can
/// add methods to it. Used because the Value hierarchy is registered as a spine
/// in Values.cpp and filled in by Instructions.cpp / Constants.cpp.
template <typename T> nb::class_<T> reopen() {
  return nb::borrow<nb::class_<T>>(nb::type<T>());
}
```

- [ ] **Step 4: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS (behaviour unchanged so far; the hook is dormant until a `Value*` reaches Python in Task 9).

- [ ] **Step 5: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add the Value type_hook from Value.def and Instruction.def"
```

---

### Task 9: `Function`, `Argument`, `BasicBlock`, and traversal iterators

**Files:**
- Modify: `src/IR/Values.cpp`, `src/IR/Context.cpp` (module traversal)
- Test: `tests/test_values.py` (append), `tests/test_context.py` (append)

**Interfaces:**
- Consumes: the registered spine from Task 8, `reopen<T>()`.
- Produces: `llvm.Module.functions -> list[Function]`, `llvm.Module.get_function(name) -> Function | None`; `llvm.Function` (`.name`, `.type`, `.function_type`, `.return_type`, `.is_var_arg`, `.args`, `.arg(i)`, `.num_args`, `.basic_blocks`, `.entry_block`, `.is_declaration`, `.append_basic_block(name="")`); `llvm.Argument` (`.arg_no`, `.parent`); `llvm.BasicBlock` (`.name`, `.parent`, `.instructions`, `.terminator`, `.context`), and `llvm.BasicBlock.create(ctx, name="", parent=None) -> BasicBlock`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_values.py`:

```python
def test_function_traversal():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        assert [f.name for f in mod.functions] == ["f"]
        f = mod.get_function("f")
        assert f is not None
        assert mod.get_function("nope") is None
        assert f.num_args == 2
        assert [a.name for a in f.args] == ["x", "y"]
        assert f.arg(0).arg_no == 0
        assert f.arg(1).parent == f
        assert not f.is_declaration
        assert str(f.return_type) == "i32"
        del f, mod
    assert_no_leaks()


def test_basic_block_and_instruction_traversal():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        blocks = f.basic_blocks
        assert [b.name for b in blocks] == ["entry"]
        entry = f.entry_block
        assert entry.name == "entry"
        assert entry == blocks[0]
        insts = entry.instructions
        # add, ret
        assert len(insts) == 2
        assert entry.terminator == insts[-1]
        # The Value type_hook downcast the add to its concrete class.
        assert type(insts[0]).__name__ == "BinaryOperator"
        del f, entry, blocks, insts, mod
    assert_no_leaks()


def test_value_users_and_operands():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        # %x is used by the add.
        assert x.num_uses == 1
        add = x.users[0]
        assert type(add).__name__ == "BinaryOperator"
        assert add.num_operands == 2
        assert add.operand(0) == x
        del f, x, add, mod
    assert_no_leaks()


def test_append_basic_block():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly("declare void @g()\n", ctx, "m")
        # Build a fresh function body by hand.
        ft = llvm.function_t(llvm.void_t(ctx), [])
        fn = llvm.Function.create(ft, "h", mod)
        bb = fn.append_basic_block("entry")
        assert bb.name == "entry"
        assert fn.entry_block == bb
        assert bb.parent == fn
        del fn, bb, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_values.py -v -k "traversal or users or append"
```

Expected: `AttributeError: 'Module' object has no attribute 'functions'`.

- [ ] **Step 3: Add `Function`, `Argument`, `BasicBlock` to `src/IR/Values.cpp`**

Add includes `<llvm/IR/Function.h>`, `<llvm/IR/Argument.h>`, `<llvm/IR/BasicBlock.h>`, `<llvm/IR/DerivedTypes.h>`, and at the end of `populate_values`:

```cpp
  reopen<llvm::Instruction>()
      .def_prop_ro("num_successors",
                   [](llvm::Instruction &self) {
                     return self.getNumSuccessors();
                   })
      .def("successor", &llvm::Instruction::getSuccessor, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("is_terminator",
                   [](llvm::Instruction &self) { return self.isTerminator(); })
      .def_prop_ro("parent", &llvm::Instruction::getParent,
                   nb::rv_policy::reference_internal);

  nb::class_<llvm::Argument, llvm::Value>(m, "Argument")
      .def_prop_ro("arg_no", &llvm::Argument::getArgNo)
      .def_prop_ro("parent",
                   [](llvm::Argument &self) { return self.getParent(); },
                   nb::rv_policy::reference_internal);

  nb::class_<llvm::BasicBlock, llvm::Value>(m, "BasicBlock")
      .def_static(
          "create",
          [](eudsl::Context &ctx, const std::string &name,
             llvm::Function *parent) {
            return llvm::BasicBlock::Create(ctx.get(), name, parent);
          },
          "context"_a, "name"_a = "", "parent"_a = nullptr,
          nb::rv_policy::reference_internal)
      .def_prop_ro("parent",
                   [](llvm::BasicBlock &self) { return self.getParent(); },
                   nb::rv_policy::reference_internal)
      .def_prop_ro("terminator",
                   [](llvm::BasicBlock &self) {
                     return self.getTerminatorOrNull();
                   },
                   nb::rv_policy::reference_internal)
      .def_prop_ro("instructions",
                   [](llvm::BasicBlock &self) {
                     std::vector<llvm::Instruction *> out;
                     for (llvm::Instruction &i : self)
                       out.push_back(&i);
                     return out;
                   });

  reopen<llvm::GlobalObject>(); // ensure it exists before Function refines it
  nb::class_<llvm::Function, llvm::GlobalObject>(m, "Function")
      .def_static(
          "create",
          [](llvm::FunctionType *ft, const std::string &name,
             eudsl::Module &mod) {
            return llvm::Function::Create(ft, llvm::GlobalValue::ExternalLinkage,
                                          name, mod.get());
          },
          "function_type"_a, "name"_a, "module"_a,
          nb::rv_policy::reference_internal, nb::keep_alive<0, 3>())
      .def_prop_ro("function_type", &llvm::Function::getFunctionType,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("return_type", &llvm::Function::getReturnType,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("is_var_arg", &llvm::Function::isVarArg)
      .def_prop_ro("is_declaration", &llvm::Function::isDeclaration)
      .def_prop_ro("num_args", &llvm::Function::arg_size)
      .def("arg", &llvm::Function::getArg, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("args",
                   [](llvm::Function &self) {
                     std::vector<llvm::Argument *> out;
                     for (llvm::Argument &a : self.args())
                       out.push_back(&a);
                     return out;
                   })
      .def_prop_ro("basic_blocks",
                   [](llvm::Function &self) {
                     std::vector<llvm::BasicBlock *> out;
                     for (llvm::BasicBlock &b : self)
                       out.push_back(&b);
                     return out;
                   })
      .def_prop_ro("entry_block",
                   [](llvm::Function &self) -> llvm::BasicBlock * {
                     if (self.empty())
                       return nullptr;
                     return &self.getEntryBlock();
                   },
                   nb::rv_policy::reference_internal)
      .def(
          "append_basic_block",
          [](llvm::Function &self, const std::string &name) {
            return llvm::BasicBlock::Create(self.getContext(), name, &self);
          },
          "name"_a = "", nb::rv_policy::reference_internal);

  m.attr("Function") = m.attr("Function"); // no-op; keeps ordering explicit
```

Remove the trailing no-op line if the linter objects. `reopen<llvm::GlobalObject>()` is only to assert existence; drop it — `GlobalObject` is already registered in Task 8's spine.

- [ ] **Step 4: Add module traversal to `src/IR/Context.cpp`**

Add `#include <llvm/IR/Function.h>` and, in the `Module` class body:

```cpp
      .def_prop_ro("functions",
                   [](eudsl::Module &self) {
                     std::vector<llvm::Function *> out;
                     for (llvm::Function &f : self.get().functions())
                       out.push_back(&f);
                     return out;
                   })
      .def(
          "get_function",
          [](eudsl::Module &self, const std::string &name) {
            return self.get().getFunction(name);
          },
          "name"_a, nb::rv_policy::reference_internal)
```

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS, including the Value downcast assertions (`BinaryOperator`), which confirm Task 8's hook fires.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind Function, Argument, BasicBlock and traversal iterators"
```

---

### Task 10: `Instruction` subclasses with accessors; `PHINode` incoming values

**Files:**
- Create: `src/IR/Instructions.cpp`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`
- Test: `tests/test_instructions.py`

**Interfaces:**
- Consumes: the registered `Instruction` spine, `reopen<T>()`, the Value type_hook.
- Produces: registered nanobind classes for every opcode class, with the intermediate bases `UnaryInstruction`, `UnaryOperator`, `BinaryOperator`, `CmpInst`, `CastInst`, `CallBase`, `FuncletPadInst`; accessors on `PHINode` (`.num_incoming`, `.incoming_value(i)`, `.incoming_block(i)`, `.add_incoming(v, bb)`), `ICmpInst`/`FCmpInst` (`.predicate`), `CallBase` (`.called_operand`, `.num_args`, `.arg_operand(i)`), `GetElementPtrInst` (`.source_element_type`), `AllocaInst` (`.allocated_type`), `LoadInst`/`StoreInst` (`.pointer_operand`), `CondBrInst`/`UncondBrInst` (`.is_conditional`, `.condition`).

- [ ] **Step 1: Write the failing tests**

`tests/test_instructions.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks

_PHI_SRC = dedent(
    """\
    define i32 @f(i1 %c) {
    entry:
      br i1 %c, label %a, label %b
    a:
      br label %join
    b:
      br label %join
    join:
      %p = phi i32 [ 1, %a ], [ 2, %b ]
      %eq = icmp eq i32 %p, 1
      ret i32 %p
    }
    """
)


def _insts_by_class(mod, name):
    out = []
    for f in mod.functions:
        for bb in f.basic_blocks:
            for i in bb.instructions:
                if type(i).__name__ == name:
                    out.append(i)
    return out


def test_phi_incoming():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_PHI_SRC, ctx, "m")
        (phi,) = _insts_by_class(mod, "PHINode")
        assert phi.num_incoming == 2
        assert phi.incoming_block(0).name == "a"
        assert phi.incoming_block(1).name == "b"
        assert str(phi.incoming_value(0)) == "i32 1"
        del phi, mod
    assert_no_leaks()


def test_icmp_predicate():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_PHI_SRC, ctx, "m")
        (icmp,) = _insts_by_class(mod, "ICmpInst")
        assert icmp.predicate == llvm.ICmpPredicate.EQ
        del icmp, mod
    assert_no_leaks()


def test_conditional_branch():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_PHI_SRC, ctx, "m")
        cbrs = _insts_by_class(mod, "CondBrInst")
        assert len(cbrs) == 1
        assert cbrs[0].is_conditional
        assert cbrs[0].num_successors == 2
        del cbrs, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_instructions.py -v
```

Expected: `AssertionError` — `type(i).__name__` is `"Instruction"` because the opcode classes are not registered yet, so `_insts_by_class` finds nothing and the unpack fails.

- [ ] **Step 3: Write `src/IR/Instructions.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>

namespace {
// Register a leaf instruction class with a given base. A macro so the base is
// spelled once; the Value type_hook already knows the name from Instruction.def.
#define LEAF(Class, Base) nb::class_<llvm::Class, llvm::Base>(m, #Class)
} // namespace

void populate_instructions(nb::module_ &m) {
  // Predicate enums.
  nb::enum_<llvm::CmpInst::Predicate>(m, "ICmpPredicate")
      .value("EQ", llvm::CmpInst::ICMP_EQ)
      .value("NE", llvm::CmpInst::ICMP_NE)
      .value("UGT", llvm::CmpInst::ICMP_UGT)
      .value("UGE", llvm::CmpInst::ICMP_UGE)
      .value("ULT", llvm::CmpInst::ICMP_ULT)
      .value("ULE", llvm::CmpInst::ICMP_ULE)
      .value("SGT", llvm::CmpInst::ICMP_SGT)
      .value("SGE", llvm::CmpInst::ICMP_SGE)
      .value("SLT", llvm::CmpInst::ICMP_SLT)
      .value("SLE", llvm::CmpInst::ICMP_SLE);
  nb::enum_<llvm::FCmpInst::Predicate>(m, "FCmpPredicate")
      .value("OEQ", llvm::CmpInst::FCMP_OEQ)
      .value("OGT", llvm::CmpInst::FCMP_OGT)
      .value("OGE", llvm::CmpInst::FCMP_OGE)
      .value("OLT", llvm::CmpInst::FCMP_OLT)
      .value("OLE", llvm::CmpInst::FCMP_OLE)
      .value("ONE", llvm::CmpInst::FCMP_ONE)
      .value("UEQ", llvm::CmpInst::FCMP_UEQ)
      .value("UNE", llvm::CmpInst::FCMP_UNE);

  // Intermediate bases (Value type_hook never names these, but leaf classes
  // need them registered as their nanobind base).
  nb::class_<llvm::UnaryInstruction, llvm::Instruction>(m, "UnaryInstruction");
  nb::class_<llvm::UnaryOperator, llvm::UnaryInstruction>(m, "UnaryOperator");
  nb::class_<llvm::BinaryOperator, llvm::Instruction>(m, "BinaryOperator");
  nb::class_<llvm::CastInst, llvm::UnaryInstruction>(m, "CastInst");
  nb::class_<llvm::FuncletPadInst, llvm::Instruction>(m, "FuncletPadInst");

  nb::class_<llvm::CmpInst, llvm::Instruction>(m, "CmpInst")
      .def_prop_ro("predicate", &llvm::CmpInst::getPredicate);

  nb::class_<llvm::CallBase, llvm::Instruction>(m, "CallBase")
      .def_prop_ro("num_args", &llvm::CallBase::arg_size)
      .def("arg_operand", &llvm::CallBase::getArgOperand, "index"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("called_operand", &llvm::CallBase::getCalledOperand,
                   nb::rv_policy::reference_internal);

  LEAF(ICmpInst, CmpInst);
  LEAF(FCmpInst, CmpInst);
  LEAF(CallInst, CallBase);
  LEAF(InvokeInst, CallBase);
  LEAF(CallBrInst, CallBase);

  nb::class_<llvm::PHINode, llvm::Instruction>(m, "PHINode")
      .def_prop_ro("num_incoming", &llvm::PHINode::getNumIncomingValues)
      .def("incoming_value", &llvm::PHINode::getIncomingValue, "index"_a,
           nb::rv_policy::reference_internal)
      .def("incoming_block",
           [](llvm::PHINode &self, unsigned i) {
             return self.getIncomingBlock(i);
           },
           "index"_a, nb::rv_policy::reference_internal)
      .def("add_incoming", &llvm::PHINode::addIncoming, "value"_a, "block"_a);

  nb::class_<llvm::AllocaInst, llvm::UnaryInstruction>(m, "AllocaInst")
      .def_prop_ro("allocated_type", &llvm::AllocaInst::getAllocatedType,
                   nb::rv_policy::reference_internal);
  nb::class_<llvm::LoadInst, llvm::UnaryInstruction>(m, "LoadInst")
      .def_prop_ro("pointer_operand", &llvm::LoadInst::getPointerOperand,
                   nb::rv_policy::reference_internal);
  nb::class_<llvm::StoreInst, llvm::Instruction>(m, "StoreInst")
      .def_prop_ro("pointer_operand", &llvm::StoreInst::getPointerOperand,
                   nb::rv_policy::reference_internal);
  nb::class_<llvm::GetElementPtrInst, llvm::Instruction>(m, "GetElementPtrInst")
      .def_prop_ro("source_element_type",
                   &llvm::GetElementPtrInst::getSourceElementType,
                   nb::rv_policy::reference_internal);

  nb::class_<llvm::ReturnInst, llvm::Instruction>(m, "ReturnInst")
      .def_prop_ro("return_value", &llvm::ReturnInst::getReturnValue,
                   nb::rv_policy::reference_internal);
  nb::class_<llvm::UncondBrInst, llvm::Instruction>(m, "UncondBrInst")
      .def_prop_ro("is_conditional", [](llvm::UncondBrInst &) { return false; });
  nb::class_<llvm::CondBrInst, llvm::Instruction>(m, "CondBrInst")
      .def_prop_ro("is_conditional", [](llvm::CondBrInst &) { return true; })
      .def_prop_ro("condition", &llvm::CondBrInst::getCondition,
                   nb::rv_policy::reference_internal);
  nb::class_<llvm::SwitchInst, llvm::Instruction>(m, "SwitchInst");
  nb::class_<llvm::IndirectBrInst, llvm::Instruction>(m, "IndirectBrInst");
  nb::class_<llvm::ResumeInst, llvm::Instruction>(m, "ResumeInst");
  nb::class_<llvm::UnreachableInst, llvm::Instruction>(m, "UnreachableInst");
  nb::class_<llvm::SelectInst, llvm::Instruction>(m, "SelectInst");
  nb::class_<llvm::VAArgInst, llvm::UnaryInstruction>(m, "VAArgInst");
  nb::class_<llvm::ExtractElementInst, llvm::Instruction>(m, "ExtractElementInst");
  nb::class_<llvm::InsertElementInst, llvm::Instruction>(m, "InsertElementInst");
  nb::class_<llvm::ShuffleVectorInst, llvm::Instruction>(m, "ShuffleVectorInst");
  nb::class_<llvm::ExtractValueInst, llvm::UnaryInstruction>(m, "ExtractValueInst");
  nb::class_<llvm::InsertValueInst, llvm::Instruction>(m, "InsertValueInst");
  nb::class_<llvm::LandingPadInst, llvm::Instruction>(m, "LandingPadInst");
  nb::class_<llvm::FreezeInst, llvm::UnaryInstruction>(m, "FreezeInst");
  nb::class_<llvm::FenceInst, llvm::Instruction>(m, "FenceInst");
  nb::class_<llvm::AtomicCmpXchgInst, llvm::Instruction>(m, "AtomicCmpXchgInst");
  nb::class_<llvm::AtomicRMWInst, llvm::Instruction>(m, "AtomicRMWInst");
  nb::class_<llvm::CleanupPadInst, llvm::FuncletPadInst>(m, "CleanupPadInst");
  nb::class_<llvm::CatchPadInst, llvm::FuncletPadInst>(m, "CatchPadInst");
  nb::class_<llvm::CatchReturnInst, llvm::Instruction>(m, "CatchReturnInst");
  nb::class_<llvm::CleanupReturnInst, llvm::Instruction>(m, "CleanupReturnInst");
  nb::class_<llvm::CatchSwitchInst, llvm::Instruction>(m, "CatchSwitchInst");
  nb::class_<llvm::TruncInst, llvm::CastInst>(m, "TruncInst");
  nb::class_<llvm::ZExtInst, llvm::CastInst>(m, "ZExtInst");
  nb::class_<llvm::SExtInst, llvm::CastInst>(m, "SExtInst");
  nb::class_<llvm::FPToUIInst, llvm::CastInst>(m, "FPToUIInst");
  nb::class_<llvm::FPToSIInst, llvm::CastInst>(m, "FPToSIInst");
  nb::class_<llvm::UIToFPInst, llvm::CastInst>(m, "UIToFPInst");
  nb::class_<llvm::SIToFPInst, llvm::CastInst>(m, "SIToFPInst");
  nb::class_<llvm::FPTruncInst, llvm::CastInst>(m, "FPTruncInst");
  nb::class_<llvm::FPExtInst, llvm::CastInst>(m, "FPExtInst");
  nb::class_<llvm::PtrToIntInst, llvm::CastInst>(m, "PtrToIntInst");
  nb::class_<llvm::PtrToAddrInst, llvm::CastInst>(m, "PtrToAddrInst");
  nb::class_<llvm::IntToPtrInst, llvm::CastInst>(m, "IntToPtrInst");
  nb::class_<llvm::BitCastInst, llvm::CastInst>(m, "BitCastInst");
  nb::class_<llvm::AddrSpaceCastInst, llvm::CastInst>(m, "AddrSpaceCastInst");
  nb::class_<llvm::FPUnaryOperator, llvm::UnaryOperator>(m, "FPUnaryOperator");
  nb::class_<llvm::FPBinaryOperator, llvm::BinaryOperator>(m, "FPBinaryOperator");
#undef LEAF
}
```

Note the opcode `Add`..`Xor` all map to `BinaryOperator` (there is no `AddInst` class); the type_hook's `Instruction.def` table returns `&typeid(llvm::BinaryOperator)` for those opcodes, matching the test's `type(insts[0]).__name__ == "BinaryOperator"`. `FNeg` maps to `FPUnaryOperator`. `FAdd`..`FRem` map to `FPBinaryOperator`. These are what `Instruction.def`'s `HANDLE_BINARY_INST(15, FAdd, FPBinaryOperator)` third column names — confirm the third-column class name equals the registered class name for each entry.

- [ ] **Step 4: Register `populate_instructions`** in `src/eudslllvm_ext.cpp` (after `populate_values`), add `src/IR/Instructions.cpp` to `nanobind_add_module`.

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind Instruction subclasses and PHINode incoming values"
```

---

### Task 11: `Constant` subclasses and constant construction

**Files:**
- Create: `src/IR/Constants.cpp`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`
- Test: `tests/test_constants.py`

**Interfaces:**
- Consumes: the `Constant`/`GlobalValue`/`GlobalObject` spine, the Value type_hook.
- Produces: registered `ConstantData`, `ConstantAggregate`, `ConstantInt` (`.value`, `.zext_value`), `ConstantFP` (`.double_value`), `UndefValue`, `PoisonValue`, `ConstantPointerNull`, `ConstantAggregateZero`, `ConstantArray`, `ConstantStruct`, `ConstantVector`, `ConstantDataArray`, `ConstantDataVector`, `ConstantExpr`, `BlockAddress`, `GlobalVariable`, `GlobalAlias`, `GlobalIFunc`; factories `llvm.const_int(ty, value, signed=False)`, `llvm.const_bool(ctx, b)`, `llvm.const_fp(ty, value)`, `llvm.undef(ty)`, `llvm.poison(ty)`, `llvm.null(ty)`, `llvm.const_null(ty)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_constants.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_const_int():
    with llvm.Context() as ctx:
        c = llvm.const_int(llvm.i32(ctx), 42)
        assert type(c).__name__ == "ConstantInt"
        assert c.value == 42
        assert str(c) == "i32 42"
        neg = llvm.const_int(llvm.i32(ctx), -1, signed=True)
        assert neg.value == -1
    assert_no_leaks()


def test_const_bool_and_fp():
    with llvm.Context() as ctx:
        t = llvm.const_bool(ctx, True)
        assert type(t).__name__ == "ConstantInt"
        assert str(t) == "i1 true"
        f = llvm.const_fp(llvm.f64(ctx), 1.5)
        assert type(f).__name__ == "ConstantFP"
        assert f.double_value == 1.5
        assert str(f) == "double 1.500000e+00"
    assert_no_leaks()


def test_undef_poison_null():
    with llvm.Context() as ctx:
        assert type(llvm.undef(llvm.i32(ctx))).__name__ == "UndefValue"
        assert type(llvm.poison(llvm.i32(ctx))).__name__ == "PoisonValue"
        assert type(llvm.null(llvm.ptr_t(ctx))).__name__ == "ConstantPointerNull"
        assert str(llvm.undef(llvm.i32(ctx))) == "i32 undef"
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_constants.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'const_int'`.

- [ ] **Step 3: Write `src/IR/Constants.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalVariable.h>

void populate_constants(nb::module_ &m) {
  nb::class_<llvm::ConstantData, llvm::Constant>(m, "ConstantData");
  nb::class_<llvm::ConstantAggregate, llvm::Constant>(m, "ConstantAggregate");

  nb::class_<llvm::ConstantInt, llvm::ConstantData>(m, "ConstantInt")
      .def_prop_ro("value",
                   [](llvm::ConstantInt &self) {
                     return self.getValue().getSExtValue();
                   })
      .def_prop_ro("zext_value", [](llvm::ConstantInt &self) {
        return self.getValue().getZExtValue();
      });

  nb::class_<llvm::ConstantFP, llvm::ConstantData>(m, "ConstantFP")
      .def_prop_ro("double_value", [](llvm::ConstantFP &self) {
        return self.getValueAPF().convertToDouble();
      });

  nb::class_<llvm::UndefValue, llvm::ConstantData>(m, "UndefValue");
  nb::class_<llvm::PoisonValue, llvm::UndefValue>(m, "PoisonValue");
  nb::class_<llvm::ConstantPointerNull, llvm::ConstantData>(
      m, "ConstantPointerNull");
  nb::class_<llvm::ConstantAggregateZero, llvm::ConstantData>(
      m, "ConstantAggregateZero");
  nb::class_<llvm::ConstantTokenNone, llvm::ConstantData>(m,
                                                          "ConstantTokenNone");
  nb::class_<llvm::ConstantArray, llvm::ConstantAggregate>(m, "ConstantArray");
  nb::class_<llvm::ConstantStruct, llvm::ConstantAggregate>(m,
                                                            "ConstantStruct");
  nb::class_<llvm::ConstantVector, llvm::ConstantAggregate>(m,
                                                            "ConstantVector");
  nb::class_<llvm::ConstantDataSequential, llvm::ConstantData>(
      m, "ConstantDataSequential");
  nb::class_<llvm::ConstantDataArray, llvm::ConstantDataSequential>(
      m, "ConstantDataArray");
  nb::class_<llvm::ConstantDataVector, llvm::ConstantDataSequential>(
      m, "ConstantDataVector");
  nb::class_<llvm::ConstantExpr, llvm::Constant>(m, "ConstantExpr");
  nb::class_<llvm::BlockAddress, llvm::Constant>(m, "BlockAddress");

  nb::class_<llvm::GlobalVariable, llvm::GlobalObject>(m, "GlobalVariable")
      .def_prop_ro("is_constant", &llvm::GlobalVariable::isConstant)
      .def_prop_ro("initializer",
                   [](llvm::GlobalVariable &self) -> llvm::Constant * {
                     return self.hasInitializer() ? self.getInitializer()
                                                  : nullptr;
                   },
                   nb::rv_policy::reference_internal);
  nb::class_<llvm::GlobalAlias, llvm::GlobalValue>(m, "GlobalAlias");
  nb::class_<llvm::GlobalIFunc, llvm::GlobalObject>(m, "GlobalIFunc");

  m.def(
      "const_int",
      [](llvm::Type *ty, int64_t value, bool isSigned) -> llvm::Constant * {
        auto *ity = llvm::cast<llvm::IntegerType>(ty);
        return llvm::ConstantInt::get(ity, static_cast<uint64_t>(value),
                                      isSigned);
      },
      "type"_a, "value"_a, "signed"_a = false,
      nb::rv_policy::reference_internal);
  m.def(
      "const_bool",
      [](eudsl::Context &ctx, bool b) -> llvm::Constant * {
        return llvm::ConstantInt::getBool(ctx.get(), b);
      },
      "context"_a, "value"_a, nb::rv_policy::reference_internal);
  m.def(
      "const_fp",
      [](llvm::Type *ty, double value) -> llvm::Constant * {
        return llvm::ConstantFP::get(ty, value);
      },
      "type"_a, "value"_a, nb::rv_policy::reference_internal);
  m.def(
      "undef",
      [](llvm::Type *ty) -> llvm::Constant * {
        return llvm::UndefValue::get(ty);
      },
      "type"_a, nb::rv_policy::reference_internal);
  m.def(
      "poison",
      [](llvm::Type *ty) -> llvm::Constant * {
        return llvm::PoisonValue::get(ty);
      },
      "type"_a, nb::rv_policy::reference_internal);
  m.def(
      "null",
      [](llvm::Type *ty) -> llvm::Constant * {
        return llvm::Constant::getNullValue(ty);
      },
      "type"_a, nb::rv_policy::reference_internal);
  m.attr("const_null") = m.attr("null");
}
```

`ConstantInt::get(IntegerType*, uint64_t, bool)` and `ConstantFP::get(Type*, double)` were verified present. `null(ptr_t)` produces a `ConstantPointerNull`, matching the test.

- [ ] **Step 4: Register `populate_constants`** in `src/eudslllvm_ext.cpp` (after `populate_instructions`), add `src/IR/Constants.cpp` to `nanobind_add_module`.

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind Constant subclasses and constant construction"
```

---

### Task 12: `IRBuilder` with an insertion-point context manager

**Files:**
- Create: `src/IR/Builder.cpp`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`
- Test: `tests/test_builder.py`

**Interfaces:**
- Consumes: `Context`, `Function`, `BasicBlock`, `Value`, `Type`, constants.
- Produces: `llvm.IRBuilder(ctx)` with `.set_insert_point(bb)`, `.insert_block`, and creation methods `.ret(v=None)`, `.br(dest)`, `.cond_br(cond, t, f)`, `.add/.fadd/.sub/.fsub/.mul/.fmul/.sdiv/.udiv/.fdiv(l, r, name="")`, `.icmp(pred, l, r, name="")`, `.fcmp(pred, l, r, name="")`, `.alloca(ty, name="")`, `.load(ty, ptr, name="")`, `.store(v, ptr)`, `.gep(ty, ptr, indices, name="")`, `.call(fn, args, name="")`, `.phi(ty, name="")`; context-manager `.at_end_of(bb)`.

- [ ] **Step 1: Write the failing test**

`tests/test_builder.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks


def test_build_add_function():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn = llvm.Function.create(llvm.function_t(i32, [i32, i32]), "add2", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            s = b.add(fn.arg(0), fn.arg(1), "s")
            b.ret(s)
        printed = str(mod)
        assert "define i32 @add2(i32 %0, i32 %1)" in printed
        assert "%s = add i32 %0, %1" in printed
        assert "ret i32 %s" in printed
        del b, fn, bb, mod
    assert_no_leaks()


def test_build_conditional_with_phi():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        i1 = llvm.i1(ctx)
        fn = llvm.Function.create(llvm.function_t(i32, [i1]), "sel", mod)
        entry = fn.append_basic_block("entry")
        a = fn.append_basic_block("a")
        b_ = fn.append_basic_block("b")
        join = fn.append_basic_block("join")
        bld = llvm.IRBuilder(ctx)
        with bld.at_end_of(entry):
            bld.cond_br(fn.arg(0), a, b_)
        with bld.at_end_of(a):
            bld.br(join)
        with bld.at_end_of(b_):
            bld.br(join)
        with bld.at_end_of(join):
            p = bld.phi(i32, "p")
            p.add_incoming(llvm.const_int(i32, 1), a)
            p.add_incoming(llvm.const_int(i32, 2), b_)
            bld.ret(p)
        printed = str(mod)
        assert "phi i32 [ 1, %a ], [ 2, %b ]" in printed
        del bld, fn, entry, a, b_, join, p, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_builder.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'IRBuilder'`.

- [ ] **Step 3: Write `src/IR/Builder.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/IRBuilder.h>

#include <memory>

namespace {
// Owns an IRBuilder plus a saved insertion point, so `with builder.at_end_of(bb)`
// restores the previous point on __exit__.
struct InsertGuard {
  llvm::IRBuilder<> *builder;
  llvm::BasicBlock *block;
};
} // namespace

void populate_builder(nb::module_ &m) {
  using B = llvm::IRBuilder<>;

  nb::class_<InsertGuard>(m, "_InsertGuard")
      .def("__enter__",
           [](InsertGuard &g) { g.builder->SetInsertPoint(g.block); })
      .def("__exit__",
           [](InsertGuard &, nb::handle, nb::handle, nb::handle) {});

  nb::class_<B>(m, "IRBuilder")
      .def("__init__",
           [](B *self, eudsl::Context &ctx) {
             new (self) B(ctx.get());
           },
           "context"_a, nb::keep_alive<1, 2>())
      .def("set_insert_point",
           [](B &self, llvm::BasicBlock *bb) { self.SetInsertPoint(bb); },
           "block"_a)
      .def("at_end_of",
           [](B &self, llvm::BasicBlock *bb) {
             return InsertGuard{&self, bb};
           },
           "block"_a, nb::keep_alive<0, 1>())
      .def_prop_ro("insert_block", &B::GetInsertBlock,
                   nb::rv_policy::reference_internal)
      .def(
          "ret",
          [](B &self, llvm::Value *v) -> llvm::Value * {
            return v ? self.CreateRet(v) : self.CreateRetVoid();
          },
          "value"_a = nullptr, nb::rv_policy::reference_internal)
      .def("br",
           [](B &self, llvm::BasicBlock *dest) -> llvm::Value * {
             return self.CreateBr(dest);
           },
           "dest"_a, nb::rv_policy::reference_internal)
      .def("cond_br",
           [](B &self, llvm::Value *c, llvm::BasicBlock *t,
              llvm::BasicBlock *f) -> llvm::Value * {
             return self.CreateCondBr(c, t, f);
           },
           "cond"_a, "true_dest"_a, "false_dest"_a,
           nb::rv_policy::reference_internal)
#define EUDSL_BIN(pyName, method)                                              \
  .def(                                                                        \
      pyName,                                                                  \
      [](B &self, llvm::Value *l, llvm::Value *r, const std::string &name)     \
          -> llvm::Value * { return self.method(l, r, name); },                \
      "lhs"_a, "rhs"_a, "name"_a = "", nb::rv_policy::reference_internal)
          EUDSL_BIN("add", CreateAdd) EUDSL_BIN("fadd", CreateFAdd)
              EUDSL_BIN("sub", CreateSub) EUDSL_BIN("fsub", CreateFSub)
                  EUDSL_BIN("mul", CreateMul) EUDSL_BIN("fmul", CreateFMul)
                      EUDSL_BIN("sdiv", CreateSDiv) EUDSL_BIN("udiv", CreateUDiv)
                          EUDSL_BIN("fdiv", CreateFDiv)
#undef EUDSL_BIN
      .def("icmp",
           [](B &self, llvm::CmpInst::Predicate p, llvm::Value *l,
              llvm::Value *r, const std::string &name) -> llvm::Value * {
             return self.CreateICmp(p, l, r, name);
           },
           "predicate"_a, "lhs"_a, "rhs"_a, "name"_a = "",
           nb::rv_policy::reference_internal)
      .def("fcmp",
           [](B &self, llvm::CmpInst::Predicate p, llvm::Value *l,
              llvm::Value *r, const std::string &name) -> llvm::Value * {
             return self.CreateFCmp(p, l, r, name);
           },
           "predicate"_a, "lhs"_a, "rhs"_a, "name"_a = "",
           nb::rv_policy::reference_internal)
      .def("alloca",
           [](B &self, llvm::Type *ty, const std::string &name)
               -> llvm::Value * { return self.CreateAlloca(ty, nullptr, name); },
           "type"_a, "name"_a = "", nb::rv_policy::reference_internal)
      .def("load",
           [](B &self, llvm::Type *ty, llvm::Value *ptr,
              const std::string &name) -> llvm::Value * {
             return self.CreateLoad(ty, ptr, name);
           },
           "type"_a, "ptr"_a, "name"_a = "",
           nb::rv_policy::reference_internal)
      .def("store",
           [](B &self, llvm::Value *v, llvm::Value *ptr) -> llvm::Value * {
             return self.CreateStore(v, ptr);
           },
           "value"_a, "ptr"_a, nb::rv_policy::reference_internal)
      .def("gep",
           [](B &self, llvm::Type *ty, llvm::Value *ptr,
              std::vector<llvm::Value *> idxs,
              const std::string &name) -> llvm::Value * {
             return self.CreateGEP(ty, ptr, idxs, name);
           },
           "type"_a, "ptr"_a, "indices"_a, "name"_a = "",
           nb::rv_policy::reference_internal)
      .def("call",
           [](B &self, llvm::Function *fn, std::vector<llvm::Value *> args,
              const std::string &name) -> llvm::Value * {
             return self.CreateCall(fn->getFunctionType(), fn, args, name);
           },
           "fn"_a, "args"_a, "name"_a = "",
           nb::rv_policy::reference_internal)
      .def("phi",
           [](B &self, llvm::Type *ty, const std::string &name) {
             return self.CreatePHI(ty, 0, name);
           },
           "type"_a, "name"_a = "", nb::rv_policy::reference_internal);
}
```

Add `#include <llvm/IR/Function.h>` for `getFunctionType`. `CreatePHI` returns `PHINode*`; the Value type_hook makes the Python object a `PHINode`, so `.add_incoming` from Task 10 works.

- [ ] **Step 4: Register `populate_builder`** in `src/eudslllvm_ext.cpp` (after `populate_constants`), add `src/IR/Builder.cpp` to `nanobind_add_module`.

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind IRBuilder with an insertion-point context manager"
```

---

### Task 13: Attributes, linkage, visibility, calling convention

**Files:**
- Create: `src/IR/Attributes.cpp`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`, `src/IR/Values.cpp` (Function attribute methods)
- Test: `tests/test_attributes.py`

**Interfaces:**
- Consumes: `Function`, `Context`.
- Produces: `llvm.Linkage` and `llvm.CallingConv` and `llvm.Visibility` enums; `llvm.Function.linkage` (rw), `.calling_conv` (rw), `.visibility` (rw), `.add_fn_attr(name, value="")`, `.has_fn_attr(name)`, `.fn_attr_value(name)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_attributes.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def _fn(ctx, mod):
    return llvm.Function.create(
        llvm.function_t(llvm.void_t(ctx), []), "f", mod
    )


def test_linkage_and_calling_conv():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.linkage = llvm.Linkage.INTERNAL
        assert f.linkage == llvm.Linkage.INTERNAL
        assert "define internal void @f()" in str(mod)
        f.calling_conv = llvm.CallingConv.FAST
        assert f.calling_conv == llvm.CallingConv.FAST
        del f, mod
    assert_no_leaks()


def test_string_fn_attribute():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f = _fn(ctx, mod)
        f.add_fn_attr("target-cpu", "znver3")
        assert f.has_fn_attr("target-cpu")
        assert f.fn_attr_value("target-cpu") == "znver3"
        assert 'target-cpu"="znver3' in str(mod)
        del f, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_attributes.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'Linkage'`.

- [ ] **Step 3: Write `src/IR/Attributes.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/AsmParser/Parser.h> // unused-safe; keeps include set uniform
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>

void populate_attributes(nb::module_ &m) {
  nb::enum_<llvm::GlobalValue::LinkageTypes>(m, "Linkage")
      .value("EXTERNAL", llvm::GlobalValue::ExternalLinkage)
      .value("INTERNAL", llvm::GlobalValue::InternalLinkage)
      .value("PRIVATE", llvm::GlobalValue::PrivateLinkage)
      .value("LINKONCE", llvm::GlobalValue::LinkOnceAnyLinkage)
      .value("LINKONCE_ODR", llvm::GlobalValue::LinkOnceODRLinkage)
      .value("WEAK", llvm::GlobalValue::WeakAnyLinkage)
      .value("COMMON", llvm::GlobalValue::CommonLinkage)
      .value("APPENDING", llvm::GlobalValue::AppendingLinkage)
      .value("EXTERNAL_WEAK", llvm::GlobalValue::ExternalWeakLinkage);

  nb::enum_<llvm::GlobalValue::VisibilityTypes>(m, "Visibility")
      .value("DEFAULT", llvm::GlobalValue::DefaultVisibility)
      .value("HIDDEN", llvm::GlobalValue::HiddenVisibility)
      .value("PROTECTED", llvm::GlobalValue::ProtectedVisibility);

  // CallingConv::ID is a plain unsigned namespace of constants, not an enum
  // class; expose the common ones as module-level ints under a submodule.
  nb::module_ cc = m.def_submodule("CallingConv");
  cc.attr("C") = (unsigned)llvm::CallingConv::C;
  cc.attr("FAST") = (unsigned)llvm::CallingConv::Fast;
  cc.attr("COLD") = (unsigned)llvm::CallingConv::Cold;
}
```

`CallingConv` values are `unsigned` constants (`llvm::CallingConv::C == 0`, `Fast == 8`), not an `enum class`. The test compares `f.calling_conv == llvm.CallingConv.FAST`; `.calling_conv` is bound as an `unsigned` in Step 4, so equality is int-vs-int. Confirm `llvm.CallingConv.FAST` is `8`.

- [ ] **Step 4: Add the Function attribute methods in `src/IR/Values.cpp`**

Add these `.def`s to the `Function` class binding (add `#include <llvm/IR/Attributes.h>`, `#include <llvm/IR/CallingConv.h>`):

```cpp
      .def_prop_rw("linkage", &llvm::Function::getLinkage,
                   &llvm::Function::setLinkage)
      .def_prop_rw("visibility", &llvm::Function::getVisibility,
                   &llvm::Function::setVisibility)
      .def_prop_rw(
          "calling_conv",
          [](llvm::Function &self) { return (unsigned)self.getCallingConv(); },
          [](llvm::Function &self, unsigned cc) {
            self.setCallingConv((llvm::CallingConv::ID)cc);
          })
      .def(
          "add_fn_attr",
          [](llvm::Function &self, const std::string &name,
             const std::string &value) { self.addFnAttr(name, value); },
          "name"_a, "value"_a = "")
      .def("has_fn_attr",
           [](llvm::Function &self, const std::string &name) {
             return self.hasFnAttribute(name);
           },
           "name"_a)
      .def("fn_attr_value",
           [](llvm::Function &self, const std::string &name) {
             return self.getFnAttribute(name).getValueAsString().str();
           },
           "name"_a)
```

`Function::addFnAttr(StringRef, StringRef)`, `hasFnAttribute(StringRef)`, `getFnAttribute(StringRef)` are the string-keyed attribute API (verified in the exploration). `getVisibility`/`setVisibility`/`getLinkage`/`setLinkage` come from `GlobalValue`.

- [ ] **Step 5: Register `populate_attributes`** in `src/eudslllvm_ext.cpp` **before** `populate_values` (the `Linkage`/`Visibility` enums must exist before `Function`'s `def_prop_rw("linkage", ...)` references them at binding time — actually the property lambdas only need the *C++* enum type, which nanobind maps to the registered `nb::enum_`; register `populate_attributes` before `populate_values` to be safe). Add `src/IR/Attributes.cpp` to `nanobind_add_module`.

- [ ] **Step 6: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind attributes, linkage, visibility, calling convention"
```

---

### Task 14: Metadata, MDNode, named metadata

**Files:**
- Create: `src/IR/Metadata.cpp`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`, `src/IR/Context.cpp` (named metadata on Module)
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `Context`, `Module`.
- Produces: `llvm.MDString(ctx, s)`, `llvm.MDNode(ctx, operands)`, `llvm.MDNode.operand(i)`, `.num_operands`; `llvm.Module.add_named_metadata(name, node)`, `llvm.Module.named_metadata(name) -> list`.

- [ ] **Step 1: Write the failing test**

`tests/test_metadata.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_named_metadata_round_trips():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        s = llvm.MDString(ctx, "hello")
        node = llvm.MDNode(ctx, [s])
        mod.add_named_metadata("my.meta", node)
        printed = str(mod)
        assert "!my.meta = !{!0}" in printed
        assert '!0 = !{!"hello"}' in printed
        got = mod.named_metadata("my.meta")
        assert len(got) == 1
        del mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_metadata.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'MDString'`.

- [ ] **Step 3: Write `src/IR/Metadata.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Metadata.h>

void populate_metadata(nb::module_ &m) {
  nb::class_<llvm::Metadata>(m, "Metadata")
      .def("__str__", [](llvm::Metadata &self) { return eudsl::toString(self); });

  nb::class_<llvm::MDString, llvm::Metadata>(m, "MDString")
      .def("__init__",
           [](llvm::MDString *, eudsl::Context &, const std::string &) {
             // MDString has no public constructor; use the factory instead.
           })
      .def_prop_ro("string",
                   [](llvm::MDString &self) { return self.getString().str(); });

  nb::class_<llvm::MDNode, llvm::Metadata>(m, "MDNode")
      .def_prop_ro("num_operands", &llvm::MDNode::getNumOperands)
      .def("operand",
           [](llvm::MDNode &self, unsigned i) -> llvm::Metadata * {
             return self.getOperand(i).get();
           },
           "index"_a, nb::rv_policy::reference_internal);

  m.def(
      "MDString",
      [](eudsl::Context &ctx, const std::string &s) -> llvm::MDString * {
        return llvm::MDString::get(ctx.get(), s);
      },
      "context"_a, "value"_a, nb::rv_policy::reference_internal);
  m.def(
      "MDNode",
      [](eudsl::Context &ctx,
         std::vector<llvm::Metadata *> ops) -> llvm::MDNode * {
        return llvm::MDNode::get(ctx.get(), ops);
      },
      "context"_a, "operands"_a, nb::rv_policy::reference_internal);
}
```

`MDString` has no public constructor, so the `__init__` stub above is wrong; drop the `.def("__init__", ...)` and rely on the free `m.def("MDString", ...)` factory (nanobind lets a free function share the class name — `llvm.MDString(ctx, "x")` calls the factory, which returns an `MDString*`). Remove the `__init__` lambda entirely.

- [ ] **Step 4: Add named metadata to `src/IR/Context.cpp`**

Add `#include <llvm/IR/Metadata.h>` and to the `Module` class:

```cpp
      .def(
          "add_named_metadata",
          [](eudsl::Module &self, const std::string &name, llvm::MDNode *node) {
            self.get().getOrInsertNamedMetadata(name)->addOperand(node);
          },
          "name"_a, "node"_a)
      .def(
          "named_metadata",
          [](eudsl::Module &self, const std::string &name) {
            std::vector<llvm::MDNode *> out;
            if (auto *nmd = self.get().getNamedMetadata(name))
              for (llvm::MDNode *op : nmd->operands())
                out.push_back(op);
            return out;
          },
          "name"_a)
```

- [ ] **Step 5: Register `populate_metadata`** in `src/eudslllvm_ext.cpp` (after `populate_constants`), add `src/IR/Metadata.cpp` to `nanobind_add_module`.

- [ ] **Step 6: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind Metadata, MDNode, MDString and named metadata"
```

---

### Task 15: Error handling — `ParseError`, `VerifyError`, fatal error handler

**Files:**
- Create: `src/IR/Errors.h`, `src/IR/Errors.cpp`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`, `src/IR/Context.cpp` (raise `ParseError`)
- Test: `tests/test_errors.py`

**Interfaces:**
- Consumes: `parse_assembly`.
- Produces: `llvm.ParseError`, `llvm.VerifyError` exception types; `parse_assembly` raises `ParseError` on bad IR; the fatal-error handler is installed at import.

- [ ] **Step 1: Write the failing tests**

`tests/test_errors.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_parse_error_is_specific():
    with llvm.Context() as ctx:
        with pytest.raises(llvm.ParseError):
            llvm.parse_assembly("this is not IR", ctx, "bad")
    assert_no_leaks()


def test_parse_error_is_an_exception_subclass():
    assert issubclass(llvm.ParseError, Exception)
    assert issubclass(llvm.VerifyError, Exception)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_errors.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'ParseError'`.

- [ ] **Step 3: Write `src/IR/Errors.h`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

namespace eudsl {

// C++ exception types raised from binding code and mapped to Python exceptions
// registered in Errors.cpp.
struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct VerifyError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void registerExceptions(nanobind::module_ &m);

} // namespace eudsl
```

- [ ] **Step 4: Write `src/IR/Errors.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Errors.h"

#include <llvm/Support/ErrorHandling.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace {
// Fatal errors abort the process; convert the message to something visible
// before LLVM calls abort(). The handler cannot return, so this is a
// best-effort last word rather than a recoverable path.
void fatalHandler(void *, const char *reason, bool) {
  PyErr_WarnEx(PyExc_RuntimeWarning,
               (std::string("LLVM fatal error: ") + reason).c_str(), 1);
}
} // namespace

namespace eudsl {

void registerExceptions(nb::module_ &m) {
  nb::exception<ParseError>(m, "ParseError");
  nb::exception<VerifyError>(m, "VerifyError");
  llvm::install_fatal_error_handler(fatalHandler);
}

} // namespace eudsl
```

`nb::exception<T>(m, "Name")` registers a Python exception and a translator that converts a thrown `T` into it. `PyErr_WarnEx` is safe to call before `abort()`.

- [ ] **Step 5: Raise `ParseError` from `parse_assembly`**

In `src/IR/Context.cpp`, add `#include "IR/Errors.h"` and change the `throw std::runtime_error(msg);` in `parse_assembly` to `throw eudsl::ParseError(msg);`.

- [ ] **Step 6: Register exceptions first in `src/eudslllvm_ext.cpp`**

```cpp
#include "IR/Errors.h"
...
NB_MODULE(eudslllvm_ext, m) {
  m.doc() = "Hand-written nanobind bindings over the LLVM C++ IR API.";
  eudsl::registerExceptions(m);
  populate_attributes(m);
  populate_context(m);
  populate_types(m);
  populate_values(m);
  populate_instructions(m);
  populate_constants(m);
  populate_metadata(m);
  populate_builder(m);
}
```

Add `src/IR/Errors.cpp` to `nanobind_add_module`.

- [ ] **Step 7: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add ParseError, VerifyError, and a fatal-error handler"
```

---

## Phase A2 — compile and run

### Task 16: `verifyModule`; bitcode read and write

**Files:**
- Modify: `src/IR/Context.cpp`, `CMakeLists.txt` (link `LLVMBitReader`, `LLVMBitWriter`)
- Test: `tests/test_verify_bitcode.py`

**Interfaces:**
- Consumes: `Module`, `Context`, `VerifyError` (Task 15).
- Produces: `llvm.Module.verify()` (raises `VerifyError` on failure, returns `None` on success), `llvm.Module.to_bitcode() -> bytes`, `llvm.parse_bitcode(data: bytes, ctx) -> Module`.

- [ ] **Step 1: Write the failing tests**

`tests/test_verify_bitcode.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_GOOD = dedent(
    """\
    define i32 @f(i32 %x) {
    entry:
      ret i32 %x
    }
    """
)


def test_verify_accepts_good_module():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_GOOD, ctx, "m")
        assert mod.verify() is None
        del mod
    assert_no_leaks()


def test_bitcode_round_trip():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_GOOD, ctx, "m")
        data = mod.to_bitcode()
        assert isinstance(data, bytes)
        assert data[:2] == b"BC"
        del mod
    with llvm.Context() as ctx2:
        mod2 = llvm.parse_bitcode(data, ctx2)
        assert "define i32 @f(i32 %x)" in str(mod2)
        del mod2
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_verify_bitcode.py -v
```

Expected: `AttributeError: 'Module' object has no attribute 'verify'`.

- [ ] **Step 3: Add verify and bitcode to `src/IR/Context.cpp`**

Add includes:

```cpp
#include "IR/Errors.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/MemoryBuffer.h>

#include <nanobind/stl/bytes.h>
```

Add to the `Module` class body:

```cpp
      .def("verify",
           [](eudsl::Module &self) {
             std::string msg;
             llvm::raw_string_ostream os(msg);
             if (llvm::verifyModule(self.get(), &os))
               throw eudsl::VerifyError(msg);
           })
      .def("to_bitcode",
           [](eudsl::Module &self) {
             std::string buf;
             llvm::raw_string_ostream os(buf);
             llvm::WriteBitcodeToFile(self.get(), os);
             os.flush();
             return nb::bytes(buf.data(), buf.size());
           })
```

Add a free function after the `parse_assembly` def:

```cpp
  m.def(
      "parse_bitcode",
      [](nb::bytes data, eudsl::Context &ctx) {
        llvm::StringRef ref(data.c_str(), data.size());
        auto buf = llvm::MemoryBuffer::getMemBuffer(ref, "<bitcode>", false);
        llvm::Expected<std::unique_ptr<llvm::Module>> mod =
            llvm::parseBitcodeFile(buf->getMemBufferRef(), ctx.get());
        if (!mod)
          throw eudsl::ParseError(llvm::toString(mod.takeError()));
        return new eudsl::Module(std::move(*mod), ctx);
      },
      "data"_a, "context"_a, nb::keep_alive<0, 2>());
```

Add `#include <nanobind/stl/bytes.h>` to `Common.h`. Add `LLVMBitReader LLVMBitWriter` to `eudslllvm_ext_libs` in `CMakeLists.txt`.

- [ ] **Step 4: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add verifyModule, bitcode read and write"
```

---

### Task 17: `PassBuilder` and pipeline execution

**Files:**
- Create: `src/IR/Passes.cpp`
- Modify: `CMakeLists.txt` (link `LLVMPasses LLVMAnalysis LLVMTransformUtils`), `src/eudslllvm_ext.cpp`
- Test: `tests/test_passes.py`

**Interfaces:**
- Consumes: `Module`.
- Produces: `llvm.run_passes(module, pipeline: str)` which builds the standard analysis managers, parses the textual pipeline, runs it, and raises `RuntimeError` on a bad pipeline string.

- [ ] **Step 1: Write the failing tests**

`tests/test_passes.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

# An always-true branch that instcombine + simplifycfg fold away.
_SRC = dedent(
    """\
    define i32 @f(i32 %x) {
    entry:
      %a = add i32 %x, 0
      ret i32 %a
    }
    """
)


def test_instcombine_removes_add_zero():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        assert "add i32 %x, 0" in str(mod)
        llvm.run_passes(mod, "instcombine")
        assert "add i32 %x, 0" not in str(mod)
        del mod
    assert_no_leaks()


def test_bad_pipeline_raises():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        with pytest.raises(RuntimeError):
            llvm.run_passes(mod, "not-a-real-pass")
        del mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_passes.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'run_passes'`.

- [ ] **Step 3: Write `src/IR/Passes.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>

void populate_passes(nb::module_ &m) {
  m.def(
      "run_passes",
      [](eudsl::Module &mod, const std::string &pipeline) {
        llvm::PassBuilder pb;
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm;
        if (llvm::Error err = pb.parsePassPipeline(mpm, pipeline))
          throw std::runtime_error(llvm::toString(std::move(err)));
        mpm.run(mod.get(), mam);
      },
      "module"_a, "pipeline"_a,
      "Parse and run a textual pass pipeline over the module in place.");
}
```

- [ ] **Step 4: Register `populate_passes`** in `src/eudslllvm_ext.cpp` (after `populate_builder`), add `src/IR/Passes.cpp` to `nanobind_add_module`, add `LLVMPasses LLVMAnalysis LLVMTransformUtils LLVMScalarOpts LLVMInstCombine` to `eudslllvm_ext_libs`.

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add PassBuilder pipeline execution via run_passes"
```

---

### Task 18: `Target`, `TargetMachine`, `DataLayout`, assembly and object emission

**Files:**
- Create: `src/IR/Target.cpp`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`, `src/IR/Ownership.h`/`.cpp` (Context::take)
- Test: `tests/test_target.py`

**Interfaces:**
- Consumes: `Module`, target libraries linked in Task 2.
- Produces: `llvm.host_triple() -> str`; `llvm.TargetMachine(triple="", cpu="", features="")` with `.triple`, `.data_layout_str`, `.emit_assembly(module) -> str`, `.emit_object(module) -> bytes`; `llvm.Module.set_data_layout_from(tm)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_target.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """
)


def test_host_triple_is_nonempty():
    assert isinstance(llvm.host_triple(), str)
    assert llvm.host_triple()


def test_emit_assembly_and_object():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        tm = llvm.TargetMachine(llvm.host_triple())
        assert tm.data_layout_str
        mod.set_data_layout_from(tm)
        asm = tm.emit_assembly(mod)
        assert "add" in asm
        obj = tm.emit_object(mod)
        assert isinstance(obj, bytes)
        assert len(obj) > 0
        del tm, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_target.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'host_triple'`.

- [ ] **Step 3: Write `src/IR/Target.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>

#include <nanobind/stl/bytes.h>

#include <memory>

namespace {
// Owns a TargetMachine.
struct TM {
  std::unique_ptr<llvm::TargetMachine> tm;
};

std::string emit(TM &self, eudsl::Module &mod, llvm::CodeGenFileType type) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  llvm::buffer_ostream bos(os);
  llvm::legacy::PassManager pm;
  if (self.tm->addPassesToEmitFile(pm, bos, nullptr, type))
    throw std::runtime_error("target cannot emit this file type");
  pm.run(mod.get());
  return buf;
}
} // namespace

void populate_target(nb::module_ &m) {
  m.def("host_triple", []() { return llvm::sys::getDefaultTargetTriple(); });

  nb::class_<TM>(m, "TargetMachine")
      .def("__init__",
           [](TM *self, const std::string &triple, const std::string &cpu,
              const std::string &features) {
             std::string tripleStr =
                 triple.empty() ? llvm::sys::getDefaultTargetTriple() : triple;
             llvm::Triple tt(tripleStr);
             std::string err;
             const llvm::Target *target =
                 llvm::TargetRegistry::lookupTarget(tt, err);
             if (!target)
               throw std::runtime_error(err);
             llvm::TargetOptions opts;
             llvm::TargetMachine *tm = target->createTargetMachine(
                 tt, cpu, features, opts, std::nullopt);
             if (!tm)
               throw std::runtime_error("could not create TargetMachine for " +
                                        tripleStr);
             new (self) TM{std::unique_ptr<llvm::TargetMachine>(tm)};
           },
           "triple"_a = "", "cpu"_a = "", "features"_a = "")
      .def_prop_ro("triple",
                   [](TM &self) {
                     return self.tm->getTargetTriple().str();
                   })
      .def_prop_ro("data_layout_str",
                   [](TM &self) {
                     return self.tm->createDataLayout().getStringRepresentation();
                   })
      .def("emit_assembly",
           [](TM &self, eudsl::Module &mod) {
             return emit(self, mod, llvm::CodeGenFileType::AssemblyFile);
           },
           "module"_a)
      .def("emit_object", [](TM &self, eudsl::Module &mod) {
        std::string obj = emit(self, mod, llvm::CodeGenFileType::ObjectFile);
        return nb::bytes(obj.data(), obj.size());
      }, "module"_a);
}
```

Add to `src/IR/Context.cpp`'s `Module` class, `set_data_layout_from`:

```cpp
      .def("set_data_layout_from",
           [](eudsl::Module &self, nb::handle tm) {
             // TargetMachine lives in Target.cpp; fetch its data-layout string
             // through the bound property to avoid a cross-file C++ dependency.
             std::string dl =
                 nb::cast<std::string>(tm.attr("data_layout_str"));
             self.get().setDataLayout(dl);
           },
           "target_machine"_a)
```

`Triple(const std::string&)` and `TargetRegistry::lookupTarget(const Triple&, std::string&)` were verified. `addPassesToEmitFile` uses the legacy pass manager (still the codegen path). `emit` returns the assembly/object as a string; `emit_object` wraps it in `bytes`.

- [ ] **Step 4: Register `populate_target`** in `src/eudslllvm_ext.cpp` (after `populate_passes`), add `src/IR/Target.cpp` to `nanobind_add_module`, add `LLVMCodeGen LLVMTarget LLVMMC` (already present from Task 2) and confirm the target libs from Task 2 are linked.

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind Target, TargetMachine, DataLayout, asm and object emission"
```

---

### Task 19: `Linker`

**Files:**
- Create: `src/IR/Linker.cpp`
- Modify: `CMakeLists.txt` (link `LLVMLinker`), `src/eudslllvm_ext.cpp`
- Test: `tests/test_linker.py`

**Interfaces:**
- Consumes: `Module` (and `Module::take()` from Task 1).
- Produces: `llvm.link_into(dest: Module, src: Module)` which consumes `src` (marking it moved-from) and links it into `dest`, raising `RuntimeError` on failure.

- [ ] **Step 1: Write the failing test**

`tests/test_linker.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_link_two_modules():
    with llvm.Context() as ctx:
        dest = llvm.parse_assembly("declare i32 @a()\n", ctx, "dest")
        src = llvm.parse_assembly(
            dedent(
                """\
                define i32 @a() {
                  ret i32 7
                }
                """
            ),
            ctx,
            "src",
        )
        llvm.link_into(dest, src)
        assert src._is_consumed
        assert "define i32 @a()" in str(dest)
        with pytest.raises(RuntimeError, match="has been consumed"):
            str(src)
        del dest, src
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_linker.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'link_into'`.

- [ ] **Step 3: Write `src/IR/Linker.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/Linker/Linker.h>

void populate_linker(nb::module_ &m) {
  m.def(
      "link_into",
      [](eudsl::Module &dest, eudsl::Module &src) {
        // linkModules consumes the source module; take() marks the Python
        // wrapper moved-from so later use raises rather than segfaults.
        std::unique_ptr<llvm::Module> srcOwned = src.take();
        if (llvm::Linker::linkModules(dest.get(), std::move(srcOwned)))
          throw std::runtime_error("linkModules failed");
      },
      "dest"_a, "src"_a);
}
```

`Linker::linkModules(Module&, std::unique_ptr<Module>)` returns `true` on error (verified). `src.take()` transfers ownership and flips `_is_consumed`.

- [ ] **Step 4: Register `populate_linker`** in `src/eudslllvm_ext.cpp` (after `populate_target`), add `src/IR/Linker.cpp` to `nanobind_add_module`, add `LLVMLinker` to `eudslllvm_ext_libs`.

- [ ] **Step 5: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind the Linker via link_into with moved-from tracking"
```

---

### Task 20: ORC `LLJIT` — add module, lookup, ctypes-callable addresses, execution

**Files:**
- Create: `src/IR/JIT.cpp`
- Modify: `CMakeLists.txt` (link `LLVMOrcJIT LLVMExecutionEngine LLVMJITLink LLVMOrcTargetProcess LLVMOrcShared`), `src/eudslllvm_ext.cpp`, `src/IR/Ownership.h`/`.cpp` (`Context::take`)
- Test: `tests/test_jit.py`

**Interfaces:**
- Consumes: `Module`, `Context`, target init from Task 2.
- Produces: `Context._take_llvm()` (relinquish the underlying `LLVMContext`, marking consumed), `llvm.LLJIT()` with `.add_module(module)` (consumes the module *and* its context) and `.lookup(name) -> int` (function address as an integer usable with `ctypes`).

- [ ] **Step 1: Write the failing test**

`tests/test_jit.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """
)


def test_jit_execute():
    ctx = llvm.Context()
    mod = llvm.parse_assembly(_SRC, ctx, "m")
    jit = llvm.LLJIT()
    jit.add_module(mod)  # consumes mod and ctx
    assert mod._is_consumed
    addr = jit.lookup("add")
    assert addr != 0
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(addr)
    assert fn(2, 40) == 42
    del jit, mod, ctx
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_jit.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'LLJIT'`.

- [ ] **Step 3: Add `Context::take` to `src/IR/Ownership.h`/`.cpp`**

Header, in `class Context`:

```cpp
  /// Relinquish the underlying LLVMContext (e.g. into a ThreadSafeModule). The
  /// live count drops here; later get() throws.
  std::unique_ptr<llvm::LLVMContext> take();
  bool isConsumed() const { return ctx == nullptr; }
```

`.cpp`:

```cpp
std::unique_ptr<llvm::LLVMContext> Context::take() {
  if (!ctx)
    throw std::runtime_error("context already consumed");
  --gLiveContexts;
  return std::move(ctx);
}
```

Change `Context::get()` to throw if consumed:

```cpp
llvm::LLVMContext &Context::get() const {
  if (!ctx)
    throw std::runtime_error("context has been consumed");
  return *ctx;
}
```

(Change the header declaration of `get()` accordingly — it is no longer inline-trivial; move the body to the `.cpp`.) Expose `_take_llvm` and `_is_consumed` on the Python `Context` in `Context.cpp`:

```cpp
      .def_prop_ro("_is_consumed", &eudsl::Context::isConsumed)
```

- [ ] **Step 4: Write `src/IR/JIT.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include <memory>

namespace {
struct JIT {
  std::unique_ptr<llvm::orc::LLJIT> jit;
};
} // namespace

void populate_jit(nb::module_ &m) {
  nb::class_<JIT>(m, "LLJIT")
      .def("__init__",
           [](JIT *self) {
             auto jit = eudsl::unwrap(llvm::orc::LLJITBuilder().create());
             new (self) JIT{std::move(jit)};
           })
      .def(
          "add_module",
          [](JIT &self, eudsl::Module &mod) {
            // Take both the module and its context into a ThreadSafeModule.
            eudsl::Context &ctx = mod.context();
            std::unique_ptr<llvm::Module> m = mod.take();
            std::unique_ptr<llvm::LLVMContext> c = ctx.take();
            llvm::orc::ThreadSafeModule tsm(std::move(m), std::move(c));
            eudsl::unwrap(self.jit->addIRModule(std::move(tsm)));
          },
          "module"_a)
      .def(
          "lookup",
          [](JIT &self, const std::string &name) {
            llvm::orc::ExecutorAddr addr =
                eudsl::unwrap(self.jit->lookup(name));
            return static_cast<uint64_t>(addr.getValue());
          },
          "name"_a);
}
```

`ThreadSafeModule(unique_ptr<Module>, unique_ptr<LLVMContext>)`, `LLJITBuilder().create()`, `addIRModule(ThreadSafeModule)`, `lookup(StringRef)`, `ExecutorAddr::getValue()` all verified. Taking the context out of `eudsl::Context` drops the live count, so `assert_no_leaks()` passes after the JIT owns everything and is deleted.

- [ ] **Step 5: Register `populate_jit`** in `src/eudslllvm_ext.cpp` (after `populate_linker`), add `src/IR/JIT.cpp` to `nanobind_add_module`, add `LLVMOrcJIT LLVMExecutionEngine LLVMJITLink LLVMOrcTargetProcess LLVMOrcShared LLVMRuntimeDyld` to `eudslllvm_ext_libs`.

- [ ] **Step 6: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS, including the ctypes execution asserting `fn(2, 40) == 42`.

- [ ] **Step 7: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind ORC LLJIT with module add, lookup, and execution tests"
```

---

### Task 21: `Intrinsic` lookup and declaration; the `llvm.intrinsics` `__getattr__` module

**Files:**
- Create: `src/IR/Intrinsics.cpp`, `src/llvm/intrinsics.py`
- Modify: `CMakeLists.txt`, `src/eudslllvm_ext.cpp`, `src/llvm/__init__.py`
- Test: `tests/test_intrinsics.py`, and reinstate `test_builder`'s intrinsic assertion

**Interfaces:**
- Consumes: `Module`, `Type`, `Function`, `Context`, `IRBuilder`.
- Produces: `llvm.lookup_intrinsic_id(name: str) -> int` (0 if unknown), `llvm.intrinsic_is_overloaded(id) -> bool`, `llvm.get_intrinsic_declaration(module, id, overload_types) -> Function`; the Python module `llvm.intrinsics` whose `__getattr__("sqrt")` resolves `llvm.sqrt`.

- [ ] **Step 1: Write the failing tests**

`tests/test_intrinsics.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
import llvm.intrinsics
from llvm.testing import assert_no_leaks


def test_lookup_intrinsic_id():
    assert llvm.lookup_intrinsic_id("llvm.sqrt") != 0
    assert llvm.lookup_intrinsic_id("llvm.not.a.real.intrinsic") == 0


def test_get_overloaded_declaration():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        sqrt_id = llvm.lookup_intrinsic_id("llvm.sqrt")
        assert llvm.intrinsic_is_overloaded(sqrt_id)
        f32 = llvm.f32(ctx)
        decl = llvm.get_intrinsic_declaration(mod, sqrt_id, [f32])
        assert decl.name == "llvm.sqrt.f32"
        assert "declare float @llvm.sqrt.f32(float)" in str(mod)
        del decl, mod
    assert_no_leaks()


def test_intrinsics_getattr_shim():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f64 = llvm.f64(ctx)
        decl = llvm.intrinsics.sqrt(mod, [f64])
        assert decl.name == "llvm.sqrt.f64"
        del decl, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_intrinsics.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'lookup_intrinsic_id'`.

- [ ] **Step 3: Write `src/IR/Intrinsics.cpp`**

```cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Intrinsics.h>

void populate_intrinsics(nb::module_ &m) {
  m.def(
      "lookup_intrinsic_id",
      [](const std::string &name) {
        return (unsigned)llvm::Intrinsic::lookupIntrinsicID(name);
      },
      "name"_a);
  m.def(
      "intrinsic_is_overloaded",
      [](unsigned id) {
        return llvm::Intrinsic::isOverloaded((llvm::Intrinsic::ID)id);
      },
      "id"_a);
  m.def(
      "get_intrinsic_declaration",
      [](eudsl::Module &mod, unsigned id,
         std::vector<llvm::Type *> overloadTypes) -> llvm::Function * {
        return llvm::Intrinsic::getOrInsertDeclaration(
            &mod.get(), (llvm::Intrinsic::ID)id, overloadTypes);
      },
      "module"_a, "id"_a, "overload_types"_a = std::vector<llvm::Type *>{},
      nb::rv_policy::reference_internal);
}
```

`lookupIntrinsicID`, `isOverloaded`, `getOrInsertDeclaration(Module*, ID, ArrayRef<Type*>)` all verified. Add `LLVMCore` (already linked) — intrinsics live in Core.

- [ ] **Step 4: Write `src/llvm/intrinsics.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Attribute-style access to LLVM intrinsics.

`llvm.intrinsics.sqrt(module, [f32])` resolves the id for `llvm.sqrt`, checks
it exists, and emits the overloaded declaration. Overload resolution happens
in C++ against LLVM's own tables.
"""

from . import (
    lookup_intrinsic_id,
    get_intrinsic_declaration,
)


def __getattr__(name):
    intrinsic_id = lookup_intrinsic_id(f"llvm.{name.replace('_', '.')}")
    if intrinsic_id == 0:
        raise AttributeError(f"unknown intrinsic llvm.{name}")

    def declare(module, overload_types=()):
        return get_intrinsic_declaration(module, intrinsic_id, list(overload_types))

    declare.__name__ = name
    return declare
```

- [ ] **Step 5: Register `populate_intrinsics`** in `src/eudslllvm_ext.cpp` (after `populate_jit`), add `src/IR/Intrinsics.cpp` to `nanobind_add_module`.

- [ ] **Step 6: Rebuild and run the suite**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS. This is the commit where the new layer overtakes the old one in capability.

- [ ] **Step 7: Commit, then submit the stack to open PR 1 (draft)**

This is the last task of Phase A. Commit it like the others, then submit the
stack to push the Phase A branch and open PR 1 as a draft (Tasks 1–21). The
Phase A branch was created with `gh stack init` before Task 1 (see **Delivery**).

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind Intrinsic lookup/declaration and the llvm.intrinsics shim"
# Push the Phase A branch and open PR 1 as a draft. Run plain `gh stack submit`
# to set the title/description interactively instead of auto-generating them.
gh stack submit --auto
```

---

## Phase B1 — typed values

**Before Task 22, add the Phase B branch to the stack** (see **Delivery**), so
every Phase B commit lands on the stacked branch:

```bash
cd $EUDSL && gh stack add users/makslevental/eudsl-llvmpy-dsl
```

The DSL emits into a *current builder*. Task 22 introduces that mechanism.

### Design note: value-caster subclasses vs. base-class dunders

**Chosen approach (implemented): MLIR-style value casters.** The DSL defines an
`ArithValue(llvm.Value)` subclass that carries the arithmetic/comparison
operators, and registers it — via a `register_value_caster(TypeID, ...)`
registry mirroring MLIR's `register_value_caster` / `PyValue::maybeDownCast` —
for the integer and floating-point type kinds. Values of those kinds are
re-wrapped as `ArithValue` by `maybe_downcast(v, parent)` (used by `@function`
on incoming args, and internally so operator results stay typed). Mechanics:

- C++ exposes one stateless primitive, `_wrap_value_as(value, py_type, parent)`,
  which calls `nb::inst_reference` to bind an existing `Value*` into a chosen
  Python (sub)type with its lifetime tied to `parent`. This is the only way to
  wrap an already-owned pointer as a Python subclass — a nanobind subclass has
  "no constructor defined" and cannot be built around an existing instance.
- The caster registry lives in Python (`llvm/dsl/casters.py`), keyed on the
  `TypeID` enum (added to `Type.type_id`). It is *not* a C++ static.
- `Type` exposes a `TypeID` nb::enum_ and `type_id` property.

**Rejected alternative: monkeypatch dunders onto the base `llvm.Value`.** Because
nanobind classes are heap types, `llvm.Value.__add__ = fn` does install the
number slot, so this works mechanically and needs no caster machinery. It was
rejected for two reasons: (1) it puts `__add__`/`__lt__`/etc. on *every* value
including pointers, labels, and void, so a nonsensical `ptr + x` fails only when
the builder rejects the operands rather than being absent; and (2) it diverges
from `eudsl-python-extras`, whose `ArithValue` is a registered value-caster
subclass — matching that keeps the two DSLs' typed-value model identical and
leaves room for user-registered casters (e.g. a downstream tensor value type).

**Rejected variant: hold the caster registry in a C++ `static`
`unordered_map<unsigned, nb::object>`.** Simpler wiring, but it retains Python
type references at static-storage duration, which triggers nanobind's
"leaked function ..." diagnostics (and risks a crash) at interpreter shutdown.
Keeping the registry in Python avoids holding any Python reference past
finalization.

This replaces the earlier plan text below (which specified base-class
`install_value_dunders`); the tasks are otherwise unchanged in scope. The DSL
still emits into a *current builder*; the operators now live on `ArithValue`.


### Task 22: Arithmetic dunders with integer/float dispatch and constant coercion

**Files:**
- Create: `src/llvm/dsl/__init__.py`, `src/llvm/dsl/context.py`, `src/llvm/dsl/values.py`
- Modify: `src/llvm/__init__.py` (install dunders on import)
- Test: `tests/test_dsl_values.py`

**Interfaces:**
- Consumes: `llvm.IRBuilder`, `llvm.Value`, `llvm.const_int`, `llvm.const_fp`.
- Produces: `llvm.dsl.context.current_builder() -> IRBuilder`, `llvm.dsl.context.building(builder)` (context manager), `llvm.dsl.values.install_value_dunders()`; `Value.__add__/__sub__/__mul__/__truediv__` etc. with int/float dispatch and Python-scalar coercion.

- [ ] **Step 1: Write the failing test**

`tests/test_dsl_values.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.dsl.context import building
from llvm.testing import assert_no_leaks


def _entry(ctx, mod, ret_ty, arg_tys, name="f"):
    fn = llvm.Function.create(llvm.function_t(ret_ty, arg_tys), name, mod)
    bb = fn.append_basic_block("entry")
    return fn, bb


def test_integer_add_and_mul():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb = _entry(ctx, mod, i32, [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            r = fn.arg(0) * fn.arg(1) + 1
            b.ret(r)
        printed = str(mod)
        assert "mul i32" in printed
        assert "add i32" in printed
        # `+ 1` coerced to an i32 constant.
        assert "add i32 %3, 1" in printed
        del b, fn, mod
    assert_no_leaks()


def test_float_add_uses_fadd():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.f32(ctx)
        fn, bb = _entry(ctx, mod, f32, [f32, f32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0) + fn.arg(1))
        assert "fadd float" in str(mod)
        del b, fn, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_values.py -v
```

Expected: `ModuleNotFoundError: No module named 'llvm.dsl'`.

- [ ] **Step 3: Write `src/llvm/dsl/context.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Thread-local DSL state: the current IRBuilder and enclosing function."""

import threading
from contextlib import contextmanager

_tls = threading.local()


def current_builder():
    b = getattr(_tls, "builder", None)
    if b is None:
        raise RuntimeError("no current IRBuilder; use `with building(builder):`")
    return b


def current_function():
    f = getattr(_tls, "function", None)
    if f is None:
        raise RuntimeError("no current function")
    return f


@contextmanager
def building(builder, function=None):
    prev_b = getattr(_tls, "builder", None)
    prev_f = getattr(_tls, "function", None)
    _tls.builder = builder
    if function is not None:
        _tls.function = function
    try:
        yield builder
    finally:
        _tls.builder = prev_b
        _tls.function = prev_f
```

- [ ] **Step 4: Write `src/llvm/dsl/values.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Operator overloading on llvm.Value, dispatching on Value.type.

Mirrors ArithValue in mlir/extras/dialects/arith.py: `+` picks add or fadd from
the operand type, Python scalars coerce to constants of the other operand's
type.
"""

from .. import Value, const_int, const_fp
from .context import current_builder


def _coerce(value, like):
    """Turn a Python int/float into a constant matching `like`'s type."""
    if isinstance(value, Value):
        return value
    ty = like.type
    if ty.is_floating_point:
        return const_fp(ty, float(value))
    if ty.is_integer:
        return const_int(ty, int(value), signed=True)
    raise TypeError(f"cannot coerce {value!r} to {ty}")


def _binary(method_int, method_float):
    def op(self, other):
        other = _coerce(other, self)
        b = current_builder()
        if self.type.is_floating_point:
            return getattr(b, method_float)(self, other)
        return getattr(b, method_int)(self, other)

    return op


def _rbinary(forward):
    def op(self, other):
        other = _coerce(other, self)
        return forward(other, self)

    return op


def install_value_dunders():
    Value.__add__ = _binary("add", "fadd")
    Value.__sub__ = _binary("sub", "fsub")
    Value.__mul__ = _binary("mul", "fmul")
    Value.__truediv__ = _binary("sdiv", "fdiv")
    Value.__radd__ = _rbinary(Value.__add__)
    Value.__rmul__ = _rbinary(Value.__mul__)
```

- [ ] **Step 5: Write `src/llvm/dsl/__init__.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
```

- [ ] **Step 6: Install the dunders in `src/llvm/__init__.py`**

Append after the `from .eudslllvm_ext import *` lines:

```python
from .dsl.values import install_value_dunders as _install_value_dunders

_install_value_dunders()
```

- [ ] **Step 7: Run the test**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests -v
```

Expected: all PASS. (No rebuild needed — Python-only change.)

- [ ] **Step 8: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add arithmetic dunders with int/float dispatch and coercion"
```

---

### Task 23: Comparison dunders to `icmp`/`fcmp`

**Files:**
- Modify: `src/llvm/dsl/values.py`
- Test: `tests/test_dsl_values.py` (append)

**Interfaces:**
- Consumes: `Value`, `llvm.ICmpPredicate`, `llvm.FCmpPredicate`, current builder.
- Produces: `Value.__lt__/__le__/__gt__/__ge__/__eq__/__ne__` emitting signed `icmp`/ordered `fcmp`. (`__eq__`/`__hash__` are already bound in C++ for identity; the DSL overrides `__lt__` etc. only, and provides `eq`/`ne` as named methods to avoid breaking hashing.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dsl_values.py`:

```python
def test_integer_comparison_signed():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb = _entry(ctx, mod, llvm.i1(ctx), [i32, i32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0) < fn.arg(1))
        assert "icmp slt i32" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_float_comparison_ordered():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.f32(ctx)
        fn, bb = _entry(ctx, mod, llvm.i1(ctx), [f32, f32])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            b.ret(fn.arg(0) > fn.arg(1))
        assert "fcmp ogt float" in str(mod)
        del b, fn, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_values.py -v -k comparison
```

Expected: FAIL — `<` currently raises `TypeError` (no `__lt__`) or returns default.

- [ ] **Step 3: Add comparison dunders to `src/llvm/dsl/values.py`**

Add imports `from .. import ICmpPredicate, FCmpPredicate` and this helper plus registrations inside `install_value_dunders`:

```python
def _cmp(icmp_pred, fcmp_pred):
    def op(self, other):
        other = _coerce(other, self)
        b = current_builder()
        if self.type.is_floating_point:
            return b.fcmp(getattr(FCmpPredicate, fcmp_pred), self, other)
        return b.icmp(getattr(ICmpPredicate, icmp_pred), self, other)

    return op
```

and inside `install_value_dunders`:

```python
    Value.__lt__ = _cmp("SLT", "OLT")
    Value.__le__ = _cmp("SLE", "OLE")
    Value.__gt__ = _cmp("SGT", "OGT")
    Value.__ge__ = _cmp("SGE", "OGE")
    # eq/ne stay as identity for hashing; expose value comparison by name.
    Value.eq = _cmp("EQ", "OEQ")
    Value.ne = _cmp("NE", "ONE")
```

Comparisons default to signed integer predicates, matching the spec (`a < b` → `icmp slt`). `__eq__`/`__ne__` are deliberately left as the C++ identity comparison so `Value` stays hashable and usable as a dict key in the traversal helpers; value-equality IR is emitted via `a.eq(b)`.

- [ ] **Step 4: Run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add comparison dunders lowering to signed icmp / ordered fcmp"
```

---

### Task 24: Pointer, GEP, load, store sugar including `__getitem__`/`__setitem__`

**Files:**
- Modify: `src/llvm/dsl/values.py`
- Test: `tests/test_dsl_values.py` (append)

**Interfaces:**
- Consumes: `Value`, current builder, `llvm.i32`, `llvm.const_int`.
- Produces: `Value.gep(result_element_type, *indices)`, `Value.load(result_type)`, `Value.store(value)`; the `@pointee(ty)` helper to record the element type for `__getitem__`/`__setitem__` on a pointer value (opaque pointers carry no element type, so the DSL requires it explicitly).

Because LLVM pointers are opaque, `ptr[i]` cannot infer the element type. The DSL therefore keys indexing off an explicitly attached element type: `p = with_element_type(ptr_value, i32); p[0]` loads an `i32`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dsl_values.py`:

```python
def test_gep_load_store_via_alloca():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        fn, bb = _entry(ctx, mod, i32, [])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            slot = b.alloca(i32, "slot")
            b.store(llvm.const_int(i32, 5), slot)
            loaded = b.load(i32, slot, "loaded")
            b.ret(loaded)
        printed = str(mod)
        assert "alloca i32" in printed
        assert "store i32 5" in printed
        assert "load i32" in printed
        del b, fn, mod
    assert_no_leaks()


def test_pointer_subscript_sugar():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        from llvm.dsl.values import with_element_type

        fn, bb = _entry(ctx, mod, i32, [llvm.ptr_t(ctx)])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            p = with_element_type(fn.arg(0), i32)
            v = p[2]  # gep + load
            b.ret(v)
        printed = str(mod)
        assert "getelementptr i32" in printed
        assert "load i32" in printed
        del b, fn, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_values.py -v -k "gep or subscript"
```

Expected: `ImportError: cannot import name 'with_element_type'`.

- [ ] **Step 3: Add pointer sugar to `src/llvm/dsl/values.py`**

```python
from .. import i32 as _i32_factory  # not used directly; kept for parity


class _TypedPointer:
    """A pointer value plus the element type needed for opaque-pointer GEP."""

    def __init__(self, ptr, element_type):
        self._ptr = ptr
        self._element_type = element_type

    def _index_const(self, i):
        ctx_ty = self._ptr.type  # PointerType; context reachable via any value
        # Build an i32 index constant from the context of the element type.
        return const_int(_i32_of(self._element_type), int(i))

    def __getitem__(self, i):
        b = current_builder()
        idx = self._index_const(i) if isinstance(i, int) else i
        gep = b.gep(self._element_type, self._ptr, [idx])
        return b.load(self._element_type, gep)

    def __setitem__(self, i, value):
        b = current_builder()
        idx = self._index_const(i) if isinstance(i, int) else i
        gep = b.gep(self._element_type, self._ptr, [idx])
        b.store(value, gep)


def _i32_of(any_type):
    # Element type's context provides i32; Type has no context accessor bound,
    # so route through the builder's insert block's parent module context.
    b = current_builder()
    ctx = b.insert_block.parent.args and None  # placeholder
    raise NotImplementedError
```

The `_i32_of` sketch is wrong: `Type` exposes no context, and index constants need a context. Fix by binding `Type.context` in C++ first. **Add to `src/IR/Types.cpp`** a `def_prop_ro("context", ...)` on `Type` returning the owning `eudsl::Context`? The `Type` only knows `LLVMContext&`, not the `eudsl::Context`. Simpler: index constants use a fixed 64-bit index type derived from the builder. **Bind `IRBuilder.i64_const(value)`** in C++ (Builder.cpp): `self.getInt64(v)` returns a `ConstantInt*`. Then the Python sugar calls `current_builder().i64_const(i)`. Add to `Builder.cpp`:

```cpp
      .def("i64_const",
           [](B &self, int64_t v) -> llvm::Value * { return self.getInt64(v); },
           "value"_a, nb::rv_policy::reference_internal)
      .def("i32_const",
           [](B &self, int32_t v) -> llvm::Value * { return self.getInt32(v); },
           "value"_a, nb::rv_policy::reference_internal)
```

Then rewrite the Python sugar cleanly:

```python
class _TypedPointer:
    def __init__(self, ptr, element_type):
        self._ptr = ptr
        self._element_type = element_type

    def _idx(self, i):
        if isinstance(i, int):
            return current_builder().i64_const(i)
        return i

    def __getitem__(self, i):
        b = current_builder()
        gep = b.gep(self._element_type, self._ptr, [self._idx(i)])
        return b.load(self._element_type, gep)

    def __setitem__(self, i, value):
        b = current_builder()
        gep = b.gep(self._element_type, self._ptr, [self._idx(i)])
        b.store(value, gep)


def with_element_type(ptr, element_type):
    return _TypedPointer(ptr, element_type)
```

Remove the broken `_i32_of`/`_TypedPointer._index_const` sketch entirely; the version above is the one to write. `IRBuilder::getInt64`/`getInt32` are standard IRBuilder helpers (confirm present; they are in `IRBuilder.h`).

- [ ] **Step 4: Rebuild (C++ changed) and run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add pointer/GEP/load/store sugar with explicit element types"
```

---

### Task 25: Aggregate construction and indexing

**Files:**
- Modify: `src/IR/Builder.cpp` (extract/insert value), `src/llvm/dsl/values.py`
- Test: `tests/test_dsl_values.py` (append)

**Interfaces:**
- Consumes: `IRBuilder`, `Value`.
- Produces: `IRBuilder.extract_value(agg, index, name="")`, `IRBuilder.insert_value(agg, value, index, name="")`; `Value.extract(index)` sugar for struct/array values.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dsl_values.py`:

```python
def test_extract_value_from_struct_arg():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        st = llvm.struct_t(ctx, [i32, i32])
        fn, bb = _entry(ctx, mod, i32, [st])
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb), building(b):
            first = b.extract_value(fn.arg(0), 0, "first")
            b.ret(first)
        assert "extractvalue { i32, i32 }" in str(mod)
        del b, fn, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_values.py -v -k extract
```

Expected: `AttributeError: 'IRBuilder' object has no attribute 'extract_value'`.

- [ ] **Step 3: Add extract/insert value to `src/IR/Builder.cpp`**

```cpp
      .def("extract_value",
           [](B &self, llvm::Value *agg, unsigned idx,
              const std::string &name) -> llvm::Value * {
             return self.CreateExtractValue(agg, {idx}, name);
           },
           "aggregate"_a, "index"_a, "name"_a = "",
           nb::rv_policy::reference_internal)
      .def("insert_value",
           [](B &self, llvm::Value *agg, llvm::Value *val, unsigned idx,
              const std::string &name) -> llvm::Value * {
             return self.CreateInsertValue(agg, val, {idx}, name);
           },
           "aggregate"_a, "value"_a, "index"_a, "name"_a = "",
           nb::rv_policy::reference_internal)
```

Add `Value.extract` sugar to `install_value_dunders` in `values.py`:

```python
    Value.extract = lambda self, index: current_builder().extract_value(self, index)
```

- [ ] **Step 4: Rebuild and run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add aggregate extract/insert value and Value.extract sugar"
```

---

## Phase B2 — control flow

The control-flow lowering reuses `eudsl-python-extras`' two-stage design: an AST
canonicalizer rewrites `if`/`while`/`for` into calls to context managers
(`if_ctx_manager`, `else_ctx_manager`, `while_`, `range_`) and `yield_`, then
those context managers emit real LLVM basic blocks and phi nodes at runtime.
The AST transformer classes in `mlir/extras/dialects/scf.py` are pure `ast`
rewrites — they rename `if` → `with if_ctx_manager(...)` and `yield` →
`yield_(...)` — so they port with only import-path edits. The *runtime* is
new: MLIR's region ops become explicit LLVM blocks + phis.

### Task 26: Vendor the AST canonicalizer and the CF transformers, with tests

**Files:**
- Create: `src/llvm/ast/__init__.py`, `src/llvm/ast/util.py`, `src/llvm/ast/py_type.py`, `src/llvm/ast/canonicalize.py`, `src/llvm/ast/cf_transformers.py`
- Test: `tests/test_ast.py`

**Interfaces:**
- Consumes: nothing (pure Python `ast`/`dis`/`opcode`).
- Produces: `llvm.ast.canonicalize.canonicalize(using=...)`, `Transformer`, `StrictTransformer`, `Canonicalizer`, `FunctionPatcher`; and the transformer classes `InsertEmptyYield`, `CanonicalizeElIfs`, `ReplaceYieldWithLLVMYield`, `ReplaceIfWithWith`, `CanonicalizeWhile` in `cf_transformers.py`.

- [ ] **Step 1: Vendor the pure-AST files**

Copy these from `$EUDSL/projects/eudsl-python-extras/mlir/extras/ast/` into `src/llvm/ast/`, with edits:

- `canonicalize.py` — copy verbatim; change `from ..ast.util import ...` to `from .util import get_module_cst, set_lineno, find_func_in_code_object`.
- `py_type.py` — copy verbatim (pure `ctypes`/`typing`, no MLIR imports).
- `util.py` — copy, then **strip the MLIR/cloudpickle parts**: delete `from cloudpickle import cloudpickle`, `from ...ir import Type`, `unpickle_mlir_type`, `MLIRTypePickler`, `copy_object`, and `copy_func` (the DSL does not copy closures). Keep `set_lineno`, `ast_call`, `get_module_cst`, `bind`, `get_localsplus_name_to_idx`, `_empty_cell_value`, `make_empty_cell`, `make_cell`, `append_hidden_node`, `find_func_in_code_object`.
- `__init__.py` — empty file with the license header.

- [ ] **Step 2: Vendor the CF transformers into `src/llvm/ast/cf_transformers.py`**

Copy the transformer classes from `mlir/extras/dialects/scf.py` (lines ~399–601: `is_yield_`, `is_yield`, `InsertEmptyYield`, `forward_yield_from_nested_if`, `CanonicalizeElIfs`, `CanonicalizeWhile`, `ReplaceYieldWithSCFYield`, `ReplaceIfWithWith`, `RemoveJumpsAndInsertGlobals`, `SCFCanonicalizer`) with these edits:

- Import from the vendored modules: `from .canonicalize import StrictTransformer, FunctionPatcher, Canonicalizer` and `from .util import ast_call, set_lineno, append_hidden_node`, plus `import ast`, `from copy import deepcopy`, `from typing import List, Union`.
- Rename `ReplaceYieldWithSCFYield` → `ReplaceYieldWithLLVMYield`; inside it, the call target changes from `yield_.__name__` to the string literal `"yield_"` (the runtime `yield_` lands in Task 27; use the literal name here so this file has no runtime dependency).
- In `ReplaceIfWithWith`, keep the references to `if_ctx_manager.__name__`, `else_ctx_manager.__name__`, `placeholder_opaque_t.__name__` as the string literals `"if_ctx_manager"`, `"else_ctx_manager"`, `"placeholder_opaque_t"`.
- In `CanonicalizeWhile`, keep the reference to `while__.__name__` as the literal `"while_"`.
- Delete `RemoveJumpsAndInsertGlobals` and `SCFCanonicalizer` from this file (the LLVM canonicalizer with its function patcher lands in Task 27, so it can inject the LLVM runtime globals). Keep only the transformer classes and the `is_yield`/`is_yield_`/`forward_yield_from_nested_if` helpers.

- [ ] **Step 3: Write the failing test**

`tests/test_ast.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ast

from llvm.ast import cf_transformers as T
from llvm.ast.canonicalize import transform_func


def _rewrite(src):
    tree = ast.parse(src)

    class C:
        cst_transformers = [
            T.CanonicalizeElIfs,
            T.InsertEmptyYield,
            T.ReplaceYieldWithLLVMYield,
            T.ReplaceIfWithWith,
        ]

    # Apply each transformer to the function node and return the unparsed body.
    node = tree.body[0]
    for ctor in C.cst_transformers:
        node = ctor(context=None, first_lineno=0).generic_visit(node)
    return ast.unparse(node)


def test_if_becomes_with_if_ctx_manager():
    src = (
        "def f():\n"
        "    if c:\n"
        "        x = a\n"
        "    else:\n"
        "        x = b\n"
    )
    out = _rewrite(src)
    assert "if_ctx_manager" in out
    assert "else_ctx_manager" in out
    assert "yield_" in out


def test_no_else_still_yields():
    src = "def f():\n    if c:\n        g()\n"
    out = _rewrite(src)
    assert "if_ctx_manager" in out
    assert "yield_" in out
```

- [ ] **Step 4: Run the test**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_ast.py -v
```

Expected: PASS after the vendored files import cleanly. If an import fails, it points at a leftover MLIR import to strip.

- [ ] **Step 5: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Vendor the AST canonicalizer and CF transformers"
```

---

### Task 27: `@function` decorator and `if`/`else` lowering to blocks and phi nodes

**Files:**
- Create: `src/llvm/dsl/cf.py`, `src/llvm/dsl/func.py`
- Modify: `src/llvm/__init__.py` (export `function`, `yield_`)
- Test: `tests/test_dsl_cf.py`

**Interfaces:**
- Consumes: `IRBuilder`, `Function`, `BasicBlock`, current builder/function, the CF transformers.
- Produces: `llvm.function(*, module, name=None)` decorator; the runtime `if_ctx_manager(cond, results)`, `else_ctx_manager(if_op)`, `yield_(*values)`, `placeholder_opaque_t()`; `llvm.LLVMCanonicalizer`.

- [ ] **Step 1: Write the failing test**

`tests/test_dsl_cf.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_if_else_produces_phi():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)

        @llvm.function(module=mod)
        def pick(c: llvm.i1, a: i32, b: i32) -> i32:
            if c:
                x = a + 1
            else:
                x = b
            return x

        printed = str(mod)
        assert "br i1" in printed
        assert "phi i32" in printed
        assert "add i32" in printed
        del mod
    assert_no_leaks()


def test_if_else_jits_correctly():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def pick(c: llvm.i1, a: i32, b: i32) -> i32:
        if c:
            x = a
        else:
            x = b
        return x

    import ctypes

    jit = llvm.LLJIT()
    jit.add_module(mod)
    addr = jit.lookup("pick")
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(addr)
    assert fn(True, 10, 20) == 10
    assert fn(False, 10, 20) == 20
    del jit, mod, ctx
    assert_no_leaks()
```

Note the annotations use type *factories* that need a context. Resolve annotations against the decorator's module context: an annotation like `i32` here is `llvm.i32(ctx)` — but the test writes `i32 = llvm.i32(ctx)` as a local and annotates with it, so the annotation is already a `Type`. The decorator accepts annotations that are either a `Type` instance or a callable `ctx -> Type`; see Step 3. `llvm.i1` is a factory (callable), so the decorator calls `llvm.i1(module.context)`.

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_cf.py -v
```

Expected: `AttributeError: module 'llvm' has no attribute 'function'`.

- [ ] **Step 3: Write `src/llvm/dsl/func.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The @function decorator: turns a Python function into an LLVM function."""

import inspect

from .. import Function, IRBuilder, Type, function_t
from .context import building
from .cf import LLVMCanonicalizer
from ..ast.canonicalize import canonicalize


def _resolve(annotation, ctx):
    if isinstance(annotation, Type):
        return annotation
    if callable(annotation):
        return annotation(ctx)
    raise TypeError(f"cannot resolve type annotation {annotation!r}")


def function(*, module, name=None):
    def decorator(f):
        ctx = module.context
        sig = inspect.signature(f)
        param_types = [
            _resolve(p.annotation, ctx) for p in sig.parameters.values()
        ]
        ret_type = _resolve(sig.return_annotation, ctx)
        fn_name = name or f.__name__

        fn = Function.create(function_t(ret_type, param_types), fn_name, module)
        entry = fn.append_basic_block("entry")
        builder = IRBuilder(ctx)

        # Rewrite Python control flow into the cf context-manager calls.
        f = canonicalize(using=LLVMCanonicalizer())(f)

        with builder.at_end_of(entry), building(builder, fn):
            args = [fn.arg(i) for i in range(len(param_types))]
            result = f(*args)
            if result is not None:
                builder.ret(result)
            elif entry.terminator is None:
                builder.ret(None)
        return fn

    return decorator
```

The `@function` decorator resolves annotations, creates the function and entry block, canonicalizes the body (so `if` becomes `with if_ctx_manager(...)`), then runs it under the current builder/function. A returned value becomes `ret`; a fallthrough with no terminator becomes `ret void`.

- [ ] **Step 4: Write `src/llvm/dsl/cf.py`**

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runtime for lowered control flow: real basic blocks and phi nodes.

The AST canonicalizer (cf_transformers) rewrites `if`/`else` into
`with if_ctx_manager(cond, results): ... yield_(x)` / `with else_ctx_manager(op): ... yield_(y)`.
These context managers create then/else/merge blocks, and the values passed to
yield_ become phi nodes at the merge block — the explicit-block analogue of
scf.if's region results.
"""

from contextlib import contextmanager

from .context import current_builder, current_function
from ..ast.canonicalize import Canonicalizer, FunctionPatcher
from ..ast import cf_transformers as _T


def placeholder_opaque_t():
    # Marks a phi-result slot in the rewritten AST; the real value is the phi.
    return None


class _IfOp:
    """Bookkeeping for one lowered if/else."""

    def __init__(self, cond):
        b = current_builder()
        fn = current_function()
        self.builder = b
        self.then_block = fn.append_basic_block("if.then")
        self.merge_block = fn.append_basic_block("if.end")
        self.else_block = None
        self.entry_block = b.insert_block
        self.cond = cond
        self.then_yields = []
        self.else_yields = []
        self._active = "then"

    def record_yield(self, values):
        if self._active == "then":
            self.then_yields = list(values)
        else:
            self.else_yields = list(values)


# Stack of active if-ops so yield_ knows where to record.
_if_stack = []


@contextmanager
def if_ctx_manager(cond, results=()):
    op = _IfOp(cond)
    _if_stack.append(op)
    b = op.builder
    # If there is no else, both edges of the branch will be filled after the
    # body runs; default the false edge to merge.
    b.set_insert_point(op.then_block)
    try:
        yield op
    finally:
        # Terminate the then block into merge if the body did not.
        if op.then_block.terminator is None:
            b.br(op.merge_block)
        _if_stack.pop()
        _finish_if(op, results)


@contextmanager
def else_ctx_manager(op):
    b = op.builder
    op.else_block = current_function().append_basic_block("if.else")
    op._active = "else"
    _if_stack.append(op)
    b.set_insert_point(op.else_block)
    try:
        yield op
    finally:
        if op.else_block.terminator is None:
            b.br(op.merge_block)
        _if_stack.pop()
        _finish_if(op, ())  # phis already created on then-finish; see note


def _finish_if(op, results):
    b = op.builder
    # Emit the conditional branch from the entry block now that both targets
    # exist (else_block may still be None -> false edge is merge).
    b.set_insert_point(op.entry_block)
    false_dest = op.else_block if op.else_block is not None else op.merge_block
    if op.entry_block.terminator is None:
        b.cond_br(op.cond, op.then_block, false_dest)
    # Build phis at the merge block for each yielded value.
    b.set_insert_point(op.merge_block)
    op.phis = []
    for i, then_val in enumerate(op.then_yields):
        phi = b.phi(then_val.type, f"if.phi.{i}")
        phi.add_incoming(then_val, op.then_block)
        if op.else_yields:
            phi.add_incoming(op.else_yields[i], op.else_block)
        op.phis.append(phi)


def yield_(*values):
    if not _if_stack:
        return values[0] if len(values) == 1 else values
    op = _if_stack[-1]
    op.record_yield(values)
    # The rewritten AST assigns `x = yield_(...)`; return the phi placeholders
    # once both branches are known. Because phis are built on if-finish, expose
    # them lazily through the merge-block read below.
    return None


class _InjectCFGlobals(FunctionPatcher):
    def patch_function(self, f):
        f.__globals__["yield_"] = yield_
        f.__globals__["if_ctx_manager"] = if_ctx_manager
        f.__globals__["else_ctx_manager"] = else_ctx_manager
        f.__globals__["placeholder_opaque_t"] = placeholder_opaque_t
        return f


class LLVMCanonicalizer(Canonicalizer):
    cst_transformers = [
        _T.CanonicalizeElIfs,
        _T.InsertEmptyYield,
        _T.ReplaceYieldWithLLVMYield,
        _T.ReplaceIfWithWith,
        _T.CanonicalizeWhile,
    ]
    function_patchers = [_InjectCFGlobals]
```

**Known gap to close during implementation:** the `yield_`/phi wiring above records yielded values but the rewritten AST assigns the *result* of the `if` (the phi) to `x`. Making `x` refer to the phi requires the phi to be the value returned to the assignment target after the `with` block completes — which the scf design achieves by having `if_ctx_manager` yield an op whose `.results` are read. Port that exactly: `ReplaceIfWithWith` assigns `(x,) = placeholder_opaque_t()` slots, and the `with if_ctx_manager(...) as __if_op__:` binds the op; the transformer's forwarding assignment reads `__if_op__.results`. Implement `_IfOp.results` to return `op.phis` and adjust the transformer port so the post-if read binds `x = __if_op__.results[0]`. Validate against `test_if_else_produces_phi` first (IR shape), then `test_if_else_jits_correctly` (execution) — the execution test is the real proof the phi is wired to `x`.

- [ ] **Step 5: Export `function` and `yield_` in `src/llvm/__init__.py`**

```python
from .dsl.func import function
from .dsl.cf import yield_
```

- [ ] **Step 6: Run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_cf.py -v
```

Expected: `test_if_else_produces_phi` and `test_if_else_jits_correctly` PASS. Iterate on the phi-to-assignment wiring (the known gap) using systematic-debugging until the execution test is green.

- [ ] **Step 7: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add @function and if/else lowering to blocks and phi nodes"
```

---

### Task 28: `elif` canonicalization

**Files:**
- Test: `tests/test_dsl_cf.py` (append)
- Modify: only if the vendored `CanonicalizeElIfs` needs an LLVM-specific fix.

**Interfaces:**
- Consumes: the CF transformers and runtime from Task 27.
- Produces: no new API; `elif` chains lower correctly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dsl_cf.py`:

```python
def test_elif_chain_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def classify(x: i32) -> i32:
        if x < 0:
            r = llvm.const_int(i32, -1)
        elif x == 0:
            r = llvm.const_int(i32, 0)
        else:
            r = llvm.const_int(i32, 1)
        return r

    import ctypes

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("classify"))
    assert fn(-5) == -1
    assert fn(0) == 0
    assert fn(7) == 1
    del jit, mod, ctx
    assert_no_leaks()
```

Note `x == 0` uses the C++ identity `__eq__`, which is wrong for value comparison. The DSL body must use `x.eq(0)` OR Task 28 must decide `__eq__` semantics inside DSL bodies. Since the canonicalizer runs on the source, rewrite the test to use `x.eq(llvm.const_int(i32, 0))` — the spec keeps `__eq__` as identity (Task 23). Update the test body to `elif x.eq(llvm.const_int(i32, 0)):` before running.

- [ ] **Step 2: Run the test to confirm it fails or passes**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_cf.py::test_elif_chain_jits -v
```

Expected: FAIL if `CanonicalizeElIfs` forwarding is not yet correct for the LLVM runtime; PASS if Task 27's port already handled nested-if forwarding.

- [ ] **Step 3: Fix `CanonicalizeElIfs` forwarding if needed**

`elif` is `else: if ...`. `CanonicalizeElIfs.forward_yield_from_nested_if` forwards the inner if's yielded name outward. If the execution test shows a missing phi incoming from the elif branch, the fix is in the vendored `forward_yield_from_nested_if` in `cf_transformers.py` — ensure the forwarded `yield_` targets the outer merge. Debug with systematic-debugging; the IR (`str(mod)`) shows which block lacks a phi incoming.

- [ ] **Step 4: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Support elif chains in DSL control flow"
```

---

### Task 29: `while` loops

**Files:**
- Modify: `src/llvm/dsl/cf.py` (add `while_` runtime)
- Test: `tests/test_dsl_cf.py` (append)

**Interfaces:**
- Consumes: current builder/function, `CanonicalizeWhile` from Task 26.
- Produces: `llvm.dsl.cf.while_(cond_value)` runtime and its `next()` protocol, wired into the injected globals.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dsl_cf.py`:

```python
def test_while_countdown_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def sum_to(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        i = llvm.const_int(i32, 0)
        while i.ne(n):
            acc = acc + i
            i = i + 1
            yield_(acc, i)
        return acc

    import ctypes

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("sum_to"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    del jit, mod, ctx
    assert_no_leaks()
```

`while` with loop-carried values (`acc`, `i`) requires phis at the loop header. This is the hardest DSL case; the `yield_(acc, i)` marks the loop-carried set.

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_cf.py::test_while_countdown_jits -v
```

Expected: `NameError: name 'while_' is not defined` or a lowering failure.

- [ ] **Step 3: Add the `while_` runtime to `src/llvm/dsl/cf.py`**

Implement a header/body/exit block structure with phis for loop-carried values:

```python
class _WhileOp:
    def __init__(self):
        b = current_builder()
        fn = current_function()
        self.builder = b
        self.preheader = b.insert_block
        self.header = fn.append_basic_block("while.header")
        self.body = fn.append_basic_block("while.body")
        self.exit = fn.append_basic_block("while.end")
        self.header_phis = None
        self.incoming_from_preheader = None
        self.first_pass = True

    # The while_ protocol: `next(while_(cond), False)` in the rewritten AST
    # drives one structural pass to place blocks and phis, then the real branch.


def while_(cond):
    # Returns an iterator whose first next() sets up the loop and returns the
    # header condition; mirrors CanonicalizeWhile's `next(w, False)` rewrite.
    ...
```

Port `CanonicalizeWhile`'s runtime contract from `scf.py`'s `while__`/`while___` (lines ~246–277) to LLVM blocks: the loop-carried values passed to `yield_` become header phis with incoming edges from the preheader (initial values) and the body (updated values). Because this is the most intricate lowering, implement it incrementally: first get a `while` with **no** loop-carried values (a side-effecting loop) passing, then add the phi wiring for carried values, validating each with the JIT execution test.

**Accepted risk flagged for the user:** loop-carried phi wiring in `while_` is the single most complex piece of this plan. If it does not converge within the task, land the no-carried-value form (green, useful) and split the carried-value form into a follow-up task rather than blocking the phase.

- [ ] **Step 4: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add while-loop lowering with loop-carried phi nodes"
```

---

### Task 30: `for` / `range_` with loop-carried values

**Files:**
- Modify: `src/llvm/dsl/cf.py` (add `range_` and `for_` runtime)
- Test: `tests/test_dsl_cf.py` (append)

**Interfaces:**
- Consumes: the `while_` machinery from Task 29.
- Produces: `llvm.dsl.cf.range_(start, stop, step=1)` and the `for_` runtime, wired into the injected globals; `range_` lowered onto the `while_` block structure with an induction-variable phi.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dsl_cf.py`:

```python
def test_for_range_sum_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def total(n: i32) -> i32:
        acc = llvm.const_int(i32, 0)
        for i in range_(0, n):
            acc = acc + i
            yield_(acc)
        return acc

    import ctypes

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("total"))
    assert fn(5) == 0 + 1 + 2 + 3 + 4
    del jit, mod, ctx
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_cf.py::test_for_range_sum_jits -v
```

Expected: `NameError: name 'range_' is not defined`.

- [ ] **Step 3: Implement `range_`/`for_` in `src/llvm/dsl/cf.py`**

`range_(start, stop, step)` builds the same header/body/exit structure as `while_`, with an induction phi `i` starting at `start`, a header comparison `i < stop` (via `b.icmp(ICmpPredicate.SLT, ...)`), and a body-end increment `i + step` feeding back into the header phi. Loop-carried values from `yield_` get additional header phis, exactly as in Task 29. Reuse `_WhileOp`'s phi-wiring helper. Add `range_` and `for_` to `_InjectCFGlobals`. `InsertEmptyYield.visit_For` (already vendored) inserts the empty yield when the body lacks one; confirm the `"range_"`/`"for_"` name check in the vendored transformer matches the injected names.

- [ ] **Step 4: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add for/range_ lowering with an induction-variable phi"
```

---

### Task 31: `break` / `continue` / early `return` detection raising `NotImplementedError`

**Files:**
- Modify: `src/llvm/ast/cf_transformers.py` (add a detector transformer)
- Test: `tests/test_dsl_cf.py` (append)

**Interfaces:**
- Consumes: the transformer base classes.
- Produces: `RejectUnsupportedJumps` transformer in `cf_transformers.py`, added to `LLVMCanonicalizer.cst_transformers` before the others, raising `NotImplementedError` with a clear message on `ast.Break`, `ast.Continue`, or a `return` inside an `if`/`while`/`for`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dsl_cf.py`:

```python
import pytest


def test_break_raises_not_implemented():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        with pytest.raises(NotImplementedError, match="break"):

            @llvm.function(module=mod)
            def bad(n: i32) -> i32:
                for i in range_(0, n):
                    if i.eq(llvm.const_int(i32, 3)):
                        break
                    yield_()
                return n

        del mod


def test_early_return_in_if_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        with pytest.raises(NotImplementedError, match="return"):

            @llvm.function(module=mod)
            def bad2(c: llvm.i1, a: i32) -> i32:
                if c:
                    return a
                return a

        del mod
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_cf.py -v -k "break or early_return"
```

Expected: FAIL — currently `break` reaches the lowering and produces wrong IR or a different error.

- [ ] **Step 3: Add `RejectUnsupportedJumps` to `src/llvm/ast/cf_transformers.py`**

```python
class RejectUnsupportedJumps(StrictTransformer):
    """Reject control-flow the phi-based lowering does not model.

    break/continue and early return inside if/while/for would need edge
    duplication and predecessor bookkeeping the yield-protocol lowering does
    not do. Detect and refuse rather than emit wrong IR.
    """

    def visit_Break(self, node):
        raise NotImplementedError(
            "`break` inside DSL control flow is not supported"
        )

    def visit_Continue(self, node):
        raise NotImplementedError(
            "`continue` inside DSL control flow is not supported"
        )

    def _visit_loop_or_if(self, node):
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                raise NotImplementedError(
                    "early `return` inside DSL control flow is not supported"
                )
        return self.generic_visit(node)

    visit_If = _visit_loop_or_if
    visit_While = _visit_loop_or_if
    visit_For = _visit_loop_or_if
```

Add `RejectUnsupportedJumps` as the **first** entry in `LLVMCanonicalizer.cst_transformers` in `cf.py`, so it runs before `ReplaceIfWithWith` rewrites the `if` away. Note: `visit_If`/`visit_While`/`visit_For` scanning for a nested `ast.Return` would also flag the *trailing* return of the function body if that return is syntactically inside the top-level construct; guard by only rejecting a `Return` that is a descendant of an `if`/loop that is itself nested (i.e. the function's final `return` is a sibling of the constructs, not inside them, so `ast.walk` over the `if` node will not reach it). Verify with `test_if_else_jits_correctly` from Task 27 still passing (its `return x` is after the `if`, not inside it).

- [ ] **Step 4: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Reject break/continue/early return in DSL control flow"
```

---

## Phase B3 — program structure

### Task 32: `@function` rework — declarations, calls via `__call__`, multiple returns

**Files:**
- Modify: `src/llvm/dsl/func.py`
- Test: `tests/test_dsl_func.py`

**Interfaces:**
- Consumes: `@function` from Task 27, `IRBuilder.call`.
- Produces: `@function` supporting an empty body (declaration, no entry block); the decorated object is callable (`f(args...)` emits a `call` in the current builder); multiple `return` statements at the top level of the body.

- [ ] **Step 1: Write the failing tests**

`tests/test_dsl_func.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes

import llvm
from llvm.testing import assert_no_leaks


def test_declaration_has_no_body():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)

        @llvm.function(module=mod)
        def extern(a: i32) -> i32: ...

        printed = str(mod)
        assert "declare i32 @extern(i32)" in printed
        del mod
    assert_no_leaks()


def test_call_between_functions_jits():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def inc(x: i32) -> i32:
        return x + 1

    @llvm.function(module=mod)
    def inc2(x: i32) -> i32:
        return inc(inc(x))

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(jit.lookup("inc2"))
    assert fn(40) == 42
    del jit, mod, ctx
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_func.py -v
```

Expected: `test_declaration_has_no_body` fails (a body-less function still gets an entry block and `ret`), and `inc(...)` is not callable.

- [ ] **Step 3: Rework `src/llvm/dsl/func.py`**

Return a small `DSLFunction` wrapper instead of the bare `Function`, and detect the empty-body (`...`/`pass`-only) case:

```python
class DSLFunction:
    def __init__(self, fn, module):
        self.fn = fn
        self.module = module

    def __call__(self, *args):
        # Emit a call in the current builder; used from inside other @functions.
        from .context import current_builder
        return current_builder().call(self.fn, list(args))
```

In `decorator`, after resolving types and creating `fn`:

```python
        body = f.__code__.co_consts  # cheap check below is via source
        is_declaration = _body_is_empty(f)
        if is_declaration:
            return DSLFunction(fn, module)  # no entry block, stays `declare`

        entry = fn.append_basic_block("entry")
        ...  # as Task 27
        return DSLFunction(fn, module)
```

Add `_body_is_empty(f)` using `inspect.getsource` + `ast` to detect a body that is only `...`, `pass`, or a docstring. Multiple top-level `return`s already work because each `return` in the (non-canonicalized top-level) body sets the terminator; the last one wins for a straight-line body. For the `inc2` case, `inc(...)` returns a call `Value`, and `return inc(inc(x))` becomes `builder.ret(call)`.

Update `src/llvm/__init__.py` if `function` now lives behind `DSLFunction` — the export stays `from .dsl.func import function`.

- [ ] **Step 4: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Rework @function: declarations, calls via __call__, returns"
```

---

### Task 33: Function attributes, linkage, calling convention, varargs on `@function`

**Files:**
- Modify: `src/llvm/dsl/func.py`
- Test: `tests/test_dsl_func.py` (append)

**Interfaces:**
- Consumes: `@function`, `Function.linkage`/`.calling_conv`/`.add_fn_attr` (Task 13).
- Produces: `@function(module=..., linkage=..., calling_conv=..., attrs={...}, var_arg=False)` keyword options.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dsl_func.py`:

```python
def test_function_options():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)

        @llvm.function(
            module=mod,
            linkage=llvm.Linkage.INTERNAL,
            attrs={"target-cpu": "znver3"},
        )
        def f(x: i32) -> i32:
            return x

        printed = str(mod)
        assert "define internal i32 @f" in printed
        assert 'target-cpu"="znver3' in printed
        del mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_dsl_func.py::test_function_options -v
```

Expected: `TypeError: function() got an unexpected keyword argument 'linkage'`.

- [ ] **Step 3: Add the options to `src/llvm/dsl/func.py`**

Extend `function(*, module, name=None, linkage=None, calling_conv=None, attrs=None, var_arg=False)`. Pass `var_arg` into `function_t(ret_type, param_types, var_arg=var_arg)`. After creating `fn`, apply:

```python
        if linkage is not None:
            fn.linkage = linkage
        if calling_conv is not None:
            fn.calling_conv = calling_conv
        for k, v in (attrs or {}).items():
            fn.add_fn_attr(k, v)
```

- [ ] **Step 4: Commit**

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Add linkage, calling convention, attrs, varargs to @function"
```

---

### Task 34: Globals, constant initializers, address spaces

**Files:**
- Modify: `src/IR/Constants.cpp` (GlobalVariable construction), `src/IR/Context.cpp` (module global accessors)
- Test: `tests/test_globals.py`

**Interfaces:**
- Consumes: `Module`, `Type`, `Constant`, `GlobalVariable` (Task 11).
- Produces: `llvm.Module.add_global(type, name, initializer=None, constant=False, address_space=0) -> GlobalVariable`, `llvm.Module.get_global(name) -> GlobalVariable | None`, `llvm.Module.globals -> list[GlobalVariable]`.

- [ ] **Step 1: Write the failing test**

`tests/test_globals.py`:

```python
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_add_global_with_initializer():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        g = mod.add_global(i32, "counter", llvm.const_int(i32, 7))
        assert type(g).__name__ == "GlobalVariable"
        assert g.name == "counter"
        assert "@counter = global i32 7" in str(mod)
        assert mod.get_global("counter") == g
        assert [x.name for x in mod.globals] == ["counter"]
        del g, mod
    assert_no_leaks()


def test_constant_global_in_address_space():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)
        g = mod.add_global(
            i32, "ro", llvm.const_int(i32, 1), constant=True, address_space=1
        )
        printed = str(mod)
        assert "addrspace(1)" in printed
        assert "constant i32 1" in printed
        del g, mod
    assert_no_leaks()
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pytest tests/test_globals.py -v
```

Expected: `AttributeError: 'Module' object has no attribute 'add_global'`.

- [ ] **Step 3: Add global accessors to `src/IR/Context.cpp`**

Add `#include <llvm/IR/GlobalVariable.h>` and to the `Module` class:

```cpp
      .def(
          "add_global",
          [](eudsl::Module &self, llvm::Type *ty, const std::string &name,
             llvm::Constant *init, bool isConstant,
             unsigned addressSpace) -> llvm::GlobalVariable * {
            auto *gv = new llvm::GlobalVariable(
                self.get(), ty, isConstant,
                llvm::GlobalValue::ExternalLinkage, init, name, nullptr,
                llvm::GlobalValue::NotThreadLocal, addressSpace);
            return gv;
          },
          "type"_a, "name"_a, "init"_a = nullptr, "constant"_a = false,
          "address_space"_a = 0, nb::rv_policy::reference_internal)
      .def(
          "get_global",
          [](eudsl::Module &self, const std::string &name) {
            return self.get().getNamedGlobal(name);
          },
          "name"_a, nb::rv_policy::reference_internal)
      .def_prop_ro("globals", [](eudsl::Module &self) {
        std::vector<llvm::GlobalVariable *> out;
        for (llvm::GlobalVariable &g : self.get().globals())
          out.push_back(&g);
        return out;
      })
```

The `GlobalVariable` constructor `(Module&, Type*, bool isConstant, LinkageTypes, Constant* Initializer, Twine Name, GlobalVariable* InsertBefore, ThreadLocalMode, unsigned AddressSpace)` inserts into the module. Confirm the exact parameter order against `GlobalVariable.h`; adjust if the signature differs.

- [ ] **Step 4: Rebuild and run the tests**

```bash
cd $EUDSL/projects/eudsl-llvmpy && $PY -m pip install -e . --no-build-isolation -v \
  && $PY -m pytest tests -v
```

Expected: all PASS.

- [ ] **Step 5: Commit, then submit the stack to open PR 2 (draft)**

This is the last task of Phase B. Commit it, then submit the stack to push the
Phase B branch, open PR 2 as a draft, and link it above PR 1 (Tasks 22–34). The
Phase B branch was added with `gh stack add` before Task 22 (see **Delivery**).

```bash
cd $EUDSL && git add -A projects/eudsl-llvmpy \
  && git commit -m "[eudsl-llvmpy] Bind module globals with initializers and address spaces"
# Push the Phase B branch, open PR 2 as a draft, and link the stack.
gh stack submit --auto
```

---

## Final verification

After Task 34, run the whole suite from a clean build and confirm every tier
from the spec's testing section is exercised:

```bash
cd $EUDSL/projects/eudsl-llvmpy
rm -rf build src/llvm/*.so
$PY -m pip install -e . --no-build-isolation -v
$PY -m pytest tests -v
```

Expected: every test passes. The three testing tiers are covered as:

1. **pytest units per module** — each `test_*.py` ends with `assert_no_leaks()` (`gc.collect()` + `Context._get_live_count() == 0`), matching `mlir/test/python/ir/*.py`.
2. **IR-text assertions** — `str(mod)` / `str(value)` substring checks throughout. (The vendored `filecheck_with_comments` fixture from `mlir/extras/testing/testing.py` can be added to `src/llvm/testing.py` if FileCheck-style checks are wanted; the substring assertions cover the same ground and need no external tool.)
3. **execution tests** — `test_jit.py` and the `*_jits` tests in `test_dsl_cf.py`/`test_dsl_func.py` JIT the function via `LLJIT`, call it through `ctypes`, and assert the numeric result. Host-target-only means these run on developer machines and CI.

CI (`.github/workflows/build_test_release_eudsl.yml`) already builds and tests
`projects/eudsl-llvmpy`; the deletion of the `eudsl-tblgen` build dependency in
Task 1 means the "Build eudsl-llvmpy" step no longer needs the `eudsl-tblgen`
wheel installed first, but leaving that install in place is harmless.

## Self-review notes

- **Spec coverage.** Every numbered work-breakdown item (1–34) maps to the
  task of the same number. The four spec decisions about test disposition
  (Task 1) are handled: `test_smoke` ported, `test_symbol_collision` ported,
  `test_from_capsule` deleted, `test_builder` deleted and reintroduced without
  `amdgcn` at Task 21/27.
- **Two flagged risks** carried from the spec, not solved by it: (a) the
  `report_fatal_error` handler cannot return, so bad data-layout/target
  strings still abort — Task 15 reports before abort but cannot recover; (b)
  loop-carried phi wiring in `while_`/`range_` (Tasks 29–30) is the highest-risk
  lowering, with an explicit fallback (land the no-carried-value form) stated
  in Task 29.
- **Deferred items** (`break`/`continue`/early `return`, DebugInfo, AMDGPU/NVPTX
  default-off, `.pyi` stubs, llvmlite shim) are out of scope per the Global
  Constraints and are not given tasks, except Task 31 which makes the first set
  raise `NotImplementedError`.
