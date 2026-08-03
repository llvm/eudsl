# Manual Test Plan: mlir-python-mcp

These tests verify the MCP server works end-to-end when connected to Claude Code.

## Prerequisites

1. Install: `pip install -e ".[test]" -f https://llvm.github.io/eudsl`
2. Add to `.mcp.json` in your project root:
   ```json
   {
     "mcpServers": {
       "mlir-python-mcp": {
         "command": "mlir-python-mcp"
       }
     }
   }
   ```
3. Start a new Claude Code session (MCP servers connect at session init)

`.mcp.json` must exist before the session starts. Adding it to a running session,
or checking out a branch that carries it, has no effect until you restart: the
server list is built once at init. `/mcp` will not show the server otherwise.

These tests target the MCP protocol surface. The pytest suite calls
`MLIRMCPServer._dispatch` directly, and the transport layer above it
(`handle_list_tools`, `handle_call_tool`, `run`, `main`) is marked
`# pragma: no cover`, so it only runs when a real client connects.

## Test 1: Parse MLIR text

Call `parse_mlir` with:
```
src: |
  func.func @add(%a: i32, %b: i32) -> i32 {
    %c = arith.addi %a, %b : i32
    return %c : i32
  }
```

Expected: "Parsed into 'module'. Top-level operations: 1"

## Test 2: Run pass pipeline

Call `run_pipeline` with:
```
mlir: |
  func.func @f() {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 1 : i32
    return
  }
pipeline: canonicalize
```

Expected: Output IR has dead constants removed (only `return` remains in body).

## Test 3: Programmatic IR construction

Call `execute_python` with:
```python
from mlir.dialects import arith, func

m = new_module()
with ir.InsertionPoint(m.body):
    i32 = ir.IntegerType.get_signless(32)
    f_type = ir.FunctionType.get([i32, i32], [i32])
    f = func.FuncOp("my_add", f_type)
    entry = f.add_entry_block()
    with ir.InsertionPoint(entry):
        result = arith.AddIOp(entry.arguments[0], entry.arguments[1])
        func.ReturnOp([result])
print(get_module_asm(m))
```

Expected: Valid `func.func @my_add(%arg0: i32, %arg1: i32) -> i32` with `arith.addi`.

## Test 4: Pipeline fluent builder

Call `execute_python` with:
```python
p = Pipeline().canonicalize().cse()
print(str(p))
```

Expected: `builtin.module(canonicalize,cse)`

## Test 5: Discovery via help()

Call `execute_python` with:
```python
from mlir.dialects import arith

import inspect
print(inspect.signature(arith.AddIOp.__init__))
```

Expected: Shows `(self, lhs: ..., rhs: ..., *, overflowFlags=None, results=None, loc=None, ip=None)`

## Test 6: List passes with filter

Call `list_passes` with:
```
filter: linalg
```

Expected: Lists passes containing "linalg" (e.g. `convert_linalg_to_loops`, `linalg_generalize_named_ops`).

## Test 7: Verify module

Call `verify_module` after Test 1.

Expected: "Verification passed."

## Test 8: Pipeline object with run_pipeline

Call `execute_python` with:
```python
m = new_module("func.func @f() { %c = arith.constant 1 : i32\n return }")
m = run_pipeline(m, Pipeline().canonicalize())
print(get_module_asm(m))
```

Expected: Dead constant removed, only `return` in body.

## Test 9: Tool descriptions match behavior

Read the `execute_python` description as the client presents it, then run the
dialect idiom it advertises. Repeat for any description that shows example code.

```python
from mlir.dialects import arith, func

print(arith.AddIOp, func.FuncOp)
```

Expected: No `NameError` or `AttributeError`. A description that names an
identifier the namespace does not provide is a bug in the description, since it
is the only API reference the model gets. This test exists because the
description once advertised `dialects.arith`, which raises `AttributeError`:
`session.py` imports the `mlir.dialects` package without importing any
submodule.

## Test 10: History, chaining, and rewind

Call `run_pipeline` with:
```
mlir: |
  func.func @f(%a: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %s = arith.addi %a, %c0 : i32
    %dead = arith.constant 7 : i32
    return %s : i32
  }
pipeline: canonicalize
```

Expected: Body reduces to `return %arg0 : i32`.

Then `chain_pipeline` with `pipeline: cse`, then `history`.

Expected:
```
History (3 entries):
  [0] initial
  [1] builtin.module(canonicalize)
  [2] builtin.module(cse)
```

Then `rewind` with `steps: 2`.

Expected: `Rewound to step 'initial'.` followed by the pre-canonicalize IR, with
`arith.constant 0`, `arith.addi`, and the dead `arith.constant 7` all back.
Confirm with `get_current_ir` that the rewound state is what later calls see.

Note that history tracks only `run_pipeline` and `chain_pipeline`. Mutations made
through `execute_python` are invisible to it and are not undone by `rewind`.

## Test 11: Navigate and mutate an existing module

Call `parse_mlir` with `var_name: mut` and:
```
func.func @g(%a: i32) -> i32 {
  %x = arith.addi %a, %a : i32
  %y = arith.muli %x, %a : i32
  return %y : i32
}
```

Then walk the tree with `walk_operations`, `var_name: mut`, `filter: arith`.

Expected: 2 operations, `arith.addi` and `arith.muli`, each reporting
`operands=2, results=1, regions=0` and an `overflowFlags` attribute.

Then `symbol_lookup` with `symbol_name: g`, `var_name: mut`, `result_var: g_op`,
and `inspect_op` with `var_name: g_op`.

Expected tree:
```
func.func
  ^bb(i32):
    arith.addi
    arith.muli
    func.return
```

Bind the op and confirm its metadata, via `execute_python`:
```python
addi_op = find_ops_by_name(mut, "arith.addi")[0]
print(len(list(addi_op.results[0].uses)))
```

Expected: `1`. Then `get_op_info` with `var_name: addi_op` reports two i32
operands, one i32 result, 0 regions.

Now do the mutation in the order that keeps the IR valid. Call
`replace_all_uses` with `old_value_expr: addi_op.results[0]` and
`new_value_expr: addi_op.operands[0]`, then `erase_op` with
`var_name: addi_op`.

Expected: `Replaced all uses.`, then `Erased operation and removed 'addi_op'
from namespace.` `get_module_asm` with `var_name: mut` should show the `addi`
gone and `muli` reading the block argument twice:
```
module {
  func.func @g(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    return %0 : i32
  }
}
```

Finish with `verify_module`, `var_name: mut`. Expected: `Verification passed.`

Order matters here. `replace_all_uses` has to run before `erase_op`, otherwise
`muli` is left holding an operand whose defining op is gone.

## Test 12: File round trip

Call `save_mlir_file` with `var_name: mut` and a real path such as
`/tmp/mcp_plan_roundtrip.mlir`, then `load_mlir_file` on the same path with
`var_name: reloaded`, then `verify_module` with `var_name: reloaded`.

Expected: `Wrote '<path>'.`, then `Loaded '<path>' into 'reloaded'.`, then
`Verification passed.` Read the file outside the server and confirm the text
matches what `get_module_asm` returned.

## Test 13: Unfiltered discovery output sizes

Call `list_passes` with no filter, then `list_ops` with `dialect: arith`, then
`list_ir_apis`, `list_rewrite_apis`, and `list_dialects`.

Nothing in the package truncates output, so these land in the model's context at
full size. Current measurements:

| Tool | chars | lines |
|---|---|---|
| `list_passes` (no filter) | 23,439 | 316 |
| `list_ops` (`arith`) | 5,025 | 54 |
| `list_ir_apis` | 3,602 | 156 |
| `list_rewrite_apis` | 1,391 | 17 |
| `list_dialects` | 249 | 38 |

Expected: `list_passes` unfiltered costs roughly 6k tokens, so prefer a filter
as in Test 6. Treat a large jump in these numbers after an LLVM bump as a signal
to add truncation or paging.
