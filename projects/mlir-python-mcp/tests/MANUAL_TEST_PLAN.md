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
m = new_module()
with InsertionPoint(m.body):
    i32 = IntegerType.get_signless(32)
    f_type = FunctionType.get([i32, i32], [i32])
    f = func.FuncOp("my_add", f_type)
    entry = f.add_entry_block()
    with InsertionPoint(entry):
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
