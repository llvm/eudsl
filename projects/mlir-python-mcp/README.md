# mlir-python-mcp

MCP (Model Context Protocol) server providing a persistent MLIR Python REPL via the upstream `mlir-python-bindings`.
Enables Claude to manipulate MLIR IR programmatically at the API/AST level instead of relying on textual edits.

## Install

```bash
pip install mlir-python-mcp -f https://llvm.github.io/eudsl
```

Or with uv (no `-f` flag needed, resolved from pyproject.toml):

```bash
uv pip install mlir-python-mcp
```

## Claude Code Configuration

Add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "mlir-python-mcp": {
      "command": "mlir-python-mcp"
    }
  }
}
```

Then start a new Claude Code session (MCP servers connect at session init).

## Tools

### Core REPL

- **`execute_python`** — Run arbitrary Python in a persistent session. The namespace is pre-loaded with all MLIR
  bindings.
- **`list_variables`** — Show user-defined variables in the session.
- **`new_session` / `list_sessions` / `delete_session`** — Manage named sessions (each with its own `ir.Context`).

### Pipeline Workflow

- **`run_pipeline`** — Set initial IR and apply passes. Initializes history.
- **`chain_pipeline`** — Apply more passes to current state (incremental lowering).
- **`get_current_ir`** — Return current IR text.
- **`rewind`** — Undo last N pass applications.
- **`history`** — Show transformation history with optional diffs.
- **`list_passes`** — List available passes.
- **`reset`** — Clear IR state and history.

> **Note:** History/rewind only tracks transformations made via `run_pipeline` and `chain_pipeline`.
> Mutations via `execute_python` (e.g. calling `run_pipeline()` directly in the REPL) are not recorded.

### Discovery

- **`list_dialects`** — List all available dialect modules. Use `help(<dialect>.<Op>)` for docs.
- **`list_ops`** — List all ops in a dialect (e.g. `arith`, `linalg`). Use `help(<dialect>.<Op>)` for full signatures.
- **`list_ir_apis`** — List all classes/functions in `mlir.ir`. Use `help(ir.<Class>)` for docs.
- **`list_rewrite_apis`** — List all classes/functions in `mlir.rewrite`. Use `help(rewrite.<Class>)` for docs.

### MLIR Python Binding API Tools

- **`parse_mlir`** — Parse MLIR text into `ir.Module`.
- **`get_module_asm`** — Get textual IR.
- **`load_mlir_file` / `save_mlir_file`** — File I/O for `.mlir` files.
- **`walk_operations`** — Walk op tree with optional name filter.
- **`get_op_info`** — Detailed info about an operation.
- **`create_func`** — Create `func.FuncOp` with entry block.
- **`symbol_lookup`** — Find operation by symbol name.
- **`verify_module`** — Run verification, report diagnostics.
- **`clone_module`** — Deep-copy a module.
- **`inspect_op`** — Print hierarchical op tree.
- **`replace_all_uses`** — SSA use-def chain manipulation.
- **`erase_op`** — Detach and erase an operation.

## Pre-loaded Namespace

The session namespace includes:

- `ir`, `passmanager`, `ctx`, `PassManager`
- Type constructors: `IntegerType`, `F32Type`, `RankedTensorType`, `MemRefType`, `VectorType`, `FunctionType`, etc.
- Attribute constructors: `IntegerAttr`, `FloatAttr`, `StringAttr`, `ArrayAttr`, `DictAttr`, `DenseElementsAttr`, etc.
- Dialect modules: `arith`, `func`, `scf`, `memref`, `tensor`, `linalg`, `vector`, `affine`, `cf`, `math`, `builtin`
- Helpers: `new_module`, `run_pipeline`, `find_ops`, `find_ops_by_name`, `print_op_tree`, `get_module_asm`,
  `list_op_names`

Use the generated builders for types, attributes, and ops:

```python
i32 = IntegerType.get_signless(32)
f32 = F32Type.get()
tensor_type = RankedTensorType.get([2, 3], f32)
c = arith.ConstantOp(i32, 42)
```

`Type.parse()` / `Attribute.parse()` are only for unregistered dialects.

## Development

```bash
pip install -e ".[test]" -f https://llvm.github.io/eudsl
pytest --cov=mlir_python_mcp --cov-report=term-missing --cov-branch --cov-fail-under=100
```
