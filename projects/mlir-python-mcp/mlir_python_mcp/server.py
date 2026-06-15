# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import difflib
import io
import pkgutil
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
import mlir.dialects as _dialects
import mlir.ir as ir
import mlir.rewrite as _rewrite
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mlir.dialects import func

from .helpers import (
    _to_operation,
    find_ops,
    get_module_asm,
    print_op_tree,
    run_pipeline,
    Pipeline,
)
from .session import MLIRSession, _NAMESPACE_KEYS


def _wrap_pipeline(pipeline: str) -> str:
    if not pipeline.strip().startswith("builtin.module"):
        pipeline = f"builtin.module({pipeline})"
    return pipeline


_TOOLS: list[types.Tool] = [
    # Group 1: Core REPL
    types.Tool(
        name="execute_python",
        description=(
            "Execute arbitrary Python code in a persistent MLIR session. "
            "The namespace is pre-loaded with: "
            "ir (mlir.ir), dialects (mlir.dialects), passmanager (mlir.passmanager), "
            "rewrite (mlir.rewrite), execution_engine (mlir.execution_engine), "
            "ctx (the session's ir.Context), Pipeline (pass pipeline builder), "
            "and helper functions (new_module, run_pipeline, find_ops, find_ops_by_name, "
            "print_op_tree, get_module_asm, list_op_names). "
            "Access types via ir.IntegerType, ir.F32Type, ir.RankedTensorType, etc. "
            "Access dialects via dialects.arith, dialects.func, dialects.scf, dialects.linalg, etc. "
            "Pipeline is a fluent pass pipeline builder with methods for every "
            "registered MLIR pass (e.g. Pipeline().canonicalize().cse().convert_func_to_llvm()). "
            "Use run_pipeline(module, Pipeline().<passes>) to apply. "
            "Discovery: use help(<class>) for docs/signatures (e.g. help(dialects.arith.AddIOp), "
            "help(Pipeline.canonicalize), help(ir.IntegerType)), "
            "dir(dialects.arith) to list available ops, "
            "and dir(Pipeline) to list all passes. "
            "Variables persist between calls. Use generated builders for types/attrs/ops "
            "(e.g. ir.IntegerType.get_signless(32), dialects.arith.AddIOp(lhs, rhs)). "
            "Note: PascalCase op classes (dialects.arith.AddIOp) return OpView objects — use .result to get the Value. "
            "snake_case helpers (dialects.arith.addi) return Values directly. "
            "Type.parse/Attribute.parse are only for unregistered dialects."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "session": {
                    "type": "string",
                    "description": "Session name",
                    "default": "default",
                },
                "reset": {
                    "type": "boolean",
                    "description": "Reset user variables before executing",
                    "default": False,
                },
            },
            "required": ["code"],
        },
    ),
    types.Tool(
        name="list_variables",
        description="List user-defined variables in a session namespace.",
        inputSchema={
            "type": "object",
            "properties": {
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    types.Tool(
        name="new_session",
        description="Create a new named MLIR session with its own ir.Context.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for the new session"},
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="list_sessions",
        description="List all active MLIR sessions.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="delete_session",
        description="Destroy a named session. Cannot delete 'default'.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Session to delete"},
            },
            "required": ["name"],
        },
    ),
    # Group 2: Pipeline Workflow
    # NOTE: History/rewind only tracks transformations made via run_pipeline and
    # chain_pipeline. Mutations via execute_python are not recorded in history.
    types.Tool(
        name="run_pipeline",
        description=(
            "Set initial MLIR IR and apply a pass pipeline. Initializes transformation history. "
            "Pipeline can be just pass names (e.g. 'canonicalize,cse') — "
            "it will be auto-wrapped with 'builtin.module(...)' if needed. "
            "Note: history/rewind only tracks this tool and chain_pipeline, not execute_python mutations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "mlir": {"type": "string", "description": "MLIR source text"},
                "pipeline": {"type": "string", "description": "Pass pipeline string"},
                "session": {"type": "string", "default": "default"},
            },
            "required": ["mlir", "pipeline"],
        },
    ),
    types.Tool(
        name="chain_pipeline",
        description=(
            "Apply additional passes to the current IR state (incremental lowering). "
            "Must have active IR from run_pipeline first."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pipeline": {"type": "string", "description": "Pass pipeline string"},
                "session": {"type": "string", "default": "default"},
            },
            "required": ["pipeline"],
        },
    ),
    types.Tool(
        name="get_current_ir",
        description="Return the current IR text from the pipeline workflow.",
        inputSchema={
            "type": "object",
            "properties": {
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    types.Tool(
        name="rewind",
        description="Undo the last N pass applications, rewinding IR state.",
        inputSchema={
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of steps to rewind",
                    "default": 1,
                    "minimum": 1,
                },
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    types.Tool(
        name="history",
        description="Show the transformation history with optional diffs.",
        inputSchema={
            "type": "object",
            "properties": {
                "show_ir": {
                    "type": "boolean",
                    "description": "Include full IR text in each entry",
                    "default": False,
                },
                "show_diff": {
                    "type": "boolean",
                    "description": "Show unified diff between consecutive states",
                    "default": False,
                },
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    types.Tool(
        name="list_passes",
        description=(
            "List available MLIR passes with descriptions. Optionally filter by substring. "
            "Each pass corresponds to a Pipeline() method (e.g. Pipeline().canonicalize()). "
            "Use execute_python with help(Pipeline.<name>) for full docs on a specific pass."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Substring to filter pass names or descriptions",
                },
            },
        },
    ),
    types.Tool(
        name="reset",
        description="Clear current IR state and transformation history.",
        inputSchema={
            "type": "object",
            "properties": {
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    # Group 2b: Discovery Tools
    types.Tool(
        name="list_dialects",
        description=(
            "List all available dialect modules in mlir.dialects. "
            "Use execute_python with help(<dialect>.<OpName>) to get full docs for any op."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Substring to filter dialect names",
                },
            },
        },
    ),
    types.Tool(
        name="list_ops",
        description=(
            "List all operation classes in a given dialect module (e.g. 'arith', 'func', 'linalg'). "
            "Use execute_python with help(<dialect>.<OpName>) for full docs, signature, and examples."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dialect": {
                    "type": "string",
                    "description": "Dialect name (e.g. 'arith', 'func', 'scf', 'linalg')",
                },
                "filter": {
                    "type": "string",
                    "description": "Substring to filter op names",
                },
            },
            "required": ["dialect"],
        },
    ),
    types.Tool(
        name="list_ir_apis",
        description=(
            "List all public classes and functions in mlir.ir (Context, Module, Operation, Type, etc.). "
            "Use execute_python with help(ir.<ClassName>) for full docs and method signatures."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Substring to filter names",
                },
            },
        },
    ),
    types.Tool(
        name="list_rewrite_apis",
        description=(
            "List all public classes and functions in mlir.rewrite (PatternRewriter, TypeConverter, etc.). "
            "Use execute_python with help(rewrite.<ClassName>) for full docs and method signatures."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": "Substring to filter names",
                },
            },
        },
    ),
    # Group 3: MLIR Python Binding API Tools
    types.Tool(
        name="parse_mlir",
        description="Parse MLIR text into an ir.Module and store as a session variable.",
        inputSchema={
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "MLIR source text"},
                "var_name": {
                    "type": "string",
                    "description": "Variable name to store the module",
                    "default": "module",
                },
                "session": {"type": "string", "default": "default"},
            },
            "required": ["src"],
        },
    ),
    types.Tool(
        name="get_module_asm",
        description="Get the MLIR assembly text of a named module/operation variable.",
        inputSchema={
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "default": "module"},
                "session": {"type": "string", "default": "default"},
                "debug_info": {
                    "type": "boolean",
                    "description": "Include debug location info",
                    "default": False,
                },
            },
        },
    ),
    types.Tool(
        name="load_mlir_file",
        description="Load a .mlir file from disk into the session as an ir.Module.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to .mlir file"},
                "var_name": {"type": "string", "default": "module"},
                "session": {"type": "string", "default": "default"},
            },
            "required": ["path"],
        },
    ),
    types.Tool(
        name="save_mlir_file",
        description="Write a module variable to a .mlir file.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Destination file path"},
                "var_name": {"type": "string", "default": "module"},
                "session": {"type": "string", "default": "default"},
            },
            "required": ["path"],
        },
    ),
    types.Tool(
        name="walk_operations",
        description=(
            "Walk the operation tree and return a structured list of operations. "
            "Optionally filter by op name pattern (substring match)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "default": "module"},
                "filter": {
                    "type": "string",
                    "description": "Op name substring filter (e.g. 'arith' or 'func.func')",
                },
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    types.Tool(
        name="get_op_info",
        description=(
            "Get detailed info about a specific operation variable: "
            "name, attributes, operand types, result types, region count."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "var_name": {
                    "type": "string",
                    "description": "Variable name holding the operation",
                },
                "session": {"type": "string", "default": "default"},
            },
            "required": ["var_name"],
        },
    ),
    types.Tool(
        name="create_func",
        description=(
            "Create a func.FuncOp with the given name and type signature, "
            "add an entry block, and store it in the session. "
            "Uses the generated func.FuncOp builder."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function name"},
                "input_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Input type strings (e.g. ['i32', 'f32']). Uses Type.parse for convenience.",
                },
                "result_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Result type strings (e.g. ['i64']). Uses Type.parse for convenience.",
                },
                "var_name": {
                    "type": "string",
                    "description": "Variable to store the FuncOp",
                    "default": "f",
                },
                "module_var": {
                    "type": "string",
                    "description": "Module to insert into (must exist in session)",
                    "default": "module",
                },
                "session": {"type": "string", "default": "default"},
            },
            "required": ["name", "input_types", "result_types"],
        },
    ),
    types.Tool(
        name="symbol_lookup",
        description="Look up an operation by symbol name (e.g. function name) in a module.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Symbol name to look up",
                },
                "var_name": {
                    "type": "string",
                    "description": "Module variable to search in",
                    "default": "module",
                },
                "result_var": {
                    "type": "string",
                    "description": "Variable to store the found operation",
                    "default": "found_op",
                },
                "session": {"type": "string", "default": "default"},
            },
            "required": ["symbol_name"],
        },
    ),
    types.Tool(
        name="verify_module",
        description="Verify a module and return diagnostics.",
        inputSchema={
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "default": "module"},
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    types.Tool(
        name="clone_module",
        description="Deep-copy a module for safe experimentation.",
        inputSchema={
            "type": "object",
            "properties": {
                "src_var": {
                    "type": "string",
                    "description": "Source module variable",
                    "default": "module",
                },
                "dest_var": {
                    "type": "string",
                    "description": "Destination variable name",
                    "default": "module_copy",
                },
                "session": {"type": "string", "default": "default"},
            },
        },
    ),
    types.Tool(
        name="inspect_op",
        description="Print a hierarchical tree of operations (op names and block args).",
        inputSchema={
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "description": "Variable to inspect"},
                "session": {"type": "string", "default": "default"},
            },
            "required": ["var_name"],
        },
    ),
    types.Tool(
        name="replace_all_uses",
        description="Replace all uses of one value with another (SSA use-def chain).",
        inputSchema={
            "type": "object",
            "properties": {
                "old_value_expr": {
                    "type": "string",
                    "description": "Python expression evaluating to the old Value (e.g. 'op.results[0]')",
                },
                "new_value_expr": {
                    "type": "string",
                    "description": "Python expression evaluating to the new Value",
                },
                "session": {"type": "string", "default": "default"},
            },
            "required": ["old_value_expr", "new_value_expr"],
        },
    ),
    types.Tool(
        name="erase_op",
        description="Detach and erase an operation from its parent.",
        inputSchema={
            "type": "object",
            "properties": {
                "var_name": {
                    "type": "string",
                    "description": "Variable holding the operation to erase",
                },
                "session": {"type": "string", "default": "default"},
            },
            "required": ["var_name"],
        },
    ),
]


def _text(t: str) -> types.TextContent:
    return types.TextContent(type="text", text=t)


class MLIRMCPServer:
    def __init__(self) -> None:
        self.server = Server("mlir-python-repl")
        self._sessions: dict[str, MLIRSession] = {"default": MLIRSession("default")}

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:  # pragma: no cover
            return _TOOLS

        @self.server.call_tool()
        async def handle_call_tool(  # pragma: no cover
            name: str, arguments: dict | None
        ) -> list[types.TextContent]:
            return await self._dispatch(name, arguments or {})

    def _get_session(self, name: str) -> MLIRSession:
        if name not in self._sessions:
            raise KeyError(f"No session named '{name}'. Use new_session to create one.")
        return self._sessions[name]

    def _user_vars(self, session: MLIRSession) -> dict[str, Any]:
        return {k: v for k, v in session.namespace.items() if k not in _NAMESPACE_KEYS}

    async def _dispatch(self, name: str, args: dict) -> list[types.TextContent]:
        try:
            return self._call(name, args)
        except Exception:
            return [_text(f"Error:\n{traceback.format_exc()}")]

    def _call(self, name: str, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")

        # Group 1: Core REPL
        if name == "execute_python":
            return self._execute_python(args)
        elif name == "list_variables":
            return self._list_variables(session_name)
        elif name == "new_session":
            return self._new_session(args)
        elif name == "list_sessions":
            return self._list_sessions()
        elif name == "delete_session":
            return self._delete_session(args)
        # Group 2: Pipeline Workflow
        elif name == "run_pipeline":
            return self._run_pipeline(args)
        elif name == "chain_pipeline":
            return self._chain_pipeline(args)
        elif name == "get_current_ir":
            return self._get_current_ir(session_name)
        elif name == "rewind":
            return self._rewind(args)
        elif name == "history":
            return self._history(args)
        elif name == "list_passes":
            return self._list_passes(args)
        elif name == "reset":
            return self._reset(session_name)
        # Group 2b: Discovery Tools
        elif name == "list_dialects":
            return self._list_dialects(args)
        elif name == "list_ops":
            return self._list_ops(args)
        elif name == "list_ir_apis":
            return self._list_ir_apis(args)
        elif name == "list_rewrite_apis":
            return self._list_rewrite_apis(args)
        # Group 3: MLIR Python Binding API Tools
        elif name == "parse_mlir":
            return self._parse_mlir(args)
        elif name == "get_module_asm":
            return self._get_module_asm(args)
        elif name == "load_mlir_file":
            return self._load_mlir_file(args)
        elif name == "save_mlir_file":
            return self._save_mlir_file(args)
        elif name == "walk_operations":
            return self._walk_operations(args)
        elif name == "get_op_info":
            return self._get_op_info(args)
        elif name == "create_func":
            return self._create_func(args)
        elif name == "symbol_lookup":
            return self._symbol_lookup(args)
        elif name == "verify_module":
            return self._verify_module(args)
        elif name == "clone_module":
            return self._clone_module(args)
        elif name == "inspect_op":
            return self._inspect_op(args)
        elif name == "replace_all_uses":
            return self._replace_all_uses(args)
        elif name == "erase_op":
            return self._erase_op(args)
        else:
            raise ValueError(f"Unknown tool: {name}")

    # --- Group 1: Core REPL ---

    def _execute_python(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        code = args.get("code", "")

        if args.get("reset", False):
            session.reset_namespace()
            if not code:
                return [_text(f"Session '{session_name}' namespace reset.")]

        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with session.ctx, ir.Location.unknown():
                    exec(code, session.namespace)
            # Restore protected namespace keys that user code may have overwritten
            session._populate_namespace()
            out = stdout.getvalue()
            err = stderr.getvalue()
            parts = []
            if out:
                parts.append(out.rstrip())
            if err:
                parts.append(f"Stderr:\n{err.rstrip()}")
            if not parts:
                try:
                    last_line = code.strip().splitlines()[-1]
                    val = eval(last_line, session.namespace)
                    if val is not None:
                        parts.append(repr(val))
                    else:
                        parts.append("Executed successfully (no output).")
                except Exception:
                    parts.append("Executed successfully (no output).")
            return [_text("\n".join(parts))]
        except Exception:
            return [_text(f"Error:\n{traceback.format_exc()}")]

    def _list_variables(self, session_name: str) -> list[types.TextContent]:
        session = self._get_session(session_name)
        user_vars = self._user_vars(session)
        if not user_vars:
            return [_text(f"No user variables in session '{session_name}'.")]
        lines = [f"Variables in session '{session_name}':"]
        for k, v in user_vars.items():
            lines.append(f"  {k} = {type(v).__name__}: {repr(v)[:120]}")
        return [_text("\n".join(lines))]

    def _new_session(self, args: dict) -> list[types.TextContent]:
        name = args["name"]
        if name in self._sessions:
            return [_text(f"Session '{name}' already exists.")]
        self._sessions[name] = MLIRSession(name)
        return [_text(f"Created session '{name}'.")]

    def _list_sessions(self) -> list[types.TextContent]:
        lines = [f"Active sessions ({len(self._sessions)}):"]
        for sname, sess in self._sessions.items():
            user_vars = self._user_vars(sess)
            var_summary = (
                ", ".join(list(user_vars.keys())[:10]) if user_vars else "(empty)"
            )
            lines.append(f"  {sname}: {var_summary}")
        return [_text("\n".join(lines))]

    def _delete_session(self, args: dict) -> list[types.TextContent]:
        name = args["name"]
        if name == "default":
            return [_text("Cannot delete the 'default' session.")]
        if name not in self._sessions:
            return [_text(f"No session '{name}'.")]
        self._sessions.pop(name).destroy()
        return [_text(f"Deleted session '{name}'.")]

    # --- Group 2: Pipeline Workflow ---

    def _run_pipeline(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        mlir_src = args["mlir"]
        pipeline = _wrap_pipeline(args["pipeline"])

        with session.ctx, ir.Location.unknown():
            module = ir.Module.parse(mlir_src)
            initial_asm = get_module_asm(module)
            session.ir_history = [("initial", initial_asm)]
            module = run_pipeline(module, pipeline)
            result_asm = get_module_asm(module)
            session.ir_history.append((pipeline, result_asm))
            session.current_ir = result_asm
            session.namespace["module"] = module
        return [_text(result_asm)]

    def _chain_pipeline(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        pipeline = _wrap_pipeline(args["pipeline"])

        if session.current_ir is None:
            return [_text("No active IR state. Use run_pipeline first.")]

        with session.ctx, ir.Location.unknown():
            module = ir.Module.parse(session.current_ir)
            module = run_pipeline(module, pipeline)
            result_asm = get_module_asm(module)
            session.ir_history.append((pipeline, result_asm))
            session.current_ir = result_asm
            session.namespace["module"] = module
        return [_text(result_asm)]

    def _get_current_ir(self, session_name: str) -> list[types.TextContent]:
        session = self._get_session(session_name)
        if session.current_ir is None:
            return [_text("No active IR state.")]
        return [_text(session.current_ir)]

    def _rewind(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        steps = args.get("steps", 1)

        if not session.ir_history:
            return [_text("No history to rewind.")]

        target_idx = max(0, len(session.ir_history) - 1 - steps)
        session.ir_history = session.ir_history[: target_idx + 1]
        desc, ir_text = session.ir_history[-1]
        session.current_ir = ir_text

        with session.ctx, ir.Location.unknown():
            session.namespace["module"] = ir.Module.parse(ir_text)

        return [_text(f"Rewound to step '{desc}'.\n{ir_text}")]

    def _history(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        show_ir = args.get("show_ir", False)
        show_diff = args.get("show_diff", False)

        if not session.ir_history:
            return [_text("No history.")]

        lines = [f"History ({len(session.ir_history)} entries):"]
        for i, (desc, ir_text) in enumerate(session.ir_history):
            lines.append(f"  [{i}] {desc}")
            if show_ir:
                for line in ir_text.splitlines():
                    lines.append(f"    {line}")
            if show_diff and i > 0:
                prev_ir = session.ir_history[i - 1][1]
                diff = difflib.unified_diff(
                    prev_ir.splitlines(),
                    ir_text.splitlines(),
                    fromfile=f"step {i - 1}",
                    tofile=f"step {i}",
                    lineterm="",
                )
                diff_text = "\n".join(diff)
                if diff_text:
                    lines.append(f"    --- diff ---")
                    for line in diff_text.splitlines():
                        lines.append(f"    {line}")
        return [_text("\n".join(lines))]

    def _list_passes(self, args: dict) -> list[types.TextContent]:
        filter_str = args.get("filter", "")

        passes = []
        for name in sorted(dir(Pipeline)):
            if name.startswith("_"):
                continue
            method = getattr(Pipeline, name, None)
            if not callable(method):
                continue  # pragma: no cover
            doc = getattr(method, "__doc__", None) or ""
            first_line = doc.strip().split("\n")[0] if doc.strip() else ""
            if (
                filter_str
                and filter_str not in name
                and filter_str not in first_line.lower()
            ):
                continue
            passes.append(f"{name} — {first_line}" if first_line else name)

        if not passes:
            return [_text("No passes found matching filter.")]
        return [_text("\n".join(passes))]

    def _reset(self, session_name: str) -> list[types.TextContent]:
        session = self._get_session(session_name)
        session.reset_history()
        return [_text("IR state and history cleared.")]

    # --- Group 2b: Discovery Tools ---

    def _list_dialects(self, args: dict) -> list[types.TextContent]:
        filter_str = args.get("filter", "")

        dialects = []
        for importer, name, ispkg in pkgutil.iter_modules(_dialects.__path__):
            if name.startswith("_"):
                continue
            if filter_str and filter_str not in name:
                continue
            dialects.append(name)
        dialects.sort()
        if not dialects:
            return [_text("No dialects found matching filter.")]
        return [_text("\n".join(dialects))]

    def _list_ops(self, args: dict) -> list[types.TextContent]:
        dialect_name = args["dialect"]
        filter_str = args.get("filter", "")

        try:
            mod = __import__(f"mlir.dialects.{dialect_name}", fromlist=[dialect_name])
        except ImportError:
            return [_text(f"Dialect '{dialect_name}' not found.")]

        ops = []
        for name in sorted(dir(mod)):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name, None)
            if not (isinstance(obj, type) and hasattr(obj, "OPERATION_NAME")):
                continue
            if "Adaptor" in name:
                continue
            if (
                filter_str
                and filter_str not in name
                and filter_str not in getattr(obj, "OPERATION_NAME", "")
            ):
                continue
            op_name = getattr(obj, "OPERATION_NAME", "")
            doc = (getattr(obj, "__doc__", "") or "").strip().split("\n")[0]
            ops.append(f"{name} ({op_name}) — {doc}" if doc else f"{name} ({op_name})")
        if not ops:
            return [_text(f"No ops found in '{dialect_name}' matching filter.")]
        return [_text("\n".join(ops))]

    def _list_ir_apis(self, args: dict) -> list[types.TextContent]:
        filter_str = args.get("filter", "")

        apis = []
        for name in sorted(dir(ir)):
            if name.startswith("_"):
                continue
            if filter_str and filter_str not in name.lower() and filter_str not in name:
                continue
            obj = getattr(ir, name, None)
            kind = (
                "class"
                if isinstance(obj, type)
                else "function" if callable(obj) else "other"
            )
            apis.append(f"{name} [{kind}]")
        if not apis:
            return [_text("No APIs found matching filter.")]
        return [_text("\n".join(apis))]

    def _list_rewrite_apis(self, args: dict) -> list[types.TextContent]:
        filter_str = args.get("filter", "")

        apis = []
        for name in sorted(dir(_rewrite)):
            if name.startswith("_"):
                continue
            if filter_str and filter_str not in name.lower() and filter_str not in name:
                continue
            obj = getattr(_rewrite, name, None)
            kind = (
                "class"
                if isinstance(obj, type)
                else "function" if callable(obj) else "other"
            )
            doc = (getattr(obj, "__doc__", "") or "").strip().split("\n")[0]
            apis.append(f"{name} [{kind}] — {doc}" if doc else f"{name} [{kind}]")
        if not apis:
            return [_text("No APIs found matching filter.")]
        return [_text("\n".join(apis))]

    # --- Group 3: MLIR Python Binding API Tools ---

    def _parse_mlir(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args.get("var_name", "module")

        with session.ctx, ir.Location.unknown():
            module = ir.Module.parse(args["src"])
        session.namespace[var_name] = module
        op_count = len(list(module.body.operations))
        return [_text(f"Parsed into '{var_name}'. Top-level operations: {op_count}")]

    def _get_module_asm(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args.get("var_name", "module")
        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]
        debug_info = args.get("debug_info", False)
        return [_text(get_module_asm(obj, debug_info=debug_info))]

    def _load_mlir_file(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        path = Path(args["path"])
        var_name = args.get("var_name", "module")
        src = path.read_text()
        with session.ctx, ir.Location.unknown():
            module = ir.Module.parse(src)
        session.namespace[var_name] = module
        return [_text(f"Loaded '{path}' into '{var_name}'.")]

    def _save_mlir_file(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args.get("var_name", "module")
        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]
        path = Path(args["path"])
        path.write_text(get_module_asm(obj))
        return [_text(f"Wrote '{path}'.")]

    def _walk_operations(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args.get("var_name", "module")
        filter_str = args.get("filter", "")
        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]

        root = _to_operation(obj)
        results = []

        def _collect(op):
            if filter_str and filter_str not in op.name:
                return False
            results.append(
                {
                    "name": op.name,
                    "num_operands": len(op.operands),
                    "num_results": len(op.results),
                    "num_regions": len(op.regions),
                    "attributes": {k: str(v) for k, v in op.attributes.items()},
                }
            )
            return False

        find_ops(root, _collect)
        lines = [f"Found {len(results)} operations:"]
        for r in results:
            attrs = ", ".join(f"{k}={v}" for k, v in r["attributes"].items())
            lines.append(
                f"  {r['name']} (operands={r['num_operands']}, "
                f"results={r['num_results']}, regions={r['num_regions']}"
                f"{', attrs: ' + attrs if attrs else ''})"
            )
        return [_text("\n".join(lines))]

    def _get_op_info(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args["var_name"]
        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]

        op = _to_operation(obj)
        lines = [
            f"Operation: {op.name}",
            f"Operands ({len(op.operands)}):",
        ]
        for i, operand in enumerate(op.operands):
            lines.append(f"  [{i}] {operand.type}")
        lines.append(f"Results ({len(op.results)}):")
        for i, result in enumerate(op.results):
            lines.append(f"  [{i}] {result.type}")
        lines.append(f"Regions: {len(op.regions)}")
        lines.append(f"Attributes:")
        for k, v in op.attributes.items():
            lines.append(f"  {k} = {v}")
        return [_text("\n".join(lines))]

    def _create_func(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        func_name = args["name"]
        input_type_strs = args["input_types"]
        result_type_strs = args["result_types"]
        var_name = args.get("var_name", "f")
        module_var = args.get("module_var", "module")

        module = session.namespace.get(module_var)
        if module is None:
            return [
                _text(
                    f"No module '{module_var}' in session. Create one with parse_mlir first."
                )
            ]

        with session.ctx, ir.Location.unknown():
            input_types = [ir.Type.parse(t) for t in input_type_strs]
            result_types = [ir.Type.parse(t) for t in result_type_strs]
            func_type = ir.FunctionType.get(input_types, result_types)

            with ir.InsertionPoint(module.body):
                func_op = func.FuncOp(func_name, func_type)
                func_op.add_entry_block()

        session.namespace[var_name] = func_op
        return [
            _text(
                f"Created func @{func_name} stored in '{var_name}'.\n{get_module_asm(func_op)}"
            )
        ]

    def _symbol_lookup(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        symbol_name = args["symbol_name"]
        var_name = args.get("var_name", "module")
        result_var = args.get("result_var", "found_op")

        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]

        module_op = _to_operation(obj)
        symbol_table = ir.SymbolTable(module_op)
        try:
            found = symbol_table[symbol_name]
        except KeyError:
            return [_text(f"Symbol '{symbol_name}' not found.")]

        session.namespace[result_var] = found
        return [
            _text(
                f"Found '{symbol_name}' -> stored in '{result_var}'.\n{get_module_asm(found)}"
            )
        ]

    def _verify_module(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args.get("var_name", "module")
        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]

        op = _to_operation(obj)
        try:
            op.verify()
            return [_text("Verification passed.")]
        except ir.MLIRError as e:
            return [_text(f"Verification failed:\n{e}")]

    def _clone_module(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        src_var = args.get("src_var", "module")
        dest_var = args.get("dest_var", "module_copy")
        obj = session.namespace.get(src_var)
        if obj is None:
            return [_text(f"No variable '{src_var}' in session '{session_name}'.")]

        with session.ctx, ir.Location.unknown():
            asm = get_module_asm(obj)
            clone = ir.Module.parse(asm)
        session.namespace[dest_var] = clone
        return [_text(f"Cloned '{src_var}' into '{dest_var}'.")]

    def _inspect_op(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args["var_name"]
        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]
        return [_text(print_op_tree(obj))]

    def _replace_all_uses(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        old_expr = args["old_value_expr"]
        new_expr = args["new_value_expr"]

        with session.ctx, ir.Location.unknown():
            old_val = eval(old_expr, session.namespace)
            new_val = eval(new_expr, session.namespace)
            old_val.replace_all_uses_with(new_val)
        return [_text("Replaced all uses.")]

    def _erase_op(self, args: dict) -> list[types.TextContent]:
        session_name = args.get("session", "default")
        session = self._get_session(session_name)
        var_name = args["var_name"]
        obj = session.namespace.get(var_name)
        if obj is None:
            return [_text(f"No variable '{var_name}' in session '{session_name}'.")]

        op = _to_operation(obj)
        op.erase()
        del session.namespace[var_name]
        return [_text(f"Erased operation and removed '{var_name}' from namespace.")]

    async def run(self) -> None:  # pragma: no cover
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mlir-python-repl",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main() -> None:  # pragma: no cover
    server = MLIRMCPServer()
    await server.run()
