# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Callable, List, Optional, Union

import mlir.ir as ir

from ._passes_base import Pipeline as _PipelineBase, run_pipeline  # noqa: F401
from .passes import Pipeline  # noqa: F811

Pipeline.__doc__ = """Fluent builder for MLIR pass pipelines.

Build pipelines by chaining pass methods, then pass to run_pipeline():

    module = run_pipeline(module, Pipeline().canonicalize().cse())

Nest passes inside specific op contexts:

    p = Pipeline().Func(Pipeline().canonicalize().cse())
    module = run_pipeline(module, p)

Common patterns:

    # Lower to LLVM
    module = run_pipeline(module, Pipeline().lower_to_llvm())

    # Bufferize
    module = run_pipeline(module, Pipeline().bufferize())

    # Custom pipeline with options
    p = Pipeline().convert_func_to_llvm(use_bare_ptr_memref_call_conv=True)

The pipeline auto-wraps with builtin.module(...) when materialized via str().
Every pass registered in MLIR has a corresponding method (auto-generated).
Each pass method has a docstring describing what it does and its options.
Use help(Pipeline.<pass_name>) to see docs for a specific pass, e.g.:

    help(Pipeline.canonicalize)
    help(Pipeline.convert_func_to_llvm)

Use dir(Pipeline) to list all available passes.
"""

__all__ = [
    "Pipeline",
    "find_ops",
    "find_ops_by_name",
    "get_module_asm",
    "list_op_names",
    "new_module",
    "print_op_tree",
    "run_pipeline",
]


def _to_operation(obj) -> ir.Operation:
    if isinstance(obj, ir.Module):
        return obj.operation
    if isinstance(obj, ir.OpView):
        return obj.operation
    return obj


def print_op_tree(op, indent: int = 0) -> str:
    op = _to_operation(op)
    lines = [" " * indent + op.name]
    for region in op.regions:
        for block in region.blocks:
            args_str = ", ".join(str(a.type) for a in block.arguments)
            lines.append(" " * (indent + 2) + f"^bb({args_str}):")
            for child_op in block.operations:
                lines.append(print_op_tree(_to_operation(child_op), indent + 4))
    return "\n".join(lines)


def find_ops(
    root,
    pred: Callable[[ir.Operation], bool],
    *,
    single: bool = False,
) -> list | ir.Operation | None:
    root = _to_operation(root)
    matches: List[ir.Operation] = []

    def _walk(op: ir.Operation) -> None:
        if single and matches:
            return
        if pred(op):
            matches.append(op)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    _walk(_to_operation(child))

    _walk(root)
    if single:
        return matches[0] if matches else None
    return matches


def find_ops_by_name(root, op_name: str) -> List[ir.Operation]:
    return find_ops(root, lambda op: op.name == op_name)


def get_module_asm(
    module_or_op, *, debug_info: bool = False, large_limit: int = 10
) -> str:
    op = _to_operation(module_or_op)
    return op.get_asm(enable_debug_info=debug_info, large_elements_limit=large_limit)


def new_module(src: Optional[str] = None) -> ir.Module:
    with ir.Location.unknown():
        if src is not None:
            return ir.Module.parse(src)
        return ir.Module.create()


def list_op_names(root) -> List[str]:
    names: set = set()
    find_ops(root, lambda op: bool(names.add(op.name)) or False)
    return sorted(names)
