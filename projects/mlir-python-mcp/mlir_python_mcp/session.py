# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlir.dialects
import mlir.execution_engine
import mlir.ir as ir
import mlir.passmanager
import mlir.rewrite

from mlir_python_mcp import helpers as _helpers

_NAMESPACE_KEYS = frozenset(
    {
        "__builtins__",
        "ir",
        "dialects",
        "execution_engine",
        "passmanager",
        "rewrite",
        "ctx",
        "helpers",
        "Pipeline",
        *_helpers.__all__,
    }
)


@dataclass
class MLIRSession:
    name: str
    ctx: ir.Context = field(default_factory=ir.Context)
    namespace: dict[str, Any] = field(default_factory=dict)
    ir_history: list[tuple[str, str]] = field(default_factory=list)
    current_ir: str | None = None

    def __post_init__(self) -> None:
        self.ctx.__enter__()
        self.ctx.enable_multithreading(False)
        self._populate_namespace()

    def _populate_namespace(self) -> None:
        ns = self.namespace
        ns["__builtins__"] = __builtins__
        ns["ctx"] = self.ctx
        ns["ir"] = ir
        ns["dialects"] = mlir.dialects
        ns["execution_engine"] = mlir.execution_engine
        ns["passmanager"] = mlir.passmanager
        ns["rewrite"] = mlir.rewrite
        ns["helpers"] = _helpers
        ns["Pipeline"] = _helpers.Pipeline
        for name in _helpers.__all__:
            ns[name] = getattr(_helpers, name)

    def reset_namespace(self) -> None:
        to_delete = [k for k in self.namespace if k not in _NAMESPACE_KEYS]
        for k in to_delete:
            del self.namespace[k]

    def reset_history(self) -> None:
        self.ir_history.clear()
        self.current_ir = None

    def destroy(self) -> None:
        try:
            self.ctx.__exit__(None, None, None)
        except Exception:  # pragma: no cover
            pass
