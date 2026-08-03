# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.dialects
import mlir.ir as ir
from mlir.dialects import arith, func

from mlir_python_mcp.session import MLIRSession


class TestSessionCreation:
    def test_creates_context(self):
        s = MLIRSession("test")
        assert s.ctx is not None
        assert isinstance(s.ctx, ir.Context)
        s.destroy()

    def test_namespace_populated(self):
        s = MLIRSession("test")
        assert "ir" in s.namespace
        assert "dialects" in s.namespace
        assert "passmanager" in s.namespace
        assert "rewrite" in s.namespace
        assert "execution_engine" in s.namespace
        assert "Pipeline" in s.namespace
        assert "ctx" in s.namespace
        assert s.namespace["ctx"] is s.ctx
        s.destroy()

    def test_helpers_in_namespace(self):
        s = MLIRSession("test")
        assert "find_ops" in s.namespace
        assert "run_pipeline" in s.namespace
        assert "new_module" in s.namespace
        assert "print_op_tree" in s.namespace
        assert "get_module_asm" in s.namespace
        assert "list_op_names" in s.namespace
        assert "find_ops_by_name" in s.namespace
        s.destroy()

    def test_history_empty(self):
        s = MLIRSession("test")
        assert s.ir_history == []
        assert s.current_ir is None
        s.destroy()


class TestSessionReset:
    def test_reset_namespace(self):
        s = MLIRSession("test")
        s.namespace["my_var"] = 42
        s.reset_namespace()
        assert "my_var" not in s.namespace
        assert "ir" in s.namespace
        s.destroy()

    def test_reset_history(self):
        s = MLIRSession("test")
        s.ir_history.append(("test", "ir text"))
        s.current_ir = "ir text"
        s.reset_history()
        assert s.ir_history == []
        assert s.current_ir is None
        s.destroy()

    def test_reset_history_already_empty(self):
        s = MLIRSession("test")
        s.reset_history()
        assert s.ir_history == []
        assert s.current_ir is None
        s.destroy()


class TestSessionDestroy:
    def test_destroy(self):
        s = MLIRSession("test")
        s.destroy()

    def test_destroy_idempotent(self):
        s = MLIRSession("test")
        s.destroy()
        s.destroy()

    def test_destroy_with_broken_context(self):
        s = MLIRSession("test")
        s.ctx.__exit__(None, None, None)
        s.destroy()


class TestSessionDialects:
    def test_dialects_module_available(self):
        s = MLIRSession("test")
        assert "dialects" in s.namespace
        assert s.namespace["dialects"] is mlir.dialects
        s.destroy()

    def test_dialects_accessible(self):
        s = MLIRSession("test")
        d = s.namespace["dialects"]
        assert d.arith is arith
        assert d.func is func
        s.destroy()
