# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import mlir.ir as ir
import pytest

from mlir_python_mcp.helpers import (
    _to_operation,
    find_ops,
    find_ops_by_name,
    get_module_asm,
    list_op_names,
    new_module,
    print_op_tree,
    run_pipeline,
)


@pytest.fixture
def ctx():
    c = ir.Context()
    c.__enter__()
    c.enable_multithreading(False)
    yield c
    c.__exit__(None, None, None)


@pytest.fixture
def simple_module(ctx):
    with ir.Location.unknown():
        return ir.Module.parse("""\
func.func @add(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
""")


class TestToOperation:
    def test_module(self, simple_module):
        op = _to_operation(simple_module)
        assert isinstance(op, ir.Operation)
        assert op.name == "builtin.module"

    def test_opview(self, simple_module):
        func_op = list(simple_module.body.operations)[0]
        op = _to_operation(func_op)
        assert isinstance(op, ir.Operation)

    def test_operation(self, simple_module):
        op = simple_module.operation
        assert _to_operation(op) is op


class TestPrintOpTree:
    def test_basic(self, simple_module):
        tree = print_op_tree(simple_module)
        assert "builtin.module" in tree
        assert "func.func" in tree
        assert "arith.addi" in tree
        assert "func.return" in tree

    def test_with_indent(self, simple_module):
        tree = print_op_tree(simple_module, indent=2)
        assert tree.startswith("  builtin.module")


class TestFindOps:
    def test_find_all(self, simple_module):
        ops = find_ops(simple_module, lambda op: True)
        assert len(ops) > 0

    def test_find_by_pred(self, simple_module):
        ops = find_ops(simple_module, lambda op: op.name == "arith.addi")
        assert len(ops) == 1
        assert ops[0].name == "arith.addi"

    def test_find_none(self, simple_module):
        ops = find_ops(simple_module, lambda op: op.name == "nonexistent.op")
        assert ops == []

    def test_single_found(self, simple_module):
        op = find_ops(simple_module, lambda op: op.name == "arith.addi", single=True)
        assert op is not None
        assert op.name == "arith.addi"

    def test_single_not_found(self, simple_module):
        op = find_ops(simple_module, lambda op: op.name == "nope", single=True)
        assert op is None


class TestFindOpsByName:
    def test_found(self, simple_module):
        ops = find_ops_by_name(simple_module, "func.func")
        assert len(ops) == 1

    def test_not_found(self, simple_module):
        ops = find_ops_by_name(simple_module, "nope")
        assert ops == []


class TestGetModuleAsm:
    def test_basic(self, simple_module):
        asm = get_module_asm(simple_module)
        assert "func.func @add" in asm
        assert "arith.addi" in asm

    def test_debug_info(self, simple_module):
        asm = get_module_asm(simple_module, debug_info=True)
        assert "loc(" in asm


class TestRunPipeline:
    def test_basic(self, ctx):
        with ir.Location.unknown():
            module = ir.Module.parse("""\
func.func @f() {
  %c = arith.constant 1 : i32
  %d = arith.constant 1 : i32
  return
}
""")
        result = run_pipeline(module, "builtin.module(canonicalize)")
        assert isinstance(result, ir.Module)

    def test_with_pipeline_object(self, ctx):
        from mlir_python_mcp.helpers import Pipeline

        with ir.Location.unknown():
            module = ir.Module.parse("""\
func.func @f() {
  return
}
""")
        p = Pipeline().canonicalize()
        run_pipeline(module, p)
        asm = get_module_asm(module)
        assert "func.func @f" in asm

    def test_invalid_pipeline(self, ctx):
        from mlir_python_mcp._passes_base import MlirCompilerError

        with ir.Location.unknown():
            module = ir.Module.parse("func.func @f() { return }")
        with pytest.raises(MlirCompilerError):
            run_pipeline(module, "builtin.module(totally-invalid-pass-xyz)")


class TestNewModule:
    def test_empty(self, ctx):
        with ir.Location.unknown():
            m = new_module()
        assert isinstance(m, ir.Module)
        assert len(list(m.body.operations)) == 0

    def test_with_src(self, ctx):
        with ir.Location.unknown():
            m = new_module("func.func @f() { return }")
        assert "func.func @f" in get_module_asm(m)


class TestListOpNames:
    def test_basic(self, simple_module):
        names = list_op_names(simple_module)
        assert "arith.addi" in names
        assert "func.func" in names
        assert "func.return" in names
        assert names == sorted(names)
