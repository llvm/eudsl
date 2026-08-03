# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from conftest import SIMPLE_MODULE, TWO_FUNC_MODULE


class TestExecutePython:
    async def test_basic_exec(self, server):
        result = await server._dispatch(
            "execute_python", {"code": "x = 1 + 1\nprint(x)"}
        )
        assert "2" in result[0].text

    async def test_print_output(self, server):
        result = await server._dispatch("execute_python", {"code": "print('hello')"})
        assert "hello" in result[0].text

    async def test_persistent_state(self, server):
        await server._dispatch("execute_python", {"code": "x = 42"})
        result = await server._dispatch("execute_python", {"code": "print(x)"})
        assert "42" in result[0].text

    async def test_error(self, server):
        result = await server._dispatch(
            "execute_python", {"code": "raise ValueError('boom')"}
        )
        assert "Error" in result[0].text
        assert "boom" in result[0].text

    async def test_reset(self, server):
        await server._dispatch("execute_python", {"code": "x = 42"})
        result = await server._dispatch("execute_python", {"code": "", "reset": True})
        assert "reset" in result[0].text.lower()

    async def test_reset_then_exec(self, server):
        await server._dispatch("execute_python", {"code": "x = 42"})
        result = await server._dispatch(
            "execute_python", {"code": "y = 10\nprint(y)", "reset": True}
        )
        assert "10" in result[0].text

    async def test_mlir_in_repl(self, server):
        result = await server._dispatch(
            "execute_python",
            {
                "code": "m = new_module('func.func @f() { return }')\nprint(get_module_asm(m))"
            },
        )
        assert "func.func @f" in result[0].text

    async def test_stderr_capture(self, server):
        result = await server._dispatch(
            "execute_python",
            {"code": "import sys; print('err', file=sys.stderr)"},
        )
        assert "err" in result[0].text

    async def test_no_output_no_return(self, server):
        result = await server._dispatch("execute_python", {"code": "_ = 1"})
        assert "Executed successfully" in result[0].text or "None" in result[0].text

    async def test_expression_result(self, server):
        result = await server._dispatch("execute_python", {"code": "1 + 1"})
        assert "2" in result[0].text

    async def test_expression_returns_none(self, server):
        result = await server._dispatch(
            "execute_python", {"code": "x = [1,2,3]\nx.append(4)"}
        )
        assert "Executed successfully" in result[0].text


class TestListVariables:
    async def test_empty(self, server):
        result = await server._dispatch("list_variables", {})
        assert "No user variables" in result[0].text

    async def test_with_vars(self, server):
        await server._dispatch("execute_python", {"code": "my_val = 123"})
        result = await server._dispatch("list_variables", {})
        assert "my_val" in result[0].text
        assert "123" in result[0].text


class TestSessions:
    async def test_new_session(self, server):
        result = await server._dispatch("new_session", {"name": "s2"})
        assert "Created" in result[0].text

    async def test_new_session_exists(self, server):
        await server._dispatch("new_session", {"name": "s2"})
        result = await server._dispatch("new_session", {"name": "s2"})
        assert "already exists" in result[0].text

    async def test_list_sessions(self, server):
        result = await server._dispatch("list_sessions", {})
        assert "default" in result[0].text

    async def test_delete_session(self, server):
        await server._dispatch("new_session", {"name": "temp"})
        result = await server._dispatch("delete_session", {"name": "temp"})
        assert "Deleted" in result[0].text

    async def test_delete_default(self, server):
        result = await server._dispatch("delete_session", {"name": "default"})
        assert "Cannot delete" in result[0].text

    async def test_delete_nonexistent(self, server):
        result = await server._dispatch("delete_session", {"name": "nope"})
        assert "No session" in result[0].text

    async def test_nonexistent_session(self, server):
        result = await server._dispatch(
            "execute_python", {"code": "x = 1", "session": "nope"}
        )
        assert "Error" in result[0].text


class TestRunPipeline:
    async def test_basic(self, server):
        result = await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        assert "func.func @add" in result[0].text

    async def test_with_anchor(self, server):
        result = await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "builtin.module(canonicalize)"},
        )
        assert "func.func @add" in result[0].text

    async def test_invalid_pipeline(self, server):
        result = await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "totally-bogus-pass-xyz"},
        )
        assert "Error" in result[0].text

    async def test_invalid_mlir(self, server):
        result = await server._dispatch(
            "run_pipeline",
            {"mlir": "not valid mlir at all", "pipeline": "canonicalize"},
        )
        assert "Error" in result[0].text


class TestChainPipeline:
    async def test_basic(self, server):
        await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        result = await server._dispatch("chain_pipeline", {"pipeline": "cse"})
        assert "func.func @add" in result[0].text

    async def test_no_state(self, server):
        result = await server._dispatch("chain_pipeline", {"pipeline": "canonicalize"})
        assert "No active IR" in result[0].text


class TestGetCurrentIR:
    async def test_no_state(self, server):
        result = await server._dispatch("get_current_ir", {})
        assert "No active IR" in result[0].text

    async def test_with_state(self, server):
        await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        result = await server._dispatch("get_current_ir", {})
        assert "func.func @add" in result[0].text


class TestRewind:
    async def test_basic(self, server):
        await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        await server._dispatch("chain_pipeline", {"pipeline": "cse"})
        result = await server._dispatch("rewind", {"steps": 1})
        assert "Rewound" in result[0].text

    async def test_no_history(self, server):
        result = await server._dispatch("rewind", {})
        assert "No history" in result[0].text

    async def test_rewind_past_start(self, server):
        await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        result = await server._dispatch("rewind", {"steps": 100})
        assert "Rewound" in result[0].text
        assert "initial" in result[0].text


class TestHistory:
    async def test_empty(self, server):
        result = await server._dispatch("history", {})
        assert "No history" in result[0].text

    async def test_basic(self, server):
        await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        result = await server._dispatch("history", {})
        assert "initial" in result[0].text
        assert "canonicalize" in result[0].text

    async def test_show_ir(self, server):
        await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        result = await server._dispatch("history", {"show_ir": True})
        assert "func.func @add" in result[0].text

    async def test_show_diff(self, server):
        await server._dispatch(
            "run_pipeline",
            {
                "mlir": "func.func @f() {\n  %c = arith.constant 1 : i32\n  %d = arith.constant 1 : i32\n  return\n}",
                "pipeline": "canonicalize",
            },
        )
        result = await server._dispatch("history", {"show_diff": True})
        assert "diff" in result[0].text or "---" in result[0].text

    async def test_show_diff_no_change(self, server):
        await server._dispatch(
            "run_pipeline",
            {
                "mlir": SIMPLE_MODULE,
                "pipeline": "canonicalize",
            },
        )
        await server._dispatch("chain_pipeline", {"pipeline": "canonicalize"})
        result = await server._dispatch("history", {"show_diff": True})
        assert "History" in result[0].text


class TestListPasses:
    async def test_basic(self, server):
        result = await server._dispatch("list_passes", {})
        assert "canonicalize" in result[0].text

    async def test_filter(self, server):
        result = await server._dispatch("list_passes", {"filter": "canonicalize"})
        assert "canonicalize" in result[0].text

    async def test_filter_no_match(self, server):
        result = await server._dispatch("list_passes", {"filter": "xyznonexistent123"})
        assert "No passes found" in result[0].text


class TestReset:
    async def test_basic(self, server):
        await server._dispatch(
            "run_pipeline",
            {"mlir": SIMPLE_MODULE, "pipeline": "canonicalize"},
        )
        result = await server._dispatch("reset", {})
        assert "cleared" in result[0].text.lower()


class TestListDialects:
    async def test_basic(self, server):
        result = await server._dispatch("list_dialects", {})
        assert "arith" in result[0].text
        assert "func" in result[0].text
        assert "scf" in result[0].text

    async def test_filter(self, server):
        result = await server._dispatch("list_dialects", {"filter": "linalg"})
        assert "linalg" in result[0].text
        assert "arith" not in result[0].text

    async def test_filter_no_match(self, server):
        result = await server._dispatch("list_dialects", {"filter": "xyznonexistent"})
        assert "No dialects found" in result[0].text


class TestListOps:
    async def test_basic(self, server):
        result = await server._dispatch("list_ops", {"dialect": "arith"})
        assert "AddIOp" in result[0].text
        assert "ConstantOp" in result[0].text

    async def test_filter(self, server):
        result = await server._dispatch(
            "list_ops", {"dialect": "arith", "filter": "Constant"}
        )
        assert "ConstantOp" in result[0].text
        assert "AddIOp" not in result[0].text

    async def test_invalid_dialect(self, server):
        result = await server._dispatch(
            "list_ops", {"dialect": "nonexistent_dialect_xyz"}
        )
        assert "not found" in result[0].text

    async def test_filter_no_match(self, server):
        result = await server._dispatch(
            "list_ops", {"dialect": "arith", "filter": "xyznonexistent"}
        )
        assert "No ops found" in result[0].text


class TestListIrApis:
    async def test_basic(self, server):
        result = await server._dispatch("list_ir_apis", {})
        assert "Context" in result[0].text
        assert "Module" in result[0].text
        assert "Operation" in result[0].text

    async def test_filter(self, server):
        result = await server._dispatch("list_ir_apis", {"filter": "Type"})
        assert "IntegerType" in result[0].text

    async def test_filter_no_match(self, server):
        result = await server._dispatch("list_ir_apis", {"filter": "xyznonexistent"})
        assert "No APIs found" in result[0].text


class TestListRewriteApis:
    async def test_basic(self, server):
        result = await server._dispatch("list_rewrite_apis", {})
        assert "PatternRewriter" in result[0].text
        assert "TypeConverter" in result[0].text

    async def test_filter(self, server):
        result = await server._dispatch("list_rewrite_apis", {"filter": "Pattern"})
        assert "PatternRewriter" in result[0].text

    async def test_filter_no_match(self, server):
        result = await server._dispatch(
            "list_rewrite_apis", {"filter": "xyznonexistent"}
        )
        assert "No APIs found" in result[0].text


class TestParseMlir:
    async def test_basic(self, server):
        result = await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        assert "Parsed" in result[0].text
        assert "module" in result[0].text

    async def test_custom_var(self, server):
        result = await server._dispatch(
            "parse_mlir", {"src": SIMPLE_MODULE, "var_name": "my_mod"}
        )
        assert "my_mod" in result[0].text

    async def test_invalid(self, server):
        result = await server._dispatch("parse_mlir", {"src": "not valid"})
        assert "Error" in result[0].text


class TestGetModuleAsm:
    async def test_basic(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("get_module_asm", {})
        assert "func.func @add" in result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch("get_module_asm", {"var_name": "nope"})
        assert "No variable" in result[0].text

    async def test_debug_info(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("get_module_asm", {"debug_info": True})
        assert "loc(" in result[0].text


class TestFileIO:
    async def test_load_and_save(self, server, tmp_path):
        src_file = tmp_path / "test.mlir"
        src_file.write_text(SIMPLE_MODULE)
        result = await server._dispatch("load_mlir_file", {"path": str(src_file)})
        assert "Loaded" in result[0].text

        dest_file = tmp_path / "out.mlir"
        result = await server._dispatch("save_mlir_file", {"path": str(dest_file)})
        assert "Wrote" in result[0].text
        assert dest_file.exists()
        assert "func.func @add" in dest_file.read_text()

    async def test_save_no_var(self, server, tmp_path):
        result = await server._dispatch(
            "save_mlir_file", {"path": str(tmp_path / "x.mlir"), "var_name": "nope"}
        )
        assert "No variable" in result[0].text


class TestWalkOperations:
    async def test_basic(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("walk_operations", {})
        assert "arith.addi" in result[0].text
        assert "func.func" in result[0].text

    async def test_filter(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("walk_operations", {"filter": "arith"})
        assert "arith.addi" in result[0].text
        assert "func.func" not in result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch("walk_operations", {"var_name": "nope"})
        assert "No variable" in result[0].text


class TestGetOpInfo:
    async def test_basic(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        await server._dispatch(
            "execute_python",
            {"code": "add_op = find_ops_by_name(module, 'arith.addi')[0]"},
        )
        result = await server._dispatch("get_op_info", {"var_name": "add_op"})
        assert "arith.addi" in result[0].text
        assert "Operands" in result[0].text
        assert "Results" in result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch("get_op_info", {"var_name": "nope"})
        assert "No variable" in result[0].text


class TestCreateFunc:
    async def test_basic(self, server):
        await server._dispatch("parse_mlir", {"src": ""})
        result = await server._dispatch(
            "create_func",
            {
                "name": "my_add",
                "input_types": ["i32", "i32"],
                "result_types": ["i32"],
            },
        )
        assert "my_add" in result[0].text

    async def test_no_module(self, server):
        result = await server._dispatch(
            "create_func",
            {
                "name": "f",
                "input_types": ["i32"],
                "result_types": ["i32"],
                "module_var": "nope",
            },
        )
        assert "No module" in result[0].text


class TestSymbolLookup:
    async def test_found(self, server):
        await server._dispatch("parse_mlir", {"src": TWO_FUNC_MODULE})
        result = await server._dispatch("symbol_lookup", {"symbol_name": "foo"})
        assert "foo" in result[0].text
        assert "found_op" in result[0].text

    async def test_not_found(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("symbol_lookup", {"symbol_name": "nonexistent"})
        assert "not found" in result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch(
            "symbol_lookup", {"symbol_name": "f", "var_name": "nope"}
        )
        assert "No variable" in result[0].text


class TestVerifyModule:
    async def test_valid(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("verify_module", {})
        assert "passed" in result[0].text.lower()

    async def test_invalid(self, server):
        await server._dispatch(
            "execute_python",
            {
                "code": (
                    "import mlir.ir as ir\n"
                    "bad_module = ir.Module.create(ir.Location.unknown())\n"
                    "with ir.InsertionPoint(bad_module.body):\n"
                    "    # Create an op that will fail verification (return outside of function)\n"
                    "    ir.Operation.create('func.return', operands=[], results=[])\n"
                )
            },
        )
        result = await server._dispatch("verify_module", {"var_name": "bad_module"})
        assert "failed" in result[0].text.lower() or "Error" in result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch("verify_module", {"var_name": "nope"})
        assert "No variable" in result[0].text


class TestCloneModule:
    async def test_basic(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("clone_module", {})
        assert "Cloned" in result[0].text
        asm_result = await server._dispatch(
            "get_module_asm", {"var_name": "module_copy"}
        )
        assert "func.func @add" in asm_result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch("clone_module", {"src_var": "nope"})
        assert "No variable" in result[0].text


class TestInspectOp:
    async def test_basic(self, server):
        await server._dispatch("parse_mlir", {"src": SIMPLE_MODULE})
        result = await server._dispatch("inspect_op", {"var_name": "module"})
        assert "builtin.module" in result[0].text
        assert "func.func" in result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch("inspect_op", {"var_name": "nope"})
        assert "No variable" in result[0].text


class TestReplaceAllUses:
    async def test_basic(self, server):
        await server._dispatch(
            "parse_mlir",
            {"src": """\
func.func @f(%a: i32) -> i32 {
  %c = arith.constant 0 : i32
  %r = arith.addi %a, %c : i32
  return %r : i32
}
"""},
        )
        await server._dispatch(
            "execute_python",
            {
                "code": (
                    "ops = find_ops_by_name(module, 'arith.constant')\n"
                    "const_op = ops[0]\n"
                    "func_op = find_ops_by_name(module, 'func.func')[0]\n"
                    "block_arg = list(func_op.regions[0].blocks[0].arguments)[0]\n"
                )
            },
        )
        result = await server._dispatch(
            "replace_all_uses",
            {
                "old_value_expr": "const_op.results[0]",
                "new_value_expr": "block_arg",
            },
        )
        assert "Replaced" in result[0].text


class TestEraseOp:
    async def test_basic(self, server):
        await server._dispatch(
            "parse_mlir",
            {"src": """\
func.func @f() {
  %c = arith.constant 0 : i32
  return
}
"""},
        )
        await server._dispatch(
            "execute_python",
            {"code": "const_op = find_ops_by_name(module, 'arith.constant')[0]"},
        )
        result = await server._dispatch("erase_op", {"var_name": "const_op"})
        assert "Erased" in result[0].text

    async def test_no_var(self, server):
        result = await server._dispatch("erase_op", {"var_name": "nope"})
        assert "No variable" in result[0].text


class TestUnknownTool:
    async def test_unknown(self, server):
        result = await server._dispatch("unknown_tool_xyz", {})
        assert "Error" in result[0].text
        assert "Unknown tool" in result[0].text
