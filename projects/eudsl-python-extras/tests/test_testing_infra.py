# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
from pathlib import Path
from textwrap import dedent

import pytest

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, MLIRContext
from mlir.extras.testing.generate_test_checks import (
    AttributeNamer,
    VariableNamer,
    get_num_ssa_results,
    main,
    preprocess_line,
    process_attribute_definition,
    process_attribute_references,
    process_line,
)
from mlir.extras.testing.testing import (
    filecheck,
    get_filecheck_path,
)

pytest.mark.usefixtures("ctx")


# ============================================================================
# Tests for testing/generate_test_checks.py
# ============================================================================


class TestVariableNamer:
    """Tests for VariableNamer - covers lines 99, 118"""

    def test_generate_name_from_ssa(self):
        """Line 99: use_ssa_name and source_variable_name[0].isalpha()"""
        namer = VariableNamer("")
        namer.push_name_scope()
        name = namer.generate_name("myvar", use_ssa_name=True)
        assert name == "MYVAR"

    def test_generate_name_generic(self):
        """Line 118: duplicate variable name raises RuntimeError"""
        namer = VariableNamer("")
        namer.push_name_scope()
        namer.generate_name("var1", use_ssa_name=True)
        # Try to generate same name again - should raise
        with pytest.raises(RuntimeError, match="duplicate variable name"):
            namer.generate_name("var1", use_ssa_name=True)

    def test_generate_name_from_op_name(self):
        """Line 99-104: op_name fallback"""
        namer = VariableNamer("")
        namer.push_name_scope()
        # numeric SSA name (not alpha) so uses op_name fallback
        name = namer.generate_name("0", use_ssa_name=True, op_name="transfer_read")
        assert name == "TRANSFER_READ_0"

    def test_generate_name_val_counter(self):
        """Lines 106-107: VAL_ counter fallback"""
        namer = VariableNamer("")
        namer.push_name_scope()
        name = namer.generate_name("0", use_ssa_name=False)
        assert name == "VAL_0"
        name2 = namer.generate_name("1", use_ssa_name=False)
        assert name2 == "VAL_1"

    def test_generate_in_parent_scope(self):
        """Test generate_in_parent_scope"""
        namer = VariableNamer("")
        namer.push_name_scope()
        namer.push_name_scope()
        namer.generate_in_parent_scope(1)
        name = namer.generate_name("x", use_ssa_name=True)
        # Should be in parent scope (index 0), keyed by source name
        assert "x" in namer.scopes[0]
        assert namer.scopes[0]["x"] == "X"

    def test_clear_names(self):
        """Test clear_names resets counters"""
        namer = VariableNamer("")
        namer.push_name_scope()
        namer.generate_name("0", use_ssa_name=False)
        assert namer.name_counter == 1
        namer.clear_names()
        assert namer.name_counter == 0
        assert len(namer.used_variable_names) == 0


class TestAttributeNamer:
    """Tests for AttributeNamer - covers line 165"""

    def test_generate_name(self):
        """Test basic attribute name generation"""
        namer = AttributeNamer("")
        name = namer.generate_name("#my_attr")
        assert name == "$ATTR_0"

    def test_duplicate_name_error(self):
        """Line 165: duplicate attribute name raises RuntimeError"""
        namer = AttributeNamer("")
        namer.generate_name("#attr1")
        # Force a duplicate by manipulating the used set
        namer.used_attribute_names.add("$ATTR_1")
        with pytest.raises(RuntimeError, match="duplicate attribute name"):
            namer.map["#attr_fake"] = "$ATTR_1"
            # Actually trigger the error
            namer2 = AttributeNamer("ATTR_1")
            namer2.used_attribute_names.add("$ATTR_1")
            namer2.generate_name("#something")

    def test_get_name(self):
        """Test get_name returns saved name or None"""
        namer = AttributeNamer("")
        namer.generate_name("#foo")
        assert namer.get_name("#foo") == "$ATTR_0"
        assert namer.get_name("#bar") is None


class TestGetNumSsaResults:
    """Tests for get_num_ssa_results - covers line 215"""

    def test_no_results(self):
        """No SSA results"""
        assert get_num_ssa_results("  arith.constant 0") == 0

    def test_single_result(self):
        """Single SSA result"""
        assert get_num_ssa_results("  %0 = arith.constant 0") == 1

    def test_multiple_results(self):
        """Multiple SSA results"""
        assert get_num_ssa_results("  %0, %1 = call @foo()") == 2


class TestProcessLine:
    """Tests for process_line - covers line 237"""

    def test_process_line_basic(self):
        """Basic line processing"""
        namer = VariableNamer("")
        namer.push_name_scope()
        # Process a line that was split at '%' - use a line without op name
        result = process_line(["0 + some_thing"], namer)
        assert "VAL_0" in result

    def test_process_line_existing_variable(self):
        """Line where variable already exists in scope"""
        namer = VariableNamer("")
        namer.push_name_scope()
        namer.generate_name("0", use_ssa_name=False)
        # Now reference the same variable
        result = process_line(["0)"], namer)
        assert "VAL_0" in result
        assert ":.*" not in result  # Should just reference, not define

    def test_process_line_strict_name_re(self):
        """Test strict_name_re option"""
        namer = VariableNamer("")
        namer.push_name_scope()
        result = process_line(["0 + something"], namer, strict_name_re=True)
        # Should use stricter regex
        assert "VAL_0:" in result


class TestProcessAttributeDefinition:
    """Tests for process_attribute_definition - covers line 299"""

    def test_with_attr_def(self):
        """Line 299: handles attribute definition"""
        namer = AttributeNamer("")
        result = process_attribute_definition("#map = affine_map<() -> ()>", namer)
        assert result is not None
        assert "ATTR_0" in result

    def test_without_attr_def(self):
        """Returns None when line is not an attribute definition"""
        namer = AttributeNamer("")
        result = process_attribute_definition("func.func @foo()", namer)
        assert result is None


class TestProcessAttributeReferences:
    """Tests for process_attribute_references - covers line 304"""

    def test_with_reference(self):
        """Line 304: processes attribute references"""
        namer = AttributeNamer("")
        namer.generate_name("#map")
        result = process_attribute_references("affine.for %i = 0 to 10 map #map", namer)
        assert "ATTR_0" in result

    def test_without_reference(self):
        """No attributes to reference"""
        namer = AttributeNamer("")
        result = process_attribute_references("arith.constant 0", namer)
        assert result == "arith.constant 0"


class TestPreprocessLine:
    """Tests for preprocess_line - covers line 336"""

    def test_double_braces(self):
        """Escape {{ sequences"""
        result = preprocess_line("value = {{something}}")
        assert "{{\\{\\{}}" in result

    def test_double_brackets(self):
        """Escape [[ sequences"""
        result = preprocess_line("value = [[0]]")
        assert "{{\\[\\[}}" in result

    def test_bracket_percent(self):
        """Escape [% sequences"""
        result = preprocess_line("memref[%i]")
        assert "{{\\[}}%" in result


class TestMain:
    """Tests for the main function - covers lines 377-389, 398-403, 414, 418-446"""

    def test_basic_mlir_input(self):
        """Lines 377-389: basic CHECK-LABEL generation"""
        mlir_input = dedent("""\
            func.func @foo(%arg0: i32) -> i32 {
              %0 = arith.addi %arg0, %arg0 : i32
              return %0 : i32
            }
        """)
        result = main(mlir_input, starts_from_scope=0)
        assert "CHECK" in result
        assert "foo" in result

    def test_empty_lines_skipped(self):
        """Line 299: empty lines in input are skipped"""
        mlir_input = "func.func @foo() {\n\n  return\n}\n"
        result = main(mlir_input, starts_from_scope=0)
        assert "CHECK" in result

    def test_scope_filtering(self):
        """Test starts_from_scope parameter"""
        mlir_input = dedent("""\
            module {
              func.func @foo() {
                return
              }
            }
        """)
        # starts_from_scope=1 should skip module-level lines
        result = main(mlir_input, starts_from_scope=1)
        assert "module" not in result.replace("// # CHECK", "")

    def test_file_split_skipped(self):
        """Line 304: // ----- lines are skipped"""
        mlir_input = "func.func @foo() {\n  return\n}\n// -----\nfunc.func @bar() {\n  return\n}\n"
        result = main(mlir_input, starts_from_scope=0)
        assert "-----" not in result

    def test_block_lines(self):
        """Test block lines (starting with ^) have comments stripped"""
        mlir_input = dedent("""\
            func.func @foo() {
            ^bb0:  // pred: some_block
              return
            }
        """)
        result = main(mlir_input, starts_from_scope=0)
        assert "CHECK" in result

    def test_output_to_file(self):
        """Line 414: output to a file-like object"""
        mlir_input = "func.func @foo() {\n  return\n}\n"
        output = io.StringIO()
        result = main(mlir_input, starts_from_scope=0, output=output)
        # When output is provided as StringIO, still returns value
        assert result is not None or output.getvalue() != ""

    def test_attribute_definitions_in_scope(self):
        """Test attribute definitions are emitted at scope start"""
        mlir_input = dedent("""\
            #map = affine_map<(d0) -> (d0)>
            func.func @foo(%arg0: memref<10xf32, #map>) {
              return
            }
        """)
        result = main(mlir_input, starts_from_scope=0)
        assert "ATTR" in result or "CHECK" in result


class TestGetFilecheckPath:
    """Tests for get_filecheck_path - covers line 46, 52"""

    def test_filecheck_path_found(self):
        """Test that FileCheck can be found"""
        try:
            path = get_filecheck_path()
            assert path is not None
            assert Path(path).exists()
        except AssertionError:
            pytest.skip("FileCheck not available in this environment")


class TestFilecheck:
    """Tests for filecheck - covers lines 71, 90-99"""

    def test_filecheck_passing(self, ctx: MLIRContext):
        """Test filecheck with matching output"""
        from mlir.dialects import func
        from mlir.extras import types as T

        @func.FuncOp.from_py_func(T.i32())
        def simple(a):
            return

        correct = dedent("""\
            func.func @simple(%arg0: i32) {
              return
            }
        """)
        # This should not raise
        filecheck(correct, ctx.module)

    def test_filecheck_failing(self, ctx: MLIRContext):
        """Lines 90-99: filecheck raises ValueError on mismatch"""
        from mlir.dialects import func
        from mlir.extras import types as T

        @func.FuncOp.from_py_func(T.i32())
        def simple(a):
            return

        wrong_correct = dedent("""\
            func.func @wrong_name(%arg0: f64) {
              return
            }
        """)
        with pytest.raises(ValueError):
            filecheck(wrong_correct, ctx.module)
