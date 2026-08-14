#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ast

from llvm.ast import cf_transformers as T


def _rewrite(src):
    tree = ast.parse(src)
    node = tree.body[0]
    for ctor in (
        T.CanonicalizeElIfs,
        T.InsertEmptyYield,
        T.ReplaceYieldWithLLVMYield,
        T.ReplaceIfWithWith,
    ):
        node = ctor(context=None, first_lineno=0).generic_visit(node)
    return ast.unparse(node)


def test_if_becomes_with_if_ctx_manager():
    src = (
        "def f():\n"
        "    if c:\n"
        "        x = a\n"
        "    else:\n"
        "        x = b\n"
    )
    out = _rewrite(src)
    assert "if_ctx_manager" in out
    assert "else_ctx_manager" in out
    assert "yield_" in out


def test_no_else_still_yields():
    src = "def f():\n    if c:\n        g()\n"
    out = _rewrite(src)
    assert "if_ctx_manager" in out
    assert "yield_" in out


def test_explicit_yield_skips_empty_yield_insertion():
    # Both branches already end in an explicit yield, so InsertEmptyYield
    # takes the "already yields" arc instead of inserting one.
    src = (
        "def f():\n"
        "    if c:\n"
        "        r = yield a\n"
        "    else:\n"
        "        r = yield b\n"
    )
    out = _rewrite(src)
    assert "if_ctx_manager" in out
    assert "else_ctx_manager" in out


def test_elif_chain_forwards_yield_from_nested_if():
    # An elif is a nested If in the outer If's orelse; when the nested arm
    # already yields, CanonicalizeElIfs must forward that yield up so the
    # outer If's own orelse also ends in a yield.
    src = (
        "def f():\n"
        "    if c1:\n"
        "        r = yield 1\n"
        "    elif c2:\n"
        "        r = yield 2\n"
        "    else:\n"
        "        r = yield 3\n"
    )
    out = _rewrite(src)
    assert out.count("yield_") >= 3


def test_nested_if_in_body_forwards_tuple_yield():
    # A nested If as the first statement of a body, itself yielding a tuple,
    # exercises the tuple-target branch of forward_yield_from_nested_if.
    src = (
        "def f():\n"
        "    if outer:\n"
        "        if inner:\n"
        "            a, b = yield 1, 2\n"
        "        x = 2\n"
    )
    out = _rewrite(src)
    assert "yield_" in out


def test_canonicalize_module_imports_clean():
    # The vendored canonicalize/util import with no MLIR deps.
    import sys
    from llvm.ast import canonicalize, util  # noqa: F401
    assert hasattr(canonicalize, "canonicalize")
    assert hasattr(util, "get_module_cst")
    assert not any(
        m.startswith("mlir") for m in sys.modules if sys.modules[m] is not None
    )


def test_if_rewrite_preserves_linenos():
    src = (
        "def f():\n"
        "    if c:\n"
        "        x = a\n"
        "    else:\n"
        "        x = b\n"
    )
    tree = ast.parse(src)
    node = tree.body[0]
    for ctor in (
        T.CanonicalizeElIfs,
        T.InsertEmptyYield,
        T.ReplaceYieldWithLLVMYield,
        T.ReplaceIfWithWith,
    ):
        node = ctor(context=None, first_lineno=0).generic_visit(node)
    # The transformed With nodes must carry line numbers from the original If.
    with_nodes = [n for n in ast.walk(node) if isinstance(n, ast.With)]
    assert len(with_nodes) == 2
    # then-branch With inherits the If's lineno via ast.copy_location.
    assert with_nodes[0].lineno == 2
    # else-branch With gets orelse[0].lineno - 1 (the `else:` keyword line).
    assert with_nodes[1].lineno == 4


def test_insert_empty_yield_linenos():
    # InsertEmptyYield appends a synthetic yield at end_lineno of the last stmt.
    src = (
        "def f():\n"       # line 1
        "    if c:\n"      # line 2
        "        x = a\n"  # line 3
    )
    tree = ast.parse(src)
    node = tree.body[0]
    node = T.InsertEmptyYield(context=None, first_lineno=0).generic_visit(node)
    # The inserted yield in the then-branch sits at end_lineno of `x = a` (line 3).
    inserted = node.body[0].body[-1]
    assert isinstance(inserted.value, ast.Yield)
    assert inserted.lineno == 3
    assert inserted.end_lineno == 3


def test_elif_adjacency_lineno_check():
    # ReplaceIfWithWith detects elif via end_lineno + 1 == orelse[0].lineno.
    # When this adjacency holds, the else-With gets copy_location from the elif.
    src = (
        "def f():\n"           # line 1
        "    if c1:\n"         # line 2
        "        r = yield 1\n"  # line 3
        "    elif c2:\n"       # line 4
        "        r = yield 2\n"  # line 5
    )
    tree = ast.parse(src)
    node = tree.body[0]
    for ctor in (
        T.CanonicalizeElIfs,
        T.InsertEmptyYield,
        T.ReplaceYieldWithLLVMYield,
        T.ReplaceIfWithWith,
    ):
        node = ctor(context=None, first_lineno=0).generic_visit(node)
    with_nodes = [n for n in ast.walk(node) if isinstance(n, ast.With)]
    # then-With at line 2 (the outer if), else-With at line 4 (the elif).
    assert with_nodes[0].lineno == 2
    assert with_nodes[1].lineno == 4


def test_canonicalize_elifs_preserves_forwarded_yield_lineno():
    # CanonicalizeElIfs uses append_hidden_node which sets lineno from
    # last_statement.end_lineno. The forwarded yield should sit at the
    # end_lineno of the last statement in the orelse body.
    src = (
        "def f():\n"           # line 1
        "    if outer:\n"      # line 2
        "        if inner:\n"  # line 3
        "            r = yield 1\n"  # line 4
        "        x = 2\n"     # line 5
    )
    tree = ast.parse(src)
    node = tree.body[0]
    node = T.CanonicalizeElIfs(context=None, first_lineno=0).generic_visit(node)
    # The forwarded yield appended to the outer if's body should have lineno
    # equal to end_lineno of `x = 2` (line 5).
    last = node.body[0].body[-1]
    assert isinstance(last.value, ast.Yield)
    assert last.lineno == 5


def test_replace_yield_with_llvm_yield_preserves_lineno():
    # ReplaceYieldWithLLVMYield uses ast.copy_location from the Yield node.
    src = (
        "def f():\n"           # line 1
        "    if c:\n"          # line 2
        "        r = yield a\n"  # line 3
    )
    tree = ast.parse(src)
    node = tree.body[0]
    node = T.ReplaceYieldWithLLVMYield(context=None, first_lineno=0).generic_visit(node)
    # The yield_ call replacing `yield a` on line 3 should keep that lineno.
    assign = node.body[0].body[0]
    assert isinstance(assign.value, ast.Call)
    assert assign.value.func.id == "yield_"
    assert assign.value.lineno == 3


def test_rewrite_produces_expected_ast_structure():
    src = (
        "def f():\n"
        "    if c:\n"
        "        x = a\n"
        "    else:\n"
        "        x = b\n"
    )
    tree = ast.parse(src)
    node = tree.body[0]
    for ctor in (
        T.CanonicalizeElIfs,
        T.InsertEmptyYield,
        T.ReplaceYieldWithLLVMYield,
        T.ReplaceIfWithWith,
    ):
        node = ctor(context=None, first_lineno=0).generic_visit(node)
    # The function body should contain exactly two With statements (then + else).
    assert all(isinstance(stmt, ast.With) for stmt in node.body)
    assert len(node.body) == 2
    # The then-branch With's context_expr must be a call to if_ctx_manager.
    then_ctx = node.body[0].items[0].context_expr
    assert isinstance(then_ctx, ast.Call)
    assert then_ctx.func.id == "if_ctx_manager"
    # The else-branch With's context_expr must be a call to else_ctx_manager.
    else_ctx = node.body[1].items[0].context_expr
    assert isinstance(else_ctx, ast.Call)
    assert else_ctx.func.id == "else_ctx_manager"
