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
    assert with_nodes[0].lineno == 2
    assert with_nodes[1].lineno == 4


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
