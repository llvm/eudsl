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


def test_canonicalize_module_imports_clean():
    # The vendored canonicalize/util import with no MLIR deps.
    from llvm.ast import canonicalize, util  # noqa: F401
    assert hasattr(canonicalize, "canonicalize")
    assert hasattr(util, "get_module_cst")
