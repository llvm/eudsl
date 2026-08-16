#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ast

from llvm.ast.canonicalize import (
    canonicalize,
    Canonicalizer,
    FunctionPatcher,
    StrictTransformer,
    find_func_in_code_object,
)
from llvm.dsl.cf import LLVMCanonicalizer


def test_canonicalize_accepts_a_sequence_of_canonicalizers():
    # The `using=` argument may be a single Canonicalizer or a sequence; this
    # exercises the sequence branch.
    def f(x):
        return x

    g = canonicalize(using=[LLVMCanonicalizer()])(f)
    assert g(5) == 5


def test_canonicalize_accepts_a_single_canonicalizer_and_skips_nested_defs():
    # A bare (non-sequence) canonicalizer exercises the non-Sequence branch of
    # canonicalize(); a nested `def` in the body exercises StrictTransformer's
    # visit_FunctionDef, which leaves nested function defs untouched.
    def f(x):
        def helper(y):
            return y * 2

        return helper(x)

    g = canonicalize(using=LLVMCanonicalizer())(f)
    assert g(3) == 6


def test_transform_ast_handles_closures():
    def outer():
        captured = 42

        def inner(x):
            return x + captured

        return inner

    inner = outer()
    assert inner.__closure__ is not None
    g = canonicalize(using=LLVMCanonicalizer())(inner)
    assert g(1) == 43


def test_find_func_in_code_object_recurses_and_skips():
    # A module with a decoy nested function before the target forces the
    # search to recurse into a non-matching code object and continue.
    src = (
        "def decoy():\n"
        "    def inner_decoy():\n"
        "        return 1\n"
        "    return inner_decoy\n"
        "def target():\n"
        "    return 2\n"
    )
    code = compile(ast.parse(src), "<s>", "exec")
    found = find_func_in_code_object(code, "target")
    assert found is not None and found.co_name == "target"
    # A name that doesn't exist returns None (search exhausts).
    assert find_func_in_code_object(code, "missing") is None


def test_find_func_in_code_object_finds_doubly_nested_target():
    # target is nested two levels deep (inside outer), so the recursive call
    # must itself return non-None and propagate it back up.
    src = "def outer():\n    def target():\n        return 1\n    return target\n"
    code = compile(ast.parse(src), "<s>", "exec")
    found = find_func_in_code_object(code, "target")
    assert found is not None and found.co_name == "target"


def test_multiple_patchers_are_chained():
    # Verifies that patch_function applies all patchers in sequence, not just
    # the last one. Each patcher wraps f in a lambda that adds to the result.

    class _AddOnePatcher(FunctionPatcher):
        def patch_function(self, original_f):
            return lambda *a, **kw: original_f(*a, **kw) + 1

    class _DoubleCanonicalizer(Canonicalizer):
        @property
        def cst_transformers(self):
            return [StrictTransformer]

        @property
        def function_patchers(self):
            return [_AddOnePatcher, _AddOnePatcher]

    def f(x):
        return x

    g = canonicalize(using=_DoubleCanonicalizer())(f)
    # f(10) = 10, +1 from first patcher = 11, +1 from second = 12
    assert g(10) == 12
