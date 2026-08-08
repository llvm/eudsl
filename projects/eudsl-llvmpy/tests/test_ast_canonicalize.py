#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ast

from llvm.ast.canonicalize import canonicalize, StrictTransformer, find_func_in_code_object
from llvm.dsl.cf import LLVMCanonicalizer


def test_canonicalize_accepts_a_sequence_of_canonicalizers():
    # The `using=` argument may be a single Canonicalizer or a sequence; this
    # exercises the sequence branch.
    def f(x):
        return x

    g = canonicalize(using=[LLVMCanonicalizer()])(f)
    assert g(5) == 5


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
