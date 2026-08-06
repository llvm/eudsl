#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The filecheck_with_comments harness itself: it matches ordered # CHECK
directives against printed IR and fails on a mismatch (unlike substring `in`)."""
import pytest

import llvm
from llvm.testing import assert_no_leaks, filecheck_with_comments

_SRC = "define i32 @f(i32 %x) {\nentry:\n  %s = add i32 %x, 1\n  ret i32 %s\n}\n"


def test_filecheck_matches_ordered():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        # CHECK: define i32 @f(i32 %x)
        # CHECK:   %s = add i32 %x, 1
        # CHECK-NEXT:   ret i32 %s
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def _check_absent_pattern(mod):
    # CHECK: this text is definitely not in the emitted IR 9f3a2b
    filecheck_with_comments(mod)


def test_filecheck_reports_mismatch():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        with pytest.raises(ValueError, match="FileCheck failed"):
            _check_absent_pattern(mod)
        del mod
    assert_no_leaks()
