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
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        # CHECK: define i32 @f(i32 %x)
        # CHECK:   %s = add i32 %x, 1
        # CHECK-NEXT:   ret i32 %s
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_filecheck_binds_capture_variable():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        # A capture var bound on one line must match the same text later --
        # this is the "SSA name binding" the docstring advertises.
        # CHECK: %[[S:.*]] = add i32 %x, 1
        # CHECK: ret i32 %[[S]]
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_filecheck_check_not_passes_when_absent():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        # CHECK: define i32 @f
        # CHECK-NOT: fdiv
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def _check_absent_pattern(mod):
    # CHECK: this text is definitely not in the emitted IR 9f3a2b
    filecheck_with_comments(mod)


def test_filecheck_reports_mismatch():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        # Match the actual content-mismatch diagnostic, not just the "FileCheck
        # failed" wrapper (which also fires for invocation errors like a missing
        # directive) -- so a broken directive extraction can't slip through.
        with pytest.raises(ValueError, match="expected string not found in input"):
            _check_absent_pattern(mod)
        del mod
    assert_no_leaks()


def _check_not_present_pattern(mod):
    # CHECK: define i32 @f
    # CHECK-NOT: add
    filecheck_with_comments(mod)


def test_filecheck_check_not_fails_when_present():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        # CHECK-NOT has inverted semantics: it must FAIL because `add` is present.
        with pytest.raises(ValueError, match="excluded string found"):
            _check_not_present_pattern(mod)
        del mod
    assert_no_leaks()


def _check_no_directives(mod):
    # no directives in this function, so FileCheck finds no check strings
    filecheck_with_comments(mod)


def test_filecheck_no_directives_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        # The safety property that stops a mistyped/directive-less call from
        # silently passing: FileCheck errors when there are no check strings.
        with pytest.raises(ValueError, match="no check strings found"):
            _check_no_directives(mod)
        del mod
    assert_no_leaks()
