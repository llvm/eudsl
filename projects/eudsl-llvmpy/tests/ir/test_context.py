#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import gc

import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_module_is_counted():
    # The module count is tied to actual destruction, so it detects a leak the
    # context count cannot: __exit__ zeroes the context count even while a
    # Module still keeps the LLVMContext alive.
    assert llvm.Context._get_live_module_count() == 0
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        assert llvm.Context._get_live_module_count() == 1
        second = llvm.Module("m2", ctx)
        assert llvm.Context._get_live_module_count() == 2
        del second
        gc.collect()
        assert llvm.Context._get_live_module_count() == 1
        del mod
    gc.collect()
    assert llvm.Context._get_live_module_count() == 0
    assert_no_leaks()


def test_leaked_module_is_detected_by_module_count():
    # A module held past the context's release is invisible to the context count
    # (release() dropped it to 0) but visible to the module count.
    ctx = llvm.Context()
    leaked = llvm.Module("leak", ctx)
    ctx.__exit__(None, None, None)  # as if leaving a `with` block
    gc.collect()
    assert llvm.Context._get_live_count() == 0  # released
    assert llvm.Context._get_live_module_count() == 1  # but the module lives
    del leaked, ctx
    assert_no_leaks()


def test_context_is_counted():
    assert llvm.Context._get_live_count() == 0
    ctx = llvm.Context()
    assert llvm.Context._get_live_count() == 1
    del ctx
    assert_no_leaks()


def test_nested_contexts_are_counted():
    with llvm.Context() as a, llvm.Context() as b:
        assert a is not b
        assert llvm.Context._get_live_count() == 2
    assert_no_leaks()


def test_module_keeps_context_alive():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    del ctx
    gc.collect()
    # The module's keep_alive kept the context object alive, so this is safe.
    assert llvm.Context._get_live_count() == 1
    assert mod.module_identifier == "m"
    del mod
    assert_no_leaks()


def test_module_rename():
    with llvm.Context() as ctx:
        mod = llvm.Module("before", ctx)
        mod.module_identifier = "after"
        assert mod.module_identifier == "after"
        assert "ModuleID = 'after'" in str(mod)
        mod.source_filename = "after.ll"
        assert mod.source_filename == "after.ll"
        assert 'source_filename = "after.ll"' in str(mod)
        del mod
    assert_no_leaks()


def test_consumed_module_raises_instead_of_crashing():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        assert mod._is_consumed is False
        mod._take()
        assert mod._is_consumed is True
        with pytest.raises(RuntimeError, match="has been consumed"):
            _ = mod.module_identifier
        with pytest.raises(RuntimeError, match="has been consumed"):
            str(mod)
        del mod
    assert_no_leaks()
