#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Authoritative leak gate for the whole suite.

Every test runs inside this fixture. The checks fire in teardown, after the
test function has returned and its frame (and every local it held) is gone, so
a nonzero count means an object outlived all its Python references -- a real
ownership/keep_alive bug, not just a value still bound to a local. This catches
the case a context-only, in-body check cannot: `Context.__exit__` drops the
context count to zero even while a leaked Module keeps its LLVMContext alive,
so the Module count (tied to actual destruction) is what makes a leak visible.
"""
import gc

import pytest

from llvm.ir import Context


@pytest.fixture(autouse=True)
def _assert_no_leaks():
    yield
    gc.collect()
    live_ctx = Context._get_live_count()
    live_mod = Context._get_live_module_count()
    assert live_ctx == 0, f"{live_ctx} Context object(s) leaked past the test"
    assert live_mod == 0, f"{live_mod} Module object(s) leaked past the test"
