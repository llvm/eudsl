#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test helpers. Not imported by the llvm package itself."""

import gc

from . import Context


def assert_no_leaks():
    """Assert every Context has been released at this point.

    This is a cheap in-body check: `Context.__exit__` calls release(), so after
    a `with` block the context count is back to zero. It does NOT prove the
    underlying objects were destroyed — a Module (and the LLVMContext it keeps
    alive) can still be referenced by a live local. The authoritative leak gate
    is the autouse fixture in tests/conftest.py, which checks both the context
    and module counts after the test frame (and its locals) is gone.
    """
    gc.collect()
    live = Context._get_live_count()
    assert live == 0, f"{live} Context object(s) still alive"
