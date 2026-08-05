#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test helpers. Not imported by the llvm package itself."""

import gc

from . import Context


def assert_no_leaks():
    """Assert every Context has been destroyed.

    Mirrors the convention in llvm-project/mlir/test/python/ir/*.py: a test
    that constructs IR must leave no live context behind.
    """
    gc.collect()
    live = Context._get_live_count()
    assert live == 0, f"{live} Context object(s) still alive"
