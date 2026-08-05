#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Thread-local DSL state: the current IRBuilder and enclosing function."""

import threading
from contextlib import contextmanager

_tls = threading.local()


def current_builder():
    b = getattr(_tls, "builder", None)
    if b is None:
        raise RuntimeError("no current IRBuilder; use `with building(builder):`")
    return b


def current_function():
    f = getattr(_tls, "function", None)
    if f is None:
        raise RuntimeError("no current function")
    return f


@contextmanager
def building(builder, function=None):
    prev_b = getattr(_tls, "builder", None)
    prev_f = getattr(_tls, "function", None)
    _tls.builder = builder
    if function is not None:
        _tls.function = function
    try:
        yield builder
    finally:
        _tls.builder = prev_b
        _tls.function = prev_f
