# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import functools
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Optional

from .. import ir


@dataclass
class MLIRContext:
    context: ir.Context
    module: ir.Module

    def __str__(self):
        return str(self.module)


@contextmanager
def mlir_mod(
    src: Optional[str] = None,
    location: ir.Location = None,
) -> ir.Module:
    with ExitStack() as stack:
        if location is None:
            location = ir.Location.unknown()
        stack.enter_context(location)
        if src is not None:
            module = ir.Module.parse(src)
        else:
            module = ir.Module.create()
        ip = ir.InsertionPoint(module.body)
        stack.enter_context(ip)
        yield module


@contextmanager
def mlir_mod_ctx(
    src: Optional[str] = None,
    context: ir.Context = None,
    location: ir.Location = None,
    allow_unregistered_dialects=False,
) -> MLIRContext:
    if context is None:
        context = ir.Context()
    if allow_unregistered_dialects:
        context.allow_unregistered_dialects = True
    with context, mlir_mod(src, location) as module:
        yield MLIRContext(context, module)

    # TODO(AJM): was this removed?
    # context._clear_live_operations()


class RAIIMLIRContext:
    context: ir.Context
    location: ir.Location
    insertion_point: Optional[ir.InsertionPoint]
    module: Optional[ir.Module]

    def __init__(
        self,
        location: Optional[ir.Location] = None,
        allow_unregistered_dialects=False,
        create_module=False,
    ):
        self.context = ir.Context()
        if allow_unregistered_dialects:
            self.context.allow_unregistered_dialects = True
        self.context.__enter__()
        if location is None:
            location = ir.Location.unknown()
        self.location = location
        self.location.__enter__()
        if create_module:
            self.module = ir.Module.create()
            self.insertion_point = ir.InsertionPoint(self.module.body)
            self.insertion_point.__enter__()
        else:
            self.module = None
            self.insertion_point = None

    def __del__(self):
        if self.insertion_point is not None:
            self.insertion_point.__exit__(None, None, None)
        self.location.__exit__(None, None, None)
        self.context.__exit__(None, None, None)
        if ir is not None:  # pragma: no cover - only False during interpreter shutdown
            assert ir.Context is not self.context


class RAIIMLIRContextModule(RAIIMLIRContext):
    def __init__(
        self, location: Optional[ir.Location] = None, allow_unregistered_dialects=False
    ):
        super().__init__(
            location=location,
            allow_unregistered_dialects=allow_unregistered_dialects,
            create_module=True,
        )


class ExplicitlyManagedModule:
    module: ir.Module
    _ip: ir.InsertionPoint

    def __init__(self):
        self.module = ir.Module.create()
        self._ip = ir.InsertionPoint(self.module.body)
        self._ip.__enter__()

    def finish(self):
        self._ip.__exit__(None, None, None)
        return self.module

    def __str__(self):
        return str(self.module)


@contextlib.contextmanager
def multithreading(context=None, enabled=True):
    from ..ir import Context

    if context is None:
        context = Context.current
    context.enable_multithreading(enabled)
    yield
    context.enable_multithreading(not enabled)


enable_multithreading = functools.partial(multithreading, enabled=True)
disable_multithreading = functools.partial(multithreading, enabled=False)


@contextlib.contextmanager
def enable_debug():
    ir._GlobalDebug.flag = True
    yield
    ir._GlobalDebug.flag = False
