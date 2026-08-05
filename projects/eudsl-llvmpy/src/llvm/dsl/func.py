#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The @function decorator: turn a Python function into an LLVM function."""

import inspect

from .. import Function, IRBuilder, Type, function_t
from ..ast.canonicalize import canonicalize
from .casters import maybe_downcast
from .cf import LLVMCanonicalizer
from .context import building


def _resolve(annotation, ctx):
    """Resolve a parameter/return annotation to an llvm.Type.

    Accepts a Type instance directly, or a callable `ctx -> Type` (e.g. the
    `llvm.i32` factory used bare as an annotation).
    """
    if isinstance(annotation, Type):
        return annotation
    if callable(annotation):
        return annotation(ctx)
    raise TypeError(f"cannot resolve type annotation {annotation!r}")


def function(*, module, name=None):
    def decorator(f):
        ctx = module.context
        sig = inspect.signature(f)
        param_types = [_resolve(p.annotation, ctx) for p in sig.parameters.values()]
        ret_type = _resolve(sig.return_annotation, ctx)
        fn_name = name or f.__name__

        fn = Function.create(function_t(ret_type, param_types), fn_name, module)
        entry = fn.append_basic_block("entry")
        builder = IRBuilder(ctx)

        # Rewrite Python control flow into the cf context-manager calls.
        f = canonicalize(using=LLVMCanonicalizer())(f)

        with builder.at_end_of(entry), building(builder, fn):
            args = [maybe_downcast(fn.arg(i), fn) for i in range(len(param_types))]
            result = f(*args)
            if result is not None:
                builder.ret(result)
            elif builder.insert_block.terminator is None:
                builder.ret(None)
        return fn

    return decorator
