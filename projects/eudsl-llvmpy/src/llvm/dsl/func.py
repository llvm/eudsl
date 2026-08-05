#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The @function decorator: turn a Python function into an LLVM function."""

import ast
import inspect

from .. import Function, IRBuilder
from ..eudslllvm_ext.types import Type
from ..eudslllvm_ext.types import function as function_t
from ..ast.util import get_module_cst
from .casters import maybe_downcast
from .context import building, current_builder


def _resolve(annotation, ctx):
    """Resolve a parameter/return annotation to an llvm.Type.

    Accepts a Type instance directly, or a callable `ctx -> Type` (e.g. the
    `llvm.types.i32` factory used bare as an annotation).
    """
    if isinstance(annotation, Type):
        return annotation
    if callable(annotation):
        return annotation(ctx)
    raise TypeError(f"cannot resolve type annotation {annotation!r}")


def _body_is_empty(f) -> bool:
    """True if f's body is only `...`, `pass`, and/or a docstring (a declaration)."""
    fn_node = get_module_cst(f).body[0]
    for stmt in fn_node.body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(
            stmt.value, ast.Constant
        ):
            # `...` (Ellipsis) or a docstring.
            continue
        return False
    return True


class DSLFunction:
    """Wraps a bound llvm.Function so it can be called from inside another
    @function body (emitting a `call`), while still exposing the Function."""

    def __init__(self, fn: Function):
        self.fn = fn

    @property
    def name(self):
        return self.fn.name

    def __call__(self, *args):
        call = current_builder().call(self.fn, list(args))
        return maybe_downcast(call, self.fn)


def function(*, module, name=None):
    def decorator(f):
        ctx = module.context
        sig = inspect.signature(f)
        param_types = [_resolve(p.annotation, ctx) for p in sig.parameters.values()]
        ret_type = _resolve(sig.return_annotation, ctx)
        fn_name = name or f.__name__

        fn = Function.create(function_t(ret_type, param_types), fn_name, module)

        # Empty body -> declaration: no entry block, stays a `declare`.
        if _body_is_empty(f):
            return DSLFunction(fn)

        entry = fn.append_basic_block("entry")
        builder = IRBuilder(ctx)

        with builder.at_end_of(entry), building(builder, fn):
            args = [maybe_downcast(fn.arg(i), fn) for i in range(len(param_types))]
            result = f(*args)
            if result is not None:
                builder.ret(result)
            elif builder.insert_block.terminator is None:
                builder.ret(None)
        return DSLFunction(fn)

    return decorator
