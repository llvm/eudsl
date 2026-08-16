#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The @function decorator: turn a Python function into an LLVM function."""

import ast
import inspect
import types
from typing import get_args

from ..eudslllvm_ext.ir import Function, IRBuilder
from ..eudslllvm_ext.types import StructType, Type
from ..eudslllvm_ext.types import function as function_t
from ..ast.util import get_module_cst
from .casters import maybe_downcast
from ..eudslllvm_ext.ir import InsertPoint, current_builder


def _evaluate_alias_arg(arg, ctx):
    """Recursively turn one subscript argument into a value `.get(...)` accepts.

    A nested alias (`ArrayType[IntegerType[32], 4]`) or bare type class becomes a
    real Type; a list/tuple is mapped element-wise (e.g. the params list in
    `FunctionType[ret, [a, b]]`); scalars (ints, bools) pass through unchanged.
    """
    if isinstance(arg, Type):
        return arg
    if type(arg) is types.GenericAlias:
        return _evaluate_alias(arg, ctx)
    if isinstance(arg, (list, tuple)):
        return type(arg)(_evaluate_alias_arg(a, ctx) for a in arg)
    if isinstance(arg, type) and issubclass(arg, Type):
        return arg.get(context=ctx)
    if callable(arg):
        # A bare factory (e.g. `llvm.types.i32`) used as a nested subscript arg,
        # mirroring _resolve's callable branch for top-level annotations.
        return arg(context=ctx)
    return arg


def _evaluate_alias(alias, ctx):
    """Evaluate a types.GenericAlias (e.g. `IntegerType[32]`) into an llvm.Type
    by forwarding its evaluated subscript args to the origin's `.get(...)`.

    `StructType` is variadic in its subscript (`StructType[i32, f64]`), so its
    element args are collected into the single list `StructType.get` expects;
    every other origin forwards its args positionally (e.g.
    `FunctionType[ret, [a, b]]` -> `FunctionType.get(ret, [a, b])`).
    """
    origin = alias.__origin__
    args = [_evaluate_alias_arg(a, ctx) for a in get_args(alias)]
    if origin is StructType:
        return origin.get(list(args), context=ctx)
    return origin.get(*args, context=ctx)


def _resolve(annotation, ctx):
    """Resolve a parameter/return annotation to an llvm.Type.

    Accepts, in order:
      - a Type instance directly;
      - a deferred `types.GenericAlias` (`IntegerType[32]`, `PointerType[0]`,
        `FunctionType[ret, [args]]`, ...) evaluated against `ctx` -- these need
        no live context to be written as annotations;
      - a bare parametric type class (`PointerType`) -> `.get(context=ctx)`;
      - a callable `ctx -> Type` (e.g. the `llvm.types.i32` factory used bare).

    The GenericAlias and bare-class cases must precede the callable case: both
    are themselves callable and would otherwise hit the wrong branch.
    """
    if annotation is inspect.Parameter.empty:
        raise TypeError(
            "missing type annotation; annotate every parameter and the return type"
        )
    if isinstance(annotation, Type):
        return annotation
    if type(annotation) is types.GenericAlias:
        return _evaluate_alias(annotation, ctx)
    if isinstance(annotation, type) and issubclass(annotation, Type):
        return annotation.get(context=ctx)
    if callable(annotation):
        return annotation(context=ctx)
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


def function(
    *, module, name=None, linkage=None, calling_conv=None, attrs=None, var_arg=False
):
    def decorator(f):
        ctx = module.context
        sig = inspect.signature(f)
        param_types = [_resolve(p.annotation, ctx) for p in sig.parameters.values()]
        ret_type = _resolve(sig.return_annotation, ctx)
        fn_name = name or f.__name__

        fn = Function.create(
            function_t(ret_type, param_types, var_arg=var_arg), fn_name, module
        )
        if linkage is not None:
            fn.linkage = linkage
        if calling_conv is not None:
            fn.calling_conv = calling_conv
        for k, v in (attrs or {}).items():
            fn.add_fn_attr(k, v)

        # Empty body -> declaration: no entry block, stays a `declare`.
        if _body_is_empty(f):
            return DSLFunction(fn)

        entry = fn.append_basic_block("entry")
        builder = IRBuilder(ctx)

        with InsertPoint(entry, builder=builder):
            args = [maybe_downcast(fn.arg(i), fn) for i in range(len(param_types))]
            result = f(*args)
            if result is not None:
                builder.ret(result)
            else:
                # No DSL `return`: the current block (entry, or a loop/if exit)
                # is never already terminated here, so close it with `ret void`.
                builder.ret(None)
        return DSLFunction(fn)

    return decorator
