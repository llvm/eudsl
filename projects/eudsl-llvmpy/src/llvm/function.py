#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

import inspect
import sys
from functools import update_wrapper, wraps
from typing import TypeVar

from . import (
    add_function,
    append_basic_block,
    position_builder_at_end,
    TypeRef,
    type_of,
    get_param,
    types_,
)
from .context import current_context


def prep_func_types(sig, return_type):
    assert not (
        not sig.return_annotation is inspect.Signature.empty and return_type
    ), f"func can use return annotation or explicit return_type but not both"
    return_type = (
        sig.return_annotation
        if not sig.return_annotation is inspect.Signature.empty
        else return_type
    )
    assert isinstance(return_type, (str, TypeRef, TypeVar)) or inspect.isfunction(
        return_type
    ), f"return type must be llvm type or string or TypeVar or lambda; {return_type=}"

    input_types = [
        p.annotation
        for p in sig.parameters.values()
        if not p.annotation is inspect.Signature.empty
    ]
    assert all(
        isinstance(r, (str, TypeRef, TypeVar)) or inspect.isfunction(r)
        for r in input_types
    ), f"all input types must be mlir types or strings or TypeVars or lambdas {input_types=}"
    return input_types, return_type


class FuncOp:
    def __init__(self, body_builder, *, return_type=None, entry_bb_name="entry"):
        assert inspect.isfunction(body_builder), body_builder

        self.body_builder = body_builder
        self.func_name = self.body_builder.__name__
        self.entry_bb_name = entry_bb_name
        self._emitted = False

        sig = inspect.signature(self.body_builder)
        self.input_types, self.return_type = prep_func_types(sig, return_type)

        if self._is_decl():
            assert len(self.input_types) == len(
                sig.parameters
            ), f"func decl needs all input types annotated"
            self.emit()

    def _is_decl(self):
        # magic constant found from looking at the code for an empty fn
        if sys.version_info.minor == 13:
            return self.body_builder.__code__.co_code == b"\x95\x00g\x00"
        elif sys.version_info.minor == 12:
            return self.body_builder.__code__.co_code == b"\x97\x00y\x00"
        elif sys.version_info.minor == 11:
            return self.body_builder.__code__.co_code == b"\x97\x00d\x00S\x00"
        elif sys.version_info.minor in {8, 9, 10}:
            return self.body_builder.__code__.co_code == b"d\x00S\x00"
        else:
            raise NotImplementedError(f"{sys.version_info.minor} not supported.")

    def __str__(self):
        return str(f"{self.__class__} {self.__dict__}")

    def emit(self, *call_args):
        if self._emitted:
            return
        ctx = current_context()
        if len(call_args) == 0:
            input_types = self.input_types[:]
            locals = {"T": types_}
            for i, v in enumerate(input_types):
                if isinstance(v, TypeVar):
                    v = v.__name__
                if isinstance(v, str):
                    input_types[i] = eval(v, self.body_builder.__globals__, locals)
                elif inspect.isfunction(v):
                    input_types[i] = v()
                else:
                    raise ValueError(f"unknown input_type {v=}")
        else:
            input_types = [type_of(a) for a in call_args]

        return_type = self.return_type
        if inspect.isfunction(return_type):
            return_type = return_type()

        function_ty = types_.function(return_type, input_types)
        function = add_function(ctx.module, self.func_name, function_ty)
        if self._is_decl():
            return

        entry_bb = append_basic_block(function, self.entry_bb_name)
        position_builder_at_end(ctx.builder, entry_bb)

        params = [get_param(function, i) for i in range(len(input_types))]
        self.body_builder(*params)
        self._emitted = True


def make_maybe_no_args_decorator(decorator):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(decorator)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return decorator(args[0])
        else:
            # decorator arguments
            return lambda realf: decorator(realf, *args, **kwargs)

    return new_dec


@make_maybe_no_args_decorator
def function(f, *, emit=False, entry_bb_name="entry") -> FuncOp:
    func_ = FuncOp(body_builder=f, entry_bb_name=entry_bb_name)
    func_ = update_wrapper(func_, f)
    if emit:
        func_.emit()
    return func_
