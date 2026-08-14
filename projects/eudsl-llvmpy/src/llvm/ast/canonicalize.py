# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ast
import difflib
import logging
import types
from abc import ABC, abstractmethod
from typing import List, Union, Sequence

from .util import get_module_cst, set_lineno, find_func_in_code_object

logger = logging.getLogger(__name__)


class Transformer(ast.NodeTransformer):
    def __init__(self, context, first_lineno):
        super().__init__()
        self.context = context
        self.first_lineno = first_lineno


class StrictTransformer(Transformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        return node


def transform_func(f, *transformer_ctors: type[Transformer]):
    module = get_module_cst(f)
    context = types.SimpleNamespace()
    for transformer_ctor in transformer_ctors:
        orig_code = ast.unparse(module)
        func_node = module.body[0]
        replace = transformer_ctor(
            context=context, first_lineno=f.__code__.co_firstlineno - 1
        )
        logger.debug("[transformer] %s", replace.__class__.__name__)
        func_node = replace.generic_visit(func_node)
        new_code = ast.unparse(func_node)

        diff = list(
            difflib.unified_diff(
                orig_code.splitlines(),  # to this
                new_code.splitlines(),  # delta from this
                lineterm="",
            )
        )
        logger.debug("[transformed code diff]\n%s", "\n" + "\n".join(diff))
        logger.debug("[transformed code]\n%s", new_code)
        module.body[0] = func_node

    logger.debug("[final transformed code]\n\n%s", new_code)

    return module


# Wraps `f` in a synthetic enclosing scope that provides its free variables,
# so the compiled code object has the correct co_freevars layout.
def insert_closed_vars(f, module):
    enclosing_mod = ast.FunctionDef(
        name="enclosing_mod",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[],
        decorator_list=[],
        type_params=[],
    )
    for var in f.__code__.co_freevars:
        enclosing_mod.body.append(
            ast.Assign(
                targets=[ast.Name(var, ctx=ast.Store())],
                value=ast.Constant(None, kind="None"),
            )
        )
    enclosing_mod = set_lineno(enclosing_mod, module.body[0].lineno)
    enclosing_mod = ast.fix_missing_locations(enclosing_mod)

    enclosing_mod.body.extend(module.body)
    module.body = [enclosing_mod]
    return module


def transform_ast(
    f, transformers: List[Union[type[Transformer], type[StrictTransformer]]]
):
    module = transform_func(f, *transformers)
    if f.__closure__:
        module = insert_closed_vars(f, module)
    module = ast.fix_missing_locations(module)
    module = ast.increment_lineno(module, f.__code__.co_firstlineno - 1)
    module_code_o = compile(module, f.__code__.co_filename, "exec")
    new_f_code_o = find_func_in_code_object(module_code_o, f.__name__)
    f.__code__ = new_f_code_o
    return f


class FunctionPatcher(ABC):
    def __init__(self, context=None):
        self.context = context

    @abstractmethod
    def patch_function(self, original_f):
        pass  # pragma: no cover


def patch_function(f, patchers: List[type[FunctionPatcher]]):
    context = types.SimpleNamespace()
    new_f = f
    for patcher in patchers:
        new_f = patcher(context).patch_function(new_f)

    return new_f


class Canonicalizer(ABC):
    @property
    @abstractmethod
    def cst_transformers(self) -> List[StrictTransformer]:
        pass  # pragma: no cover

    @property
    @abstractmethod
    def function_patchers(self) -> List[FunctionPatcher]:
        pass  # pragma: no cover


def canonicalize(*, using: Union[Canonicalizer, Sequence[Canonicalizer]]):
    if not isinstance(using, Sequence):
        using = [using]
    cst_transformers = []
    function_patchers = []
    for u in using:
        cst_transformers.extend(u.cst_transformers)
        function_patchers.extend(u.function_patchers)

    def wrapper(f):
        f = transform_ast(f, cst_transformers)
        f = patch_function(f, function_patchers)
        return f

    return wrapper
