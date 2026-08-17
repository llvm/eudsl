#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Vendored from eudsl-python-extras' mlir/extras/ast/util.py, with the MLIR- and
# cloudpickle-specific helpers (MLIRTypePickler, copy_object, copy_func,
# unpickle_mlir_type) removed: the DSL canonicalizer needs only the pure-ast
# helpers below.
import ast
import inspect
import types
from textwrap import dedent


def set_lineno(node, n=1):
    for child in ast.walk(node):
        child.lineno = n
        child.end_lineno = n
    return node


def ast_call(name, args=None, keywords=None):
    if args is None:
        args = []
    if keywords is None:
        keywords = []
    call = ast.Call(
        func=ast.Name(name, ctx=ast.Load()),
        args=args,
        keywords=keywords,
    )
    return call


def get_module_cst(f):
    f_src = dedent(inspect.getsource(f))
    tree = ast.parse(f_src)
    assert isinstance(
        tree.body[0], ast.FunctionDef
    ), f"unexpected ast node {tree.body[0]}"
    return tree


def append_hidden_node(node_body, new_node):
    last_statement = node_body[-1]
    assert (
        last_statement.end_lineno is not None
    ), f"last_statement {ast.unparse(last_statement)} must have end_lineno"
    new_node = ast.fix_missing_locations(
        set_lineno(new_node, last_statement.end_lineno)
    )
    node_body.append(new_node)
    return node_body


def find_func_in_code_object(co, func_name):
    for c in co.co_consts:
        if type(c) is types.CodeType:
            if c.co_name == func_name:
                return c
            else:
                f = find_func_in_code_object(c, func_name)
                if f is not None:
                    return f
