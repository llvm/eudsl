# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
from ...ast.util import ast_call, set_lineno, append_hidden_node
from .dialect import *
from bytecode import ConcreteBytecode
from ...ast.canonicalize import BytecodePatcher, Canonicalizer, StrictTransformer


def is_yield_(last_statement):
    return (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == yield_.__name__
    ) or (
        isinstance(last_statement, ast.Assign)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == yield_.__name__
    )


def is_yield(last_statement):
    return (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Yield)
    ) or (
        isinstance(last_statement, ast.Assign)
        and isinstance(last_statement.value, ast.Yield)
    )

class InsertEmptyYield(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> ast.If:
        updated_node = self.generic_visit(updated_node)

        new_yield = ast.Expr(ast.Yield(value=None))
        if not is_yield(updated_node.body[-1]):
            updated_node.body = append_hidden_node(
                updated_node.body, deepcopy(new_yield)
            )
        if updated_node.orelse and not is_yield(updated_node.orelse[-1]):
            updated_node.orelse = append_hidden_node(
                updated_node.orelse, deepcopy(new_yield)
            )

        updated_node = ast.fix_missing_locations(updated_node)
        return updated_node

    def visit_For(self, updated_node: ast.For) -> ast.For:
        # TODO(max): this isn't robust at all...
        line = ast.dump(updated_node.iter.func)
        if "range_" in line or "for_" in line:
            updated_node = self.generic_visit(updated_node)
            new_yield = ast.Expr(ast.Yield(value=None))
            if not is_yield(updated_node.body[-1]):
                updated_node.body = append_hidden_node(updated_node.body, new_yield)
            updated_node = ast.fix_missing_locations(updated_node)
        return updated_node


def forward_yield_from_nested_if(node_body):
    last_statement = node_body[0].body[-1]
    if isinstance(last_statement.targets[0], ast.Tuple):
        res = ast.Tuple(
            [ast.Name(t.id, ast.Load()) for t in last_statement.targets[0].elts],
            ast.Load(),
        )
        targets = [
            ast.Tuple(
                [ast.Name(t.id, ast.Store()) for t in last_statement.targets[0].elts],
                ast.Store(),
            )
        ]
    else:
        res = ast.Name(last_statement.targets[0].id, ast.Load())
        targets = [ast.Name(last_statement.targets[0].id, ast.Store())]
    forwarding_yield = ast.Assign(
        targets=targets,
        value=ast.Yield(res),
    )
    return append_hidden_node(node_body, forwarding_yield)


class CanonicalizeElIfs(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> ast.If:
        # postorder
        updated_node = self.generic_visit(updated_node)
        needs_forward = lambda body: (
            body
            and isinstance(body[0], ast.If)
            and is_yield(body[0].body[-1])
            and not is_yield(body[-1])
        )
        if needs_forward(updated_node.body):
            updated_node.body = forward_yield_from_nested_if(updated_node.body)

        if needs_forward(updated_node.orelse):
            updated_node.orelse = forward_yield_from_nested_if(updated_node.orelse)
        updated_node = ast.fix_missing_locations(updated_node)
        return updated_node


class CanonicalizeWhile(StrictTransformer):
    def visit_While(self, updated_node: ast.While) -> List[ast.AST]:
        # postorder
        updated_node = self.generic_visit(updated_node)
        if isinstance(updated_node.test, ast.NamedExpr):
            test = updated_node.test.value
        else:
            test = updated_node.test
        w = ast_call(while__.__name__, [test])
        w = ast.copy_location(w, updated_node)
        assign = ast.Assign(
            targets=[ast.Name(f"w_{updated_node.lineno}", ctx=ast.Store())],
            value=w,
        )
        assign = ast.fix_missing_locations(ast.copy_location(assign, updated_node))

        next_ = ast_call(
            next.__name__,
            [
                ast.Name(f"w_{updated_node.lineno}", ctx=ast.Load()),
                ast.Constant(False, kind="bool"),
            ],
        )
        next_ = ast.fix_missing_locations(ast.copy_location(next_, updated_node))
        if isinstance(updated_node.test, ast.NamedExpr):
            updated_node.test.value = next_
        else:
            new_test = ast.NamedExpr(
                target=ast.Name(f"__init__{updated_node.lineno}"), value=next_
            )
            new_test = ast.copy_location(new_test, updated_node)
            updated_node.test = new_test

        updated_node = ast.fix_missing_locations(updated_node)
        assign = ast.fix_missing_locations(assign)

        return [assign, updated_node]


class ReplaceYieldWithSCFYield(StrictTransformer):
    def visit_Yield(self, node: ast.Yield) -> ast.Expr:
        if isinstance(node.value, ast.Tuple):
            args = node.value.elts
        else:
            args = [node.value] if node.value else []
        call = ast.copy_location(ast_call(yield_.__name__, args), node)
        call = ast.fix_missing_locations(call)
        return call


class ReplaceIfWithWith(StrictTransformer):
    def visit_If(self, updated_node: ast.If) -> Union[ast.With, List[ast.With]]:
        is_elif = (
            len(updated_node.orelse) >= 1
            and isinstance(updated_node.orelse[0], ast.If)
            and updated_node.body[-1].end_lineno + 1 == updated_node.orelse[0].lineno
        )

        updated_node = self.generic_visit(updated_node)
        last_statement = updated_node.body[-1]
        assert is_yield_(last_statement) or is_yield(
            last_statement
        ), f"{last_statement=}"

        test = updated_node.test
        num_results = max(
            len(last_statement.value.args),
            # if lhs of assign is a tuple unpacking
            (
                len(last_statement.targets[0].elts)
                if isinstance(last_statement, ast.Assign)
                and isinstance(last_statement.targets[0], ast.Tuple)
                else 0
            ),
        )
        results = [ast_call(placeholder_opaque_t.__name__) for _ in range(num_results)]
        results = ast.fix_missing_locations(
            ast.copy_location(ast.Tuple(results, ctx=ast.Load()), test)
        )

        if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Store())
        withitem = ast.withitem(
            context_expr=ast_call(if_ctx_manager.__name__, args=[test, results]),
            optional_vars=if_op_name,
        )
        then_with = ast.With(items=[withitem])
        then_with = ast.copy_location(then_with, updated_node)
        then_with = ast.fix_missing_locations(then_with)
        then_with.body = updated_node.body

        if updated_node.orelse:
            if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Load())
            withitem = ast.withitem(
                context_expr=ast_call(else_ctx_manager.__name__, args=[if_op_name]),
                optional_vars=None,
            )
            else_with = ast.With(items=[withitem])
            if is_elif:
                else_with = ast.copy_location(else_with, updated_node.orelse[0])
            else:
                else_with = set_lineno(else_with, updated_node.orelse[0].lineno - 1)
            else_with = ast.fix_missing_locations(else_with)
            else_with.body = updated_node.orelse
            return [then_with, else_with]
        else:
            return then_with


class RemoveJumpsAndInsertGlobals(BytecodePatcher):
    def patch_bytecode(self, code: ConcreteBytecode, f):
        # TODO(max): this is bad and should be in the closure rather than as a global
        f.__globals__[yield_.__name__] = yield_
        f.__globals__[if_ctx_manager.__name__] = if_ctx_manager
        f.__globals__[else_ctx_manager.__name__] = else_ctx_manager
        f.__globals__[placeholder_opaque_t.__name__] = placeholder_opaque_t
        return code


class SCFCanonicalizer(Canonicalizer):
    cst_transformers = [
        CanonicalizeElIfs,
        InsertEmptyYield,
        ReplaceYieldWithSCFYield,
        ReplaceIfWithWith,
        CanonicalizeWhile,
    ]

    bytecode_patchers = [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()
