#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Pure-AST control-flow transformers, vendored from eudsl-python-extras'
# mlir/extras/dialects/scf.py (the transformer classes only). They rewrite
# Python if/elif/else/while/for into calls to the runtime context managers
# (if_ctx_manager / else_ctx_manager / while_ / range_) and yield_ defined in
# llvm/dsl/cf.py. The runtime names are referenced here as string literals so
# this module has no dependency on the LLVM runtime; the LLVMCanonicalizer in
# cf.py supplies the function-patcher that injects those globals.
import ast
from copy import deepcopy
from typing import List, Union

from .canonicalize import StrictTransformer
from .util import ast_call, set_lineno, append_hidden_node


def is_yield_(last_statement):
    return (
        isinstance(last_statement, ast.Expr)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == "yield_"
    ) or (
        isinstance(last_statement, ast.Assign)
        and isinstance(last_statement.value, ast.Call)
        and isinstance(last_statement.value.func, ast.Name)
        and last_statement.value.func.id == "yield_"
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


class ReplaceYieldWithLLVMYield(StrictTransformer):
    def visit_Yield(self, node: ast.Yield) -> ast.Expr:
        if isinstance(node.value, ast.Tuple):
            args = node.value.elts
        else:
            args = [node.value] if node.value else []
        call = ast.copy_location(ast_call("yield_", args), node)
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
            (
                len(last_statement.targets[0].elts)
                if isinstance(last_statement, ast.Assign)
                and isinstance(last_statement.targets[0], ast.Tuple)
                else 0
            ),
        )
        results = [ast_call("placeholder_opaque_t") for _ in range(num_results)]
        results = ast.fix_missing_locations(
            ast.copy_location(ast.Tuple(results, ctx=ast.Load()), test)
        )

        if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Store())
        withitem = ast.withitem(
            context_expr=ast_call("if_ctx_manager", args=[test, results]),
            optional_vars=if_op_name,
        )
        then_with = ast.With(items=[withitem])
        then_with = ast.copy_location(then_with, updated_node)
        then_with = ast.fix_missing_locations(then_with)
        then_with.body = updated_node.body

        if updated_node.orelse:
            if_op_name = ast.Name(f"__if_op__{updated_node.lineno}", ctx=ast.Load())
            withitem = ast.withitem(
                context_expr=ast_call("else_ctx_manager", args=[if_op_name]),
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


def _reject_nested_control_flow(body_stmts, where):
    """The loop transforms lift the body into a nested function the if/else
    transformers do not revisit, so control flow nested inside a loop body is
    not lowered. Detect and refuse rather than miscompile."""
    for stmt in body_stmts:
        for child in ast.walk(stmt):
            if isinstance(child, (ast.If, ast.For, ast.While)):
                raise NotImplementedError(
                    f"control flow nested inside a `{where}` loop body is not "
                    "supported"
                )


def _carried_from_yield(yield_value):
    """Names carried by a trailing `yield a, b` (or `yield a`)."""
    if yield_value is None:
        return []
    if isinstance(yield_value, ast.Tuple):
        elts = yield_value.elts
    else:
        elts = [yield_value]
    names = []
    for e in elts:
        if not isinstance(e, ast.Name):
            raise NotImplementedError(
                "loop yield must list plain loop-carried variable names, "
                f"got {ast.dump(e)}"
            )
        names.append(e.id)
    return names


class WhileToWhileLoop(StrictTransformer):
    """Rewrite `while COND: BODY; yield carried` into a while_loop call.

        while COND:
            BODY
            yield acc, i
        # ->
        def __wcond_L__(acc, i): return COND
        def __wbody_L__(acc, i):
            BODY
            return (acc, i)
        (acc, i) = while_loop(__wcond_L__, __wbody_L__, (acc, i))

    The loop-carried variables (from the trailing yield) become the parameters
    and results of the nested cond/body functions, so the while_loop runtime can
    feed header phis in and thread body results back as phi incomings without
    rebinding Python closure variables. Straight-line loop bodies only: control
    flow nested inside a loop body is not lowered (the nested functions are not
    revisited by the if/else transformers).
    """

    def visit_While(self, node: ast.While) -> list:
        node = self.generic_visit(node)
        line = node.lineno
        last = node.body[-1]
        if not (isinstance(last, ast.Expr) and isinstance(last.value, ast.Yield)):
            raise NotImplementedError(
                "a DSL `while` body must end with `yield <loop-carried vars>`"
            )
        carried = _carried_from_yield(last.value.value)
        body_stmts = node.body[:-1]
        _reject_nested_control_flow(body_stmts, "while")

        def params():
            return ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=n) for n in carried],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            )

        carried_load = ast.Tuple(
            [ast.Name(n, ast.Load()) for n in carried], ast.Load()
        )
        carried_store = ast.Tuple(
            [ast.Name(n, ast.Store()) for n in carried], ast.Store()
        )

        cond_name = f"__wcond_{line}__"
        body_name = f"__wbody_{line}__"

        cond_fn = ast.FunctionDef(
            name=cond_name,
            args=params(),
            body=[ast.Return(node.test)],
            decorator_list=[],
            type_params=[],
        )
        body_fn = ast.FunctionDef(
            name=body_name,
            args=params(),
            body=list(body_stmts) + [ast.Return(carried_load)],
            decorator_list=[],
            type_params=[],
        )
        call = ast.Assign(
            targets=[carried_store],
            value=ast.Call(
                func=ast.Name("while_loop", ast.Load()),
                args=[
                    ast.Name(cond_name, ast.Load()),
                    ast.Name(body_name, ast.Load()),
                    ast.Tuple(
                        [ast.Name(n, ast.Load()) for n in carried], ast.Load()
                    ),
                ],
                keywords=[],
            ),
        )
        out = [cond_fn, body_fn, call]
        for n in out:
            ast.copy_location(n, node)
            ast.fix_missing_locations(n)
        return out


class ForToForLoop(StrictTransformer):
    """Rewrite `for i in range_(...): BODY; yield carried` into a for_loop call.

        for i in range_(start, stop, step):
            BODY
            yield acc
        # ->
        def __fbody_L__(i, acc):
            BODY
            return (acc,)
        (acc,) = for_loop(start, stop, step, __fbody_L__, (acc,))

    Like WhileToWhileLoop, the loop-carried variables come from the trailing
    yield and become the body function's parameters/results (after the induction
    variable `i`), so no closure rebinding is needed. Only `range_(...)` iterables
    are handled; other `for` iterables are left untouched.
    """

    def visit_For(self, node: ast.For) -> object:
        node = self.generic_visit(node)
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range_"
        ):
            return node
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("for target must be a single name")
        iv = node.target.id
        line = node.lineno

        args = list(it.args)
        if len(args) == 1:
            start = ast.Constant(0)
            stop = args[0]
            step = ast.Constant(1)
        elif len(args) == 2:
            start, stop = args
            step = ast.Constant(1)
        elif len(args) == 3:
            start, stop, step = args
        else:
            raise NotImplementedError("range_ takes 1-3 arguments")

        last = node.body[-1]
        if not (isinstance(last, ast.Expr) and isinstance(last.value, ast.Yield)):
            raise NotImplementedError(
                "a DSL `for` body must end with `yield <loop-carried vars>`"
            )
        carried = _carried_from_yield(last.value.value)
        body_stmts = node.body[:-1]
        _reject_nested_control_flow(body_stmts, "for")

        body_name = f"__fbody_{line}__"
        params = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=iv)] + [ast.arg(arg=n) for n in carried],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        carried_load = ast.Tuple(
            [ast.Name(n, ast.Load()) for n in carried], ast.Load()
        )
        carried_store = ast.Tuple(
            [ast.Name(n, ast.Store()) for n in carried], ast.Store()
        )
        body_fn = ast.FunctionDef(
            name=body_name,
            args=params,
            body=list(body_stmts) + [ast.Return(carried_load)],
            decorator_list=[],
            type_params=[],
        )
        call = ast.Assign(
            targets=[carried_store],
            value=ast.Call(
                func=ast.Name("for_loop", ast.Load()),
                args=[
                    start,
                    stop,
                    step,
                    ast.Name(body_name, ast.Load()),
                    ast.Tuple(
                        [ast.Name(n, ast.Load()) for n in carried], ast.Load()
                    ),
                ],
                keywords=[],
            ),
        )
        out = [body_fn, call]
        for n in out:
            ast.copy_location(n, node)
            ast.fix_missing_locations(n)
        return out


class RejectUnsupportedJumps(StrictTransformer):
    """Reject control flow the phi-based lowering does not model.

    break/continue and early `return` inside if/while/for would need edge
    duplication and predecessor bookkeeping the yield-protocol lowering does not
    do (and the loop transforms lift bodies into nested functions, where a bare
    return/break/continue would silently mean the wrong thing). Detect and
    refuse rather than emit wrong IR. Runs before the loop/if transformers, so
    the constructs are still in their original ast form.
    """

    def visit_Break(self, node):
        raise NotImplementedError(
            "`break` inside DSL control flow is not supported"
        )

    def visit_Continue(self, node):
        raise NotImplementedError(
            "`continue` inside DSL control flow is not supported"
        )

    def _reject_nested_return(self, node):
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                raise NotImplementedError(
                    "early `return` inside DSL control flow is not supported"
                )
        return self.generic_visit(node)

    visit_If = _reject_nested_return
    visit_While = _reject_nested_return
    visit_For = _reject_nested_return
