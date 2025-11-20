# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
import copy
from . import *
from bytecode import ConcreteBytecode
from ...ast.canonicalize import StrictTransformer, Canonicalizer, BytecodePatcher

class CanonicalizeFMA(StrictTransformer):
    def visit_AnnAssign(
        self, updated_node: ast.AnnAssign
    ) -> Union[ast.AnnAssign, ast.Assign]:
        updated_node: ast.AnnAssign = self.generic_visit(updated_node)
        target = copy.deepcopy(updated_node.target)
        target.ctx = ast.Load()
        arith_constant_func_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="arith_dialect", ctx=ast.Load()),
                attr="constant",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword("result", updated_node.annotation),
                ast.keyword("value", updated_node.value),
            ],
        )
        updated_node = ast.Assign(
            targets=[updated_node.target], value=arith_constant_func_call
        )
        updated_node = ast.fix_missing_locations(updated_node)
        return updated_node

    def visit_AugAssign(
        self, updated_node: ast.AugAssign
    ) -> Union[ast.AugAssign, ast.Assign]:
        updated_node: ast.AugAssign = self.generic_visit(updated_node)
        if (
            isinstance(updated_node.op, ast.Add)
            and isinstance(updated_node.value, ast.BinOp)
            and isinstance(updated_node.value.op, ast.Mult)
        ):
            target = copy.deepcopy(updated_node.target)
            target.ctx = ast.Load()
            math_fma_func_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="math_dialect", ctx=ast.Load()),
                    attr="fma",
                    ctx=ast.Load(),
                ),
                args=[
                    updated_node.value.left,
                    updated_node.value.right,
                    target,
                ],
                keywords=[],
            )
            updated_node = ast.Assign(
                targets=[updated_node.target], value=math_fma_func_call
            )
            updated_node = ast.fix_missing_locations(updated_node)

        return updated_node


class ArithPatchByteCode(BytecodePatcher):
    def patch_bytecode(self, code: ConcreteBytecode, f):
        # TODO(max): this is bad and should be in the closure rather than as a global
        from ....dialects import arith, math

        f.__globals__["math_dialect"] = math
        f.__globals__["arith_dialect"] = arith
        return code


class ArithCanonicalizer(Canonicalizer):
    cst_transformers = [CanonicalizeFMA]
    bytecode_patchers = [ArithPatchByteCode]


canonicalizer = ArithCanonicalizer()
