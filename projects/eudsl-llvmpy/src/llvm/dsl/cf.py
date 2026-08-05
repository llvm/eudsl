#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runtime for lowered control flow: real basic blocks and phi nodes.

The AST canonicalizer (llvm.ast.cf_transformers) rewrites

    if c:
        r = yield a + 1
    else:
        r = yield b
    return r

into

    with if_ctx_manager(c, (placeholder_opaque_t(),)) as __if_op__:
        r = yield_(a + 1)
    with else_ctx_manager(__if_op__):
        r = yield_(b)
    return r

The MLIR analogue uses regions whose results become block/region results. Here
we emit explicit LLVM blocks: `if_ctx_manager` creates then/merge blocks and a
provisional conditional branch (false edge -> merge); `else_ctx_manager`
repoints that false edge to a fresh else block; and the values passed to
`yield_` become phi nodes at the merge block. Because Python's `r = yield_(...)`
executes in the else branch last, the else-branch `yield_` is the one that
builds the phis (both incoming edges are then known) and returns them, so `r`
binds to the phi that `return r` uses.
"""

from contextlib import contextmanager

from ..ast.canonicalize import Canonicalizer, FunctionPatcher
from ..ast import cf_transformers as _T
from .casters import maybe_downcast
from .context import current_builder, current_function


def placeholder_opaque_t():
    # Marks a phi-result slot in the rewritten AST. The real value is the phi
    # built by yield_; the placeholder itself is never inspected here.
    return None


class _IfOp:
    """Bookkeeping for one lowered if/else."""

    def __init__(self, cond):
        b = current_builder()
        fn = current_function()
        self.builder = b
        self.cond = cond
        self.entry_block = b.insert_block
        self.then_block = fn.append_basic_block("if.then")
        self.merge_block = fn.append_basic_block("if.end")
        self.else_block = None
        # Provisional branch; else_ctx_manager repoints successor 1 (false edge).
        self.cond_br = b.cond_br(cond, self.then_block, self.merge_block)
        self.then_vals = None
        self.else_vals = None
        # The block that actually reaches merge on each edge. For a plain branch
        # this is then_block/else_block, but with a nested if in a branch the
        # real predecessor is whatever block is current when the branch's
        # trailing yield_ runs (e.g. an inner merge block) -- so capture it
        # there rather than assuming the branch's entry block.
        self.then_pred = None
        self.else_pred = None
        self.active = "then"

    def _terminate_current(self):
        b = self.builder
        pred = b.insert_block
        if b.insert_block.terminator is None:
            b.br(self.merge_block)
        return pred

    def record_and_maybe_phi(self, values):
        """Called by yield_. Record this branch's values; on the else branch
        (both edges known) build the phis and return them."""
        b = self.builder
        if self.active == "then":
            self.then_vals = list(values)
            self.then_pred = self._terminate_current()
            # No phis yet; the value bound here is overwritten by the else
            # branch's assignment (or unused for a side-effecting if). Match the
            # else-branch return shape (scalar for a single value).
            vals = list(values)
            if len(vals) == 1:
                return vals[0]
            return tuple(vals)
        # else branch
        self.else_vals = list(values)
        self.else_pred = self._terminate_current()
        b.set_insert_point(self.merge_block)
        phis = []
        for i, tv in enumerate(self.then_vals):
            phi = b.phi(tv.type, f"if.phi.{i}")
            phi.add_incoming(tv, self.then_pred)
            phi.add_incoming(self.else_vals[i], self.else_pred)
            phis.append(maybe_downcast(phi, current_function()))
        self.phis = phis
        if len(phis) == 1:
            return phis[0]
        return tuple(phis)


_if_stack = []


@contextmanager
def if_ctx_manager(cond, results=()):
    op = _IfOp(cond)
    _if_stack.append(op)
    op.builder.set_insert_point(op.then_block)
    try:
        yield op
    finally:
        # Each branch body is terminated by its trailing yield_ (the
        # canonicalizer guarantees one); do not terminate here or the merge
        # block would get a self-branch. Just leave the builder at merge for
        # whatever follows (a no-else if continues here; an else block enters
        # next and repositions).
        _if_stack.pop()
        op.builder.set_insert_point(op.merge_block)


@contextmanager
def else_ctx_manager(op):
    b = op.builder
    fn = current_function()
    op.else_block = fn.append_basic_block("if.else")
    # Repoint the entry conditional branch's false edge to the else block.
    op.cond_br.set_successor(1, op.else_block)
    op.active = "else"
    _if_stack.append(op)
    b.set_insert_point(op.else_block)
    try:
        yield op
    finally:
        # else body already terminated + phis built by its trailing yield_.
        _if_stack.pop()
        b.set_insert_point(op.merge_block)


def yield_(*values):
    if not _if_stack:
        return values[0] if len(values) == 1 else values
    return _if_stack[-1].record_and_maybe_phi(values)


class _InjectCFGlobals(FunctionPatcher):
    def patch_function(self, f):
        g = f.__globals__
        g["yield_"] = yield_
        g["if_ctx_manager"] = if_ctx_manager
        g["else_ctx_manager"] = else_ctx_manager
        g["placeholder_opaque_t"] = placeholder_opaque_t
        return f


class LLVMCanonicalizer(Canonicalizer):
    cst_transformers = [
        _T.CanonicalizeElIfs,
        _T.InsertEmptyYield,
        _T.ReplaceYieldWithLLVMYield,
        _T.ReplaceIfWithWith,
    ]
    function_patchers = [_InjectCFGlobals]
