#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generic-MIR runtime for lowered if/else: MachineBasicBlocks and G_PHI.

The MIR analogue of llvm.dsl.cf's if/else lowering. The same AST canonicalizer
rewrites

    if c:
        r = yield a + 1
    else:
        r = yield b
    return r

into if_ctx_manager/else_ctx_manager/yield_ (see llvm.ast.cf_transformers); this
module emits the blocks and phis. Since MIR's conditional branch (G_BRCOND) has
a single target, the entry block gets `G_BRCOND cond -> then` plus a fall-through
`G_BR -> false-target` whose target (initially the merge block) is repointed to
the else block by else_ctx_manager -- the MIR equivalent of the IR builder's
cond_br.set_successor. The values passed to yield_ become the merge block's
G_PHIs, built on the else branch (when both incoming edges are known), so
`r = yield_(...)` binds to the phi that `return r` uses.

Only if/else is lowered here: a for/while in a @machine_function body is
rejected (MIRCanonicalizer installs no loop transformers).
"""

from contextlib import contextmanager

from ..ast import cf_transformers as _T
from ..ast.canonicalize import Canonicalizer, FunctionPatcher
from .machine import MachineValue, current_machine_builder


def placeholder_opaque_t():
    # Marks a phi-result slot in the rewritten AST; the real value is the phi
    # built by yield_. Never inspected here.
    return None


class _MIRIfOp:
    """Bookkeeping for one lowered if/else, emitting MIR blocks and G_PHIs."""

    def __init__(self, cond):
        b = current_machine_builder()
        mf = b.machine_function
        self.builder = b
        self.entry_block = b.insert_block
        self.then_block = mf.create_block()
        self.merge_block = mf.create_block()
        self.else_block = None
        # G_BRCOND cond -> then, then a fall-through G_BR whose target starts at
        # merge; else_ctx_manager repoints it to the else block.
        b.build_brcond(cond.reg, self.then_block)
        self.false_br = b.build_br(self.merge_block)
        self.entry_block.add_successor(self.then_block)
        self.entry_block.add_successor(self.merge_block)
        self.then_vals = None
        self.else_vals = None
        self.then_pred = None
        self.else_pred = None
        self.active = "then"

    def _terminate_current(self):
        # A branch body never self-terminates (its trailing yield_ calls this),
        # so the current block always branches to merge. Capture the real
        # predecessor (b.insert_block), which nested control flow may have moved.
        b = self.builder
        pred = b.insert_block
        b.build_br(self.merge_block)
        pred.add_successor(self.merge_block)
        return pred

    def _repoint_false_edge(self, new_target):
        """Atomically move the entry's false edge (CFG successor + the G_BR
        terminator's target) from the merge block to `new_target`. MIR tracks
        the successor list and the terminator operand separately, so both must
        change together or the CFG and terminators silently disagree."""
        self.entry_block.replace_successor(self.merge_block, new_target)
        self.false_br.set_branch_target(new_target)

    def record_and_maybe_phi(self, values):
        """Called by yield_. Record this branch's values; on the else branch
        (both edges known) build the G_PHIs and return them."""
        b = self.builder
        if self.active == "then":
            self.then_vals = list(values)
            self.then_pred = self._terminate_current()
            vals = list(values)
            return vals[0] if len(vals) == 1 else tuple(vals)
        # else branch: both edges known, build the phis at the merge block.
        self.else_vals = list(values)
        self.else_pred = self._terminate_current()
        if len(self.then_vals) != len(self.else_vals):
            raise ValueError(
                f"then/else branches yield different numbers of values "
                f"({len(self.then_vals)} vs {len(self.else_vals)})"
            )
        b.set_block(self.merge_block)
        phis = []
        for tv, ev in zip(self.then_vals, self.else_vals):
            if tv.llt != ev.llt:
                raise TypeError(
                    f"then/else values have mismatched types: {tv.llt} and " f"{ev.llt}"
                )
            reg = b.build_phi(
                tv.llt, [(tv.reg, self.then_pred), (ev.reg, self.else_pred)]
            )
            phis.append(MachineValue(reg, tv.llt))
        return phis[0] if len(phis) == 1 else tuple(phis)


_if_stack = []


@contextmanager
def if_ctx_manager(cond, results=()):
    op = _MIRIfOp(cond)
    _if_stack.append(op)
    op.builder.set_block(op.then_block)
    try:
        yield op
    finally:
        _if_stack.pop()
        op.builder.set_block(op.merge_block)


@contextmanager
def else_ctx_manager(op):
    b = op.builder
    op.else_block = b.machine_function.create_block()
    # Repoint the entry's fall-through branch (and CFG edge) merge -> else.
    op._repoint_false_edge(op.else_block)
    op.active = "else"
    _if_stack.append(op)
    b.set_block(op.else_block)
    try:
        yield op
    finally:
        _if_stack.pop()
        b.set_block(op.merge_block)


def yield_(*values):
    # _InjectMIRCFGlobals injects yield_ into the traced function's globals, so
    # a stray yield_ outside an if body is reachable; fail with a clear message
    # rather than a bare IndexError.
    if not _if_stack:
        raise RuntimeError(
            "yield_ used outside a lowered if/else body (no active if op)"
        )
    return _if_stack[-1].record_and_maybe_phi(values)


class _InjectMIRCFGlobals(FunctionPatcher):
    def patch_function(self, f):
        g = f.__globals__
        g["yield_"] = yield_
        g["if_ctx_manager"] = if_ctx_manager
        g["else_ctx_manager"] = else_ctx_manager
        g["placeholder_opaque_t"] = placeholder_opaque_t
        return f


class _RejectLoops(_T.StrictTransformer):
    """This runtime lowers only if/else; a for/while would otherwise fall
    through the canonicalizer untouched and silently unroll (a Python `range`)
    or hang the trace (a truthy MachineValue condition). Reject it loudly."""

    def visit_For(self, node):
        raise NotImplementedError(
            "`for` loops in a @machine_function body are not supported"
        )

    def visit_While(self, node):
        raise NotImplementedError(
            "`while` loops in a @machine_function body are not supported"
        )


class MIRCanonicalizer(Canonicalizer):
    # if/else lowering; mirrors LLVMCanonicalizer's if/else passes so the
    # rewritten AST shape is identical. _RejectLoops turns an unsupported
    # for/while into a clear error instead of silently wrong MIR.
    cst_transformers = [
        _T.RejectUnsupportedJumps,
        _RejectLoops,
        _T.CanonicalizeElIfs,
        _T.InsertEmptyYield,
        _T.ReplaceYieldWithLLVMYield,
        _T.ReplaceIfWithWith,
    ]
    function_patchers = [_InjectMIRCFGlobals]
