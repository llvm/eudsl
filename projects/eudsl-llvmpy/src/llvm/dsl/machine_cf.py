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

Loops (for/while) lower the same way as llvm.dsl.cf._Loop: an inline
preheader -> header(phis) -> body -> exit shape whose header G_PHIs carry the
induction variable and loop-carried values.
"""

from contextlib import contextmanager

from ..ast import cf_transformers as _T
from ..ast.canonicalize import Canonicalizer, FunctionPatcher
from ..eudslllvm_ext.mir import LLT
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


def _as_mv(x, witness):
    """Coerce a Python int to a G_CONSTANT MachineValue of `witness`'s type;
    pass a MachineValue through."""
    if isinstance(x, MachineValue):
        return x
    reg = current_machine_builder().build_constant(witness.llt, int(x))
    return MachineValue(reg, witness.llt)


_loop_stack = []


class _MIRLoop:
    """A lowered for/while loop: preheader -> header(phis) -> body -> exit.

    Mirrors llvm.dsl.cf._Loop with MIR blocks. The header's induction-variable
    and loop-carried G_PHIs are built empty at the header top on __enter__ (so
    they precede the loop condition) and seeded with the preheader incoming;
    __exit__ adds their back-edge incoming from whatever block the body ended in
    (nested control flow may have moved the builder) and parks the builder at the
    exit block. The loop-carried phis (not the induction phi) are the live-outs,
    exposed as `.results`. The body stays inline, so control flow nested in it
    lowers normally.
    """

    def __init__(
        self, kind, *, start=None, stop=None, step=None, cond_fn=None, iter_args=()
    ):
        self.kind = kind
        self.start = start
        self.stop = stop
        self.step = step
        self.cond_fn = cond_fn
        self.iter_args = list(iter_args)
        self.next_carried = []
        self.results = ()

    def __enter__(self):
        b = current_machine_builder()
        mf = b.machine_function
        self.builder = b
        preheader = b.insert_block
        self.header = mf.create_block()
        body = mf.create_block()
        self.exit_block = mf.create_block()
        b.build_br(self.header)
        preheader.add_successor(self.header)

        b.set_block(self.header)
        # The type witness is the first MachineValue among the bounds/carried
        # inits; a loop with an int to coerce but no MachineValue has no LLT.
        witness_candidates = list(self.iter_args)
        if self.kind == "for":
            witness_candidates = [self.start, self.stop] + witness_candidates
        witness = next(
            (c for c in witness_candidates if isinstance(c, MachineValue)), None
        )

        def coerce(x):
            if witness is None and not isinstance(x, MachineValue):
                raise NotImplementedError(
                    "a DSL loop needs at least one MachineValue bound or "
                    "loop-carried value to infer the LLT"
                )
            return _as_mv(x, witness)

        self.iter_args = [coerce(a) for a in self.iter_args]

        self.iv_phi = None
        iv_val = None
        if self.kind == "for":
            start = coerce(self.start)
            self.stop = coerce(self.stop)
            self.iv_phi = b.build_empty_phi(start.llt)
            self.iv_phi.add_phi_incoming(start.reg, preheader)
            iv_val = MachineValue(self.iv_phi.operand(0).reg, start.llt)
        self.carried_phis = []
        carried_typed = []
        for a in self.iter_args:
            phi = b.build_empty_phi(a.llt)
            phi.add_phi_incoming(a.reg, preheader)
            self.carried_phis.append(phi)
            carried_typed.append(MachineValue(phi.operand(0).reg, a.llt))
        carried_typed = tuple(carried_typed)
        self._iv_val = iv_val

        if self.kind == "for":
            # step is validated to be a nonzero int by range_(), so its sign
            # reliably selects the compare direction.
            descending = self.step < 0
            cond = iv_val > self.stop if descending else iv_val < self.stop
        else:
            cond = self.cond_fn(*carried_typed)
            if not isinstance(cond, MachineValue) or cond.llt != LLT.scalar(1):
                raise TypeError(
                    "while condition must evaluate to an i1 MachineValue "
                    "(e.g. a comparison like `a < b`)"
                )
        b.build_brcond(cond.reg, body)
        b.build_br(self.exit_block)
        self.header.add_successor(body)
        self.header.add_successor(self.exit_block)

        b.set_block(body)
        _loop_stack.append(self)
        if self.kind == "for":
            return (iv_val, carried_typed) if carried_typed else iv_val
        return carried_typed

    def __exit__(self, exc_type, exc, tb):
        _loop_stack.pop()
        if exc_type is not None:
            return False
        b = self.builder
        body_end = b.insert_block  # nested control flow may have moved us
        next_iv = self._iv_val + self.step if self.kind == "for" else None
        b.build_br(self.header)
        body_end.add_successor(self.header)
        if self.kind == "for":
            self.iv_phi.add_phi_incoming(next_iv.reg, body_end)
        if len(self.next_carried) != len(self.carried_phis):
            raise ValueError(
                f"loop body yielded {len(self.next_carried)} carried value(s) "
                f"but the loop carries {len(self.carried_phis)}"
            )
        for phi, nxt in zip(self.carried_phis, self.next_carried):
            phi.add_phi_incoming(nxt.reg, body_end)
        b.set_block(self.exit_block)
        self.results = tuple(
            MachineValue(phi.operand(0).reg, arg.llt)
            for phi, arg in zip(self.carried_phis, self.iter_args)
        )
        return False


def loop_yield(*values):
    """Record this iteration's updated carried values (the loop analogue of
    yield_), read by _MIRLoop.__exit__ to wire the back-edge phi incomings."""
    if not _loop_stack:
        raise RuntimeError("loop_yield used outside a loop body (no active loop)")
    _loop_stack[-1].next_carried = list(values)


def range_(start, stop=None, step=1, *, iter_args=()):
    """`for i in range_(...)` loop builder; the AST transform supplies iter_args
    and drives this via `with`."""
    if stop is None:
        start, stop = 0, start
    # The step's sign selects the compare direction at trace time, so it must be
    # a known int -- a MachineValue/float step can't pick a direction (and a
    # float would silently truncate). A zero step never advances (infinite loop).
    if not isinstance(step, int):
        raise TypeError("range_ step must be an int")
    if step == 0:
        raise ValueError("range_ step must not be zero")
    return _MIRLoop("for", start=start, stop=stop, step=step, iter_args=iter_args)


def while_(cond_fn, *, iter_args=()):
    """`while COND` loop builder; the AST transform lifts COND into cond_fn
    (evaluated in the header against the carried phis)."""
    return _MIRLoop("while", cond_fn=cond_fn, iter_args=iter_args)


class _InjectMIRCFGlobals(FunctionPatcher):
    def patch_function(self, f):
        g = f.__globals__
        g["yield_"] = yield_
        g["if_ctx_manager"] = if_ctx_manager
        g["else_ctx_manager"] = else_ctx_manager
        g["placeholder_opaque_t"] = placeholder_opaque_t
        g["loop_yield"] = loop_yield
        g["range_"] = range_
        g["while_"] = while_
        return f


class MIRCanonicalizer(Canonicalizer):
    # Mirrors LLVMCanonicalizer's pass order: loops first (a loop's trailing
    # yield becomes its carried values before the if/else passes turn remaining
    # yields into yield_()), then the if/else lowering.
    cst_transformers = [
        _T.RejectUnsupportedJumps,
        _T.ForToInline,
        _T.WhileToInline,
        _T.CanonicalizeElIfs,
        _T.InsertEmptyYield,
        _T.ReplaceYieldWithLLVMYield,
        _T.ReplaceIfWithWith,
    ]
    function_patchers = [_InjectMIRCFGlobals]
