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

Loops follow the same "keep the body inline, thread carried values as phis"
idea (see `_Loop`): `for i in range_(...)` / `while COND` become a `with`
statement over a loop object whose header phis carry the loop-carried values.
Because the body stays inline (not lifted into a function), control flow nested
inside a loop body -- and loops nested inside `if` branches -- lower normally.
"""

from contextlib import contextmanager

from ..eudslllvm_ext.ir import Value, const_int
from ..ast.canonicalize import Canonicalizer, FunctionPatcher
from ..ast import cf_transformers as _T
from .casters import maybe_downcast
from ..eudslllvm_ext.ir import current_builder, current_function


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
        # A branch body never self-terminates (its trailing yield_ is what calls
        # this), so the current block always needs the branch to merge.
        b = self.builder
        pred = b.insert_block
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
    # Only ever runs inside if_ctx_manager/else_ctx_manager (the if transform
    # rewrites branch yields to yield_(); loop yields are consumed by the loop
    # transforms), so an if-op is always on the stack.
    return _if_stack[-1].record_and_maybe_phi(values)


class _InjectCFGlobals(FunctionPatcher):
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


class LLVMCanonicalizer(Canonicalizer):
    cst_transformers = [
        # Reject unsupported jumps first, on the original AST: break/continue and
        # early `return` inside control flow are still not modeled.
        _T.RejectUnsupportedJumps,
        # Loops next, so a loop's trailing `yield` is consumed as its carried
        # values (rewritten to loop_yield) before the if/else passes turn any
        # remaining yields into scf-style yield_(). The loop body stays inline
        # (in a `with` block), so control flow nested in it is lowered by the
        # passes below and any nested loop by these two passes.
        _T.ForToInline,
        _T.WhileToInline,
        _T.CanonicalizeElIfs,
        _T.InsertEmptyYield,
        _T.ReplaceYieldWithLLVMYield,
        _T.ReplaceIfWithWith,
    ]
    function_patchers = [_InjectCFGlobals]


def _as_value(x, like):
    """Coerce a Python int to a constant of `like`'s type; pass Values through."""
    if isinstance(x, Value):
        return x
    return const_int(like.type, int(x), signed=True)


_loop_stack = []


class _Loop:
    """A lowered for/while loop: preheader -> header(phis) -> body -> exit.

    Built on ``__enter__``, which appends the blocks, seeds the header phis from
    the preheader, emits the entry condition, and leaves the persistent builder
    inside the body block. ``__exit__`` adds the back-edge phi incomings from
    whatever block the body actually ended in (so nested control flow that moved
    the builder is handled, just like `_IfOp`), then parks the builder at the
    exit block. The loop-carried values are the header phis; they are exposed as
    ``.results`` (valid at the exit block) so post-loop code binds to the loop
    result rather than the body's last recomputed value.

    The body is left inline in the caller's frame (the AST transform wraps it in
    ``with``), so `if`/`while`/`for` nested in a loop body lower normally.
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
        b = current_builder()
        fn = current_function()
        self.builder = b
        preheader = b.insert_block
        self.header = fn.append_basic_block(f"{self.kind}.header")
        body = fn.append_basic_block(f"{self.kind}.body")
        self.exit_block = fn.append_basic_block(f"{self.kind}.end")
        b.br(self.header)

        b.set_insert_point(self.header)
        # Coerce any Python-int bounds/carried inits to constants. The type
        # witness is the first Value among the bounds and carried inits; a loop
        # with an int to coerce but no Value anywhere has no type to infer.
        witness_candidates = list(self.iter_args)
        if self.kind == "for":
            witness_candidates = [self.start, self.stop] + witness_candidates
        witness = next((c for c in witness_candidates if isinstance(c, Value)), None)

        def coerce(x):
            if witness is None and not isinstance(x, Value):
                raise NotImplementedError(
                    "a DSL loop needs at least one Value bound or loop-carried "
                    "value to infer the IR type"
                )
            return _as_value(x, witness)

        self.iter_args = [coerce(a) for a in self.iter_args]

        self.iv_phi = None
        iv_typed = None
        if self.kind == "for":
            start = coerce(self.start)
            stop = coerce(self.stop)
            self.stop = stop
            self.iv_phi = b.phi(start.type, "for.iv")
            self.iv_phi.add_incoming(start, preheader)
            iv_typed = maybe_downcast(self.iv_phi, fn)
        self.carried_phis = []
        for k, a in enumerate(self.iter_args):
            p = b.phi(a.type, f"{self.kind}.{k}")
            p.add_incoming(a, preheader)
            self.carried_phis.append(p)
        carried_typed = tuple(maybe_downcast(p, fn) for p in self.carried_phis)
        self._iv_typed = iv_typed

        if self.kind == "for":
            descending = isinstance(self.step, int) and self.step < 0
            cond = iv_typed > self.stop if descending else iv_typed < self.stop
        else:
            cond = self.cond_fn(*carried_typed)
        b.cond_br(cond, body, self.exit_block)

        b.set_insert_point(body)
        _loop_stack.append(self)
        if self.kind == "for":
            return (iv_typed, carried_typed) if carried_typed else iv_typed
        return carried_typed

    def __exit__(self, exc_type, exc, tb):
        _loop_stack.pop()
        if exc_type is not None:
            return False
        b = self.builder
        body_end = b.insert_block  # nested control flow may have moved us
        next_iv = self._iv_typed + self.step if self.kind == "for" else None
        b.br(self.header)
        if self.kind == "for":
            self.iv_phi.add_incoming(next_iv, body_end)
        for phi, nxt in zip(self.carried_phis, self.next_carried):
            phi.add_incoming(nxt, body_end)
        b.set_insert_point(self.exit_block)
        # The header phis hold the final carried values on the exit edge and the
        # header dominates the exit block, so they are the loop's live-out result.
        self.results = tuple(
            maybe_downcast(p, current_function()) for p in self.carried_phis
        )
        return False


def loop_yield(*values):
    """Record this iteration's updated carried values (the loop analogue of
    yield_). Read by `_Loop.__exit__` to wire the back-edge phi incomings."""
    _loop_stack[-1].next_carried = list(values)


def range_(start, stop=None, step=1, *, iter_args=()):
    """`for i in range_(...)` loop builder. The AST transform supplies
    `iter_args` (the loop-carried values) and iterates this via `with`."""
    if stop is None:
        start, stop = 0, start
    return _Loop("for", start=start, stop=stop, step=step, iter_args=iter_args)


def while_(cond_fn, *, iter_args=()):
    """`while COND` loop builder. The AST transform lifts COND into `cond_fn`
    (evaluated in the header against the carried phis) and supplies iter_args."""
    return _Loop("while", cond_fn=cond_fn, iter_args=iter_args)
