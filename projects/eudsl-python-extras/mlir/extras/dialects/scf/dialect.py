# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import List, Union, Optional, Sequence

from ..arith import constant as _ext_arith_constant, index_cast
from ...meta import region_op
from ...util import get_user_code_loc, region_adder
from ....dialects._ods_common import (
    _cext,
    get_default_loc_context,
    get_op_result_or_op_results,
)
from ....dialects.linalg.opdsl.lang.emitter import _is_index_type

# gotta come first
from ....dialects.scf import *
from ....dialects.scf import _Dialect, yield_ as yield__
from ....ir import (
    Attribute,
    IndexType,
    InsertionPoint,
    OpResultList,
    OpView,
    OpaqueType,
    Operation,
    Value,
    _denseI64ArrayAttr,
)

logger = logging.getLogger(__name__)

opaque = lambda dialect_namespace, buffer: OpaqueType.get(dialect_namespace, buffer)


def canonicalize_start_stop_step(start, stop, step):
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    params = [start, stop, step]
    type = IndexType.get()
    maybe_types = {p.type for p in params if isinstance(p, Value)}
    if maybe_types:
        if len(maybe_types) > 1:
            raise ValueError(
                f"all {start=} and {stop=} and {step=} ir.Value objects must have the same type"
            )
        type = maybe_types.pop()

    for i, p in enumerate(params):
        if isinstance(p, int):
            p = _ext_arith_constant(p, type=type)
        assert isinstance(p, Value)
        params[i] = p

    return params[0], params[1], params[2]


def _build_for(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    start, stop, step = canonicalize_start_stop_step(start, stop, step)
    return ForOp(start, stop, step, iter_args, loc=loc, ip=ip)


def range_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    for_op = _build_for(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = for_op.induction_variable
    iter_args = tuple(for_op.inner_iter_args)
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args, for_op.results
        elif len(iter_args) == 1:
            yield iv, iter_args[0], for_op.results[0]
        else:
            yield iv


def placeholder_opaque_t():
    return opaque("scf", "placeholder")


for__ = region_op(_build_for, terminator=yield__)


def _parfor(op_ctor):
    def _base(
        lower_bounds, upper_bounds=None, steps=None, *, loc=None, ip=None, **kwargs
    ):
        if upper_bounds is None:
            upper_bounds = lower_bounds
            lower_bounds = [0] * len(upper_bounds)
        if steps is None:
            steps = [1] * len(lower_bounds)

        params = [list(lower_bounds), list(upper_bounds), list(steps)]
        for i, p in enumerate(params):
            for j, pp in enumerate(p):
                if isinstance(pp, int):
                    pp = _ext_arith_constant(pp, index=True)
                if not _is_index_type(pp.type):
                    pp = index_cast(pp)
                p[j] = pp
            params[i] = p

        return op_ctor(*params, loc=loc, ip=ip, **kwargs)

    return _base


@region_op
def in_parallel():
    return InParallelOp()


def in_parallel_(parallel_insert_slice=None):
    if isinstance(parallel_insert_slice, (tuple, list)):
        assert (
            len(parallel_insert_slice) <= 1
        ), "expected at most one parallel_insert_slice op"
        if len(parallel_insert_slice) == 1:
            parallel_insert_slice = parallel_insert_slice[0]
        else:
            parallel_insert_slice = None

    @in_parallel
    def foo():
        if parallel_insert_slice is not None:
            parallel_insert_slice()
        return


def parallel_insert_slice(
    source,
    dest,
    static_offsets=None,
    static_sizes=None,
    static_strides=None,
    offsets=None,
    sizes=None,
    strides=None,
):
    from . import tensor

    @in_parallel
    def foo():
        tensor.parallel_insert_slice(
            source,
            dest,
            offsets,
            sizes,
            strides,
            static_offsets,
            static_sizes,
            static_strides,
        )


forall_ = region_op(_parfor(ForallOp), terminator=in_parallel_)


def _parfor_context_manager(op_ctor):
    def _base(*args, **kwargs):
        for_op = _parfor(op_ctor)(*args, **kwargs)
        block = for_op.regions[0].blocks[0]
        block_args = tuple(block.arguments)
        with InsertionPoint(block):
            yield block_args

    return _base


forall = _parfor_context_manager(ForallOp)


class ParallelOp(ParallelOp):
    def __init__(
        self,
        lower_bounds,
        upper_bounds,
        steps,
        *,
        inits: Optional[Union[Operation, OpView, Sequence[Value]]] = None,
        loc=None,
        ip=None,
    ):
        assert len(lower_bounds) == len(upper_bounds) == len(steps)
        if inits is None:
            inits = []
        results = [i.type for i in inits]
        iv_types = [IndexType.get()] * len(lower_bounds)
        super().__init__(
            results,
            lower_bounds,
            upper_bounds,
            steps,
            inits,
            loc=loc,
            ip=ip,
        )
        self.regions[0].blocks.append(*iv_types)

    @property
    def body(self):
        return self.regions[0].blocks[0]

    @property
    def induction_variables(self):
        return self.body.arguments


def _parallel_terminator(xs):
    if len(xs):
        raise ValueError(
            "default return->parallel terminator does not support operands; use scf.reduce_ instead"
        )
    return reduce_()


parallel_ = region_op(_parfor(ParallelOp), terminator=_parallel_terminator)
parallel = _parfor_context_manager(ParallelOp)


def while___(cond: Value, *, loc=None, ip=None):
    def wrapper():
        nonlocal ip
        inits = list(cond.owner.operands)
        results_ = [i.type for i in inits]
        while_op = WhileOp(results_, inits, loc=loc, ip=ip)
        while_op.regions[0].blocks.append(*[i.type for i in inits])
        before = while_op.regions[0].blocks[0]
        while_op.regions[1].blocks.append(*[i.type for i in inits])
        after = while_op.regions[1].blocks[0]
        with InsertionPoint(before) as ip:
            cond_ = condition(cond, list(before.arguments))
            cond.owner.move_before(cond_)
        with InsertionPoint(after):
            yield inits

    if hasattr(while___, "wrapper"):
        # needed in order to exit the `after` insertion point
        next(while___.wrapper, False)
        del while___.wrapper
        return False
    else:
        while___.wrapper = wrapper()
        # enter `after` insertion point
        return next(while___.wrapper)


def while__(cond: Value, *, loc=None, ip=None):
    yield while___(cond, loc=loc, ip=ip)
    yield while___(cond, loc=loc, ip=ip)


class ReduceOp(ReduceOp):
    def __init__(self, operands, num_reductions, *, loc=None, ip=None):
        super().__init__(operands, num_reductions, loc=loc, ip=ip)
        for i in range(num_reductions):
            self.regions[i].blocks.append(operands[i].type, operands[i].type)


def reduce_(*operands, loc=None, ip=None):
    return ReduceOp(operands, len(operands), loc=loc, ip=ip)


reduce = region_op(reduce_, terminator=lambda xs: reduce_return(*xs))


@region_adder(terminator=lambda xs: reduce_return(*xs))
def another_reduce(reduce_op):
    for r in reduce_op.regions:
        if len(r.blocks[0].operations) == 0:
            return r


def yield_(*args, results_=None):
    if len(args):
        assert results_ is None, "must provide results_ or args"
    if results_ is not None:
        args = results_
    if len(args) == 1 and isinstance(args[0], (list, OpResultList)):
        args = list(args[0])
    y = yield__(args)
    parent_op = y.operation.parent.opview
    if len(parent_op.results):
        results = get_op_result_or_op_results(parent_op)
        assert (
            isinstance(results, (OpResultList, Value))
            or isinstance(results, list)
            and all(isinstance(r, Value) for r in results)
        ), f"api has changed: {results=}"
        if isinstance(results, Value):
            results = [results]
        unpacked_args = args
        if any(isinstance(a, OpResultList) for a in unpacked_args):
            assert len(unpacked_args) == 1
            unpacked_args = list(unpacked_args[0])

        for i, r in enumerate(results):
            if r.type == placeholder_opaque_t():
                r.set_type(unpacked_args[i].type)

        if len(results) > 1:
            return results
        return results[0]
    elif len(args):
        raise RuntimeError(f"can't yield from parent_op which has no results")
    return None


def _if(cond, results=None, *, has_else=False, loc=None, ip=None):
    if results is None:
        results = []
    if results:
        has_else = True
    return IfOp(cond, results, hasElse=has_else, loc=loc, ip=ip)


if_ = region_op(_if, terminator=yield__)


@contextmanager
def if_ctx_manager(cond, results=None, *, has_else=False, loc=None, ip=None):
    if_op = _if(cond, results, has_else=has_else, loc=loc, ip=ip)
    with InsertionPoint(if_op.regions[0].blocks[0]):
        yield if_op


@contextmanager
def else_ctx_manager(if_op):
    if len(if_op.regions[1].blocks) == 0:
        if_op.regions[1].blocks.append(*[])
    with InsertionPoint(if_op.regions[1].blocks[0]):
        yield


@region_adder(terminator=yield__)
def else_(ifop):
    return ifop.regions[1]


execute_region = region_op(ExecuteRegionOp)
