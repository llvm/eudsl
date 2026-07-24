# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ast
import copy
import inspect
from typing import List, Optional, Tuple, Union, Sequence

# noinspection PyUnresolvedReferences
import numpy as np

from ._shaped_value import (
    ShapedValue,
    _indices_to_indexer,
    _is_scalar,
    _is_int_arraylike,
)
from .arith import ArithValue, ScalarValue, constant
from .. import types as T
from ..ast.canonicalize import Canonicalizer, FunctionPatcher, StrictTransformer
from ..ast.util import ast_call
from ..util import (
    _unpack_sizes_element_type,
    _update_caller_vars,
    get_user_code_loc,
    mlir_type_to_np_dtype,
)
from ..._mlir_libs._mlir import register_value_caster
from ...dialects import tensor
from ...dialects._ods_common import _dispatch_mixed_values, get_op_result_or_op_results
from ...dialects.tensor import *
from ...dialects.transform.structured import _get_int_array_array_attr
from ...ir import RankedTensorType, ShapedType, Type, Value, IndexType

S = ShapedType.get_dynamic_size()


def empty(*sizes: Union[int, Value], element_type: Type = None, loc=None, ip=None):
    if element_type is None:
        sizes, element_type = _unpack_sizes_element_type(sizes)
    return get_op_result_or_op_results(
        tensor.EmptyOp(sizes, element_type, loc=loc, ip=ip)
    )


def extract_slice(
    source: "TensorValue",
    offsets: Optional[Sequence[Value]] = None,
    strides: Optional[Sequence[Value]] = None,
    static_offsets: Optional[Sequence[int]] = None,
    static_sizes: Optional[Sequence[int]] = None,
    static_strides: Optional[Sequence[int]] = None,
    *,
    loc=None,
    ip=None,
):
    if offsets is None:
        offsets = []
    if strides is None:
        strides = []
    assert static_sizes, f"this convenience method only handles static sizes"
    assert offsets or static_offsets and bool(offsets) != bool(static_offsets)
    assert strides or static_strides and bool(strides) != bool(static_strides)
    sizes = []
    result = T.tensor(*static_sizes, source.dtype)
    return tensor.extract_slice(
        result,
        source,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
        ip=ip,
    )


def insert_slice(
    source: Value,
    dest: Value,
    offsets: Optional[Sequence[Value]] = None,
    strides: Optional[Sequence[Value]] = None,
    static_offsets: Optional[Sequence[int]] = None,
    static_sizes: Optional[Sequence[int]] = None,
    static_strides: Optional[Sequence[int]] = None,
    *,
    loc=None,
    ip=None,
):
    if offsets is None:
        offsets = []
    if strides is None:
        strides = []
    assert static_sizes, f"this convenience method only handles static sizes"
    assert offsets or static_offsets and bool(offsets) != bool(static_offsets)
    assert strides or static_strides and bool(strides) != bool(static_strides)
    sizes = []
    return tensor.insert_slice(
        source,
        dest,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
        loc=loc,
        ip=ip,
    )


def _is_index_tensor(x):
    """Returns True if x is a TensorValue with index dtype, False otherwise."""
    return (
        isinstance(x, Value)
        and isinstance(x, TensorValue)
        and isinstance(x.dtype, IndexType)
    )


# TODO(max): unify vector/memref/tensor
@register_value_caster(RankedTensorType.static_typeid)
@ShapedValue
class TensorValue(ArithValue):
    def __getitem__(self, idx: tuple) -> "TensorValue":
        loc = get_user_code_loc()

        assert self.has_rank(), "only ranked tensor slicing/indexing supported"

        if idx is None:
            return expand_dims(self, (0,), loc=loc)
        if idx == Ellipsis or idx == slice(None):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) for i in idx):
            return self
        if isinstance(idx, tuple) and all(i == slice(None) or i is None for i in idx):
            nones = [i for i, n in enumerate(idx) if n is None]
            return expand_dims(self, nones, loc=loc)

        if isinstance(idx, Value) and not isinstance(idx, ScalarValue):
            raise ValueError("indexing by tensor is not currently supported")

        idx = list((idx,) if isinstance(idx, (int, ScalarValue, slice)) else idx)
        for i, d in enumerate(idx):
            if isinstance(d, int):
                idx[i] = constant(d, index=True, loc=loc)

        if all(isinstance(d, ScalarValue) for d in idx) and len(idx) == len(self.shape):
            return tensor.extract(self, idx, loc=loc)
        else:
            if any(_is_index_tensor(i) or _is_int_arraylike(i) for i in idx):
                raise ValueError("indexing by tensor is not currently supported")

            indexer = _indices_to_indexer(tuple(idx), self.shape)
            out = self

            if indexer.is_full():
                out = out
            elif indexer.is_constant():
                out = extract_slice(
                    out,
                    static_offsets=indexer.static_offsets(),
                    static_sizes=indexer.static_sizes(),
                    static_strides=indexer.static_strides(),
                    loc=loc,
                    ip=None,
                )
            else:
                raise ValueError(f"non-constant indices not supported {indexer}")

            # This adds newaxis/None dimensions.
            return expand_dims(out, indexer.newaxis_dims, loc=loc, ip=None)

    def __setitem__(self, idx, source):
        res = _setitem(self, idx, source)
        # early-return cases (full-slice/ellipsis) leave the tensor untouched, so
        # there's nothing to rebind in the caller.
        if res is self:
            return
        # Fallback for code that isn't rewritten by TensorCanonicalizer (e.g. an
        # undecorated function or the REPL): reach into the caller and rebind the
        # variable(s) that pointed at `self` to the new SSA value. Inside a
        # @canonicalize(using=tensor.canonicalizer) function this path isn't taken
        # because `t[idx] = source` is rewritten to `t = __mlir_extras_setitem__(...)`.
        previous_frame = inspect.currentframe().f_back
        _update_caller_vars(previous_frame, [self], [res])

    def coerce(
        self,
        other,
        *,
        loc=None,
        ip=None,
    ) -> Tuple["TensorValue", "TensorValue"]:
        if isinstance(other, np.ndarray):
            other = TensorValue(other)
            return other
        elif _is_scalar(other):
            if not self.has_static_shape():
                raise ValueError(
                    f"can't coerce {other=} because {self=} doesn't have static shape"
                )
            if isinstance(other, (int, float)):
                np_dtype = mlir_type_to_np_dtype(self.dtype)
                other = TensorValue(
                    np.full(self.shape, other, dtype=np_dtype),
                    dtype=self.dtype,
                    loc=loc,
                    ip=ip,
                )
                return other
            elif isinstance(other, ScalarValue):
                other = tensor.splat(
                    RankedTensorType.get(self.shape, other.dtype),
                    other,
                    [],
                    loc=loc,
                    ip=ip,
                )
                return other

        raise ValueError(f"can't coerce unknown {other=}")


def compute_result_shape_reassoc_list(inp_shape, newaxis_dims):
    newaxis_dims = sorted(newaxis_dims)
    if len(set(newaxis_dims)) != len(newaxis_dims):
        raise ValueError(f"repeated axis in expand_dims: {newaxis_dims}")

    ndim_out = len(inp_shape) + len(newaxis_dims)
    if not all(0 <= d < ndim_out for d in newaxis_dims):
        raise ValueError("no negative dims allowed")
    result_shape = list(inp_shape)
    for i in reversed(newaxis_dims):
        result_shape.insert(i, 1)
    reassoc_list = [[i] for i in range(len(inp_shape))]
    for i, d in enumerate(newaxis_dims):
        reassoc_list.append([len(inp_shape) + i])
        if d == 0:
            d = 1
        reassoc_list[max(d - 1, 0)].extend(reassoc_list.pop(d))

    reassoc_list = _get_int_array_array_attr(reassoc_list)
    return result_shape, reassoc_list


def expand_dims(
    inp,
    newaxis_dims,
    *,
    loc=None,
    ip=None,
) -> TensorValue:
    """Expand the shape of a tensor.

    Insert a new axis that will appear at the `axis` position in the expanded
    tensor shape.

    Args:
      inp: Input tensor-like.
      axis: Position in the expanded axes where the new axis (or axes) is placed.

    Returns:
       View of `a` with the number of dimensions increased.

    """

    if len(newaxis_dims) == 0:
        return inp

    result_shape, reassoc_list = compute_result_shape_reassoc_list(
        inp.shape, newaxis_dims
    )
    if inp.fold():
        return TensorValue(inp.literal_value.reshape(result_shape))

    return TensorValue(
        tensor.expand_shape(
            RankedTensorType.get(result_shape, inp.dtype),
            inp,
            reassoc_list,
            output_shape=[],
            static_output_shape=result_shape,
            loc=loc,
            ip=ip,
        )
    )


def parallel_insert_slice(
    source,
    dest,
    offsets=None,
    sizes=None,
    strides=None,
    static_offsets=None,
    static_sizes=None,
    static_strides=None,
):
    from ...dialects.memref import _is_constant_int_like

    for s in [offsets, sizes, strides]:
        if s is not None:
            for idx, i in enumerate(s):
                if _is_constant_int_like(i):
                    s[idx] = i.owner.opview.literal_value

    if offsets is not None and static_offsets is None:
        offsets, _, static_offsets = _dispatch_mixed_values(offsets)
    if sizes is not None and static_sizes is None:
        sizes, _, static_sizes = _dispatch_mixed_values(sizes)
    if strides is not None and static_strides is None:
        strides, _, static_strides = _dispatch_mixed_values(strides)
    if offsets is None:
        offsets = []
    if sizes is None:
        sizes = []
    if strides is None:
        strides = []

    return tensor.parallel_insert_slice(
        source,
        dest,
        offsets,
        sizes,
        strides,
        static_offsets,
        static_sizes,
        static_strides,
    )


def pad_(
    source: Value,
    low: List[int],
    high: List[int],
    *,
    nofold=None,
    loc=None,
    ip=None,
):
    assert all(
        isinstance(l, int) for l in low
    ), f"only literal pad values supported: {low=}"
    assert all(
        isinstance(l, int) for l in high
    ), f"only literal pad values supported: {high=}"

    dim_sizes = []
    source_type = source.type
    for dim in range(source_type.rank):
        dim_sizes.append(source_type.get_dim_size(dim) + low[dim] + high[dim])
    result_type = RankedTensorType.get(dim_sizes, source_type.element_type)

    return tensor.PadOp(
        result_type,
        source,
        [],
        [],
        low,
        high,
        nofold=nofold,
        loc=loc,
        ip=ip,
    )


pad = region_op(pad_, terminator=lambda args: tensor.YieldOp(args[0]))

generate = region_op(
    lambda result, dynamic_extents: tensor.GenerateOp(result, dynamic_extents)
)


def _setitem(dest: "TensorValue", idx, source) -> "TensorValue":
    """Functional core of ``TensorValue.__setitem__``.

    Emits the ``tensor.insert``/``tensor.insert_slice`` op and returns the *new*
    SSA value (tensors are immutable). Returns ``dest`` unchanged for full-slice/
    ellipsis writes (nothing to do). Unlike ``__setitem__`` this does no caller-frame
    rebinding -- the caller is responsible for binding the result.
    """
    loc = get_user_code_loc()

    assert dest.has_rank(), "only ranked tensor slicing/indexing supported"
    assert source.has_rank(), "only ranked tensor slicing/indexing supported"

    if (
        idx == Ellipsis
        or idx == slice(None)
        or (isinstance(idx, tuple) and all(i == slice(None) for i in idx))
    ):
        assert (
            dest.shape == source.shape
        ), f"Expected matching shape for dest slice {dest.shape=} and source {source.shape=}"
        return dest

    if isinstance(idx, Value) and not isinstance(idx, ScalarValue):
        raise ValueError("indexing by tensor is not currently supported")

    idx = list((idx,) if isinstance(idx, (int, ScalarValue, slice)) else idx)
    for i, d in enumerate(idx):
        if isinstance(d, int):
            idx[i] = constant(d, index=True, loc=loc)

    if all(isinstance(d, ScalarValue) and d.fold() for d in idx) and len(idx) == len(
        dest.shape
    ):
        assert isinstance(
            source, ScalarValue
        ), "coordinate insert requires scalar element"
        res = tensor.insert(source, dest, idx, loc=loc)
    else:
        if any(_is_index_tensor(i) or _is_int_arraylike(i) for i in idx):
            raise ValueError("indexing by tensor is not currently supported")
        indexer = _indices_to_indexer(tuple(idx), dest.shape)
        if indexer.is_constant():
            assert (
                indexer.static_sizes() == source.shape
            ), f"Expected matching shape for dest slice {indexer.static_sizes()=} and source {source.shape=}"
            res = insert_slice(
                source,
                dest,
                static_offsets=indexer.static_offsets(),
                static_sizes=indexer.static_sizes(),
                static_strides=indexer.static_strides(),
                loc=loc,
                ip=None,
            )
        else:
            raise ValueError(f"non-constant indices not supported {indexer}")

    return res


def __mlir_extras_setitem__(dest, idx, source):
    """Runtime dispatch target that ``CanonicalizeSetItem`` rewrites ``dest[idx] = source`` into.

    Because the rewrite is purely syntactic it fires on *every* subscript assignment
    inside a canonicalized function, so this dispatches at runtime: tensors get the
    functional insert (and the new SSA value is returned to be rebound), while plain
    Python containers / mutable MLIR values (e.g. ``memref``) keep normal in-place
    ``__setitem__`` semantics and rebind to themselves (a harmless no-op).
    """
    if isinstance(dest, TensorValue):
        return _setitem(dest, idx, source)
    dest[idx] = source
    return dest


def _reconstruct_index(slice_node: ast.expr) -> ast.expr:
    """Turn a ``Subscript.slice`` AST node back into an ordinary index expression.

    An ``ast.Slice`` is only valid inside a subscript, so to pass it as a call
    argument it must be rebuilt as a ``slice(...)`` call -- mirroring what CPython
    hands to ``__setitem__`` at runtime. Tuples (extended indexing) are rebuilt
    element-wise; everything else (names, constants, ...) passes through unchanged.
    """
    if isinstance(slice_node, ast.Slice):
        return ast.Call(
            func=ast.Name(id="slice", ctx=ast.Load()),
            args=[
                slice_node.lower or ast.Constant(value=None),
                slice_node.upper or ast.Constant(value=None),
                slice_node.step or ast.Constant(value=None),
            ],
            keywords=[],
        )
    if isinstance(slice_node, ast.Tuple):
        return ast.Tuple(
            elts=[_reconstruct_index(e) for e in slice_node.elts], ctx=ast.Load()
        )
    return slice_node


class CanonicalizeSetItem(StrictTransformer):
    """Rewrite ``t[idx] = source`` into ``t = __mlir_extras_setitem__(t, idx, source)``.

    This replaces the caller-frame rebinding hack (``_update_caller_vars``) with a
    lexical rebinding performed by the compiler, so it also works through attribute
    and container aliases (``obj.t[i] = x``, ``lst[k][i] = x``) that identity-based
    frame rewriting can't reach.
    """

    def visit_Assign(self, updated_node: ast.Assign) -> ast.Assign:
        updated_node = self.generic_visit(updated_node)
        # Only plain single-target subscript assignments; skip chained/tuple targets.
        if len(updated_node.targets) != 1:
            return updated_node
        target = updated_node.targets[0]
        if not isinstance(target, ast.Subscript):
            return updated_node
        base = target.value
        # The base has to be a valid assignment target so we can rebind it; e.g.
        # `foo()[i] = x` can't become `foo() = ...`.
        if not isinstance(base, (ast.Name, ast.Attribute, ast.Subscript)):
            return updated_node

        base_load = copy.deepcopy(base)
        base_load.ctx = ast.Load()
        base_store = copy.deepcopy(base)
        base_store.ctx = ast.Store()

        call = ast_call(
            __mlir_extras_setitem__.__name__,
            args=[base_load, _reconstruct_index(target.slice), updated_node.value],
        )
        new_node = ast.Assign(targets=[base_store], value=call)
        new_node = ast.copy_location(new_node, updated_node)
        new_node = ast.fix_missing_locations(new_node)
        return new_node


class TensorPatchFunction(FunctionPatcher):
    def patch_function(self, f):
        f.__globals__[__mlir_extras_setitem__.__name__] = __mlir_extras_setitem__
        return f


class TensorCanonicalizer(Canonicalizer):
    cst_transformers = [CanonicalizeSetItem]
    function_patchers = [TensorPatchFunction]


canonicalizer = TensorCanonicalizer()
