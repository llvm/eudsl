# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import platform
from textwrap import dedent

import numpy as np
import pytest

import mlir.extras.types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects import arith
from mlir.extras.dialects._shaped_value import _Indexer
from mlir.extras.dialects.arith import ScalarValue, constant
from mlir.extras.dialects.scf import (
    range_,
    yield_,
    canonicalizer,
)
from mlir.extras.dialects.tensor import (
    TensorValue,
    empty,
    expand_dims,
    pad,
    compute_result_shape_reassoc_list,
    parallel_insert_slice,
)

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)
from mlir.ir import ShapedType

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    arr = np.random.randint(0, 10, (10, 10))
    ten = TensorValue(arr)
    assert np.array_equal(arr, ten.literal_value)


def test_simple_literal_indexing(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())

    w = ten[0]
    w = ten[2, 4]
    w = ten[2, 4, 6]
    w = ten[2, 4, 6, 8]
    assert isinstance(w, ScalarValue)

    # bare ScalarValue as index (not in a tuple)
    idx = constant(3, index=True)
    w = ten[idx]

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 4 : index
    # CHECK:  %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_0]][2, 4, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x4444xi32>
    # CHECK:  %[[VAL_6:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_7:.*]] = arith.constant 4 : index
    # CHECK:  %[[VAL_8:.*]] = arith.constant 6 : index
    # CHECK:  %[[VAL_9:.*]] = tensor.extract_slice %[[VAL_0]][2, 4, 6, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x1x4444xi32>
    # CHECK:  %[[VAL_10:.*]] = arith.constant 2 : index
    # CHECK:  %[[VAL_11:.*]] = arith.constant 4 : index
    # CHECK:  %[[VAL_12:.*]] = arith.constant 6 : index
    # CHECK:  %[[VAL_13:.*]] = arith.constant 8 : index
    # CHECK:  %[[VAL_14:.*]] = tensor.extract %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]]] : tensor<10x22x333x4444xi32>

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())

    w = ten[...]
    assert w == ten
    w = ten[:]
    assert w == ten
    w = ten[:, :]
    assert w == ten
    w = ten[:, :, :]
    assert w == ten
    w = ten[:, :, :, :]
    assert w == ten

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_1(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())
    w = ten[1, ...]
    w = ten[1, :, ...]
    w = ten[1, :, :, ...]

    try:
        w = ten[1, :, :, :, :]
    except IndexError as e:
        assert (
            str(e)
            == "Too many indices for shaped type with rank: 5 non-None/Ellipsis indices for dim 4."
        )

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_2(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())
    w = ten[1, :]
    w = ten[1, :, :]
    w = ten[1, :, :, :]
    w = ten[:, 1]
    w = ten[:, :, 1]

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_6:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 0] [1, 22, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x4444xi32>
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 0, 0] [10, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x333x4444xi32>
    # CHECK:  %[[VAL_9:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_10:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 1, 0] [10, 22, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x1x4444xi32>

    filecheck_with_comments(ctx.module)


def test_ellipsis_and_full_slice_plus_coordinate_3(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())

    w = ten[:, :, :, 1]
    w = ten[:, 1, :, 1]
    w = ten[1, :, :, 1]
    w = ten[1, 1, :, :]
    w = ten[:, :, 1, 1]
    w = ten[:, 1, 1, :]
    w = ten[1, :, 1, :]
    w = ten[1, 1, :, 1]
    w = ten[1, :, 1, 1]
    w = ten[:, 1, 1, 1]
    w = ten[1, 1, 1, :]

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 1] [10, 22, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x333x1xi32>
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 0, 1] [10, 1, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x333x1xi32>
    # CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 0, 1] [1, 22, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x333x1xi32>
    # CHECK:  %[[VAL_9:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_11:.*]] = tensor.extract_slice %[[VAL_0]][1, 1, 0, 0] [1, 1, 333, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x4444xi32>
    # CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_13:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 1, 1] [10, 22, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x22x1x1xi32>
    # CHECK:  %[[VAL_15:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_16:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_17:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 1, 0] [10, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x1x4444xi32>
    # CHECK:  %[[VAL_18:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_19:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 1, 0] [1, 22, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x1x4444xi32>
    # CHECK:  %[[VAL_21:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_22:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_23:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_24:.*]] = tensor.extract_slice %[[VAL_0]][1, 1, 0, 1] [1, 1, 333, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x333x1xi32>
    # CHECK:  %[[VAL_25:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_26:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_27:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_28:.*]] = tensor.extract_slice %[[VAL_0]][1, 0, 1, 1] [1, 22, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x22x1x1xi32>
    # CHECK:  %[[VAL_29:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_30:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_31:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_32:.*]] = tensor.extract_slice %[[VAL_0]][0, 1, 1, 1] [10, 1, 1, 1] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<10x1x1x1xi32>
    # CHECK:  %[[VAL_33:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_34:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_35:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_36:.*]] = tensor.extract_slice %[[VAL_0]][1, 1, 1, 0] [1, 1, 1, 4444] [1, 1, 1, 1] : tensor<10x22x333x4444xi32> to tensor<1x1x1x4444xi32>

    filecheck_with_comments(ctx.module)


def test_none_indices(ctx: MLIRContext):
    ten = empty(10, 22, 333, 4444, T.i32())
    w = ten[None]
    w = ten[:, None]
    w = ten[None, None]
    w = ten[:, :, None]
    w = ten[:, :, :, None]
    w = ten[:, :, :, :, None]
    w = ten[..., None]
    w = ten[:, None, :, :, None]
    w = ten[:, None, None, :, None]
    w = ten[:, None, None, None, None]
    w = ten[None, None, None, None, None]
    try:
        w = ten[None, None, None, None, None, None]
        print(w.owner)
    except IndexError as e:
        assert str(e) == "pop index out of range"

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<10x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2], [3], [4]] output_shape [1, 10, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<1x10x22x333x4444xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2], [3], [4]] output_shape [10, 1, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2], [3], [4], [5]] output_shape [1, 10, 1, 22, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x333x4444xi32>
    # CHECK:  %[[VAL_4:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1, 2], [3], [4]] output_shape [10, 22, 1, 333, 4444] : tensor<10x22x333x4444xi32> into tensor<10x22x1x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2, 3], [4]] output_shape [10, 22, 333, 1, 4444] : tensor<10x22x333x4444xi32> into tensor<10x22x333x1x4444xi32>
    # CHECK:  %[[VAL_6:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [10, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
    # CHECK:  %[[VAL_7:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [10, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x22x333x4444x1xi32>
    # CHECK:  %[[VAL_8:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2], [3], [4, 5]] output_shape [10, 1, 22, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x333x4444x1xi32>
    # CHECK:  %[[VAL_9:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3], [4], [5, 6]] output_shape [10, 1, 22, 1, 333, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x4444x1xi32>
    # CHECK:  %[[VAL_10:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2, 3], [4, 5], [6, 7]] output_shape [10, 1, 22, 1, 333, 1, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<10x1x22x1x333x1x4444x1xi32>
    # CHECK:  %[[VAL_11:.*]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1, 2], [3, 4], [5, 6], [7, 8]] output_shape [1, 10, 1, 22, 1, 333, 1, 4444, 1] : tensor<10x22x333x4444xi32> into tensor<1x10x1x22x1x333x1x4444x1xi32>

    filecheck_with_comments(ctx.module)


def test_nontrivial_slices(ctx: MLIRContext):
    ten = empty(7, 22, 333, 4444, T.i32())
    w = ten[:, 0:22:2]
    w = ten[:, 0:22:2, 0:330:30]
    w = ten[:, 0:22:2, 0:330:30, 0:4400:400]
    w = ten[:, :, 100:200:5, 1000:2000:50]

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x333x4444xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x11x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x22x333x4444xi32> to tensor<7x11x11x11xi32>
    # CHECK:  %[[VAL_4:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x333x4444xi32> to tensor<7x22x20x20xi32>

    filecheck_with_comments(ctx.module)


def test_nontrivial_slices_insertion(ctx: MLIRContext):
    ten = empty(7, 22, 333, 4444, T.i32())
    w = ten[:, 0:22:2]
    ten[:, 0:22:2] = w
    w = ten[:, 0:22:2, 0:330:30]
    ten[:, 0:22:2, 0:330:30] = w
    w = ten[:, 0:22:2, 0:330:30, 0:4400:400]
    ten[:, 0:22:2, 0:330:30, 0:4400:400] = w
    w = ten[:, :, 100:200:5, 1000:2000:50]
    ten[:, :, 100:200:5, 1000:2000:50] = w

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x333x4444xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.insert_slice %[[VAL_1]] into %[[VAL_0]][0, 0, 0, 0] [7, 11, 333, 4444] [1, 2, 1, 1] : tensor<7x11x333x4444xi32> into tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_3:.*]] = tensor.extract_slice %[[VAL_2]][0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x22x333x4444xi32> to tensor<7x11x11x4444xi32>
    # CHECK:  %[[VAL_4:.*]] = tensor.insert_slice %[[VAL_3]] into %[[VAL_2]][0, 0, 0, 0] [7, 11, 11, 4444] [1, 2, 30, 1] : tensor<7x11x11x4444xi32> into tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_5:.*]] = tensor.extract_slice %[[VAL_4]][0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x22x333x4444xi32> to tensor<7x11x11x11xi32>
    # CHECK:  %[[VAL_6:.*]] = tensor.insert_slice %[[VAL_5]] into %[[VAL_4]][0, 0, 0, 0] [7, 11, 11, 11] [1, 2, 30, 400] : tensor<7x11x11x11xi32> into tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_6]][0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x333x4444xi32> to tensor<7x22x20x20xi32>
    # CHECK:  %[[VAL_8:.*]] = tensor.insert_slice %[[VAL_7]] into %[[VAL_6]][0, 0, 100, 1000] [7, 22, 20, 20] [1, 1, 5, 50] : tensor<7x22x20x20xi32> into tensor<7x22x333x4444xi32>

    filecheck_with_comments(ctx.module)


def test_move_slice(ctx: MLIRContext):
    ten = empty(8, 8, T.i32())
    w = ten[0:4, 0:4]
    ten[4:8, 4:8] = w

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<8x8xi32>
    # CHECK:  %[[VAL_1:.*]] = tensor.extract_slice %[[VAL_0]][0, 0] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
    # CHECK:  %[[VAL_2:.*]] = tensor.insert_slice %[[VAL_1]] into %[[VAL_0]][4, 4] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>

    filecheck_with_comments(ctx.module)


def test_fold_1(ctx: MLIRContext):
    ten_arr = np.random.randint(0, 10, (10, 10)).astype(np.int32)
    x_arr = ten_arr + ten_arr
    y_arr = x_arr * x_arr
    z_arr = y_arr - x_arr

    ten = TensorValue(ten_arr, fold=True)
    x = ten + ten
    y = x * x
    z = y - x
    assert np.array_equal(z_arr, z.literal_value)
    correct = dedent(f"""\
    module {{
      %cst = arith.constant dense<{ten_arr.tolist()}> : tensor<10x10xi32>
      %cst_0 = arith.constant dense<{x_arr.tolist()}> : tensor<10x10xi32>
      %cst_1 = arith.constant dense<{y_arr.tolist()}> : tensor<10x10xi32>
      %cst_2 = arith.constant dense<{z_arr.tolist()}> : tensor<10x10xi32>
    }}
    """)
    filecheck(correct, ctx.module)


def test_for_loops(ctx: MLIRContext):
    ten = empty(7, 22, 333, 4444, T.i32())
    for i, r1, _ in range_(0, 10, iter_args=[ten]):
        y = r1 + r1
        res = yield_(y)

    assert str(res) == "TensorValue(%1, tensor<7x22x333x4444xi32>)"
    assert res.owner.name == "scf.for"

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_6:.*]] = %[[VAL_0]]) -> (tensor<7x22x333x4444xi32>) {
    # CHECK:    %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : tensor<7x22x333x4444xi32>
    # CHECK:    scf.yield %[[VAL_7]] : tensor<7x22x333x4444xi32>
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_for_loops_canonicalizer(ctx: MLIRContext):
    @canonicalize(using=canonicalizer)
    def tenfoo():
        ten = empty(7, 22, 333, 4444, T.i32())
        for i, r1, _ in range_(0, 10, iter_args=[ten]):
            y = r1 + r1
            res = yield y

        assert str(res) == "TensorValue(%1, tensor<7x22x333x4444xi32>)"
        assert res.owner.name == "scf.for"

    tenfoo()

    # CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x333x4444xi32>
    # CHECK:  %[[VAL_1:.*]] = arith.constant 0 : index
    # CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : index
    # CHECK:  %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_6:.*]] = %[[VAL_0]]) -> (tensor<7x22x333x4444xi32>) {
    # CHECK:    %[[VAL_7:.*]] = arith.addi %[[VAL_6]], %[[VAL_6]] : tensor<7x22x333x4444xi32>
    # CHECK:    scf.yield %[[VAL_7]] : tensor<7x22x333x4444xi32>
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_promotion_int_arr(ctx: MLIRContext):
    ten_arr = np.random.randint(0, 10, (10, 10)).astype(np.int32)
    ten = TensorValue(ten_arr)
    other = np.random.randint(0, 10, (10, 10)).astype(np.int32)

    x = ten + other
    y = ten - other
    z = ten / other
    w = ten // other
    v = ten % other

    ctx.module.operation.verify()
    correct = dedent(f"""\
    module {{
      %cst = arith.constant dense<{ten_arr.tolist()}> : tensor<10x10xi32>
      %cst_0 = arith.constant dense<{other.tolist()}> : tensor<10x10xi32>
      %0 = arith.addi %cst, %cst_0 : tensor<10x10xi32>
      %cst_1 = arith.constant dense<{TensorValue(y.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %1 = arith.subi %cst, %cst_1 : tensor<10x10xi32>
      %cst_2 = arith.constant dense<{TensorValue(z.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %2 = arith.divsi %cst, %cst_2 : tensor<10x10xi32>
      %cst_3 = arith.constant dense<{TensorValue(w.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %3 = arith.floordivsi %cst, %cst_3 : tensor<10x10xi32>
      %cst_4 = arith.constant dense<{TensorValue(v.owner.operands[1]).literal_value.tolist()}> : tensor<10x10xi32>
      %4 = arith.remsi %cst, %cst_4 : tensor<10x10xi32>
    }}
    """)
    filecheck(correct, ctx.module)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="windows has index here whereas linux/mac has i64",
)
def test_promotion_python_constant(ctx: MLIRContext):
    ten_arr_int = np.random.randint(0, 10, (10, 10)).astype(int)
    ten = TensorValue(ten_arr_int)

    x = ten + 1
    y = ten - 1
    z = ten / 1
    w = ten // 1
    v = ten % 1

    ten_arr_float = np.random.randint(0, 10, (10, 10)).astype(float)
    ten = TensorValue(ten_arr_float)
    xx = ten + 1.0
    yy = ten - 1.0
    zz = ten / 1.0
    vv = ten % 1.0

    ctx.module.operation.verify()
    # windows in CI...
    bits = np.dtype(int).itemsize * 8
    correct = dedent(f"""\
    module {{
      %cst = arith.constant dense<{ten_arr_int.tolist()}> : tensor<10x10xi{bits}>
      %cst_0 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %0 = arith.addi %cst, %cst_0 : tensor<10x10xi{bits}>
      %cst_1 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %1 = arith.subi %cst, %cst_1 : tensor<10x10xi{bits}>
      %cst_2 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %2 = arith.divsi %cst, %cst_2 : tensor<10x10xi{bits}>
      %cst_3 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %3 = arith.floordivsi %cst, %cst_3 : tensor<10x10xi{bits}>
      %cst_4 = arith.constant dense<1> : tensor<10x10xi{bits}>
      %4 = arith.remsi %cst, %cst_4 : tensor<10x10xi{bits}>
      %cst_5 = arith.constant dense<{ten_arr_float.tolist()}> : tensor<10x10xf64>
      %cst_6 = arith.constant dense<1.0> : tensor<10x10xf64>
      %5 = arith.addf %cst_5, %cst_6 : tensor<10x10xf64>
      %cst_7 = arith.constant dense<1.0> : tensor<10x10xf64>
      %6 = arith.subf %cst_5, %cst_7 : tensor<10x10xf64>
      %cst_8 = arith.constant dense<1.0> : tensor<10x10xf64>
      %7 = arith.divf %cst_5, %cst_8 : tensor<10x10xf64>
      %cst_9 = arith.constant dense<1.0> : tensor<10x10xf64>
      %8 = arith.remf %cst_5, %cst_9 : tensor<10x10xf64>
    }}
    """)
    filecheck(correct, str(ctx.module).replace("00000e+00", ""))


@pytest.mark.skipif(
    platform.system() != "Windows",
    reason="windows has index here whereas linux/mac has i64",
)
def test_promotion_python_constant_win(ctx: MLIRContext):
    ten_arr_int = np.random.randint(0, 10, (10, 10)).astype(int)
    ten = TensorValue(ten_arr_int)

    x = ten + 1
    y = ten - 1
    z = ten / 1
    w = ten // 1
    v = ten % 1

    ten_arr_float = np.random.randint(0, 10, (10, 10)).astype(float)
    ten = TensorValue(ten_arr_float)
    xx = ten + 1.0
    yy = ten - 1.0
    zz = ten / 1.0
    vv = ten % 1.0

    ctx.module.operation.verify()
    # windows in CI...
    bits = np.dtype(int).itemsize * 8
    correct = dedent(f"""\
    module {{
      %cst = arith.constant dense<{ten_arr_int.tolist()}> : tensor<10x10xindex>
      %cst_0 = arith.constant dense<1> : tensor<10x10xindex>
      %0 = arith.addi %cst, %cst_0 : tensor<10x10xindex>
      %cst_1 = arith.constant dense<1> : tensor<10x10xindex>
      %1 = arith.subi %cst, %cst_1 : tensor<10x10xindex>
      %cst_2 = arith.constant dense<1> : tensor<10x10xindex>
      %2 = arith.divsi %cst, %cst_2 : tensor<10x10xindex>
      %cst_3 = arith.constant dense<1> : tensor<10x10xindex>
      %3 = arith.floordivsi %cst, %cst_3 : tensor<10x10xindex>
      %cst_4 = arith.constant dense<1> : tensor<10x10xindex>
      %4 = arith.remsi %cst, %cst_4 : tensor<10x10xindex>
      %cst_5 = arith.constant dense<{ten_arr_float.tolist()}> : tensor<10x10xf64>
      %cst_6 = arith.constant dense<1.0> : tensor<10x10xf64>
      %5 = arith.addf %cst_5, %cst_6 : tensor<10x10xf64>
      %cst_7 = arith.constant dense<1.0> : tensor<10x10xf64>
      %6 = arith.subf %cst_5, %cst_7 : tensor<10x10xf64>
      %cst_8 = arith.constant dense<1.0> : tensor<10x10xf64>
      %7 = arith.divf %cst_5, %cst_8 : tensor<10x10xf64>
      %cst_9 = arith.constant dense<1.0> : tensor<10x10xf64>
      %8 = arith.remf %cst_5, %cst_9 : tensor<10x10xf64>
    }}
    """)
    filecheck(correct, str(ctx.module).replace("00000e+00", ""))


def test_promotion_arith(ctx: MLIRContext):
    ten_arr_int = np.random.randint(0, 10, (2, 2)).astype(np.int32)
    ten = TensorValue(ten_arr_int)
    one = arith.constant(1, type=T.i32())
    x = ten + one

    ten_arr_float = np.random.randint(0, 10, (3, 3)).astype(np.float32)
    ten = TensorValue(ten_arr_float)
    one = arith.constant(1.0, type=T.f32())
    x = ten + one

    ctx.module.operation.verify()
    correct = dedent(f"""\
    module {{
      %cst = arith.constant dense<{ten_arr_int.tolist()}> : tensor<2x2xi32>
      %c1_i32 = arith.constant 1 : i32
      %splat = tensor.splat %c1_i32 : tensor<2x2xi32>
      %0 = arith.addi %cst, %splat : tensor<2x2xi32>
      %cst_0 = arith.constant dense<{ten_arr_float.tolist()}> : tensor<3x3xf32>
      %cst_1 = arith.constant 1.0 : f32
      %splat_2 = tensor.splat %cst_1 : tensor<3x3xf32>
      %1 = arith.addf %cst_0, %splat_2 : tensor<3x3xf32>
    }}
    """)
    filecheck(correct, str(ctx.module).replace("00000e+00", ""))


def test_tensor_arithmetic(ctx: MLIRContext):
    one = arith.constant(1)
    assert isinstance(one, ScalarValue)
    two = arith.constant(2)
    assert isinstance(two, ScalarValue)
    three = one + two
    assert isinstance(three, ScalarValue)

    ten1 = empty(10, 10, 10, T.f32())
    assert isinstance(ten1, TensorValue)
    ten2 = empty(10, 10, 10, T.f32())
    assert isinstance(ten2, TensorValue)
    ten3 = ten1 + ten2
    assert isinstance(ten3, TensorValue)

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]] = tensor.empty() : tensor<10x10x10xf32>
    # CHECK:  %[[VAL_4:.*]] = tensor.empty() : tensor<10x10x10xf32>
    # CHECK:  %[[VAL_5:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : tensor<10x10x10xf32>

    filecheck_with_comments(ctx.module)


def test_shaped_value_literal_value_non_constant(ctx: MLIRContext):
    # Line 27 in _shaped_value.py is unreachable because is_constant is a method
    # reference (always truthy). The test verifies the actual behavior.
    ten = empty(10, 10, T.i32())
    # is_constant() returns False, but `self.is_constant` (without parens) is the
    # bound method and always truthy, so the guard is never triggered.
    # Instead it falls through and hits AttributeError on the owner.opview.value
    with pytest.raises(AttributeError):
        _ = ten.literal_value


def test_negative_step_slice_raises(ctx: MLIRContext):
    # Exercises line 195 in _shaped_value.py: negative step raises IndexError
    # Use tuple indexing to get through TensorValue.__getitem__
    ten = empty(10, 10, T.i32())
    with pytest.raises(
        IndexError, match="Negative step indexing mode not yet supported"
    ):
        _ = ten[::-1, :]


def test_multiple_ellipses_raises(ctx: MLIRContext):
    # Exercises line 267 in _shaped_value.py: multiple ellipses raises IndexError
    ten = empty(10, 10, T.i32())
    with pytest.raises(IndexError, match="Multiple ellipses"):
        _ = ten[..., ..., 0]


def test_unsupported_indexing_mode_raises(ctx: MLIRContext):
    # Exercises line 224 in _shaped_value.py: unsupported indexing type raises IndexError
    ten = empty(10, 10, T.i32())
    with pytest.raises(IndexError, match="Indexing mode not yet supported"):
        _ = ten[{1, 2}, 0]


def test_setitem_full_slice(ctx: MLIRContext):
    # Covers lines 183-186: __setitem__ with full slice (Ellipsis) returns self
    ten = empty(4, 4, T.i32())
    source = empty(4, 4, T.i32())
    ten[...] = source
    ten[:] = source
    ten[:, :] = source

    # Only empty ops, no insert_slice since full-slice setitem returns self
    # CHECK: %{{.*}} = tensor.empty() : tensor<4x4xi32>
    # CHECK: %{{.*}} = tensor.empty() : tensor<4x4xi32>

    filecheck_with_comments(ctx.module)


def test_setitem_slice_insert(ctx: MLIRContext):
    # Covers lines 191, 202-216: __setitem__ with slice (insert_slice path)
    ten = empty(4, 4, T.i32())
    source = empty(2, 4, T.i32())
    ten[0:2,] = source

    # CHECK: %[[TEN:.*]] = tensor.empty() : tensor<4x4xi32>
    # CHECK: %[[SRC:.*]] = tensor.empty() : tensor<2x4xi32>
    # CHECK: %{{.*}} = tensor.insert_slice %[[SRC]] into %[[TEN]][0, 0] [2, 4] [1, 1]

    filecheck_with_comments(ctx.module)


def test_setitem_index_tensor_raises(ctx: MLIRContext):
    # Covers line 202: __setitem__ with index tensor raises ValueError
    ten = empty(10, T.index())
    source = empty(5, T.index())
    with pytest.raises(
        ValueError, match="indexing by tensor is not currently supported"
    ):
        ten[ten] = source


def test_setitem_non_constant_indices_raises(ctx: MLIRContext):
    # Covers line 218: __setitem__ with non-constant indices raises ValueError
    ten = empty(10, T.i32())
    source = empty(5, T.i32())
    c1 = constant(1, index=True)
    c2 = constant(2, index=True)
    dyn = c1 * c2
    with pytest.raises(ValueError, match="non-constant indices not supported"):
        ten[dyn:] = source


def test_getitem_index_tensor_raises(ctx: MLIRContext):
    # Covers line 148: __getitem__ with index tensor raises ValueError
    ten = empty(10, T.index())
    with pytest.raises(
        ValueError, match="indexing by tensor is not currently supported"
    ):
        _ = ten[ten]


def test_getitem_non_constant_indices_raises(ctx: MLIRContext):
    # Covers line 165: __getitem__ with non-constant indices raises ValueError
    ten = empty(10, T.i32())
    c1 = constant(1, index=True)
    c2 = constant(2, index=True)
    dyn = c1 * c2
    with pytest.raises(ValueError, match="non-constant indices not supported"):
        _ = ten[dyn:]


def test_coerce_unknown_raises(ctx: MLIRContext):
    # Covers line 257: coerce with unknown type raises ValueError
    ten = empty(4, 4, T.i32())
    with pytest.raises(ValueError, match="can't coerce unknown"):
        ten.coerce("not_a_valid_type")


def test_compute_result_shape_reassoc_list_repeated_axis():
    # Covers line 263: repeated axis in expand_dims raises ValueError
    with pytest.raises(ValueError, match="repeated axis"):
        compute_result_shape_reassoc_list((10, 20), (0, 0))


def test_compute_result_shape_reassoc_list_negative_dims():
    # Covers line 267: negative dims raises ValueError
    with pytest.raises(ValueError, match="no negative dims allowed"):
        compute_result_shape_reassoc_list((10, 20), (-1,))


def test_expand_dims_with_fold(ctx: MLIRContext):
    # Covers line 310: expand_dims with a foldable (constant) tensor
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ten = TensorValue(arr, fold=True)
    result = expand_dims(ten, (0,))
    # When fold() is true, it reshapes the literal_value
    expected = arr.reshape(1, 2, 2)
    assert np.array_equal(result.literal_value, expected)


def test_pad(ctx: MLIRContext):
    # Covers lines 375-388: pad_ function (via the region_op wrapper `pad`)
    ten = empty(4, 16, T.f32())
    pad_value = constant(0.0, type=T.f32())

    @pad(ten, [1, 2], [3, 4])
    def padded(i: T.index(), j: T.index()):
        return pad_value

    # CHECK: %[[TEN:.*]] = tensor.empty() : tensor<4x16xf32>
    # CHECK: %[[PAD_VAL:.*]] = arith.constant 0.000000e+00 : f32
    # CHECK: %{{.*}} = tensor.pad %[[TEN]] low[1, 2] high[3, 4]
    # CHECK:   tensor.yield %[[PAD_VAL]] : f32
    # CHECK: } : tensor<4x16xf32> to tensor<8x22xf32>

    filecheck_with_comments(ctx.module)


def test_indexer_static_offsets_unsupported_index():
    # Exercises line 105 in _shaped_value.py: static_offsets raises on unsupported idx
    # Construct an _Indexer with an unsupported index type directly
    indexer = _Indexer(
        indices=(object(),),
        newaxis_dims=(),
        in_shape=(10,),
    )
    with pytest.raises(ValueError, match="not supported with static offsets"):
        indexer.static_offsets()


def test_indexer_static_sizes_unsupported_index():
    # Exercises lines 122-125 in _shaped_value.py: static_sizes raises on unsupported idx
    indexer = _Indexer(
        indices=(object(),),
        newaxis_dims=(),
        in_shape=(10,),
    )
    with pytest.raises(ValueError, match="not supported with static sizes"):
        indexer.static_sizes()


def test_indexer_static_strides_unsupported_index():
    # Exercises line 137 in _shaped_value.py: static_strides raises on unsupported idx
    indexer = _Indexer(
        indices=(object(),),
        newaxis_dims=(),
        in_shape=(10,),
    )
    with pytest.raises(ValueError, match="not supported with static strides"):
        indexer.static_strides()


def test_slice_on_dynamic_dim_raises(ctx: MLIRContext):
    # Exercises lines 208-212 in _shaped_value.py: slice on dynamic dimension
    S = ShapedType.get_dynamic_size()
    ten = empty(S, 10, T.i32())
    with pytest.raises(IndexError, match="Cannot use NumPy slice indexing"):
        _ = ten[0:5, :]


def test_maybe_compute_size_mul_add_pattern(ctx: MLIRContext):
    # Exercises line 350 in _shaped_value.py: _maybe_compute_size with mul/add pattern
    # The pattern is: start = x * D, stop = (x + 1) * D
    # We test this indirectly via tensor slicing with computed indices
    from mlir.extras.dialects._shaped_value import _compute_size

    ten = empty(64, 64, T.i32())
    D = constant(8, index=True)
    i = constant(2, index=True)
    one = constant(1, index=True)
    start = i * D  # MulIOp
    stop = (i + one) * D  # MulIOp(AddIOp(i, 1), D)
    # Call _maybe_compute_size directly to hit line 350
    result = _compute_size(start, stop, 1)
    # result should be D (the constant 8)
    assert result is not None


def test_getitem_partial_index_with_index_tensor(ctx: MLIRContext):
    # Covers line 151: __getitem__ else branch with index tensor in partial indexing
    ten = empty(10, 20, 30, T.i32())
    idx_tensor = empty(5, T.index())
    # ten has 3 dims, idx is (scalar, index_tensor) - only 2 indices, partial indexing
    # The scalar goes through constant conversion, then the else branch is entered
    # because not all are ScalarValues (idx_tensor is a TensorValue), and then
    # line 150 checks _is_index_tensor and raises
    with pytest.raises(
        ValueError, match="indexing by tensor is not currently supported"
    ):
        _ = ten[0, idx_tensor]


def test_setitem_index_tensor_in_else_branch(ctx: MLIRContext):
    # Covers line 208: __setitem__ else branch with index tensor raises ValueError
    ten = empty(10, 20, T.i32())
    idx_tensor = empty(5, T.index())
    source = empty(5, T.i32())
    # idx_tensor is NOT a ScalarValue, and len(idx) != len(self.shape),
    # so we hit the else branch where _is_index_tensor check fires
    with pytest.raises(
        ValueError, match="indexing by tensor is not currently supported"
    ):
        ten[idx_tensor,] = source


def test_coerce_non_static_shape_raises(ctx: MLIRContext):
    # Covers line 241: coerce raises when tensor doesn't have static shape
    S = ShapedType.get_dynamic_size()
    ten = empty(S, 10, T.i32())
    with pytest.raises(ValueError, match="can't coerce .* doesn't have static shape"):
        ten.coerce(42)


def test_setitem_int_to_constant_conversion(ctx: MLIRContext):
    """Line 194: __setitem__ converts int idx to constant via insert_slice path"""
    ten = empty(10, 20, T.i32())
    source = empty(1, 20, T.i32())
    # idx = (5, slice(None)) -> int 5 gets converted to constant on line 194
    # Then not all ScalarValue (slice is present), hits insert_slice path
    ten[5, :] = source

    ctx.module.operation.verify()


def test_parallel_insert_slice_with_static_args(ctx: MLIRContext):
    """Lines 338-355: parallel_insert_slice with static offsets/sizes/strides (None dynamic args)"""
    from mlir.extras.dialects.scf import forall_

    ten = empty(10, 10, T.i32())

    @forall_([0, 0], [10, 10], [1, 1], shared_outs=[ten])
    def forfoo(i, j, shared_outs):
        return lambda: parallel_insert_slice(
            ten,
            shared_outs,
            static_offsets=[0, 0],
            static_sizes=[10, 10],
            static_strides=[1, 1],
        )

    ctx.module.operation.verify()


def test_parallel_insert_slice_with_dynamic_args(ctx: MLIRContext):
    """Lines 342-346, 348-352: parallel_insert_slice with dynamic offsets/sizes/strides"""
    from mlir.extras.dialects.scf import forall_

    ten = empty(10, 10, T.i32())
    ten_size = constant(10, index=True)
    one = constant(1, index=True)

    @forall_([0, 0], [10, 10], [1, 1], shared_outs=[ten])
    def forfoo(i, j, shared_outs):
        return lambda: parallel_insert_slice(
            ten,
            shared_outs,
            offsets=[i, j],
            sizes=[ten_size, ten_size],
            strides=[one, one],
        )

    ctx.module.operation.verify()


def test_empty_with_explicit_element_type(ctx: MLIRContext):
    """Branch 35->37: empty() with explicit element_type (skips _unpack_sizes_element_type)"""
    ten = empty(10, 20, element_type=T.f32())
    assert ten is not None


def test_extract_slice_with_offsets_and_strides(ctx: MLIRContext):
    """Branches 53->55, 55->57: extract_slice with explicit offsets/strides"""
    from mlir.extras.dialects.tensor import extract_slice

    ten = empty(10, 20, T.f32())
    off0 = constant(0, index=True)
    off1 = constant(0, index=True)
    st0 = constant(1, index=True)
    st1 = constant(1, index=True)
    S = ShapedType.get_dynamic_size()
    result = extract_slice(
        ten,
        offsets=[off0, off1],
        strides=[st0, st1],
        static_sizes=[5, 10],
        static_offsets=[S, S],
        static_strides=[S, S],
    )
    ctx.module.operation.verify()


def test_insert_slice_with_offsets_and_strides(ctx: MLIRContext):
    """Branches 88->90, 90->92: insert_slice with explicit offsets/strides"""
    from mlir.extras.dialects.tensor import insert_slice

    dest = empty(10, 20, T.f32())
    source = empty(5, 10, T.f32())
    off0 = constant(0, index=True)
    off1 = constant(0, index=True)
    st0 = constant(1, index=True)
    st1 = constant(1, index=True)
    S = ShapedType.get_dynamic_size()
    result = insert_slice(
        source,
        dest,
        offsets=[off0, off1],
        strides=[st0, st1],
        static_sizes=[5, 10],
        static_offsets=[S, S],
        static_strides=[S, S],
    )
    ctx.module.operation.verify()


def test_coerce_unknown_type_raises(ctx: MLIRContext):
    """Branch 250->260: coerce with an unknown type raises ValueError"""
    ten = empty(10, 20, T.f32())
    with pytest.raises(ValueError, match="can't coerce unknown"):
        ten.coerce(object())


def test_getitem_slice_with_nonzero_remainder(ctx: MLIRContext):
    """Branch 117->119: slice where (stop-start) % step != 0"""
    ten = empty(10, T.f32())
    # 0:5:2 -> indices [0, 2, 4] -> 3 elements
    # (5-0)//2 + 1 = 3, (5-0)%2 = 1 != 0 so no decrement
    result = ten[0:5:2]
    ctx.module.operation.verify()
