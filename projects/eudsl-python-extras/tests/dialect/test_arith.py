# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from mlir.extras.dialects import arith, func
from mlir.extras.testing import (
    filecheck_with_comments,
    MLIRContext,
    mlir_mod_ctx,
)
from mlir.extras import types as T
from mlir.extras.ast.canonicalize import canonicalize

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)


def test_arith_value(ctx: MLIRContext):
    i32 = T.i32()

    @func.FuncOp.from_py_func(i32, i32)
    def foo(a: i32, b: i32) -> i32:
        c1 = arith.constant(1)
        return c1 * a + b - c1 / a // c1 % b

    # CHECK-LABEL:   func.func @foo(
    # CHECK-SAME:                   %[[ARG0:.*]]: i32,
    # CHECK-SAME:                   %[[ARG1:.*]]: i32) -> i32 {
    # CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
    # CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_0]], %[[ARG0]] : i32
    # CHECK:           %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG1]] : i32
    # CHECK:           %[[DIVSI_0:.*]] = arith.divsi %[[CONSTANT_0]], %[[ARG0]] : i32
    # CHECK:           %[[FLOORDIVSI_0:.*]] = arith.floordivsi %[[DIVSI_0]], %[[CONSTANT_0]] : i32
    # CHECK:           %[[REMSI_0:.*]] = arith.remsi %[[FLOORDIVSI_0]], %[[ARG1]] : i32
    # CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ADDI_0]], %[[REMSI_0]] : i32
    # CHECK:           return %[[SUBI_0]] : i32
    # CHECK:         }
    filecheck_with_comments(ctx.module)


def test_arith_constant_canonicalizer(ctx: MLIRContext):
    @func.func(emit=True)
    @canonicalize(using=arith.canonicalizer)
    def foo():
        # CHECK: %c0_i32 = arith.constant 0 : i32
        row_m: T.i32() = 0
        # CHECK: %cst = arith.constant 0.000000e+00 : f32
        row_l: T.f32() = 0.0

    filecheck_with_comments(ctx.module)


def test_arithmetic(ctx: MLIRContext):
    one = arith.constant(1)
    two = arith.constant(2)
    one + two
    one - two
    one / two
    one // two
    one % two

    two = arith.constant(2, index=True)
    one + two
    one - two
    one / two
    one // two
    one % two

    one = arith.constant(1.0)
    two = arith.constant(2.0)
    one + two
    one - two
    one / two
    try:
        one // two
    except ValueError as e:
        assert (
            str(e)
            == "floordiv not supported for lhs=ScalarValue(%cst = arith.constant 1.000000e+00 : f32)"
        )
    one % two

    ctx.module.operation.verify()

    # CHECK:         %[[CONSTANT_0:.*]] = arith.constant 1 : i32
    # CHECK:         %[[CONSTANT_1:.*]] = arith.constant 2 : i32
    # CHECK:         %[[ADDI_0:.*]] = arith.addi %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[DIVSI_0:.*]] = arith.divsi %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[FLOORDIVSI_0:.*]] = arith.floordivsi %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[REMSI_0:.*]] = arith.remsi %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CONSTANT_2:.*]] = arith.constant 2 : index
    # CHECK:         %[[INDEX_CAST_0:.*]] = arith.index_cast %[[CONSTANT_2]] : index to i32
    # CHECK:         %[[ADDI_1:.*]] = arith.addi %[[CONSTANT_0]], %[[INDEX_CAST_0]] : i32
    # CHECK:         %[[INDEX_CAST_1:.*]] = arith.index_cast %[[CONSTANT_2]] : index to i32
    # CHECK:         %[[SUBI_1:.*]] = arith.subi %[[CONSTANT_0]], %[[INDEX_CAST_1]] : i32
    # CHECK:         %[[INDEX_CAST_2:.*]] = arith.index_cast %[[CONSTANT_2]] : index to i32
    # CHECK:         %[[DIVSI_1:.*]] = arith.divsi %[[CONSTANT_0]], %[[INDEX_CAST_2]] : i32
    # CHECK:         %[[INDEX_CAST_3:.*]] = arith.index_cast %[[CONSTANT_2]] : index to i32
    # CHECK:         %[[FLOORDIVSI_1:.*]] = arith.floordivsi %[[CONSTANT_0]], %[[INDEX_CAST_3]] : i32
    # CHECK:         %[[INDEX_CAST_4:.*]] = arith.index_cast %[[CONSTANT_2]] : index to i32
    # CHECK:         %[[REMSI_1:.*]] = arith.remsi %[[CONSTANT_0]], %[[INDEX_CAST_4]] : i32
    # CHECK:         %[[CONSTANT_3:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:         %[[CONSTANT_4:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:         %[[ADDF_0:.*]] = arith.addf %[[CONSTANT_3]], %[[CONSTANT_4]] : f32
    # CHECK:         %[[SUBF_0:.*]] = arith.subf %[[CONSTANT_3]], %[[CONSTANT_4]] : f32
    # CHECK:         %[[DIVF_0:.*]] = arith.divf %[[CONSTANT_3]], %[[CONSTANT_4]] : f32
    # CHECK:         %[[REMF_0:.*]] = arith.remf %[[CONSTANT_3]], %[[CONSTANT_4]] : f32

    filecheck_with_comments(ctx.module)


def test_r_arithmetic(ctx: MLIRContext):
    one = arith.constant(1)
    two = arith.constant(2)
    one - two
    two - one

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]] : i32

    filecheck_with_comments(ctx.module)


def test_arith_cmpi(ctx: MLIRContext):
    for kind1, kind2 in [({}, {}), ({'index': True}, {'index': True}), ({'index': True}, {}), ({}, {'index': True})]:
        one = arith.constant(1, **kind1)
        two = arith.constant(2, **kind2)
        one < two
        one <= two
        one > two
        one >= two
        one == two
        one != two
        one & two
        one | two
        assert one._ne(two)
        assert not one._eq(two)

    # CHECK:         %[[CONSTANT_0:.*]] = arith.constant 1 : i32
    # CHECK:         %[[CONSTANT_1:.*]] = arith.constant 2 : i32
    # CHECK:         %[[CMPI_0:.*]] = arith.cmpi slt, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_1:.*]] = arith.cmpi sle, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_2:.*]] = arith.cmpi sgt, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_3:.*]] = arith.cmpi sge, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_4:.*]] = arith.cmpi eq, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_5:.*]] = arith.cmpi ne, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[ANDI_0:.*]] = arith.andi %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[ORI_0:.*]] = arith.ori %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CONSTANT_2:.*]] = arith.constant 1 : index
    # CHECK:         %[[CONSTANT_3:.*]] = arith.constant 2 : index
    # CHECK:         %[[CMPI_6:.*]] = arith.cmpi ult, %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[CMPI_7:.*]] = arith.cmpi ule, %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[CMPI_8:.*]] = arith.cmpi ugt, %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[CMPI_9:.*]] = arith.cmpi uge, %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[CMPI_10:.*]] = arith.cmpi eq, %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[CMPI_11:.*]] = arith.cmpi ne, %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[ANDI_1:.*]] = arith.andi %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[ORI_1:.*]] = arith.ori %[[CONSTANT_2]], %[[CONSTANT_3]] : index
    # CHECK:         %[[CONSTANT_4:.*]] = arith.constant 1 : index
    # CHECK:         %[[CONSTANT_5:.*]] = arith.constant 2 : i32
    # CHECK:         %[[INDEX_CAST_0:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[CMPI_12:.*]] = arith.cmpi ult, %[[CONSTANT_4]], %[[INDEX_CAST_0]] : index
    # CHECK:         %[[INDEX_CAST_1:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[CMPI_13:.*]] = arith.cmpi ule, %[[CONSTANT_4]], %[[INDEX_CAST_1]] : index
    # CHECK:         %[[INDEX_CAST_2:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[CMPI_14:.*]] = arith.cmpi ugt, %[[CONSTANT_4]], %[[INDEX_CAST_2]] : index
    # CHECK:         %[[INDEX_CAST_3:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[CMPI_15:.*]] = arith.cmpi uge, %[[CONSTANT_4]], %[[INDEX_CAST_3]] : index
    # CHECK:         %[[INDEX_CAST_4:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[CMPI_16:.*]] = arith.cmpi eq, %[[CONSTANT_4]], %[[INDEX_CAST_4]] : index
    # CHECK:         %[[INDEX_CAST_5:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[CMPI_17:.*]] = arith.cmpi ne, %[[CONSTANT_4]], %[[INDEX_CAST_5]] : index
    # CHECK:         %[[INDEX_CAST_6:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[ANDI_2:.*]] = arith.andi %[[CONSTANT_4]], %[[INDEX_CAST_6]] : index
    # CHECK:         %[[INDEX_CAST_7:.*]] = arith.index_cast %[[CONSTANT_5]] : i32 to index
    # CHECK:         %[[ORI_2:.*]] = arith.ori %[[CONSTANT_4]], %[[INDEX_CAST_7]] : index
    # CHECK:         %[[CONSTANT_6:.*]] = arith.constant 1 : i32
    # CHECK:         %[[CONSTANT_7:.*]] = arith.constant 2 : index
    # CHECK:         %[[INDEX_CAST_8:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[CMPI_18:.*]] = arith.cmpi slt, %[[CONSTANT_6]], %[[INDEX_CAST_8]] : i32
    # CHECK:         %[[INDEX_CAST_9:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[CMPI_19:.*]] = arith.cmpi sle, %[[CONSTANT_6]], %[[INDEX_CAST_9]] : i32
    # CHECK:         %[[INDEX_CAST_10:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[CMPI_20:.*]] = arith.cmpi sgt, %[[CONSTANT_6]], %[[INDEX_CAST_10]] : i32
    # CHECK:         %[[INDEX_CAST_11:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[CMPI_21:.*]] = arith.cmpi sge, %[[CONSTANT_6]], %[[INDEX_CAST_11]] : i32
    # CHECK:         %[[INDEX_CAST_12:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[CMPI_22:.*]] = arith.cmpi eq, %[[CONSTANT_6]], %[[INDEX_CAST_12]] : i32
    # CHECK:         %[[INDEX_CAST_13:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[CMPI_23:.*]] = arith.cmpi ne, %[[CONSTANT_6]], %[[INDEX_CAST_13]] : i32
    # CHECK:         %[[INDEX_CAST_14:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[ANDI_3:.*]] = arith.andi %[[CONSTANT_6]], %[[INDEX_CAST_14]] : i32
    # CHECK:         %[[INDEX_CAST_15:.*]] = arith.index_cast %[[CONSTANT_7]] : index to i32
    # CHECK:         %[[ORI_3:.*]] = arith.ori %[[CONSTANT_6]], %[[INDEX_CAST_15]] : i32
    filecheck_with_comments(ctx.module)

def test_arith_cmpf(ctx: MLIRContext):

    one = arith.constant(1.0)
    two = arith.constant(2.0)
    one < two
    one <= two
    one > two
    one >= two
    one == two
    one != two
    assert one._ne(two)
    assert not one._eq(two)

    ctx.module.operation.verify()

    # CHECK:         %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:         %[[CONSTANT_1:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:         %[[CMPF_0:.*]] = arith.cmpf olt, %[[CONSTANT_0]], %[[CONSTANT_1]] : f32
    # CHECK:         %[[CMPF_1:.*]] = arith.cmpf ole, %[[CONSTANT_0]], %[[CONSTANT_1]] : f32
    # CHECK:         %[[CMPF_2:.*]] = arith.cmpf ogt, %[[CONSTANT_0]], %[[CONSTANT_1]] : f32
    # CHECK:         %[[CMPF_3:.*]] = arith.cmpf oge, %[[CONSTANT_0]], %[[CONSTANT_1]] : f32
    # CHECK:         %[[CMPF_4:.*]] = arith.cmpf oeq, %[[CONSTANT_0]], %[[CONSTANT_1]] : f32
    # CHECK:         %[[CMPF_5:.*]] = arith.cmpf one, %[[CONSTANT_0]], %[[CONSTANT_1]] : f32
    filecheck_with_comments(ctx.module)


def test_arith_cmp_enum_values(ctx: MLIRContext):
    one = arith.constant(1)
    two = arith.constant(2)
    for pred in arith.CmpIPredicate.__members__.values():
        arith.cmpi(pred, one, two)

    one, two = arith.constant(1.0), arith.constant(2.0)
    for pred in arith.CmpFPredicate.__members__.values():
        arith.cmpf(pred, one, two)

    # CHECK:         %[[CONSTANT_0:.*]] = arith.constant 1 : i32
    # CHECK:         %[[CONSTANT_1:.*]] = arith.constant 2 : i32
    # CHECK:         %[[CMPI_0:.*]] = arith.cmpi eq, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_1:.*]] = arith.cmpi ne, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_2:.*]] = arith.cmpi slt, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_3:.*]] = arith.cmpi sle, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_4:.*]] = arith.cmpi sgt, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_5:.*]] = arith.cmpi sge, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_6:.*]] = arith.cmpi ult, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_7:.*]] = arith.cmpi ule, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_8:.*]] = arith.cmpi ugt, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CMPI_9:.*]] = arith.cmpi uge, %[[CONSTANT_0]], %[[CONSTANT_1]] : i32
    # CHECK:         %[[CONSTANT_2:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:         %[[CONSTANT_3:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:         %[[CMPF_0:.*]] = arith.cmpf false, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_1:.*]] = arith.cmpf oeq, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_2:.*]] = arith.cmpf ogt, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_3:.*]] = arith.cmpf oge, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_4:.*]] = arith.cmpf olt, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_5:.*]] = arith.cmpf ole, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_6:.*]] = arith.cmpf one, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_7:.*]] = arith.cmpf ord, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_8:.*]] = arith.cmpf ueq, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_9:.*]] = arith.cmpf ugt, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_10:.*]] = arith.cmpf uge, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_11:.*]] = arith.cmpf ult, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_12:.*]] = arith.cmpf ule, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_13:.*]] = arith.cmpf une, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_14:.*]] = arith.cmpf uno, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32
    # CHECK:         %[[CMPF_15:.*]] = arith.cmpf true, %[[CONSTANT_2]], %[[CONSTANT_3]] : f32

    ctx.module.operation.verify()
    filecheck_with_comments(ctx.module)


def test_arith_cmp_literals(ctx: MLIRContext):
    one = arith.constant(1)
    two = 2
    one < two
    one <= two
    one > two
    one >= two
    one == two
    one != two
    one & two
    one | two

    one = arith.constant(1.0)
    two = 2.0
    one < two
    one <= two
    one > two
    one >= two
    one == two
    one != two

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_4:.*]] = arith.cmpi sle, %[[VAL_0]], %[[VAL_3]] : i32
    # CHECK:  %[[VAL_5:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_6:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_5]] : i32
    # CHECK:  %[[VAL_7:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_8:.*]] = arith.cmpi sge, %[[VAL_0]], %[[VAL_7]] : i32
    # CHECK:  %[[VAL_9:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_9]] : i32
    # CHECK:  %[[VAL_11:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_11]] : i32
    # CHECK:  %[[VAL_13:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_14:.*]] = arith.andi %[[VAL_0]], %[[VAL_13]] : i32
    # CHECK:  %[[VAL_15:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_16:.*]] = arith.ori %[[VAL_0]], %[[VAL_15]] : i32
    # CHECK:  %[[VAL_17:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_18:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_19:.*]] = arith.cmpf olt, %[[VAL_17]], %[[VAL_18]] : f32
    # CHECK:  %[[VAL_20:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_21:.*]] = arith.cmpf ole, %[[VAL_17]], %[[VAL_20]] : f32
    # CHECK:  %[[VAL_22:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_23:.*]] = arith.cmpf ogt, %[[VAL_17]], %[[VAL_22]] : f32
    # CHECK:  %[[VAL_24:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_25:.*]] = arith.cmpf oge, %[[VAL_17]], %[[VAL_24]] : f32
    # CHECK:  %[[VAL_26:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_27:.*]] = arith.cmpf oeq, %[[VAL_17]], %[[VAL_26]] : f32
    # CHECK:  %[[VAL_28:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_29:.*]] = arith.cmpf one, %[[VAL_17]], %[[VAL_28]] : f32

    filecheck_with_comments(ctx.module)


def test_scalar_promotion(ctx: MLIRContext):
    one = arith.constant(1)
    one + 2
    one - 2
    one / 2
    one // 2
    one % 2

    one = arith.constant(1.0)
    one + 2.0
    one - 2.0
    one / 2.0
    one % 2.0

    ctx.module.operation.verify()

    # CHECK:  %[[C1_I32:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_0:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.addi %[[C1_I32]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_2:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_3:.*]] = arith.subi %[[C1_I32]], %[[VAL_2]] : i32
    # CHECK:  %[[VAL_4:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_5:.*]] = arith.divsi %[[C1_I32]], %[[VAL_4]] : i32
    # CHECK:  %[[VAL_6:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_7:.*]] = arith.floordivsi %[[C1_I32]], %[[VAL_6]] : i32
    # CHECK:  %[[VAL_8:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_9:.*]] = arith.remsi %[[C1_I32]], %[[VAL_8]] : i32
    # CHECK:  %[[VAL_10:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_11:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_12:.*]] = arith.addf %[[VAL_10]], %[[VAL_11]] : f32
    # CHECK:  %[[VAL_13:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_14:.*]] = arith.subf %[[VAL_10]], %[[VAL_13]] : f32
    # CHECK:  %[[VAL_15:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_16:.*]] = arith.divf %[[VAL_10]], %[[VAL_15]] : f32
    # CHECK:  %[[VAL_17:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_18:.*]] = arith.remf %[[VAL_10]], %[[VAL_17]] : f32

    filecheck_with_comments(ctx.module)


def test_rscalar_promotion(ctx: MLIRContext):
    one = arith.constant(1)
    2 + one
    2 - one
    2 / one
    2 // one
    2 % one

    one = arith.constant(1.0)
    2.0 + one
    2.0 - one
    2.0 / one
    2.0 % one

    ctx.module.operation.verify()
    # CHECK:  %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.addi %[[VAL_1]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_4:.*]] = arith.subi %[[VAL_3]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_5:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_6:.*]] = arith.divsi %[[VAL_5]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_7:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_8:.*]] = arith.floordivsi %[[VAL_7]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_9:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_10:.*]] = arith.remsi %[[VAL_9]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_11:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_12:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_13:.*]] = arith.addf %[[VAL_12]], %[[VAL_11]] : f32
    # CHECK:  %[[VAL_14:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_15:.*]] = arith.subf %[[VAL_14]], %[[VAL_11]] : f32
    # CHECK:  %[[VAL_16:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_17:.*]] = arith.divf %[[VAL_16]], %[[VAL_11]] : f32
    # CHECK:  %[[VAL_18:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_19:.*]] = arith.remf %[[VAL_18]], %[[VAL_11]] : f32

    filecheck_with_comments(ctx.module)


def test_arith_rcmp_literals(ctx: MLIRContext):
    one = 1
    two = arith.constant(2)
    one < two
    one <= two
    one > two
    one >= two
    one == two
    one != two
    one & two
    one | two

    one = 1.0
    two = arith.constant(2.0)
    one < two
    one <= two
    one > two
    one >= two
    one == two
    one != two

    ctx.module.operation.verify()

    # CHECK:  %[[VAL_0:.*]] = arith.constant 2 : i32
    # CHECK:  %[[VAL_1:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_2:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_1]] : i32
    # CHECK:  %[[VAL_3:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_4:.*]] = arith.cmpi sge, %[[VAL_0]], %[[VAL_3]] : i32
    # CHECK:  %[[VAL_5:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_5]] : i32
    # CHECK:  %[[VAL_7:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_8:.*]] = arith.cmpi sle, %[[VAL_0]], %[[VAL_7]] : i32
    # CHECK:  %[[VAL_9:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_0]], %[[VAL_9]] : i32
    # CHECK:  %[[VAL_11:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_0]], %[[VAL_11]] : i32
    # CHECK:  %[[VAL_13:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_14:.*]] = arith.andi %[[VAL_13]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_15:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_16:.*]] = arith.ori %[[VAL_15]], %[[VAL_0]] : i32
    # CHECK:  %[[VAL_17:.*]] = arith.constant 2.000000e+00 : f32
    # CHECK:  %[[VAL_18:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_19:.*]] = arith.cmpf ogt, %[[VAL_17]], %[[VAL_18]] : f32
    # CHECK:  %[[VAL_20:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_21:.*]] = arith.cmpf oge, %[[VAL_17]], %[[VAL_20]] : f32
    # CHECK:  %[[VAL_22:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_23:.*]] = arith.cmpf olt, %[[VAL_17]], %[[VAL_22]] : f32
    # CHECK:  %[[VAL_24:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_25:.*]] = arith.cmpf ole, %[[VAL_17]], %[[VAL_24]] : f32
    # CHECK:  %[[VAL_26:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_27:.*]] = arith.cmpf oeq, %[[VAL_17]], %[[VAL_26]] : f32
    # CHECK:  %[[VAL_28:.*]] = arith.constant 1.000000e+00 : f32
    # CHECK:  %[[VAL_29:.*]] = arith.cmpf one, %[[VAL_17]], %[[VAL_28]] : f32

    filecheck_with_comments(ctx.module)


if __name__ == "__main__":
    with mlir_mod_ctx() as c:
        test_arith_value(c)
