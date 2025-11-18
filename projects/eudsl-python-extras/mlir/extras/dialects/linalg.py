# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from . import arith
from ... import ir
from ...dialects import linalg
from ...dialects._ods_common import (
    _dispatch_mixed_values,
    _cext,
    get_op_results_or_values,
    get_default_loc_context,
    get_op_result_or_op_results,
    get_default_loc_context,
    segmented_accessor,
)

# noinspection PyUnresolvedReferences
from ...dialects.linalg import *
from ...extras import types as T


def abs(I, O, *, loc=None, ip=None):
    return linalg.abs(I, loc=loc, ip=ip, outs=[O])


def add(lhs, rhs, O, *, loc=None, ip=None):
    return linalg.add(lhs, rhs, loc=loc, ip=ip, outs=[O])


def batch_matmul(A, B, C, *, loc=None, ip=None):
    return linalg.batch_matmul(A, B, loc=loc, ip=ip, outs=[C])


def batch_matmul_transpose_a(A, B, C, *, loc=None, ip=None):
    return linalg.batch_matmul_transpose_a(A, B, loc=loc, ip=ip, outs=[C])


def batch_matmul_transpose_b(A, B, C, *, loc=None, ip=None):
    return linalg.batch_matmul_transpose_b(A, B, loc=loc, ip=ip, outs=[C])


def batch_matvec(A, B, C, *, loc=None, ip=None):
    return linalg.batch_matvec(A, B, loc=loc, ip=ip, outs=[C])


def batch_mmt4d(lhs, rhs, accum, *, loc=None, ip=None):
    return linalg.batch_mmt4d(lhs, rhs, loc=loc, ip=ip, outs=[accum])


def batch_reduce_matmul(A, B, C, *, loc=None, ip=None):
    return linalg.batch_reduce_matmul(A, B, loc=loc, ip=ip, outs=[C])


def batch_vecmat(A, B, C, *, loc=None, ip=None):
    return linalg.batch_vecmat(A, B, loc=loc, ip=ip, outs=[C])


def ceil(I, O, *, loc=None, ip=None):
    return linalg.ceil(I, loc=loc, ip=ip, outs=[O])


def conv_1d(I, K, O, *, loc=None, ip=None):
    return linalg.conv_1d(I, K, loc=loc, ip=ip, outs=[O])


def conv_1d_ncw_fcw(I, K, O, *, loc=None, ip=None):
    return linalg.conv_1d_ncw_fcw(I, K, loc=loc, ip=ip, outs=[O])


def conv_1d_nwc_wcf(I, K, O, *, loc=None, ip=None):
    return linalg.conv_1d_nwc_wcf(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nchw_fchw(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d_nchw_fchw(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_ngchw_fgchw(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d_ngchw_fgchw(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_ngchw_gfchw(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d_ngchw_gfchw(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_fhwc(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d_nhwc_fhwc(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_fhwc_q(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d_nhwc_fhwc_q(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_hwcf(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d_nhwc_hwcf(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_hwcf_q(I, K, O, *, loc=None, ip=None):
    return linalg.conv_2d_nhwc_hwcf_q(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d(I, K, O, *, loc=None, ip=None):
    return linalg.conv_3d(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d_ncdhw_fcdhw(I, K, O, *, loc=None, ip=None):
    return linalg.conv_3d_ncdhw_fcdhw(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d_ndhwc_dhwcf(I, K, O, *, loc=None, ip=None):
    return linalg.conv_3d_ndhwc_dhwcf(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d_ndhwc_dhwcf_q(I, K, O, *, loc=None, ip=None):
    return linalg.conv_3d_ndhwc_dhwcf_q(I, K, loc=loc, ip=ip, outs=[O])


def copy(I, O, *, loc=None, ip=None):
    return linalg.copy(I, loc=loc, ip=ip, outs=[O])


def depthwise_conv_1d_ncw_cw(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_1d_ncw_cw(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_1d_nwc_wc(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_1d_nwc_wc(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_1d_nwc_wcm(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_1d_nwc_wcm(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nchw_chw(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_2d_nchw_chw(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwc(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_2d_nhwc_hwc(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwc_q(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_2d_nhwc_hwc_q(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwcm(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_2d_nhwc_hwcm(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwcm_q(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_2d_nhwc_hwcm_q(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_3d_ncdhw_cdhw(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_3d_ncdhw_cdhw(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_3d_ndhwc_dhwc(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_3d_ndhwc_dhwc(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_3d_ndhwc_dhwcm(I, K, O, *, loc=None, ip=None):
    return linalg.depthwise_conv_3d_ndhwc_dhwcm(I, K, loc=loc, ip=ip, outs=[O])


def div(lhs, rhs, O, *, loc=None, ip=None):
    return linalg.div(lhs, rhs, loc=loc, ip=ip, outs=[O])


def div_unsigned(lhs, rhs, O, *, loc=None, ip=None):
    return linalg.div_unsigned(lhs, rhs, loc=loc, ip=ip, outs=[O])


def dot(A, B, C, *, loc=None, ip=None):
    return linalg.dot(A, B, loc=loc, ip=ip, outs=[C])


def elemwise_binary(lhs, rhs, O, *, loc=None, ip=None):
    return linalg.elemwise_binary(lhs, rhs, loc=loc, ip=ip, outs=[O])


def elemwise_unary(I, O, *, loc=None, ip=None):
    return linalg.elemwise_unary(I, loc=loc, ip=ip, outs=[O])


def exp(I, O, *, loc=None, ip=None):
    return linalg.exp(I, loc=loc, ip=ip, outs=[O])


def fill(v, O, *, loc=None, ip=None):
    if isinstance(v, (float, int, bool)):
        v = arith.constant(v)
    return linalg.fill(v, loc=loc, ip=ip, outs=[O])


def fill_rng_2d(min, max, seed, O, *, loc=None, ip=None):
    params = [min, max]
    for i, m in enumerate(params):
        if isinstance(m, (float, int)):
            params[i] = arith.constant(m, type=T.f64())
    min, max = params
    if isinstance(seed, int):
        seed = arith.constant(seed, T.i32())
    return linalg.fill_rng_2d(min, max, seed, loc=loc, ip=ip, outs=[O])


def floor(I, O, *, loc=None, ip=None):
    return linalg.floor(I, loc=loc, ip=ip, outs=[O])


def log(I, O, *, loc=None, ip=None):
    return linalg.log(I, loc=loc, ip=ip, outs=[O])


@linalg.linalg_structured_op
def _matmul_generic(
    A=TensorDef(T1, S.M, S.K),
    B=TensorDef(T2, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    domain(D.m, D.n, D.k)
    implements(ContractionOpInterface)
    C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


_matmul_generic.op_name = "matmul"


def matmul(A, B, C, *, loc=None, ip=None):
    op_configs = linalg.LinalgOpConfig.from_linalg_op_def(
        _matmul_generic.op_def, context=ir.Context.current
    )
    op_config = op_configs[0]
    (
        _all_arg_defs,
        _in_arg_defs,
        _out_arg_defs,
        _outs,
        result_types,
        _type_mapping,
        indexing_maps_attr,
        _iterator_types_attr,
        _index_attrs,
        _fn_attr_mapping,
        _block_arg_types,
    ) = linalg.opdsl.lang.emitter.prepare_common_structured_op(
        op_config.structured_op, A, B, outs=[C], loc=loc, ip=ip
    )
    named_op = linalg.MatmulOp(
        result_types,
        inputs=[A, B],
        outputs=[C],
        indexing_maps=indexing_maps_attr,
        cast=linalg.TypeFn.cast_signed,
        loc=loc,
        ip=ip,
    )
    linalg.fill_builtin_region(named_op.operation)
    if len(named_op.results):
        return named_op.results
    else:
        return named_op


def matmul_transpose_a(A, B, C, *, loc=None, ip=None):
    return linalg.matmul_transpose_a(A, B, loc=loc, ip=ip, outs=[C])


def matmul_transpose_b(A, B, C, *, loc=None, ip=None):
    return linalg.matmul_transpose_b(A, B, loc=loc, ip=ip, outs=[C])


def matmul_unsigned(A, B, C, *, loc=None, ip=None):
    return linalg.matmul_unsigned(A, B, loc=loc, ip=ip, outs=[C])


def matvec(A, y, x, *, loc=None, ip=None):
    return linalg.matvec(A, y, loc=loc, ip=ip, outs=[x])


def max(lhs, rhs, O, *, loc=None, ip=None):
    return linalg.max(lhs, rhs, loc=loc, ip=ip, outs=[O])


def mmt4d(lhs, rhs, accum, *, loc=None, ip=None):
    return linalg.mmt4d(lhs, rhs, loc=loc, ip=ip, outs=[accum])


def mul(lhs, rhs, O, *, loc=None, ip=None):
    return linalg.mul(lhs, rhs, loc=loc, ip=ip, outs=[O])


def negf(I, O, *, loc=None, ip=None):
    return linalg.negf(I, loc=loc, ip=ip, outs=[O])


def pooling_nchw_max(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nchw_max(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nchw_sum(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nchw_sum(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_ncw_max(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_ncw_max(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_ncw_sum(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_ncw_sum(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_ndhwc_max(I, K, O, strides, dilations, *, loc=None, ip=None):
    return linalg.pooling_ndhwc_max(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_ndhwc_min(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_ndhwc_min(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_ndhwc_sum(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_ndhwc_sum(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nhwc_max(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nhwc_max(
        I, K, strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nhwc_max_unsigned(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nhwc_max_unsigned(
        I, K, strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nhwc_min(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nhwc_min(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nhwc_min_unsigned(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nhwc_min_unsigned(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nhwc_sum(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nhwc_sum(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nwc_max(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nwc_max(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nwc_max_unsigned(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nwc_max_unsigned(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nwc_min(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nwc_min(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nwc_min_unsigned(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nwc_min_unsigned(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def pooling_nwc_sum(I, K, O, *, strides, dilations, loc=None, ip=None):
    return linalg.pooling_nwc_sum(
        I, K, strides=strides, dilations=dilations, loc=loc, ip=ip, outs=[O]
    )


def quantized_batch_matmul(A, B, C, *, loc=None, ip=None):
    return linalg.quantized_batch_matmul(A, B, loc=loc, ip=ip, outs=[C])


def quantized_matmul(A, B, C, *, loc=None, ip=None):
    return linalg.quantized_matmul(A, B, loc=loc, ip=ip, outs=[C])


def sub(lhs, rhs, O, *, loc=None, ip=None):
    return linalg.sub(lhs, rhs, loc=loc, ip=ip, outs=[O])


def vecmat(y, A, x, *, loc=None, ip=None):
    return linalg.vecmat(y, A, loc=loc, ip=ip, outs=[x])


@linalg.linalg_structured_op
def _pooling_ncdhw_max(
    I=TensorDef(
        T1,
        S.N,
        S.C,
        S.OD * S.SD + S.KD * S.DD,
        S.OH * S.SH + S.KH * S.DH,
        S.OW * S.SW + S.KW * S.DW,
    ),
    K=TensorDef(T2, S.KD, S.KH, S.KW, index_dims=[D.kd, D.kh, D.kw]),
    O=TensorDef(U, S.N, S.C, S.OD, S.OH, S.OW, output=True),
    strides=IndexAttrDef(S.SD, S.SH, S.SW, default=[1, 1, 1]),
    dilations=IndexAttrDef(S.DD, S.DH, S.DW, default=[1, 1, 1]),
):
    """Performs 3D max pooling.

    Numeric casting is performed on the input operand, promoting it to the same
    data type as the accumulator/output.
    """
    implements(ConvolutionOpInterface)
    domain(D.n, D.c, D.od, D.oh, D.ow, D.kd, D.kh, D.kw)
    O[D.n, D.c, D.od, D.oh, D.ow] = ReduceFn.max_signed[D.kd, D.kh, D.kw](
        TypeFn.cast_signed(
            U,
            I[
                D.n,
                D.c,
                D.od * S.SD + D.kd * S.DD,
                D.oh * S.SH + D.kh * S.DH,
                D.ow * S.SW + D.kw * S.DW,
            ],
        )
    )


def pooling_ncdhw_max(I, K, O, *, strides, dilations, loc=None, ip=None):
    op_configs = linalg.LinalgOpConfig.from_linalg_op_def(
        _pooling_ncdhw_max.op_def, context=ir.Context.current
    )
    op_config = op_configs[0]
    (
        _all_arg_defs,
        _in_arg_defs,
        _out_arg_defs,
        _outs,
        result_types,
        _type_mapping,
        indexing_maps_attr,
        iterator_types_attr,
        _index_attrs,
        _fn_attr_mapping,
        _block_arg_types,
    ) = linalg.opdsl.lang.emitter.prepare_common_structured_op(
        op_config.structured_op,
        I,
        K,
        strides=strides,
        dilations=dilations,
        outs=[O],
        loc=loc,
        ip=ip,
    )

    @linalg.generic(
        [I, K],
        [O],
        indexing_maps=indexing_maps_attr,
        iterator_types=iterator_types_attr,
        loc=loc,
        ip=ip,
    )
    def payload(inp, _kern, outp):
        return arith.maximumf(inp, outp)

    if len(payload.results):
        return payload.results
    else:
        return payload
