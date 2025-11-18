# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..util import get_user_code_loc
from ... import ir
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
from ...dialects.rocdl import *

_wmma_f16_16x16x16_f16 = wmma_f16_16x16x16_f16


def wmma_f16_16x16x16_f16(A, B, C, *, opsel=False, loc=None, ip=None) -> ir.Value:
    v16 = ir.VectorType.get((16,), ir.F16Type.get())
    return _wmma_f16_16x16x16_f16(v16, A, B, C, opsel=opsel, loc=loc, ip=ip).result
