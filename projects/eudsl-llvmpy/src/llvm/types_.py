#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.
from typing import Sequence

from . import (
    array_type2,
    TypeRef,
    b_float_type_in_context,
    double_type_in_context,
    float_type_in_context,
    fp128_type_in_context,
    function_type,
    half_type_in_context,
    int_type_in_context,
    int1_type_in_context,
    int8_type_in_context,
    int16_type_in_context,
    int32_type_in_context,
    int64_type_in_context,
    int_ptr_type_in_context,
    int128_type_in_context,
    TargetDataRef,
    label_type_in_context,
    metadata_type_in_context,
    pointer_type_in_context,
    ppcfp128_type_in_context,
    scalable_vector_type,
    struct_type_in_context,
    token_type_in_context,
    vector_type,
    void_type_in_context,
    x86_amx_type_in_context,
    x86_fp80_type_in_context,
)
from .context import current_context


def array(element_type: TypeRef, element_count: int):
    return array_type2(element_type, element_count)


def bfloat():
    return b_float_type_in_context(current_context().context)


def double():
    return double_type_in_context(current_context().context)


def float():
    return float_type_in_context(current_context().context)


def fp128():
    return fp128_type_in_context(current_context().context)


def function(
    return_type: TypeRef,
    param_types: Sequence[TypeRef],
    is_var_arg: bool = False,
):
    return function_type(return_type, param_types, is_var_arg)


def half():
    return half_type_in_context(current_context().context)


def int128():
    return int128_type_in_context(current_context().context)


def int16():
    return int16_type_in_context(current_context().context)


def int1():
    return int1_type_in_context(current_context().context)


def int32():
    return int32_type_in_context(current_context().context)


def int64():
    return int64_type_in_context(current_context().context)


def int8():
    return int8_type_in_context(current_context().context)


def int_ptr(td: TargetDataRef):
    return int_ptr_type_in_context(current_context().context, td)


def int(num_bits: int):
    return int_type_in_context(current_context().context, num_bits)


def label():
    return label_type_in_context(current_context().context)


def metadata():
    return metadata_type_in_context(current_context().context)


def pointer(address_space: int):
    return pointer_type_in_context(current_context().context, address_space)


def ppcfp128_type():
    return ppcfp128_type_in_context(current_context().context)


def scalable_vector(element_type: TypeRef, element_count: int):
    return scalable_vector_type(element_type, element_count)


def struct(element_types: Sequence[TypeRef], packed: bool):
    return struct_type_in_context(
        current_context().context, element_types, packed
    )


def token():
    return token_type_in_context(current_context().context)


def vector(element_type: TypeRef, element_count: int):
    return vector_type(element_type, element_count)


def void():
    return void_type_in_context(current_context().context)


def x86_amx():
    return x86_amx_type_in_context(current_context().context)


def x86_fp80():
    return x86_fp80_type_in_context(current_context().context)
