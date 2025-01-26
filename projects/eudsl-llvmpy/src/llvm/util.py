#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

from . import (
    get_intrinsic_declaration,
    lookup_intrinsic_id,
    type_of,
    build_call2,
    intrinsic_is_overloaded,
    intrinsic_get_type,
)
from .context import current_context


def call_intrinsic(intr_name, *args, **kwargs):
    intr_id = lookup_intrinsic_id(intr_name, len(intr_name))
    types = [type_of(a) for a in args]
    if intrinsic_is_overloaded(intr_id):
        intr_decl_fn = get_intrinsic_declaration(
            current_context().module, intr_id, types
        )
        intr_decl_fn_ty = intrinsic_get_type(current_context().context, intr_id, types)
    else:
        intr_decl_fn = get_intrinsic_declaration(current_context().module, intr_id, [])
        intr_decl_fn_ty = intrinsic_get_type(current_context().context, intr_id, [])

    name = kwargs.pop("name", "")
    return build_call2(
        current_context().builder,
        intr_decl_fn_ty,
        intr_decl_fn,
        args,
        name,
    )
