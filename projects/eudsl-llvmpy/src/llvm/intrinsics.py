#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Attribute-style access to LLVM intrinsics.

`llvm.intrinsics.sqrt(module, [f32])` resolves the id for `llvm.sqrt`, checks
it exists, and emits the overloaded declaration. Overload resolution happens
in C++ against LLVM's own tables.
"""

from .eudslllvm_ext.intrinsics import (  # noqa: F401
    lookup_intrinsic_id,
    intrinsic_is_overloaded,
    get_intrinsic_declaration,
)


def __getattr__(name):
    intrinsic_id = lookup_intrinsic_id(f"llvm.{name.replace('_', '.')}")
    if intrinsic_id == 0:
        raise AttributeError(f"unknown intrinsic llvm.{name}")

    def declare(module, overload_types=()):
        return get_intrinsic_declaration(module, intrinsic_id, list(overload_types))

    declare.__name__ = name
    return declare
