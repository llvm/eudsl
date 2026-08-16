#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Value-caster registry, backed by C++ (analogous to MLIR's register_value_caster).

The C++ type_hook already downcasts a returned Value* to its concrete bound
C++ class. This layer adds a user-extensible second step: re-wrapping the same
Value* as a Python subclass (e.g. ArithValue) keyed on LLVM Type::TypeID.
"""

from ..eudslllvm_ext.ir import (
    register_value_caster as _register,
    maybe_downcast,
    Value,
)


def register_value_caster(type_id, caster=None):
    """Register `caster` for values whose type has the given TypeID.

    Usable directly, `register_value_caster(TypeID.Integer, ArithValue)`, or as
    a decorator, `@register_value_caster(TypeID.Integer)`.
    """

    def decorator(c):
        _register(type_id.value if hasattr(type_id, 'value') else int(type_id), c)
        return c

    if caster is not None:
        return decorator(caster)
    return decorator
