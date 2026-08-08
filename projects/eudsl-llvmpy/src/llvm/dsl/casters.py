#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Python value-caster registry, analogous to MLIR's register_value_caster.

The C++ type_hook already downcasts a returned Value* to its concrete bound
C++ class (Instruction, ConstantInt, ...). This layer adds a user-extensible
step on top: register a Python callable (typically a Value subclass such as the
DSL's ArithValue) keyed on an llvm Type kind (Type.type_id), and maybe_downcast
re-wraps a value of that type as the subclass.

The registry lives here in Python (not in a C++ static) so nothing holds Python
references at interpreter-shutdown time. C++ provides only the stateless
`_wrap_value_as` primitive (nb::inst_reference).
"""

from ..eudslllvm_ext import Value, _wrap_value_as

# Type.type_id (a TypeID enum value) -> caster callable.
_casters: dict = {}


def register_value_caster(type_id, caster=None):
    """Register `caster` for values whose type has TypeID `type_id`.

    Usable directly, `register_value_caster(TypeID.Integer, ArithValue)`, or as
    a decorator, `@register_value_caster(TypeID.Integer)`.
    """

    def decorator(c):
        _casters[type_id] = c
        return c

    if caster is not None:
        return decorator(caster)
    return decorator


def maybe_downcast(value: Value, parent=None) -> Value:
    """Re-wrap `value` as its registered caster subclass, if any.

    `parent` (a Context/Module/Value) ties the wrapper's lifetime to the owner
    of the underlying pointer, matching the reference_internal convention used
    throughout the bindings.
    """
    caster = _casters.get(value.type.type_id)
    if caster is None:
        return value
    return _wrap_value_as(value, caster, parent)
