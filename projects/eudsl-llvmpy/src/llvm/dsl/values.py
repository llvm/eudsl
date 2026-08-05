#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Operator overloading on llvm.Value, dispatching on Value.type.

Mirrors ArithValue in mlir/extras/dialects/arith.py: `+` picks add or fadd from
the operand type, and Python scalars coerce to constants of the other operand's
type. nanobind classes are heap types, so assigning these to the bound Value
class installs the number slots correctly.
"""

from .. import Value, const_int, const_fp, ICmpPredicate, FCmpPredicate
from .context import current_builder


def _coerce(value, like):
    """Turn a Python int/float into a constant matching `like`'s type."""
    if isinstance(value, Value):
        return value
    ty = like.type
    if ty.is_floating_point:
        return const_fp(ty, float(value))
    if ty.is_integer:
        return const_int(ty, int(value), signed=True)
    raise TypeError(f"cannot coerce {value!r} to {ty}")


def _binary(method_int, method_float):
    def op(self, other):
        other = _coerce(other, self)
        b = current_builder()
        if self.type.is_floating_point:
            return getattr(b, method_float)(self, other)
        return getattr(b, method_int)(self, other)

    return op


def _rbinary(forward):
    def op(self, other):
        other = _coerce(other, self)
        return forward(other, self)

    return op


def _cmp(icmp_pred, fcmp_pred):
    def op(self, other):
        other = _coerce(other, self)
        b = current_builder()
        if self.type.is_floating_point:
            return b.fcmp(getattr(FCmpPredicate, fcmp_pred), self, other)
        return b.icmp(getattr(ICmpPredicate, icmp_pred), self, other)

    return op


def install_value_dunders():
    Value.__add__ = _binary("add", "fadd")
    Value.__sub__ = _binary("sub", "fsub")
    Value.__mul__ = _binary("mul", "fmul")
    Value.__truediv__ = _binary("sdiv", "fdiv")
    Value.__radd__ = _rbinary(Value.__add__)
    Value.__rmul__ = _rbinary(Value.__mul__)
    # Comparisons default to signed integer / ordered float predicates.
    Value.__lt__ = _cmp("SLT", "OLT")
    Value.__le__ = _cmp("SLE", "OLE")
    Value.__gt__ = _cmp("SGT", "OGT")
    Value.__ge__ = _cmp("SGE", "OGE")
    # __eq__/__ne__ stay as the C++ identity comparison so Value remains
    # hashable and usable as a dict key; value comparison is by name.
    Value.eq = _cmp("EQ", "OEQ")
    Value.ne = _cmp("NE", "ONE")
