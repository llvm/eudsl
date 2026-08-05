#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Typed values: ArithValue, an llvm.Value subclass carrying operator overloads.

Mirrors ArithValue in mlir/extras/dialects/arith.py and uses the same
register_value_caster mechanism: integer- and float-typed values arrive from the
bindings (via maybe_downcast) as ArithValue, so `a + b`, `a < b`, etc. exist
only on values where they make sense rather than being monkey-patched onto every
Value. `+` picks add or fadd from the operand type; Python scalars coerce to
constants of the other operand's type; results are re-wrapped as ArithValue so
chaining stays typed.
"""

from ..eudslllvm_ext import (
    Value,
    TypeID,
    ICmpPredicate,
    FCmpPredicate,
    const_int,
    const_fp,
)
from .casters import register_value_caster, maybe_downcast
from .context import current_builder


class ArithValue(Value):
    """A Value that supports arithmetic and comparison operators."""

    def _coerce(self, other):
        """Turn a Python int/float into a constant matching this value's type."""
        if isinstance(other, Value):
            return other
        ty = self.type
        if ty.is_floating_point:
            return const_fp(ty, float(other))
        if ty.is_integer:
            return const_int(ty, int(other), signed=True)
        raise TypeError(f"cannot coerce {other!r} to {ty}")

    def _wrap(self, v):
        # Re-wrap builder results as ArithValue so `(a + b) + c` stays typed.
        return maybe_downcast(v, self)

    def _binary(self, other, int_method, float_method):
        other = self._coerce(other)
        b = current_builder()
        method = float_method if self.type.is_floating_point else int_method
        return self._wrap(getattr(b, method)(self, other))

    def _cmp(self, other, icmp_pred, fcmp_pred):
        other = self._coerce(other)
        b = current_builder()
        if self.type.is_floating_point:
            return self._wrap(
                b.fcmp(getattr(FCmpPredicate, fcmp_pred), self, other)
            )
        return self._wrap(b.icmp(getattr(ICmpPredicate, icmp_pred), self, other))

    def __add__(self, other):
        return self._binary(other, "add", "fadd")

    def __sub__(self, other):
        return self._binary(other, "sub", "fsub")

    def __mul__(self, other):
        return self._binary(other, "mul", "fmul")

    def __truediv__(self, other):
        return self._binary(other, "sdiv", "fdiv")

    def __radd__(self, other):
        return self._binary(other, "add", "fadd")

    def __rmul__(self, other):
        return self._binary(other, "mul", "fmul")

    # Comparisons default to signed-integer / ordered-float predicates.
    def __lt__(self, other):
        return self._cmp(other, "SLT", "OLT")

    def __le__(self, other):
        return self._cmp(other, "SLE", "OLE")

    def __gt__(self, other):
        return self._cmp(other, "SGT", "OGT")

    def __ge__(self, other):
        return self._cmp(other, "SGE", "OGE")

    # __eq__/__ne__ stay as the base Value identity comparison so the value
    # remains hashable and usable as a dict key in traversal helpers; value
    # comparison is exposed by name.
    __hash__ = Value.__hash__

    def eq(self, other):
        return self._cmp(other, "EQ", "OEQ")

    def ne(self, other):
        return self._cmp(other, "NE", "ONE")


# Register ArithValue for the integer and floating-point type kinds. Casters are
# keyed on Type.type_id (a TypeID), which is width- and interning-independent, so
# one registration per family covers every width.
def install_value_casters():
    for tid in (
        TypeID.Integer,
        TypeID.Half,
        TypeID.BFloat,
        TypeID.Float,
        TypeID.Double,
    ):
        register_value_caster(tid, ArithValue)
