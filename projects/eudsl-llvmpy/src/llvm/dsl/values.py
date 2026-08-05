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
    ICmpPredicate,
    FCmpPredicate,
    const_int,
    const_fp,
)
from ..eudslllvm_ext.types import TypeID
from .casters import register_value_caster, maybe_downcast
from .context import current_builder


class ArithValue(Value):
    """A Value that supports arithmetic and comparison operators."""

    def _coerce(self, other):
        """Turn a Python int/float into a constant matching this value's type.

        ArithValue is only registered for integer and floating-point type kinds,
        so one of the two branches below always applies.
        """
        if isinstance(other, Value):
            return other
        ty = self.type
        if ty.is_floating_point:
            return const_fp(ty, float(other))
        return const_int(ty, int(other), signed=True)

    def _wrap(self, v):
        # Re-wrap builder results as ArithValue so `(a + b) + c` stays typed.
        return maybe_downcast(v, self)

    def _binary(self, other, int_method, float_method):
        other = self._coerce(other)
        if self.type != other.type:
            raise TypeError(f"mismatched types: {self.type} and {other.type}")
        b = current_builder()
        method = float_method if self.type.is_floating_point else int_method
        return self._wrap(getattr(b, method)(self, other))

    def _cmp(self, other, icmp_pred, fcmp_pred):
        other = self._coerce(other)
        if self.type != other.type:
            raise TypeError(f"mismatched types: {self.type} and {other.type}")
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

    def __rsub__(self, other):
        other = self._coerce(other)
        if self.type != other.type:
            raise TypeError(f"mismatched types: {other.type} and {self.type}")
        b = current_builder()
        method = "fsub" if self.type.is_floating_point else "sub"
        return self._wrap(getattr(b, method)(other, self))

    def __rmul__(self, other):
        return self._binary(other, "mul", "fmul")

    def __rtruediv__(self, other):
        other = self._coerce(other)
        if self.type != other.type:
            raise TypeError(f"mismatched types: {other.type} and {self.type}")
        b = current_builder()
        method = "fdiv" if self.type.is_floating_point else "sdiv"
        return self._wrap(getattr(b, method)(other, self))

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


class TypedPointer:
    """A pointer Value plus the element type needed for opaque-pointer GEP.

    LLVM pointers are opaque, so `ptr[i]` cannot infer what it points to. The
    DSL keys subscripting off an explicitly attached element type:
    `p = with_element_type(ptr, i32); p[2]` loads the i32 at offset 2.
    """

    def __init__(self, ptr, element_type):
        self._ptr = ptr
        self._element_type = element_type

    def _idx(self, i):
        if isinstance(i, int):
            return current_builder().i64_const(i)
        return i

    def gep(self, i):
        b = current_builder()
        return b.gep(self._element_type, self._ptr, [self._idx(i)])

    def __getitem__(self, i):
        b = current_builder()
        return maybe_downcast(b.load(self._element_type, self.gep(i)), self._ptr)

    def __setitem__(self, i, value):
        current_builder().store(value, self.gep(i))


def with_element_type(ptr, element_type):
    """Attach an element type to a pointer value for subscript/GEP sugar."""
    return TypedPointer(ptr, element_type)


def extract(aggregate, index):
    """extractvalue sugar: pull field `index` out of a struct/array value.

    Returned as its registered caster subclass when applicable (e.g. an
    integer/float field becomes an ArithValue).
    """
    return maybe_downcast(
        current_builder().extract_value(aggregate, index), aggregate
    )
