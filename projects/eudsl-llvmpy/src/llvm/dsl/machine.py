#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The @machine_function decorator and MachineValue: a Pythonic front end for
building generic (GlobalISel) MIR, mirroring the IR DSL's @function/ArithValue.

The "current builder" is tracked in C++ on a thread-local stack (the same
mechanism the IR builder uses): `with MachineIRBuilder(mf):` makes it current
and current_machine_builder() reads it back. A MachineValue wraps a Register
plus its LLT and overloads `+ - *` onto build_add/build_sub/build_mul; Python
ints coerce to a G_CONSTANT of the other operand's type.

A Register now carries the MachineFunction that minted it, so feeding a value
from one @machine_function body into another's builder is rejected in C++ (the
builder compares the register's owner against its own function) -- the DSL does
not need a separate Python-side anchor for that.
"""

import inspect

from ..eudslllvm_ext.ir import ICmpPredicate
from ..eudslllvm_ext.mir import (
    LLT,
    MachineIRBuilder,
    create_machine_function,
    current_machine_builder,
)


class MachineValue:
    """A generic-MIR value: a Register plus its LLT, with operator overloads."""

    def __init__(self, reg, llt):
        self.reg = reg
        self.llt = llt

    def _coerce(self, other):
        """A MachineValue passes through; a Python int becomes a G_CONSTANT of
        this value's LLT. Only an exact int (not bool, not float) is accepted --
        int(other) would silently truncate a float or parse a str into a
        wrong-typed constant that survives into the release wheel."""
        if isinstance(other, MachineValue):
            return other
        if type(other) is not int:
            raise TypeError(
                f"cannot coerce {other!r} to a {self.llt} constant; "
                "expected a MachineValue or an int"
            )
        reg = current_machine_builder().build_constant(self.llt, other)
        return MachineValue(reg, self.llt)

    def _binary(self, other, method, *, reflected=False):
        other = self._coerce(other)
        if self.llt != other.llt:
            raise TypeError(f"mismatched types: {self.llt} and {other.llt}")
        b = current_machine_builder()
        lhs, rhs = (other, self) if reflected else (self, other)
        reg = getattr(b, method)(self.llt, lhs.reg, rhs.reg)
        return MachineValue(reg, self.llt)

    def __add__(self, other):
        return self._binary(other, "build_add")

    def __sub__(self, other):
        return self._binary(other, "build_sub")

    def __mul__(self, other):
        return self._binary(other, "build_mul")

    def __radd__(self, other):
        return self._binary(other, "build_add", reflected=True)

    def __rmul__(self, other):
        return self._binary(other, "build_mul", reflected=True)

    def __rsub__(self, other):
        return self._binary(other, "build_sub", reflected=True)

    def _cmp(self, other, predicate):
        """Emit a G_ICMP, producing an i1 (LLT.scalar(1)) MachineValue.
        Comparisons default to signed-integer predicates; the unsigned
        ult/ule/ugt/uge methods request the unsigned ones."""
        other = self._coerce(other)
        if self.llt != other.llt:
            raise TypeError(f"mismatched types: {self.llt} and {other.llt}")
        i1 = LLT.scalar(1)
        reg = current_machine_builder().build_icmp(predicate, i1, self.reg, other.reg)
        return MachineValue(reg, i1)

    def __lt__(self, other):
        return self._cmp(other, ICmpPredicate.SLT)

    def __le__(self, other):
        return self._cmp(other, ICmpPredicate.SLE)

    def __gt__(self, other):
        return self._cmp(other, ICmpPredicate.SGT)

    def __ge__(self, other):
        return self._cmp(other, ICmpPredicate.SGE)

    # __eq__/__ne__ stay identity (so a MachineValue is hashable and usable in
    # traversal); value equality is exposed by name, like ArithValue. NOTE: in a
    # @machine_function body `if a == b:` therefore compares Python identity (a
    # compile-time False for two distinct values), NOT a G_ICMP -- use a.eq(b) /
    # a.ne(b) for a value comparison, and the ult/ule/ugt/uge methods for
    # unsigned ordering (the < <= > >= operators are signed).
    __hash__ = object.__hash__

    def eq(self, other):
        return self._cmp(other, ICmpPredicate.EQ)

    def ne(self, other):
        return self._cmp(other, ICmpPredicate.NE)

    def ult(self, other):
        return self._cmp(other, ICmpPredicate.ULT)

    def ule(self, other):
        return self._cmp(other, ICmpPredicate.ULE)

    def ugt(self, other):
        return self._cmp(other, ICmpPredicate.UGT)

    def uge(self, other):
        return self._cmp(other, ICmpPredicate.UGE)


class DSLMachineFunction:
    """The result of @machine_function: the built MachineFunction plus the
    MirModule that owns it."""

    def __init__(self, mmi, name, machine_function):
        self.mmi = mmi
        self.name = name
        self._machine_function = machine_function

    @property
    def machine_function(self):
        return self._machine_function

    def to_mir(self):
        return self.mmi.to_mir()


def _resolve_llt(annotation):
    if isinstance(annotation, LLT):
        return annotation
    raise TypeError(
        f"@machine_function parameters must be annotated with an LLT, got {annotation!r}"
    )


def machine_function(*, module, target, name=None):
    """Build a generic-MIR MachineFunction by tracing `f`. Each parameter is
    annotated with an LLT and arrives as a MachineValue over a fresh generic
    vreg; the body emits instructions through the contextual builder.

    The body's return value is not turned into a terminator -- this traces a
    straight-line generic-MIR fragment, so `return a + b` just ends tracing.
    If the body raises, the partially-built MachineFunction is left in `mmi`
    (there is no rollback), but the exception propagates unchanged.
    """

    def decorator(f):
        params = list(inspect.signature(f).parameters.values())
        for p in params:
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise TypeError(
                    f"@machine_function parameter {p.name!r} must be positional; "
                    "*args, **kwargs, and keyword-only parameters are not supported"
                )
            if p.annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"@machine_function parameter {p.name!r} is missing an LLT "
                    "annotation"
                )
        param_llts = [_resolve_llt(p.annotation) for p in params]
        fn_name = name or f.__name__

        mmi = create_machine_function(module, target, fn_name)
        mf = mmi.machine_function(fn_name)
        with MachineIRBuilder(mf):
            args = [
                MachineValue(mf.create_generic_virtual_register(llt), llt)
                for llt in param_llts
            ]
            f(*args)
        return DSLMachineFunction(mmi, fn_name, mf)

    return decorator
