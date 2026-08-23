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
"""

import inspect

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
        # Anchor to the builder tracing this value's MachineFunction. A
        # MachineValue carries a bare reg + llt, so a value that escapes its
        # @machine_function body could otherwise emit into a different (e.g.
        # nested) function's builder, referencing vregs it doesn't own. A
        # MachineValue is only ever constructed while a builder is current.
        self._builder = current_machine_builder()

    def _require_current(self):
        if self._builder is not current_machine_builder():
            raise RuntimeError(
                "MachineValue used outside the @machine_function body that "
                "created it (it belongs to a different MachineFunction)"
            )

    def _coerce(self, other):
        """A MachineValue passes through; a Python int becomes a G_CONSTANT of
        this value's LLT. Only an exact int (not bool, not float) is accepted --
        int(other) would silently truncate a float or parse a str into a
        wrong-typed constant that survives into the release wheel."""
        self._require_current()
        if isinstance(other, MachineValue):
            other._require_current()
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


class DSLMachineFunction:
    """The result of @machine_function: the built MachineFunction plus the
    MachineModuleInfo that owns it."""

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
