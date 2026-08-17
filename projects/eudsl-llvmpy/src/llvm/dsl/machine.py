#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The @machine_function decorator and MachineValue: a Pythonic front end for
building generic (GlobalISel) MIR, mirroring the IR DSL's @function/ArithValue.

Unlike the IR builder, MachineIRBuilder is not wired into the C++ InsertPoint
stack, so the "current builder" is tracked here with a plain stack pushed for
the duration of tracing a @machine_function body. A MachineValue wraps a
Register plus its LLT and overloads `+ - *` onto build_add/build_sub/build_mul;
Python ints coerce to a G_CONSTANT of the other operand's type.
"""

import inspect

from ..eudslllvm_ext.mir import LLT, MachineIRBuilder, create_machine_function

_builder_stack = []


def current_machine_builder():
    """The MachineIRBuilder for the @machine_function body being traced."""
    if not _builder_stack:
        raise RuntimeError(
            "no current MachineIRBuilder; use this only inside a @machine_function body"
        )
    return _builder_stack[-1]


class MachineValue:
    """A generic-MIR value: a Register plus its LLT, with operator overloads."""

    def __init__(self, reg, llt):
        self.reg = reg
        self.llt = llt

    def _coerce(self, other):
        """A MachineValue passes through; a Python int becomes a G_CONSTANT."""
        if isinstance(other, MachineValue):
            return other
        reg = current_machine_builder().build_constant(self.llt, int(other))
        return MachineValue(reg, self.llt)

    def _binary(self, other, method, *, reflected=False):
        other = self._coerce(other)
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

    def __init__(self, mmi, name):
        self.mmi = mmi
        self.name = name

    @property
    def machine_function(self):
        return self.mmi.machine_function(self.name)

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
    vreg; the body emits instructions through the contextual builder."""

    def decorator(f):
        sig = inspect.signature(f)
        param_llts = [_resolve_llt(p.annotation) for p in sig.parameters.values()]
        fn_name = name or f.__name__

        mmi = create_machine_function(module, target, fn_name)
        mf = mmi.machine_function(fn_name)
        builder = MachineIRBuilder(mf)

        _builder_stack.append(builder)
        try:
            args = [
                MachineValue(mf.create_generic_virtual_register(llt), llt)
                for llt in param_llts
            ]
            f(*args)
        finally:
            _builder_stack.pop()
        return DSLMachineFunction(mmi, fn_name)

    return decorator
