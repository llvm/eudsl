#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Free-function forms of the IRBuilder instruction emitters.

Each function mirrors the matching ``IRBuilder`` method and takes an optional
keyword-only ``builder``; when omitted, the contextual builder is used (the one
made current by ``with builder:`` or ``with InsertPoint(...):``, via
``current_builder()``). So inside such a context you can write ``load(i32, p)``
instead of ``b.load(i32, p)``.

Operands that are plain Python numbers are materialized into LLVM constants
before the instruction is emitted, so ``add(x, 1)``, ``ret(0)``, or
``call(f, [5])`` work without spelling out ``i32_const``. The constant's type is
inferred from the surrounding operand: a sibling SSA operand for binary
ops/comparisons/select, the enclosing function's return type for ``ret``, and
the callee's parameter type for ``call``. Integer indices to ``gep``,
``extract_element`` and ``insert_element`` become ``i32`` constants. Where no
type can be inferred (e.g. both operands are numbers), a ``TypeError`` is
raised; pass an explicit constant in that case.
"""

from .ir import const_fp, const_int, current_builder

__all__ = [
    # terminators
    "ret",
    "br",
    "cond_br",
    "unreachable",
    "switch_",
    "indirect_br",
    "resume",
    # integer / float binary
    "add",
    "fadd",
    "sub",
    "fsub",
    "mul",
    "fmul",
    "sdiv",
    "udiv",
    "fdiv",
    "srem",
    "urem",
    "frem",
    "shl",
    "lshr",
    "ashr",
    "and_",
    "or_",
    "xor",
    # unary
    "neg",
    "fneg",
    "not_",
    # comparisons
    "icmp",
    "fcmp",
    # casts
    "trunc",
    "zext",
    "sext",
    "fptoui",
    "fptosi",
    "uitofp",
    "sitofp",
    "fptrunc",
    "fpext",
    "ptrtoint",
    "inttoptr",
    "bitcast",
    "addrspacecast",
    "ptrtoaddr",
    # memory / atomics
    "alloca",
    "load",
    "store",
    "gep",
    "fence",
    "atomic_rmw",
    "atomic_cmpxchg",
    # calls, phis, constants
    "call",
    "call_intrinsic",
    "phi",
    "i64_const",
    "i32_const",
    # aggregates / vectors
    "extract_value",
    "insert_value",
    "extract_element",
    "insert_element",
    "shuffle_vector",
    # misc
    "select",
    "freeze",
    "va_arg",
]


def _builder(builder):
    return current_builder() if builder is None else builder


def _is_number(x):
    # bool is a subclass of int; both are handled by _const_of.
    return isinstance(x, (int, float))


def _const_of(value, ref_type):
    """Materialize a Python number as an LLVM constant of ``ref_type``.

    ``ref_type`` (the operand/return/parameter type the constant is matched to)
    must be a scalar integer or floating-point type; anything else (e.g. a
    vector) raises, since there is no unambiguous constant to build. A float is
    rejected for an integer type rather than silently truncated."""
    if ref_type.is_integer:
        if isinstance(value, float):
            raise TypeError(
                f"cannot use float {value!r} as an integer constant of " f"{ref_type}"
            )
        # Interpret negatives as signed and non-negatives as unsigned so the
        # full [-2**(w-1), 2**w) range is accepted (e.g. the mask 0xFF into i8).
        return const_int(ref_type, int(value), signed=value < 0)
    if ref_type.is_floating_point:
        return const_fp(ref_type, float(value))
    raise TypeError(
        f"cannot build a constant of type {ref_type} from {value!r}; "
        "pass an explicit value"
    )


def _coerce_pair(lhs, rhs):
    """Coerce whichever of ``lhs``/``rhs`` is a Python number to a constant of
    the other's type. Both being numbers leaves no type to infer."""
    lhs_num, rhs_num = _is_number(lhs), _is_number(rhs)
    if lhs_num and rhs_num:
        raise TypeError(
            "at least one operand must be an SSA value so the constant's type "
            "can be inferred; pass an explicit value"
        )
    if lhs_num:
        lhs = _const_of(lhs, rhs.type)
    elif rhs_num:
        rhs = _const_of(rhs, lhs.type)
    return lhs, rhs


# --- Terminators --------------------------------------------------------------


def ret(value=None, *, builder=None):
    b = _builder(builder)
    if _is_number(value):
        value = _const_of(value, b.insert_block.parent.return_type)
    return b.ret(value)


def br(dest, *, builder=None):
    return _builder(builder).br(dest)


def cond_br(cond, true_dest, false_dest, *, builder=None):
    return _builder(builder).cond_br(cond, true_dest, false_dest)


def unreachable(*, builder=None):
    return _builder(builder).unreachable()


def switch_(value, default_dest, num_cases=10, *, builder=None):
    return _builder(builder).switch_(value, default_dest, num_cases)


def indirect_br(address, num_dests=10, *, builder=None):
    return _builder(builder).indirect_br(address, num_dests)


def resume(exn, *, builder=None):
    return _builder(builder).resume(exn)


# --- Binary arithmetic / bitwise ----------------------------------------------


def add(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).add(lhs, rhs, name)


def fadd(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).fadd(lhs, rhs, name)


def sub(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).sub(lhs, rhs, name)


def fsub(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).fsub(lhs, rhs, name)


def mul(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).mul(lhs, rhs, name)


def fmul(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).fmul(lhs, rhs, name)


def sdiv(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).sdiv(lhs, rhs, name)


def udiv(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).udiv(lhs, rhs, name)


def fdiv(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).fdiv(lhs, rhs, name)


def srem(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).srem(lhs, rhs, name)


def urem(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).urem(lhs, rhs, name)


def frem(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).frem(lhs, rhs, name)


def shl(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).shl(lhs, rhs, name)


def lshr(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).lshr(lhs, rhs, name)


def ashr(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).ashr(lhs, rhs, name)


def and_(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).and_(lhs, rhs, name)


def or_(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).or_(lhs, rhs, name)


def xor(lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).xor(lhs, rhs, name)


# --- Unary --------------------------------------------------------------------


def neg(value, name="", *, builder=None):
    return _builder(builder).neg(value, name)


def fneg(value, name="", *, builder=None):
    return _builder(builder).fneg(value, name)


def not_(value, name="", *, builder=None):
    return _builder(builder).not_(value, name)


# --- Comparisons --------------------------------------------------------------


def icmp(predicate, lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).icmp(predicate, lhs, rhs, name)


def fcmp(predicate, lhs, rhs, name="", *, builder=None):
    lhs, rhs = _coerce_pair(lhs, rhs)
    return _builder(builder).fcmp(predicate, lhs, rhs, name)


# --- Casts --------------------------------------------------------------------


def trunc(value, dest_type, name="", *, builder=None):
    return _builder(builder).trunc(value, dest_type, name)


def zext(value, dest_type, name="", *, builder=None):
    return _builder(builder).zext(value, dest_type, name)


def sext(value, dest_type, name="", *, builder=None):
    return _builder(builder).sext(value, dest_type, name)


def fptoui(value, dest_type, name="", *, builder=None):
    return _builder(builder).fptoui(value, dest_type, name)


def fptosi(value, dest_type, name="", *, builder=None):
    return _builder(builder).fptosi(value, dest_type, name)


def uitofp(value, dest_type, name="", *, builder=None):
    return _builder(builder).uitofp(value, dest_type, name)


def sitofp(value, dest_type, name="", *, builder=None):
    return _builder(builder).sitofp(value, dest_type, name)


def fptrunc(value, dest_type, name="", *, builder=None):
    return _builder(builder).fptrunc(value, dest_type, name)


def fpext(value, dest_type, name="", *, builder=None):
    return _builder(builder).fpext(value, dest_type, name)


def ptrtoint(value, dest_type, name="", *, builder=None):
    return _builder(builder).ptrtoint(value, dest_type, name)


def inttoptr(value, dest_type, name="", *, builder=None):
    return _builder(builder).inttoptr(value, dest_type, name)


def bitcast(value, dest_type, name="", *, builder=None):
    return _builder(builder).bitcast(value, dest_type, name)


def addrspacecast(value, dest_type, name="", *, builder=None):
    return _builder(builder).addrspacecast(value, dest_type, name)


def ptrtoaddr(value, name="", *, builder=None):
    return _builder(builder).ptrtoaddr(value, name)


# --- Memory / atomics ---------------------------------------------------------


def alloca(type, name="", *, builder=None):
    return _builder(builder).alloca(type, name)


def load(type, ptr, name="", *, builder=None):
    return _builder(builder).load(type, ptr, name)


def store(value, ptr, *, builder=None):
    return _builder(builder).store(value, ptr)


def gep(type, ptr, indices, name="", *, builder=None):
    b = _builder(builder)
    # i32 is valid for both array/pointer and (mandatory for) struct indices.
    indices = [b.i32_const(i) if isinstance(i, int) else i for i in indices]
    return b.gep(type, ptr, indices, name)


def fence(ordering, single_thread=False, *, builder=None):
    return _builder(builder).fence(ordering, single_thread)


def atomic_rmw(op, ptr, value, ordering, single_thread=False, name="", *, builder=None):
    return _builder(builder).atomic_rmw(op, ptr, value, ordering, single_thread, name)


def atomic_cmpxchg(
    ptr,
    cmp,
    new_value,
    success_ordering,
    failure_ordering,
    single_thread=False,
    name="",
    *,
    builder=None,
):
    return _builder(builder).atomic_cmpxchg(
        ptr, cmp, new_value, success_ordering, failure_ordering, single_thread, name
    )


# --- Calls, phis, constants ---------------------------------------------------


def call(fn, args, name="", *, builder=None):
    # A number arg is coerced to its declared parameter type. Args beyond a
    # varargs callee's fixed params have no declared type, so a number there is
    # rejected (pass an explicit constant) rather than handed to the binding raw.
    fty = fn.function_type
    n = fty.num_params
    coerced = []
    for i, a in enumerate(args):
        if _is_number(a):
            if i >= n:
                raise TypeError(
                    f"cannot infer a type for number argument {a!r} at "
                    f"position {i}, beyond the callee's {n} fixed parameter(s); "
                    "pass an explicit constant"
                )
            a = _const_of(a, fty.param_type(i))
        coerced.append(a)
    return _builder(builder).call(fn, coerced, name)


def call_intrinsic(intrinsic_id, overload_types, args, name="", *, builder=None):
    return _builder(builder).call_intrinsic(intrinsic_id, overload_types, args, name)


def phi(type, name="", *, builder=None):
    return _builder(builder).phi(type, name)


def i64_const(value, *, builder=None):
    return _builder(builder).i64_const(value)


def i32_const(value, *, builder=None):
    return _builder(builder).i32_const(value)


# --- Aggregates / vectors -----------------------------------------------------


def extract_value(aggregate, index, name="", *, builder=None):
    return _builder(builder).extract_value(aggregate, index, name)


def insert_value(aggregate, value, index, name="", *, builder=None):
    return _builder(builder).insert_value(aggregate, value, index, name)


def extract_element(vector, index, name="", *, builder=None):
    b = _builder(builder)
    if isinstance(index, int):
        index = b.i32_const(index)
    return b.extract_element(vector, index, name)


def insert_element(vector, element, index, name="", *, builder=None):
    b = _builder(builder)
    if isinstance(index, int):
        index = b.i32_const(index)
    return b.insert_element(vector, element, index, name)


def shuffle_vector(v1, v2, mask, name="", *, builder=None):
    return _builder(builder).shuffle_vector(v1, v2, mask, name)


# --- Misc ---------------------------------------------------------------------


def select(cond, true_value, false_value, name="", *, builder=None):
    true_value, false_value = _coerce_pair(true_value, false_value)
    return _builder(builder).select(cond, true_value, false_value, name)


def freeze(value, name="", *, builder=None):
    return _builder(builder).freeze(value, name)


def va_arg(arg_list, type, name="", *, builder=None):
    return _builder(builder).va_arg(arg_list, type, name)
