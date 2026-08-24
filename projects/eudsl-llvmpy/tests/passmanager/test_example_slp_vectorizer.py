#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A minimal SLP-style vectorizer written as a Python function pass, exercising
the object-level bindings end to end: it finds two independent scalar `add`s in
the entry block, packs their operands into <2 x T> vectors, replaces them with a
single vector `add`, rewires the scalar uses to extractelements, and erases the
dead scalar adds. No LLVM analyses are used -- the (trivial) legality check is
done in Python."""

import ctypes
from textwrap import dedent

import pytest

import llvm
from llvm import ir, jit, types
from llvm import instructions as I
from llvm.testing import assert_no_leaks

_SRC = dedent("""\
    define void @f(ptr %pa, ptr %pb, ptr %pc, ptr %pd) {
    entry:
      %a0 = load i32, ptr %pa
      %b0 = load i32, ptr %pb
      %a1 = load i32, ptr %pc
      %b1 = load i32, ptr %pd
      %s0 = add i32 %a0, %b0
      %s1 = add i32 %a1, %b1
      store i32 %s0, ptr %pa
      store i32 %s1, ptr %pc
      ret void
    }
    """)


def _reaches(user, target, block):
    """True if `user` transitively consumes `target` within `block` (a data
    dependence walk up the operand chains -- no LLVM analysis)."""
    seen, stack = set(), [user]
    while stack:
        v = stack.pop()
        for op in v.operands:
            if op is target:
                return True
            if isinstance(op, ir.Instruction) and op.parent is block and op not in seen:
                seen.add(op)
                stack.append(op)
    return False


def slp_vectorize(fn):
    """Combine the first two scalar `add`s in the entry block into one <2 x T>
    `add`, but only when it is legal to do so."""
    bb = fn.entry_block
    adds = [i for i in bb.instructions if i.opcode_name == "add"]
    if len(adds) < 2:
        return False
    x, y = adds[0], adds[1]

    # Legality (checked, not assumed): same element type; both pure (no memory
    # traffic or other side effects); and mutually independent -- neither
    # transitively feeds the other -- so a single vector op can replace both.
    # The independence walk is the load-bearing check here and is exercised in
    # both the direct and transitive tests. The purity guard is a general-
    # legality illustration: candidates are integer `add`s, which are always
    # pure, so for this candidate set it never fires -- a real SLP pass over a
    # wider opcode set (e.g. loads/calls) would need it.
    if x.type is not y.type:
        return False
    if any(i.may_read_or_write_memory or i.may_have_side_effects for i in (x, y)):
        return False
    if _reaches(x, y, bb) or _reaches(y, x, bb):
        return False

    # Insert after whichever add comes later, so all four scalar operands
    # dominate the new vector op. `adds` is in program order, so x precedes y
    # here and `last` is always y; comes_before is used to keep the rule correct
    # if candidate selection ever stops being ordered.
    last = y if x.comes_before(y) else x
    vec = types.vector(x.type, 2)  # element type taken from the scalar op
    b = ir.IRBuilder()
    with ir.InsertPoint.after(last, builder=b):
        lhs = I.insert_element(
            I.insert_element(ir.poison(vec), x.operand(0), 0), y.operand(0), 1
        )
        rhs = I.insert_element(
            I.insert_element(ir.poison(vec), x.operand(1), 0), y.operand(1), 1
        )
        vsum = I.add(lhs, rhs)
        lane0 = I.extract_element(vsum, 0)
        lane1 = I.extract_element(vsum, 1)

    x.replace_all_uses_with(lane0)
    y.replace_all_uses_with(lane1)
    x.erase_from_parent()  # dead now that their uses are rewired
    y.erase_from_parent()
    return True  # IR changed


def test_slp_vectorizer_combines_two_adds():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_python_pass_on_function(mod, slp_vectorize)
        text = str(mod)
        assert "add <2 x i32>" in text  # a real vector op was emitted
        assert "add i32" not in text  # both scalar adds were erased
        assert text.count("extractelement") == 2  # the two lanes feed the stores
        mod.verify()  # the rewritten IR is well-formed
        del mod
    assert_no_leaks()


def test_slp_vectorizer_preserves_semantics():
    # String+verify can't tell a correct vectorization from one that swaps
    # lanes or mixes operands (all produce "add <2 x i32>" + two extracts that
    # verify()). JIT-execute @f and assert the stored results to pin the actual
    # data flow: *pa must be a0+b0 and *pc must be a1+b1.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_python_pass_on_function(mod, slp_vectorize)
        assert "add <2 x i32>" in str(mod)  # vectorization did happen
        mod.verify()
        j = jit.LLJIT()
        j.add_module(mod)  # consumes the module
        f = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        )(j.lookup("f"))
        pa = ctypes.c_int32(10)  # a0
        pb = ctypes.c_int32(3)  # b0
        pc = ctypes.c_int32(20)  # a1
        pd = ctypes.c_int32(7)  # b1
        f(ctypes.byref(pa), ctypes.byref(pb), ctypes.byref(pc), ctypes.byref(pd))
        assert pa.value == 13  # store i32 %s0 -> %pa, s0 = a0 + b0 = 10 + 3
        assert pc.value == 27  # store i32 %s1 -> %pc, s1 = a1 + b1 = 20 + 7
    assert_no_leaks()


def test_slp_vectorizer_noop_when_single_add():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                define i32 @g(i32 %x, i32 %y) {
                  %s = add i32 %x, %y
                  ret i32 %s
                }
                """),
            ctx,
            "m",
        )
        llvm.passmanager.run_python_pass_on_function(mod, slp_vectorize)
        assert "add <2" not in str(mod)  # nothing to combine
        assert "add i32" in str(mod)  # scalar add untouched
        mod.verify()
        del mod
    assert_no_leaks()


def test_slp_vectorizer_skips_dependent_adds():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                define i32 @dep(i32 %p, i32 %q) {
                  %s0 = add i32 %p, %q
                  %s1 = add i32 %s0, %q
                  ret i32 %s1
                }
                """),
            ctx,
            "m",
        )
        # %s1 uses %s0, so the two adds are not independent -> not vectorized.
        llvm.passmanager.run_python_pass_on_function(mod, slp_vectorize)
        assert "add <2" not in str(mod)
        assert str(mod).count("add i32") == 2  # both scalar adds untouched
        mod.verify()
        del mod
    assert_no_leaks()


def test_slp_vectorizer_skips_transitively_dependent_adds():
    # The first two `add`s are %s0 and %s2; %s2 depends on %s0 only *through*
    # %s1 (%s2 = add %s1, ...; %s1 = add %s0, ...). This forces _reaches to walk
    # past the direct operands (push %s1, then reach %s0) -- the direct-only
    # test never exercises that transitive walk.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                define i32 @deep(i32 %p, i32 %q) {
                  %s0 = add i32 %p, %q
                  %s1 = mul i32 %s0, %q
                  %s2 = add i32 %s1, %q
                  ret i32 %s2
                }
                """),
            ctx,
            "m",
        )
        # adds[0]=%s0, adds[1]=%s2; %s2 transitively consumes %s0 via %s1.
        llvm.passmanager.run_python_pass_on_function(mod, slp_vectorize)
        assert "add <2" not in str(mod)
        assert str(mod).count("add i32") == 2  # both scalar adds untouched
        mod.verify()
        del mod
    assert_no_leaks()
