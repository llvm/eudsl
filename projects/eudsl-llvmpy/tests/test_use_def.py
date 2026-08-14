#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-parity use-def APIs: Value.uses (Use edges), replace_all_uses_except,
and Function.walk()."""
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_EDGE_SRC = (
    "define i32 @f(i32 %x, i32 %y) {\n"
    "entry:\n  %sum = add i32 %x, %y\n  ret i32 %sum\n}\n"
)

_RAUW_SRC = dedent(
    """\
    define i32 @f(i32 %x) {
    entry:
      %a = add i32 %x, 1
      %b = add i32 %a, 2
      %c = add i32 %a, 3
      ret i32 %b
    }
    """
)


def test_use_edges():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_EDGE_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        add = x.users[0]
        uses = x.uses
        assert len(uses) == x.num_uses == 1
        assert uses[0].user == add
        # The Use.user edge is downcast to its concrete subclass, not left as
        # the base Value -- pointer-equality above cannot catch that regression.
        assert type(uses[0].user).__name__ == "BinaryOperator"
        assert uses[0].operand_number == 0  # %x is operand 0 of the add
        # %y is operand 1.
        assert f.arg(1).uses[0].operand_number == 1
        # A value with no uses: the (void) terminator.
        term = f.entry_block.terminator
        assert term.num_uses == 0
        assert list(term.uses) == []
        del f, x, add, uses, term, mod
    assert_no_leaks()


def test_uses_usable_without_value_handle():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_EDGE_SRC, ctx, "m")
        # The module owns the use list, so `.uses` (and its Use edges) stay
        # valid without holding the intermediate function/argument handle -- as
        # with every other reference accessor, lifetime is bounded by the module.
        uses = mod.get_function("f").arg(0).uses
        assert len(uses) == 1
        assert uses[0].operand_number == 0
        assert type(uses[0].user).__name__ == "BinaryOperator"
        del uses, mod
    assert_no_leaks()


def test_replace_all_uses_except():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_RAUW_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        a, b, c = list(f.entry_block)[:3]
        # Consistency of the count accessors on a >1-use value.
        assert a.num_uses == len(a.uses) == 2  # used by %b and %c
        assert x.num_uses == 1  # used by %a
        # Replace with a non-constant (the argument) so BOTH sides are checkable
        # via the API: %a loses a use, %x gains one. (Constants don't track uses.)
        a.replace_all_uses_except(x, [b])
        assert a.num_uses == 1  # only %b now
        assert x.num_uses == 2  # %a and the rewritten %c
        printed = str(mod)
        assert "%b = add i32 %a, 2" in printed  # kept (b was excepted)
        assert "%c = add i32 %x, 3" in printed  # replaced
        del f, x, a, b, c, mod
    assert_no_leaks()


def test_replace_all_uses_except_empty_is_full_rauw():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_RAUW_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        a, b, c = list(f.entry_block)[:3]
        # No exceptions -> every use of %a is replaced (== replace_all_uses_with).
        a.replace_all_uses_except(x, [])
        assert a.num_uses == 0
        assert x.num_uses == 3  # %a plus the rewritten %b and %c
        del f, x, a, b, c, mod
    assert_no_leaks()


def test_replace_all_uses_except_all_users_excepted_is_noop():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_RAUW_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        a, b, c = list(f.entry_block)[:3]
        # Every user of %a is excepted -> nothing changes.
        a.replace_all_uses_except(x, [b, c])
        assert a.num_uses == 2
        assert x.num_uses == 1
        del f, x, a, b, c, mod
    assert_no_leaks()


def test_replace_all_uses_except_type_mismatch_raises():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_RAUW_SRC, ctx, "m")
        f = mod.get_function("f")
        a, b, _ = list(f.entry_block)[:3]
        # A mismatched type would trip LLVM's assert and abort the interpreter;
        # the binding must reject it as a Python error instead.
        wrong_type = llvm.const_int(llvm.types.i1(ctx), 0)
        with pytest.raises(ValueError, match="type does not match"):
            a.replace_all_uses_except(wrong_type, [b])
        assert a.num_uses == 2  # unchanged
        del f, a, b, wrong_type, mod
    assert_no_leaks()


def test_function_walk():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            dedent(
                """\
                define i32 @f(i32 %x) {
                entry:
                  %a = add i32 %x, 1
                  br label %next
                next:
                  %b = mul i32 %a, 2
                  ret i32 %b
                }
                """
            ),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        # walk() flattens all instructions across every block, in order.
        walked = list(f.walk())
        assert [type(i).__name__ for i in walked] == [
            "BinaryOperator",  # add (entry)
            "UncondBrInst",  # br  (entry)
            "BinaryOperator",  # mul (next)
            "ReturnInst",  # ret (next)
        ]
        # Same instructions as concatenating the blocks.
        by_block = [i for bb in f for i in bb]
        assert walked == by_block
        del f, walked, by_block, mod
    assert_no_leaks()


def test_function_walk_declaration_is_empty():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly("declare i32 @ext(i32)\n", ctx, "m")
        ext = mod.get_function("ext")
        # A declaration has no body -> walk() yields nothing.
        assert list(ext.walk()) == []
        del ext, mod
    assert_no_leaks()
