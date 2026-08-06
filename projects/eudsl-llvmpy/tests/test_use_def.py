#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-parity use-def APIs: Value.uses (Use edges), replace_all_uses_except,
and Function.walk()."""
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks


def test_use_edges():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            "define i32 @f(i32 %x, i32 %y) {\n"
            "entry:\n  %sum = add i32 %x, %y\n  ret i32 %sum\n}\n",
            ctx,
            "m",
        )
        f = mod.get_function("f")
        x = f.arg(0)
        add = x.users[0]
        uses = x.uses
        assert len(uses) == 1
        assert uses[0].user == add
        assert uses[0].operand_number == 0  # %x is operand 0 of the add
        # %y is operand 1.
        assert f.arg(1).uses[0].operand_number == 1
        del f, x, add, uses, mod
    assert_no_leaks()


def test_replace_all_uses_except():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            dedent(
                """\
                define i32 @f(i32 %x) {
                entry:
                  %a = add i32 %x, 1
                  %b = add i32 %a, 2
                  %c = add i32 %a, 3
                  ret i32 %b
                }
                """
            ),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        insts = list(f.entry_block)
        a, b, c = insts[0], insts[1], insts[2]
        assert a.num_uses == 2  # used by %b and %c
        zero = llvm.const_int(llvm.i32(ctx), 0)
        a.replace_all_uses_except(zero, [b])
        printed = str(mod)
        assert "%b = add i32 %a, 2" in printed  # kept (b was excepted)
        assert "%c = add i32 0, 3" in printed  # replaced
        assert a.num_uses == 1
        del f, insts, a, b, c, zero, mod
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
