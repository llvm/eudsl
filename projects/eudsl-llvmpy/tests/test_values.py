#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @f(i32 %x, i32 %y) {
    entry:
      %sum = add i32 %x, %y
      ret i32 %sum
    }
    """
)


def test_value_and_user_registered():
    # Value/User accessors only become reachable once a Value can be obtained
    # from Python (functions()/traversal in Task 9). This test confirms
    # populate_values did not break module round-tripping and that the classes
    # exist on the module.
    assert hasattr(llvm, "Value")
    assert hasattr(llvm, "User")
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        assert "define i32 @f(i32 %x, i32 %y)" in str(mod)
        del mod
    assert_no_leaks()


def test_function_traversal():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        assert [f.name for f in mod.functions] == ["f"]
        f = mod.get_function("f")
        assert f is not None
        assert mod.get_function("nope") is None
        assert f.num_args == 2
        assert [a.name for a in f.args] == ["x", "y"]
        assert f.arg(0).arg_no == 0
        assert f.arg(1).parent == f
        assert not f.is_declaration
        assert str(f.return_type) == "i32"
        del f, mod
    assert_no_leaks()


def test_basic_block_and_instruction_traversal():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        blocks = f.basic_blocks
        assert [b.name for b in blocks] == ["entry"]
        entry = f.entry_block
        assert entry.name == "entry"
        assert entry == blocks[0]
        insts = entry.instructions
        # add, ret
        assert len(insts) == 2
        assert entry.terminator == insts[-1]
        # Instruction is registered (the spine); concrete opcode downcasting
        # (BinaryOperator) activates once the instruction classes are bound.
        assert type(insts[0]).__name__ == "Instruction"
        del f, entry, blocks, insts, mod
    assert_no_leaks()


def test_value_users_and_operands():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        # %x is used by the add.
        assert x.num_uses == 1
        add = x.users[0]
        # Concrete opcode downcast (BinaryOperator) activates in Task 10; here
        # the add arrives as a registered base (User/Instruction).
        assert add.num_operands == 2
        assert add.operand(0) == x
        del f, x, add, mod
    assert_no_leaks()


def test_append_basic_block():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly("declare void @g()\n", ctx, "m")
        ft = llvm.function_t(llvm.void_t(ctx), [])
        fn = llvm.Function.create(ft, "h", mod)
        bb = fn.append_basic_block("entry")
        assert bb.name == "entry"
        assert fn.entry_block == bb
        assert bb.parent == fn
        del fn, bb, mod
    assert_no_leaks()
