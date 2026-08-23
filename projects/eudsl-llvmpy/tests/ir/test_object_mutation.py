#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Object-level IR mutation/inspection bindings that map 1:1 to LLVM C++:
Instruction.opcode/opcode_name, erase_from_parent/remove_from_parent, clone,
move_before/insert_before/insert_after, User.set_operand, and
BasicBlock.split_basic_block/erase_from_parent/remove_from_parent."""

from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks
from llvm import ir

_SRC = dedent("""\
    define i32 @f(i32 %x) {
    entry:
      %s = add i32 %x, 1
      %m = mul i32 %s, 2
      ret i32 %m
    }
    """)


def _insts(fn):
    return list(fn.entry_block.instructions)


def test_opcode_and_opcode_name_distinguish_binops():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        add, mul, ret = _insts(mod.get_function("f"))
        assert add.opcode_name == "add"
        assert mul.opcode_name == "mul"
        assert ret.opcode_name == "ret"
        # opcode is the stable integer; add != mul, and it agrees with the name.
        assert isinstance(add.opcode, int)
        assert add.opcode != mul.opcode
        del mod
    assert_no_leaks()


def test_set_operand_rewires_use():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        add, mul, ret = _insts(fn)
        # mul's first operand is %s (the add); rewire it to the argument %x.
        assert mul.operand(0) is add
        mul.set_operand(0, fn.arg(0))
        assert mul.operand(0) is fn.arg(0)
        assert "mul i32 %x, 2" in str(mod)
        del mod
    assert_no_leaks()


def test_clone_and_insert_before_duplicates_instruction():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        add, mul, ret = _insts(fn)
        dup = add.clone()  # unattached copy of `%s = add i32 %x, 1`
        dup.insert_before(mul)
        insts = _insts(fn)
        assert len(insts) == 4  # add, dup, mul, ret
        assert insts[1] is dup
        assert dup.opcode_name == "add"
        del mod
    assert_no_leaks()


def test_insert_after_places_instruction():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        add, mul, ret = _insts(fn)
        dup = add.clone()
        dup.insert_after(add)
        assert _insts(fn)[1] is dup
        del mod
    assert_no_leaks()


def test_move_before_reorders():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        add, mul, ret = _insts(fn)
        dup = add.clone()  # independent, unused
        dup.insert_after(mul)  # order: add, mul, dup, ret
        assert _insts(fn)[2] is dup
        dup.move_before(add)  # order: dup, add, mul, ret
        assert _insts(fn)[0] is dup
        del mod
    assert_no_leaks()


def test_erase_from_parent_removes_instruction():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        add, mul, ret = _insts(fn)
        dup = add.clone()
        dup.insert_before(mul)
        assert len(_insts(fn)) == 4
        dup.erase_from_parent()  # dup has no uses
        assert len(_insts(fn)) == 3
        del mod
    assert_no_leaks()


def test_remove_from_parent_then_reinsert():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        add, mul, ret = _insts(fn)
        dup = add.clone()
        dup.insert_before(mul)
        dup.remove_from_parent()  # detached, not deleted
        assert len(_insts(fn)) == 3
        dup.insert_before(mul)  # re-attach elsewhere
        assert len(_insts(fn)) == 4
        del mod
    assert_no_leaks()


def test_split_basic_block():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        add, mul, ret = _insts(fn)
        tail = fn.entry_block.split_basic_block(mul, "tail")
        assert len(list(fn.basic_blocks)) == 2
        assert tail.name == "tail"
        # split leaves an unconditional branch to the tail as the entry's terminator
        assert fn.entry_block.terminator.opcode_name == "br"
        assert list(tail.instructions)[0] is mul
        del mod
    assert_no_leaks()


def test_basic_block_erase_from_parent():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        fn = mod.get_function("f")
        # an unreachable, unused block is safe to erase
        dead = ir.BasicBlock.create("dead", fn)
        assert len(list(fn.basic_blocks)) == 2
        dead.erase_from_parent()
        assert len(list(fn.basic_blocks)) == 1
        del mod
    assert_no_leaks()


def test_comes_before_orders_instructions():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        add, mul, ret = _insts(mod.get_function("f"))
        assert add.comes_before(mul)
        assert mul.comes_before(ret)
        assert not mul.comes_before(add)
        del mod
    assert_no_leaks()


def test_memory_and_side_effect_predicates():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                define void @g(ptr %p) {
                  %v = load i32, ptr %p
                  %a = add i32 %v, 1
                  store i32 %a, ptr %p
                  ret void
                }
                """),
            ctx,
            "m",
        )
        load, add, store, ret = _insts(mod.get_function("g"))
        assert load.may_read_or_write_memory
        assert load.may_read_from_memory
        assert not load.may_write_to_memory
        assert not add.may_read_or_write_memory
        assert store.may_have_side_effects
        assert store.may_write_to_memory
        assert not add.may_have_side_effects
        del mod
    assert_no_leaks()


def test_cfg_predecessors_and_successors():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                define void @cfg(i1 %c) {
                entry:
                  br i1 %c, label %a, label %b
                a:
                  br label %exit
                b:
                  br label %exit
                exit:
                  ret void
                }
                """),
            ctx,
            "m",
        )
        blocks = {bb.name: bb for bb in mod.get_function("cfg").basic_blocks}
        assert [b.name for b in blocks["entry"].successors] == ["a", "b"]
        assert blocks["entry"].predecessors == []
        assert sorted(b.name for b in blocks["exit"].predecessors) == ["a", "b"]
        del mod
    assert_no_leaks()
