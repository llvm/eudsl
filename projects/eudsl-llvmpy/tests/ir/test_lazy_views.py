#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""The traversal accessors return lazy sequence views (not materialized lists):
len(), negative/bounds-checked indexing, slicing, and iteration, computed on
demand. Covers every Sequence<T> instantiation."""
from textwrap import dedent

import gc

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @f(i32 %x, i32 %y) {
    entry:
      %s = add i32 %x, %y
      ret i32 %s
    }
    define void @g() {
    entry:
      ret void
    }
    """
)


def test_lazy_sequence_views():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        # Module.functions is a lazy view, not a list.
        fns = mod.functions
        assert type(fns).__name__ == "FunctionSequence"
        assert not isinstance(fns, list)
        assert len(fns) == 2
        assert fns[0].name == "f"
        assert fns[-1].name == "g"
        assert [x.name for x in fns] == ["f", "g"]  # protocol iteration
        assert [x.name for x in fns[0:1]] == ["f"]  # slice -> list
        assert [x.name for x in fns[::-1]] == ["g", "f"]  # negative step
        with pytest.raises(IndexError):
            _ = fns[9]
        with pytest.raises(IndexError):
            _ = fns[-100]  # still out of range after the negative-index adjustment
        with pytest.raises(TypeError):
            _ = fns["nope"]  # non-integer, non-slice index

        g = mod.get_function("g")
        assert len(g.args) == 0  # empty view
        assert list(g.args) == []
        with pytest.raises(IndexError):
            _ = g.args[0]

        f = mod.get_function("f")

        bbs = f.basic_blocks  # BasicBlockSequence
        assert type(bbs).__name__ == "BasicBlockSequence"
        assert len(bbs) == 1 and isinstance(bbs[0], llvm.ir.BasicBlock)

        args = f.args  # ArgumentSequence
        assert type(args).__name__ == "ArgumentSequence"
        assert len(args) == 2 and isinstance(args[0], llvm.ir.Argument)
        assert len(args[:1]) == 1

        insts = f.entry_block.instructions  # InstructionSequence
        assert type(insts).__name__ == "InstructionSequence"
        # element downcasts to the concrete class, lazily
        assert isinstance(insts[0], llvm.ir.BinaryOperator)

        add = insts[0]
        ops = add.operands  # ValueSequence
        assert type(ops).__name__ == "ValueSequence"
        assert len(ops) == 2 and isinstance(ops[0], llvm.ir.Value)
        assert ops[0] == args[0]
        assert ops[1] == args[1]
        # Distinct operands compare unequal -- not an identity-collapsed tautology.
        assert ops[0] != ops[1]
        assert ops[0] != add

        users = args[0].users  # UserSequence
        assert type(users).__name__ == "UserSequence"
        assert len(users) == 1
        assert users[0] == add

        uses = args[0].uses  # UseSequence
        assert type(uses).__name__ == "UseSequence"
        assert uses[0].operand_number == 0
        with pytest.raises(IndexError):
            _ = uses[5]

        del f, g, fns, bbs, args, insts, add, ops, users, uses, mod
    assert_no_leaks()


def test_view_reflects_live_mutation():
    # "Lazy" behaviorally: a view created before a mutation reflects it, so it
    # is a live computed-on-demand window, not an eager snapshot.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.ir.Function.create(llvm.types.function(i32, [i32]), "f", mod)
        bb = fn.append_basic_block("entry")
        insts = bb.instructions  # view taken while the block is empty
        assert len(insts) == 0
        b = llvm.ir.IRBuilder(ctx)
        with b.at_end_of(bb):
            b.ret(fn.arg(0))
        assert len(insts) == 1  # same view now sees the appended instruction
        assert isinstance(insts[0], llvm.ir.ReturnInst)
        del b, fn, bb, insts, mod
    assert_no_leaks()


def test_container_view_keeps_module_alive():
    # Module.functions' parent IS the owning module, so holding only the view
    # keeps the module alive; dropping it releases the module.
    with llvm.ir.Context() as ctx:
        fns = llvm.ir.parse_assembly(_SRC, ctx, "m").functions  # sole handle
        gc.collect()
        assert llvm.ir.Context._get_live_module_count() == 1
        assert len(fns) == 2  # still usable with no other reference to the module
        del fns
        gc.collect()
        assert llvm.ir.Context._get_live_module_count() == 0
    assert_no_leaks()
