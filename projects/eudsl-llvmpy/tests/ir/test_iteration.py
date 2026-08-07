#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-style iteration ergonomics: len(), indexing, and `for x in container`.

Module iterates its functions, Function its basic blocks, BasicBlock its
instructions, and User its operands. Each supports len(), negative and
bounds-checked __getitem__ (IndexError past the ends), and iteration.
"""
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @f(i32 %x, i32 %y) {
    entry:
      %sum = add i32 %x, %y
      ret i32 %sum
    }
    define void @g() {
    entry:
      ret void
    }
    """
)


def test_iteration_and_indexing():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")

        # Module -> functions
        fns = list(mod)
        assert [f.name for f in fns] == ["f", "g"]
        assert len(mod) == 2
        assert mod[0] == fns[0]
        assert mod[-1] == fns[1]
        with pytest.raises(IndexError):
            _ = mod[2]

        f = mod.get_function("f")

        # Function -> basic blocks
        bbs = list(f)
        assert len(f) == 1
        assert f[0] == f.entry_block
        assert f[-1] == bbs[-1]
        with pytest.raises(IndexError):
            _ = f[5]

        # BasicBlock -> instructions
        entry = f.entry_block
        insts = list(entry)
        assert len(entry) == 2  # add, ret
        assert entry[0] == insts[0]
        assert entry[-1] == entry.terminator
        with pytest.raises(IndexError):
            _ = entry[2]

        # User -> operands
        add = insts[0]
        ops = list(add)
        assert len(add) == add.num_operands == 2
        assert add[0] == add.operand(0)
        assert add[-1] == add.operand(1)
        assert ops[0] == f.arg(0)
        with pytest.raises(IndexError):
            _ = add[2]

        del f, entry, fns, bbs, insts, add, ops, mod
    assert_no_leaks()
