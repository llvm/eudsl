#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_named_metadata_round_trips():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        s = llvm.md_string("hello", context=ctx)
        node = llvm.md_node([s], context=ctx)
        mod.add_named_metadata("my.meta", node)
        printed = str(mod)
        assert "!my.meta = !{!0}" in printed
        assert '!0 = !{!"hello"}' in printed
        got = mod.named_metadata("my.meta")
        assert len(got) == 1
        retrieved = got[0]
        assert isinstance(retrieved, llvm.MDNode)
        assert retrieved.num_operands == 1
        op = retrieved.operand(0)
        assert isinstance(op, llvm.MDString)
        assert op.string == "hello"
        del mod
    assert_no_leaks()


def test_metadata_str():
    with llvm.Context() as ctx:
        s = llvm.md_string("hi", context=ctx)
        node = llvm.md_node([s], context=ctx)
        assert "hi" in str(s)
        assert "hi" in str(node)
    assert_no_leaks()
