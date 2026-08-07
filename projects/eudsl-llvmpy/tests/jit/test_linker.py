#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_link_two_modules():
    with llvm.Context() as ctx:
        dest = llvm.parse_assembly("declare i32 @a()\n", ctx, "dest")
        src = llvm.parse_assembly(
            dedent(
                """\
                define i32 @a() {
                  ret i32 7
                }
                """
            ),
            ctx,
            "src",
        )
        llvm.link_into(dest, src)
        assert src._is_consumed
        assert "define i32 @a()" in str(dest)
        with pytest.raises(RuntimeError, match="has been consumed"):
            str(src)
        del dest, src
    assert_no_leaks()
