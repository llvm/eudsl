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
