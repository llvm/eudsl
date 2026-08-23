#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent("""\
    define i32 @f(i32 %x) {
    entry:
      ret i32 %x
    }
    """)


def test_module_pass_is_invoked_once_with_the_module():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        seen = []
        llvm.passmanager.run_python_pass_on_module(mod, seen.append)
        assert len(seen) == 1
        # The callback receives the very same Module object, not a fresh wrapper.
        assert seen[0] is mod
        del mod
    assert_no_leaks()


def test_module_pass_can_mutate_ir():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        def rename(m):
            m.get_function("f").name = "g"

        llvm.passmanager.run_python_pass_on_module(mod, rename)
        assert "@g(" in str(mod)
        assert "@f(" not in str(mod)
        del mod
    assert_no_leaks()


def test_exception_in_pass_propagates_and_leaves_module_usable():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        def boom(m):
            raise ValueError("boom in pass")

        with pytest.raises(ValueError, match="boom in pass"):
            llvm.passmanager.run_python_pass_on_module(mod, boom)
        # The trampoline captured the exception and re-raised it after the
        # pipeline returned; the module is still usable afterward.
        assert "@f(" in str(mod)
        del mod
    assert_no_leaks()


def test_exception_from_unboolable_return_propagates():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        class Unboolable:
            def __bool__(self):
                raise ValueError("no truth value")

        # A truthiness that raises exercises the PyObject_IsTrue(<0) branch,
        # which must propagate rather than be silently coerced or lost.
        with pytest.raises(ValueError, match="no truth value"):
            llvm.passmanager.run_python_pass_on_module(mod, lambda m: Unboolable())
        assert "@f(" in str(mod)
        del mod
    assert_no_leaks()
