#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent("""\
    define i32 @sub(i32 %a, i32 %b) {
    entry:
      %s = sub i32 %a, %b
      ret i32 %s
    }
    """)


def test_jit_execute():
    ctx = llvm.ir.Context()
    mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)  # consumes mod
    assert mod._is_consumed
    addr = jit.lookup("sub")
    assert addr != 0
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(addr)
    assert fn(50, 8) == 42
    assert fn(10, 3) == 7
    del jit, mod, ctx
    assert_no_leaks()


def test_consumed_module_raises_on_use():
    ctx = llvm.ir.Context()
    mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)
    with pytest.raises(RuntimeError, match="has been consumed"):
        str(mod)
    with pytest.raises(RuntimeError, match="has been consumed"):
        jit.add_module(mod)
    del jit, mod, ctx
    assert_no_leaks()
