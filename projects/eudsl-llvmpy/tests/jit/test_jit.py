#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes
from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %s = add i32 %a, %b
      ret i32 %s
    }
    """
)


def test_jit_execute():
    ctx = llvm.ir.Context()
    mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
    jit = llvm.jit.LLJIT()
    jit.add_module(mod)  # consumes mod
    assert mod._is_consumed
    addr = jit.lookup("add")
    assert addr != 0
    fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(addr)
    assert fn(2, 40) == 42
    del jit, mod, ctx
    assert_no_leaks()
