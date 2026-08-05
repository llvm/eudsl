#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes

import llvm
from llvm.testing import assert_no_leaks


def test_if_else_produces_phi():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.i32(ctx)

        @llvm.function(module=mod)
        def pick(c: llvm.i1, a: i32, b: i32) -> i32:
            if c:
                r = yield a + 1
            else:
                r = yield b
            return r

        printed = str(mod)
        assert "br i1" in printed
        assert "phi i32" in printed
        assert "add i32" in printed
        del mod
    assert_no_leaks()


def test_if_else_jits_correctly():
    ctx = llvm.Context()
    mod = llvm.Module("m", ctx)
    i32 = llvm.i32(ctx)

    @llvm.function(module=mod)
    def pick(c: llvm.i1, a: i32, b: i32) -> i32:
        if c:
            r = yield a
        else:
            r = yield b
        return r

    jit = llvm.LLJIT()
    jit.add_module(mod)
    fn = ctypes.CFUNCTYPE(
        ctypes.c_int32, ctypes.c_bool, ctypes.c_int32, ctypes.c_int32
    )(jit.lookup("pick"))
    assert fn(True, 10, 20) == 10
    assert fn(False, 10, 20) == 20
    del jit, mod, ctx, pick, i32, fn
    assert_no_leaks()
