#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Unsupported control flow must raise NotImplementedError, not miscompile."""
import pytest

import llvm
from llvm.ast.canonicalize import canonicalize
from llvm.dsl.cf import LLVMCanonicalizer
from llvm.testing import assert_no_leaks


def test_break_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="break"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                for i in range_(0, n):
                    if i.eq(llvm.ir.const_int(i32, 3)):
                        break
                    yield i
                return n

        del mod


def test_continue_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="continue"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                for i in range_(0, n):
                    if i.eq(llvm.ir.const_int(i32, 3)):
                        continue
                    yield i
                return n

        del mod


def test_early_return_in_if_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="return"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad2(c: llvm.types.i1, a: i32) -> i32:
                if c:
                    return a
                return a

        del mod


def test_nested_control_flow_in_loop_raises():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="nested inside"):

            @llvm.dsl.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad3(n: i32) -> i32:
                acc = llvm.ir.const_int(i32, 0)
                for i in range_(0, n):
                    if i < n:
                        acc = yield i
                    yield acc
                return acc

        del mod


def test_trailing_return_is_allowed():
    # The function's own trailing return must NOT be rejected.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)

        @llvm.dsl.function(module=mod)
        @canonicalize(using=LLVMCanonicalizer())
        def ok(c: llvm.types.i1, a: i32, b: i32) -> i32:
            if c:
                r = yield a
            else:
                r = yield b
            return r

        assert "phi i32" in str(mod)
        del mod
    assert_no_leaks()
