#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

import llvm
from llvm.ast.canonicalize import canonicalize
from llvm.dsl.cf import LLVMCanonicalizer
from llvm.testing import assert_no_leaks


def test_break_raises_not_implemented():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="break"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n):
                    if i.eq(llvm.const_int(i32, 3)):
                        r = yield
                    break
                    yield acc
                return acc

        del mod
    assert_no_leaks()


def test_early_return_in_if_raises():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="return"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad2(c: llvm.types.i1, a: i32) -> i32:
                if c:
                    return a
                return a

        del mod
    assert_no_leaks()


def test_continue_raises_not_implemented():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        with pytest.raises(NotImplementedError, match="continue"):

            @llvm.function(module=mod)
            @canonicalize(using=LLVMCanonicalizer())
            def bad3(n: i32) -> i32:
                acc = llvm.const_int(i32, 0)
                for i in range_(0, n):
                    continue
                    yield acc
                return acc

        del mod
    assert_no_leaks()
