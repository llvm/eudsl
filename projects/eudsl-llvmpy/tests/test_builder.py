#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_build_add_function():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [i32, i32]), "add2", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            s = b.add(fn.arg(0), fn.arg(1), "s")
            b.ret(s)
        printed = str(mod)
        assert "define i32 @add2(i32 %0, i32 %1)" in printed
        assert "%s = add i32 %0, %1" in printed
        assert "ret i32 %s" in printed
        del b, fn, bb, mod
    assert_no_leaks()


def test_build_conditional_with_phi():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        i1 = llvm.types.i1(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, [i1]), "sel", mod)
        entry = fn.append_basic_block("entry")
        a = fn.append_basic_block("a")
        b_ = fn.append_basic_block("b")
        join = fn.append_basic_block("join")
        bld = llvm.IRBuilder(ctx)
        with bld.at_end_of(entry):
            bld.cond_br(fn.arg(0), a, b_)
        with bld.at_end_of(a):
            bld.br(join)
        with bld.at_end_of(b_):
            bld.br(join)
        with bld.at_end_of(join):
            p = bld.phi(i32, "p")
            p.add_incoming(llvm.const_int(i32, 1), a)
            p.add_incoming(llvm.const_int(i32, 2), b_)
            bld.ret(p)
        printed = str(mod)
        assert "phi i32 [ 1, %a ], [ 2, %b ]" in printed
        del bld, fn, entry, a, b_, join, p, mod
    assert_no_leaks()


def test_alloca_load_store():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, []), "f", mod)
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            slot = b.alloca(i32, "slot")
            b.store(llvm.const_int(i32, 5), slot)
            loaded = b.load(i32, slot, "loaded")
            b.ret(loaded)
        printed = str(mod)
        assert "alloca i32" in printed
        assert "store i32 5" in printed
        assert "load i32" in printed
        del b, fn, bb, mod
    assert_no_leaks()
