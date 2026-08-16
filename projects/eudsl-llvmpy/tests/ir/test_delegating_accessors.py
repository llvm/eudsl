#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Behavioral tests for bindings that line coverage cannot force.

Two kinds of binding are invisible to the line-coverage gate:

1. Method pointers (`.def("x", &llvm::T::x)`): the implementation is in the
   LLVM libraries, not src/IR, so the only src/IR line is the registration.
2. Accessor/op LAMBDAS (`.def("x", [](...){ ... })`): the lambda body often
   shares a source line with the enclosing `.def(...)` call, which runs at
   import. Line coverage marks that line covered even though the lambda body
   never executes.

The C++ coverage gate now also enforces FUNCTION coverage (each lambda is its
own function), so a binding whose body is never called fails the gate. This
file exercises those bindings and checks their results.
"""

from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks, filecheck_with_comments

_SRC = dedent("""\
    define i32 @f(i32 %x, i32 %y) {
    entry:
      %sum = add i32 %x, %y
      ret i32 %sum
    }
    """)


def test_type_is_pointer_and_is_label():
    with llvm.ir.Context() as ctx:
        assert llvm.types.ptr(context=ctx).is_pointer
        assert not llvm.types.i32(ctx).is_pointer
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        # A basic block is a Value; its type is the label type.
        assert f.entry_block.type.is_label
        assert not f.arg(0).type.is_label
        del f, mod
    assert_no_leaks()


def test_vector_type_element_type():
    with llvm.ir.Context() as ctx:
        assert llvm.types.vector(llvm.types.f32(ctx), 8).element_type == llvm.types.f32(
            ctx
        )
    assert_no_leaks()


def test_value_type_accessor():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        assert f.arg(0).type == llvm.types.i32(ctx)
        del f, mod
    assert_no_leaks()


def test_replace_all_uses_with():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        add = f.arg(0).users[0]  # %sum = add, used by the ret
        assert "ret i32 %sum" in str(mod)
        zero = llvm.ir.const_int(llvm.types.i32(ctx), 0)
        add.replace_all_uses_with(zero)
        # The ret now returns the constant; the (dead) add is untouched.
        # CHECK: %sum = add i32 %x, %y
        # CHECK: ret i32 0
        filecheck_with_comments(mod)
        del f, add, zero, mod
    assert_no_leaks()


def test_instruction_successor():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                define void @f(i1 %c) {
                entry:
                  br i1 %c, label %a, label %b
                a:
                  ret void
                b:
                  ret void
                }
                """),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        br = f.entry_block.terminator
        assert br.successor(0).name == "a"
        assert br.successor(1).name == "b"
        del f, br, mod
    assert_no_leaks()


def test_function_type_and_var_arg():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                declare i32 @printf(ptr, ...)
                define i32 @f(i32 %x) {
                  ret i32 %x
                }
                """),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        assert str(f.function_type) == "i32 (i32)"
        assert not f.is_var_arg
        assert mod.get_function("printf").is_var_arg
        del f, mod
    assert_no_leaks()


def test_function_visibility_roundtrip():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly("define void @f() {\n ret void\n}\n", ctx, "m")
        f = mod.get_function("f")
        assert f.visibility == llvm.ir.Visibility.DEFAULT
        f.visibility = llvm.ir.Visibility.HIDDEN
        assert f.visibility == llvm.ir.Visibility.HIDDEN
        assert "hidden" in str(mod)
        del f, mod
    assert_no_leaks()


def test_global_variable_is_constant():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                @c = constant i32 5
                @v = global i32 0
                define i32 @f() {
                entry:
                  %a = load i32, ptr @c
                  %b = load i32, ptr @v
                  ret i32 %a
                }
                """),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        loads = [
            i for i in f.entry_block.instructions if type(i).__name__ == "LoadInst"
        ]
        assert loads[0].pointer_operand.is_constant  # @c
        assert not loads[1].pointer_operand.is_constant  # @v
        del f, loads, mod
    assert_no_leaks()


def test_builder_binary_ops_and_insert_point():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        f32 = llvm.types.f32(ctx)
        fn = llvm.ir.Function.create(
            llvm.types.function(i32, [i32, i32, f32, f32]), "f", mod
        )
        bb = fn.append_basic_block("entry")
        b = llvm.ir.IRBuilder(ctx)
        b.set_insert_point(bb)  # set_insert_point
        assert b.insert_block == bb  # insert_block property
        ia, ib, fa, fb = fn.arg(0), fn.arg(1), fn.arg(2), fn.arg(3)
        b.add(ia, ib)
        b.sub(ia, ib)
        b.mul(ia, ib)
        b.sdiv(ia, ib)
        b.udiv(ia, ib)
        b.fadd(fa, fb)
        b.fsub(fa, fb)
        b.fmul(fa, fb)
        b.fdiv(fa, fb)
        b.ret(ia)
        printed = str(mod)
        for op in (
            "add i32",
            "sub i32",
            "mul i32",
            "sdiv i32",
            "udiv i32",
            "fadd float",
            "fsub float",
            "fmul float",
            "fdiv float",
        ):
            assert op in printed, op
        del b, bb, fn, ia, ib, fa, fb, mod
    assert_no_leaks()


def test_instruction_accessors():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                declare i32 @g(i32)
                define i32 @f(i1 %c, ptr %p) {
                entry:
                  %s = alloca i32
                  store i32 7, ptr %s
                  %v = load i32, ptr %s
                  %r = call i32 @g(i32 %v)
                  br i1 %c, label %a, label %b
                a:
                  br label %b
                b:
                  ret i32 %r
                }
                """),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        insts = f.entry_block.instructions

        def first(kind):
            return next(i for i in insts if type(i).__name__ == kind)

        alloca = first("AllocaInst")
        store = first("StoreInst")
        call = first("CallInst")
        condbr = f.entry_block.terminator
        assert str(alloca.allocated_type) == "i32"  # AllocaInst.allocated_type
        assert store.pointer_operand == alloca  # StoreInst.pointer_operand
        assert call.called_operand.name == "g"  # CallBase.called_operand
        assert condbr.is_conditional  # CondBrInst.is_conditional
        assert condbr.condition == f.arg(0)  # CondBrInst.condition (== %c)
        a_block = next(b for b in f.basic_blocks if b.name == "a")
        assert not a_block.terminator.is_conditional  # UncondBrInst.is_conditional
        b_block = next(b for b in f.basic_blocks if b.name == "b")
        assert b_block.terminator.return_value == call  # ReturnInst.return_value
        del f, insts, alloca, store, call, condbr, a_block, b_block, mod
    assert_no_leaks()


def test_value_name_setter_and_instruction_props():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        entry = f.entry_block
        add = entry.instructions[0]
        ret = entry.terminator
        assert add.is_terminator is False  # Instruction.is_terminator
        assert ret.is_terminator is True
        assert add.parent == entry  # Instruction.parent
        add.name = "renamed"  # Value.name setter
        assert add.name == "renamed"
        # CHECK: %renamed = add i32 %x, %y
        filecheck_with_comments(mod)
        del f, entry, add, ret, mod
    assert_no_leaks()


def test_instruction_set_successor():
    # set_successor is used by the DSL elif lowering (dsl/cf.py) and validated
    # indirectly by the elif JIT tests; assert it directly here too.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                define void @f(i1 %cond) {
                entry:
                  br i1 %cond, label %a, label %b
                a:
                  ret void
                b:
                  ret void
                c:
                  ret void
                }
                """),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        br = f.entry_block.terminator
        c_block = next(b for b in f.basic_blocks if b.name == "c")
        br.set_successor(0, c_block)  # redirect the true edge from a to c
        assert br.successor(0).name == "c"
        # The true edge now targets %c; the false edge is unchanged.
        # CHECK: br i1 %cond, label %c, label %b
        filecheck_with_comments(mod)
        del f, br, c_block, mod
    assert_no_leaks()
