#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Behavioral tests for bindings that delegate straight to an LLVM method.

These are bound as method pointers (`.def("x", &llvm::T::x)`), so their
implementation lives in the LLVM libraries, not in src/IR. The only line in
src/IR is the registration, which runs at `import llvm` and is therefore
"covered" by any test. Line coverage can't force these to be exercised, so a
regression (wrong method bound, wrong signature, wrong result) would not show
up as a coverage drop. This file exercises each one and checks its result.
"""
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


def test_type_is_pointer_and_is_label():
    with llvm.Context() as ctx:
        assert llvm.ptr_t(ctx).is_pointer
        assert not llvm.i32(ctx).is_pointer
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        # A basic block is a Value; its type is the label type.
        assert f.entry_block.type.is_label
        assert not f.arg(0).type.is_label
        del f, mod
    assert_no_leaks()


def test_vector_type_element_type():
    with llvm.Context() as ctx:
        assert llvm.vector_t(llvm.f32(ctx), 8).element_type == llvm.f32(ctx)
    assert_no_leaks()


def test_value_type_accessor():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        assert f.arg(0).type == llvm.i32(ctx)
        del f, mod
    assert_no_leaks()


def test_replace_all_uses_with():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        add = f.arg(0).users[0]  # %sum = add, used by the ret
        assert "ret i32 %sum" in str(mod)
        zero = llvm.const_int(llvm.i32(ctx), 0)
        add.replace_all_uses_with(zero)
        # The ret now returns the constant; the (dead) add is untouched.
        assert "ret i32 0" in str(mod)
        assert "%sum = add" in str(mod)
        del f, add, zero, mod
    assert_no_leaks()


def test_instruction_successor():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            dedent(
                """\
                define void @f(i1 %c) {
                entry:
                  br i1 %c, label %a, label %b
                a:
                  ret void
                b:
                  ret void
                }
                """
            ),
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
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            dedent(
                """\
                declare i32 @printf(ptr, ...)
                define i32 @f(i32 %x) {
                  ret i32 %x
                }
                """
            ),
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
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly("define void @f() {\n ret void\n}\n", ctx, "m")
        f = mod.get_function("f")
        assert f.visibility == llvm.Visibility.DEFAULT
        f.visibility = llvm.Visibility.HIDDEN
        assert f.visibility == llvm.Visibility.HIDDEN
        assert "hidden" in str(mod)
        del f, mod
    assert_no_leaks()


def test_global_variable_is_constant():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(
            dedent(
                """\
                @c = constant i32 5
                @v = global i32 0
                define i32 @f() {
                entry:
                  %a = load i32, ptr @c
                  %b = load i32, ptr @v
                  ret i32 %a
                }
                """
            ),
            ctx,
            "m",
        )
        f = mod.get_function("f")
        loads = [
            i
            for i in f.entry_block.instructions
            if type(i).__name__ == "LoadInst"
        ]
        assert loads[0].pointer_operand.is_constant  # @c
        assert not loads[1].pointer_operand.is_constant  # @v
        del f, loads, mod
    assert_no_leaks()
