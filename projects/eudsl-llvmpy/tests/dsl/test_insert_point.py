#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-style `with InsertPoint(target, builder=b)`: make `b` the current builder
for the block, positioned at the target, and restore it on exit. Nested
InsertPoints omit `builder` to reuse the current one. Also the at_block_begin /
at_block_terminator / after factories, InsertPoint.current, and current_builder /
current_function derived from the active InsertPoint."""

import pytest

import llvm
from llvm.ir import InsertPoint, IRBuilder, current_builder, current_function
from llvm.testing import assert_no_leaks, filecheck_with_comments


def _fn(ctx, name="f"):
    mod = llvm.ir.Module("m", ctx)
    fn = llvm.ir.Function.create(llvm.types.function(llvm.types.void(), []), name, mod)
    return mod, fn


def test_insert_point_restores_previous_position():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        i32 = llvm.types.i32()
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b):
            b.alloca(i32, "a")  # at end
            with InsertPoint.at_block_begin(entry):  # reuse current builder
                b.alloca(i32, "b")  # before %a
            b.alloca(i32, "c")  # builder restored -> after %a
            b.ret(None)
        # The block order is B (begin), A, C (restored to end), then the ret.
        # CHECK: %b = alloca
        # CHECK: %a = alloca
        # CHECK: %c = alloca
        # CHECK: ret void
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_insert_point_before_instruction():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        i32 = llvm.types.i32()
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b):
            a = b.alloca(i32, "a")
            b.ret(None)
            with InsertPoint(a):  # an Instruction target -> insert before `a`
                b.alloca(i32, "before_a")
        # CHECK: %before_a = alloca
        # CHECK: %a = alloca
        # CHECK: ret void
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_insert_point_after_instruction():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        i32 = llvm.types.i32()
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b):
            a = b.alloca(i32, "a")
            b.ret(None)
            with InsertPoint.after(a):
                b.alloca(i32, "after_a")
        # CHECK: %a = alloca
        # CHECK: %after_a = alloca
        # CHECK: ret void
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_insert_point_at_block_terminator():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        i32 = llvm.types.i32()
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b):
            b.ret(None)
            with InsertPoint.at_block_terminator(entry):
                b.alloca(i32, "pre_ret")
        # CHECK: %pre_ret = alloca
        # CHECK: ret void
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_insert_point_current_and_current_builder():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b) as ip:
            assert InsertPoint.current is ip
            assert ip.block == entry
            assert ip.is_set
            assert current_builder() is b
            assert current_function() is fn
        b.set_insert_point(entry)
        b.ret(None)
        del mod
    assert_no_leaks()


def test_insert_point_nesting_restores_to_outer():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        i32 = llvm.types.i32()
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b):
            b.ret(None)
            with InsertPoint.at_block_begin(entry) as outer:
                b.alloca(i32, "x")  # before ret
                with InsertPoint.at_block_terminator(entry):
                    b.alloca(i32, "y")  # before ret (after x)
                # Inner exit pops the frame and restores the builder to where it
                # was on inner enter (just before ret, i.e. after x/y) -- NOT to
                # the outer factory's begin position.
                assert InsertPoint.current is outer
                b.alloca(i32, "z")  # before ret, after y
        # Named CHECKs pin the actual order x, y, z, ret.
        # CHECK: %x = alloca
        # CHECK: %y = alloca
        # CHECK: %z = alloca
        # CHECK: ret void
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_insert_point_restores_on_exception():
    # __exit__ pops the frame and restores the builder even when the body raises.
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        outer_bb = fn.append_basic_block("outer")
        inner_bb = fn.append_basic_block("inner")
        b = IRBuilder(ctx)
        with InsertPoint(outer_bb, builder=b):
            with pytest.raises(ValueError, match="boom"):
                with InsertPoint(inner_bb):
                    assert b.insert_block == inner_bb
                    raise ValueError("boom")
            assert InsertPoint.current.block == outer_bb  # frame popped
            assert b.insert_block == outer_bb  # builder restored
            b.ret(None)
        del mod
    assert_no_leaks()


def test_current_function_follows_builder_across_functions():
    # current_function() is *derived* from the current builder's block, so it
    # follows the builder into a different function rather than tracking the
    # InsertPoint's original block.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        i32 = llvm.types.i32()
        f1 = llvm.ir.Function.create(llvm.types.function(i32, []), "f1", mod)
        f2 = llvm.ir.Function.create(llvm.types.function(i32, []), "f2", mod)
        e1 = f1.append_basic_block("entry")
        e2 = f2.append_basic_block("entry")
        b = IRBuilder(ctx)
        with InsertPoint(e1, builder=b):
            assert current_function() is f1
            b.set_insert_point(e2)  # move the builder into f2
            assert current_function() is f2
            b.ret(llvm.ir.const_int(i32, 0))
            b.set_insert_point(e1)
            b.ret(llvm.ir.const_int(i32, 0))
        del mod
    assert_no_leaks()


def test_factory_with_explicit_builder_as_outermost():
    # A factory used as the outermost context manager with an explicit builder
    # (no enclosing InsertPoint to resolve the current one from).
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        i32 = llvm.types.i32()
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        b.set_insert_point(entry)
        b.ret(None)
        with InsertPoint.at_block_terminator(entry, builder=b):
            assert current_builder() is b
            b.alloca(i32, "pre")
        # CHECK: %pre = alloca
        # CHECK: ret void
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_irbuilder_is_current_via_context_manager():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with b:  # entering the builder makes it the current one
            assert current_builder() is b
        with pytest.raises(RuntimeError, match="no current IRBuilder"):
            current_builder()
        with b:  # a bare builder frame has no active InsertPoint
            with pytest.raises(ValueError, match="no current InsertPoint"):
                _ = InsertPoint.current
        b.set_insert_point(entry)
        b.ret(None)
        del mod
    assert_no_leaks()


def test_insert_point_resolves_builder_from_enclosing_with_builder():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        i32 = llvm.types.i32()
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with b:  # b is the current builder
            with InsertPoint(entry):  # no explicit builder -> resolves b
                assert current_builder() is b
                b.alloca(i32, "a")
                b.ret(None)
        # CHECK: %a = alloca
        # CHECK: ret void
        filecheck_with_comments(mod)
        del mod
    assert_no_leaks()


def test_current_function_from_bare_builder():
    # current_function() derives from a bare `with builder:` frame too: it raises
    # while the builder has no insertion block, and returns the block's function
    # once positioned.
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with b:
            with pytest.raises(RuntimeError, match="no current function"):
                current_function()  # not positioned yet
            b.set_insert_point(entry)
            assert current_function() is fn
            b.ret(None)
        del mod
    assert_no_leaks()


def test_irbuilder_unbalanced_exit_raises():
    # IRBuilder.__exit__ is symmetric with InsertPoint.__exit__: exiting the
    # builder while an InsertPoint frame is on top raises rather than silently
    # popping the wrong frame.
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        b.__enter__()
        ip = InsertPoint(entry)  # builder=None -> resolves b
        ip.__enter__()
        with pytest.raises(ValueError, match="unbalanced IRBuilder"):
            b.__exit__(None, None, None)  # top is the InsertPoint frame
        ip.__exit__(None, None, None)
        b.__exit__(None, None, None)
        b.set_insert_point(entry)
        b.ret(None)
        del mod
    assert_no_leaks()


def test_insert_point_current_raises_when_none():
    with pytest.raises(ValueError, match="no current InsertPoint"):
        _ = InsertPoint.current


def test_current_builder_raises_when_none():
    with pytest.raises(RuntimeError, match="no current IRBuilder"):
        current_builder()


def test_current_function_raises_when_none():
    with pytest.raises(RuntimeError, match="no current function"):
        current_function()


def test_insert_point_no_builder_and_none_current_raises():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        entry = fn.append_basic_block("entry")
        # No builder= and no enclosing InsertPoint -> nothing to reposition.
        with pytest.raises(ValueError, match="no current IRBuilder"):
            with InsertPoint(entry):
                pass
        del mod
    assert_no_leaks()


def test_insert_point_bad_target_raises():
    with pytest.raises(TypeError, match="BasicBlock .* or an Instruction"):
        InsertPoint(123)


def test_at_block_terminator_without_terminator_raises():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        entry = fn.append_basic_block("entry")  # no terminator
        with pytest.raises(ValueError, match="no terminator"):
            InsertPoint.at_block_terminator(entry)
        del mod
    assert_no_leaks()


def test_insert_point_unbalanced_exit_raises():
    # Guards the balance check: exiting out of order raises rather than
    # restoring the wrong builder position.
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx)
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        ip1 = InsertPoint(entry, builder=b)
        ip2 = InsertPoint(entry, builder=b)
        ip1.__enter__()
        ip2.__enter__()
        with pytest.raises(ValueError, match="unbalanced"):
            ip1.__exit__(None, None, None)  # top is ip2, not ip1
        ip2.__exit__(None, None, None)
        ip1.__exit__(None, None, None)
        b.set_insert_point(entry)
        b.ret(None)
        del mod
    assert_no_leaks()
