#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-style implicit current context: `with Context():` sets Context.current,
so the primitive type factories can be called without an explicit context."""

import threading

import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_current_context_stack():
    assert llvm.ir.Context.current() is None
    with llvm.ir.Context() as a:
        assert llvm.ir.Context.current() is a
        with llvm.ir.Context() as b:
            assert llvm.ir.Context.current() is b  # innermost wins
        assert llvm.ir.Context.current() is a  # popped back to a
    assert llvm.ir.Context.current() is None


def test_current_context_is_thread_local():
    # The current-context stack is `thread_local`: a context entered on the
    # parent thread is invisible to a child thread, and the child cannot
    # corrupt the parent's stack. Without thread-locality the child would see
    # `parent` (or clobber it), so this pins the documented guarantee.
    with llvm.ir.Context() as parent:
        assert llvm.ir.Context.current() is parent
        seen = []
        error = []

        def child():
            try:
                seen.append(llvm.ir.Context.current())
                with llvm.ir.Context() as own:
                    seen.append(llvm.ir.Context.current() is own)
                seen.append(llvm.ir.Context.current())
            except Exception as e:  # surface failures on the main thread
                error.append(e)

        t = threading.Thread(target=child)
        t.start()
        t.join()
        assert not error, error
        # Fresh thread: empty stack -> None; its own context wins while held;
        # None again after it exits.
        assert seen == [None, True, None]
        # The child never perturbed the parent thread's current context.
        assert llvm.ir.Context.current() is parent
    assert llvm.ir.Context.current() is None
    assert_no_leaks()


def test_exit_restores_current_on_exception():
    # An exception propagating out of the `with` body must still pop the stack
    # (guards against a regression where __exit__ only ran on the happy path).
    assert llvm.ir.Context.current() is None
    with llvm.ir.Context() as outer:
        with pytest.raises(ValueError, match="boom"):
            with llvm.ir.Context() as inner:
                assert llvm.ir.Context.current() is inner
                raise ValueError("boom")
        # inner.__exit__ popped despite the exception.
        assert llvm.ir.Context.current() is outer
    assert llvm.ir.Context.current() is None
    assert_no_leaks()


def test_same_context_reentered():
    # Re-entering the *same* Context nests two pushes of one object; each exit
    # pops one, so the stack unwinds cleanly and the live count returns to 0.
    with llvm.ir.Context() as a:
        assert llvm.ir.Context.current() is a
        with a:
            assert llvm.ir.Context.current() is a  # pushed a second time
        assert llvm.ir.Context.current() is a  # one pop, still a
    assert llvm.ir.Context.current() is None
    assert_no_leaks()


def test_implicit_context_factories():
    with llvm.ir.Context() as ctx:
        # No explicit context -> uses the current one.
        assert llvm.types.i32() == llvm.types.i32(ctx)
        assert str(llvm.types.f32()) == "float"
        assert str(llvm.types.void()) == "void"
        assert llvm.types.i64().bit_width == 64
        # The remaining single-arg primitives resolve the current context too.
        assert str(llvm.types.label()) == "label"
        assert llvm.types.i1().bit_width == 1
        assert llvm.types.i8().bit_width == 8
        assert llvm.types.i16().bit_width == 16
        assert str(llvm.types.f16()) == "half"
        # Explicit context still works and agrees.
        assert llvm.types.f64() == llvm.types.f64(ctx)
        # Reordered multi-arg factories also default to the current context.
        assert llvm.types.int(7) == llvm.types.int(7, context=ctx)
        assert str(llvm.types.ptr()) == "ptr"
        assert llvm.types.struct([llvm.types.i32()]).num_elements == 1
        assert "hi" in str(llvm.ir.md_string("hi"))
    assert_no_leaks()


def test_no_current_context_raises():
    assert llvm.ir.Context.current() is None
    with pytest.raises(RuntimeError, match="no context given and no current"):
        llvm.types.i32()
