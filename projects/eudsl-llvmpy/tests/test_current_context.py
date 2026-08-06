#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-style implicit current context: `with Context():` sets Context.current,
so the primitive type factories can be called without an explicit context."""
import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_current_context_stack():
    assert llvm.Context.current() is None
    with llvm.Context() as a:
        assert llvm.Context.current() is a
        with llvm.Context() as b:
            assert llvm.Context.current() is b  # innermost wins
        assert llvm.Context.current() is a  # popped back to a
    assert llvm.Context.current() is None


def test_implicit_context_factories():
    with llvm.Context() as ctx:
        # No explicit context -> uses the current one.
        assert llvm.i32() == llvm.i32(ctx)
        assert str(llvm.f32()) == "float"
        assert str(llvm.void_t()) == "void"
        assert llvm.i64().bit_width == 64
        # Explicit context still works and agrees.
        assert llvm.f64() == llvm.f64(ctx)
        # Reordered multi-arg factories also default to the current context.
        assert llvm.int_t(7) == llvm.int_t(7, context=ctx)
        assert str(llvm.ptr_t()) == "ptr"
        assert llvm.struct_t([llvm.i32()]).num_elements == 1
        assert "hi" in str(llvm.md_string("hi"))
    assert_no_leaks()


def test_no_current_context_raises():
    assert llvm.Context.current() is None
    with pytest.raises(RuntimeError, match="no context given and no current"):
        llvm.i32()
