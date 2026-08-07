#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-style implicit current context: `with Context():` sets Context.current,
so the primitive type factories can be called without an explicit context."""
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


def test_implicit_context_factories():
    with llvm.ir.Context() as ctx:
        # No explicit context -> uses the current one.
        assert llvm.types.i32() == llvm.types.i32(ctx)
        assert str(llvm.types.f32()) == "float"
        assert str(llvm.types.void()) == "void"
        assert llvm.types.i64().bit_width == 64
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
