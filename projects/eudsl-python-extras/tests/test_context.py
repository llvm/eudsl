# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mlir import ir
from mlir.extras.context import (
    mlir_mod,
    mlir_mod_ctx,
    RAIIMLIRContext,
    RAIIMLIRContextModule,
    ExplicitlyManagedModule,
    enable_multithreading,
    disable_multithreading,
    enable_debug,
)
from mlir.extras.testing import mlir_ctx as ctx, MLIRContext

pytest.mark.usefixtures("ctx")


def test_mlir_context_str(ctx: MLIRContext):
    """Test MLIRContext.__str__ (line 19)."""
    s = str(ctx)
    assert "module" in s


def test_mlir_mod_with_src():
    """Test mlir_mod with src argument (line 32)."""
    with ir.Context(), mlir_mod(src="module {}") as module:
        assert module is not None
        assert "module" in str(module.operation)


def test_raii_mlir_context_allow_unregistered():
    """Test RAIIMLIRContext with allow_unregistered_dialects=True (line 67)."""
    raii = RAIIMLIRContext(allow_unregistered_dialects=True)
    assert raii.context.allow_unregistered_dialects is True
    del raii


def test_raii_mlir_context_module_allow_unregistered():
    """Test RAIIMLIRContextModule with allow_unregistered_dialects=True (line 93)."""
    raii = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    assert raii.context.allow_unregistered_dialects is True
    assert raii.module is not None
    del raii


def test_enable_multithreading():
    """Test enable_multithreading context manager (lines 131-137)."""
    with ir.Context() as context:
        with ir.Location.unknown():
            with enable_multithreading(context):
                pass


def test_enable_multithreading_no_arg():
    """Test enable_multithreading with no context arg (lines 131-137)."""
    with ir.Context() as context:
        with ir.Location.unknown():
            with enable_multithreading():
                pass


def test_disable_multithreading():
    """Test disable_multithreading context manager (lines 142-149)."""
    with ir.Context() as context:
        with ir.Location.unknown():
            with disable_multithreading(context):
                pass


def test_disable_multithreading_no_arg():
    """Test disable_multithreading with no context arg (lines 142-149)."""
    with ir.Context() as context:
        with ir.Location.unknown():
            with disable_multithreading():
                pass


def test_explicitly_managed_module():
    """Test ExplicitlyManagedModule (lines 117-119, 122-123, 126)."""
    with ir.Context(), ir.Location.unknown():
        emm = ExplicitlyManagedModule()
        assert emm.module is not None
        s = str(emm)
        assert "module" in s
        module = emm.finish()
        assert module is not None


def test_enable_debug():
    """Test enable_debug context manager (lines 154-156)."""
    with enable_debug():
        assert ir._GlobalDebug.flag is True
    assert ir._GlobalDebug.flag is False


def test_mlir_mod_with_location():
    """Branch 28->30: mlir_mod with explicit location."""
    with ir.Context():
        loc = ir.Location.unknown()
        with mlir_mod(location=loc) as module:
            assert module is not None


def test_mlir_mod_ctx_with_context():
    """Branch 47->49: mlir_mod_ctx with explicit context."""
    context = ir.Context()
    with mlir_mod_ctx(context=context) as ctx:
        assert ctx.module is not None


def test_raii_mlir_context_with_location():
    """Branch 69->71: RAIIMLIRContext with explicit location."""
    with ir.Context():
        loc = ir.Location.unknown()
    ctx = RAIIMLIRContext(location=loc)
    assert ctx.location is loc


def test_raii_mlir_context_module_with_location():
    """Branch 95->97: RAIIMLIRContextModule with explicit location."""
    with ir.Context():
        loc = ir.Location.unknown()
    ctx = RAIIMLIRContextModule(location=loc)
    assert ctx.location is loc
