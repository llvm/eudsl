# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.dialects import llvm
from mlir.extras.dialects.func import func
from mlir.extras.dialects.llvm import mlir_constant

# noinspection PyUnresolvedReferences
from mlir.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
from util import llvm_bindings_not_installed, llvm_amdgcn_bindings_not_installed

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


@pytest.mark.skipif(
    llvm_bindings_not_installed() or llvm_amdgcn_bindings_not_installed(),
    reason="llvm bindings not installed or llvm_amdgcn bindings not installed",
)
def test_call_instrinsic(ctx: MLIRContext):
    @func(emit=True)
    def sum(a: T.i32(), b: T.i32(), c: T.f32()):
        e = llvm.amdgcn.cvt_pk_i16(a, b)
        f = llvm.amdgcn.frexp_mant(c)

    correct = dedent("""
    module {
      func.func @sum(%arg0: i32, %arg1: i32, %arg2: f32) {
        %0 = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.i16"(%arg0, %arg1) : (i32, i32) -> vector<2xi16>
        %1 = llvm.call_intrinsic "llvm.amdgcn.frexp.mant"(%arg2) : (f32) -> f32
        return
      }
    }
    """)
    filecheck(correct, ctx.module)


def test_mlir_constant_int(ctx: MLIRContext):
    # Test mlir_constant with int value and no explicit type (lines 29-33)
    c = mlir_constant(42)
    assert c is not None

    # Test mlir_constant with explicit type
    c2 = mlir_constant(7, type=T.i32())
    assert c2 is not None

    correct = dedent("""\
    module {
      %0 = llvm.mlir.constant(42 : i32) : i32
      %1 = llvm.mlir.constant(7 : i32) : i32
    }
    """)
    filecheck(correct, ctx.module)


def test_mlir_constant_float(ctx: MLIRContext):
    # Test mlir_constant with float value (lines 34-35)
    c = mlir_constant(3.14)
    assert c is not None

    c2 = mlir_constant(2.5, type=T.f32())
    assert c2 is not None

    correct = dedent("""\
    module {
      %0 = llvm.mlir.constant(3.140000e+00 : f32) : f32
      %1 = llvm.mlir.constant(2.500000e+00 : f32) : f32
    }
    """)
    filecheck(correct, ctx.module)


def test_mlir_constant_unsupported_type(ctx: MLIRContext):
    # Test mlir_constant with unsupported type (lines 36-37)
    # Need to pass type explicitly to bypass infer_mlir_type
    with pytest.raises(NotImplementedError, match="is not a valid type"):
        mlir_constant("hello", type=T.i32())
