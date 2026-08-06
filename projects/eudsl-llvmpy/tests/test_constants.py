#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm
from llvm.testing import assert_no_leaks


def test_const_int():
    with llvm.Context() as ctx:
        c = llvm.const_int(llvm.types.i32(ctx), 42)
        assert type(c).__name__ == "ConstantInt"
        assert c.value == 42
        assert str(c) == "i32 42"
        neg = llvm.const_int(llvm.types.i32(ctx), -1, signed=True)
        assert neg.value == -1
    assert_no_leaks()


def test_const_int_signed_flag_is_currently_inert():
    # const_int takes an int64: a negative value is always built signed
    # (isSigned || value < 0), and a non-negative int64 has bit 63 clear, so
    # sign- and zero-extension coincide. So the `signed` flag does not change
    # the resulting constant for any representable value. Pin that so a future
    # change to the flag's meaning is a visible, deliberate break.
    with llvm.Context() as ctx:
        for ty in (llvm.i32(ctx), llvm.int_t(ctx, 128)):
            for v in (-1, 7, 0):
                a = llvm.const_int(ty, v, signed=True)
                b = llvm.const_int(ty, v, signed=False)
                assert str(a) == str(b)
    assert_no_leaks()


def test_const_bool_and_fp():
    with llvm.Context() as ctx:
        t = llvm.const_bool(ctx, True)
        assert type(t).__name__ == "ConstantInt"
        assert str(t) == "i1 true"
        f = llvm.const_fp(llvm.types.f64(ctx), 1.5)
        assert type(f).__name__ == "ConstantFP"
        assert f.double_value == 1.5
        assert str(f) == "double 1.500000e+00"
    assert_no_leaks()


def test_undef_poison_null():
    with llvm.Context() as ctx:
        assert type(llvm.undef(llvm.types.i32(ctx))).__name__ == "UndefValue"
        assert type(llvm.poison(llvm.types.i32(ctx))).__name__ == "PoisonValue"
        assert type(llvm.null(llvm.types.ptr(ctx))).__name__ == "ConstantPointerNull"
        assert str(llvm.undef(llvm.types.i32(ctx))) == "i32 undef"
    assert_no_leaks()
