#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

import llvm
from llvm.testing import assert_no_leaks


def test_parse_error_is_specific():
    with llvm.ir.Context() as ctx:
        with pytest.raises(llvm.ir.ParseError):
            llvm.ir.parse_assembly("this is not IR", ctx, "bad")
    assert_no_leaks()


def test_parse_error_is_an_exception_subclass():
    assert issubclass(llvm.ir.ParseError, Exception)
    assert issubclass(llvm.ir.VerifyError, Exception)
