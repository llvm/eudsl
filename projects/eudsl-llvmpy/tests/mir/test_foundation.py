#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Foundation for the MIR (Machine IR) submodule.

These pin the bare plumbing of the new `llvm.mir` submodule plus the
target-independent LowLevelType (LLT), which is the first thing bound because
it proves the CodeGen/CodeGenTypes libraries actually link into the extension.
LLT is a value type (not uniqued in a Context), so no `with Context()` is
needed and there is nothing to leak-check here.
"""

import llvm
from llvm import mir


def test_mir_submodule_is_importable():
    assert mir is llvm.mir


def test_llt_scalar_size_in_bits():
    assert mir.LLT.scalar(32).size_in_bits == 32
    assert mir.LLT.scalar(1).size_in_bits == 1


def test_llt_scalar_predicates():
    s = mir.LLT.scalar(32)
    assert s.is_scalar
    assert not s.is_pointer
    assert not s.is_vector
    assert s.is_valid


def test_llt_pointer():
    p = mir.LLT.pointer(0, 64)
    assert p.is_pointer
    assert not p.is_scalar
    assert p.size_in_bits == 64
    assert p.address_space == 0


def test_llt_fixed_vector():
    v = mir.LLT.fixed_vector(4, 32)
    assert v.is_vector
    assert v.num_elements == 4
    assert v.scalar_size_in_bits == 32
    assert v.size_in_bits == 128


def test_llt_equality():
    assert mir.LLT.scalar(32) == mir.LLT.scalar(32)
    assert mir.LLT.scalar(32) != mir.LLT.scalar(64)


def test_llt_str():
    assert str(mir.LLT.scalar(32)) == "s32"
    assert str(mir.LLT.fixed_vector(4, 32)) == "<4 x s32>"
