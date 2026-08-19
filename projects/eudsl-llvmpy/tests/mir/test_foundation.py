#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Foundation for the MIR (Machine IR) submodule.

These pin the bare plumbing of the new `llvm.mir` submodule plus the
target-independent LowLevelType (LLT), which is the first thing bound because
it proves the LLVMCodeGenTypes library actually links into the extension.
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
    # scalar() yields an ANY_SCALAR (neither INTEGER nor FLOAT kind), so both
    # kind predicates are False; they exist to mirror the full LLT API.
    assert not s.is_integer
    assert not s.is_float


def test_llt_scalar_size_in_bits_on_scalar_and_pointer():
    assert mir.LLT.scalar(32).scalar_size_in_bits == 32
    assert mir.LLT.pointer(0, 64).scalar_size_in_bits == 64


def test_llt_pointer():
    p = mir.LLT.pointer(0, 64)
    assert p.is_pointer
    assert not p.is_scalar
    assert p.size_in_bits == 64
    assert p.address_space == 0


def test_llt_pointer_nonzero_address_space():
    p = mir.LLT.pointer(3, 64)
    assert p.address_space == 3
    assert str(p) == "p3"


def test_llt_fixed_vector():
    v = mir.LLT.fixed_vector(4, 32)
    assert v.is_vector
    assert v.num_elements == 4
    assert v.scalar_size_in_bits == 32
    assert v.size_in_bits == 128


def test_llt_equality():
    assert mir.LLT.scalar(32) == mir.LLT.scalar(32)
    assert mir.LLT.scalar(32) != mir.LLT.scalar(64)
    assert not (mir.LLT.scalar(32) == mir.LLT.scalar(64))
    assert not (mir.LLT.scalar(32) != mir.LLT.scalar(32))


def test_llt_equality_with_non_llt():
    # Comparing against a non-LLT operand returns False/True rather than raising.
    assert mir.LLT.scalar(32) != 5
    assert not (mir.LLT.scalar(32) == 5)


def test_llt_hashable():
    # Equal LLTs hash equal and are usable as set/dict keys.
    assert hash(mir.LLT.scalar(32)) == hash(mir.LLT.scalar(32))
    assert hash(mir.LLT.scalar(32)) != hash(mir.LLT.scalar(64))
    types = {mir.LLT.scalar(32), mir.LLT.scalar(32), mir.LLT.fixed_vector(4, 32)}
    assert len(types) == 2
    assert mir.LLT.scalar(32) in types


def test_llt_str():
    assert str(mir.LLT.scalar(32)) == "s32"
    assert str(mir.LLT.fixed_vector(4, 32)) == "<4 x s32>"
