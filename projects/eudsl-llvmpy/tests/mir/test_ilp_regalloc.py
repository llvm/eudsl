#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ILP register allocators: pure-helper unit tests + end-to-end + comparison."""

import pytest
import llvm
from llvm import mir_ilp_model as model


def test_require_ortools_returns_cp_model():
    cp = model._require_ortools()
    assert hasattr(cp, "CpModel")


def test_scale_weight_is_positive_int():
    assert model.scale_weight(0.0) >= 1
    assert isinstance(model.scale_weight(1.5), int)
    assert model.scale_weight(2.0) > model.scale_weight(1.0)
