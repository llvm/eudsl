#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes, math, platform
import pytest
import llvm
from llvm import ir, jit, mir
from llvm.testing import assert_no_leaks

pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)
_AARCH64_LINUX = "aarch64-unknown-linux-gnu"
_IS_AARCH64 = platform.machine() in ("arm64", "aarch64")


def test_register_regalloc_requires_regallocbase_subclass():
    class NotAnAllocator:
        def select_or_split(self, li): ...

    with pytest.raises(TypeError, match="RegAllocBase"):
        mir.register_regalloc("ra-bad", NotAnAllocator)
