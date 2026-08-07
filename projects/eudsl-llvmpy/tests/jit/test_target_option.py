#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm


def test_only_host_targets_are_linked():
    # TargetRegistry reports the short target names (lowercase), not the LLVM
    # library names (AArch64/X86).
    targets = llvm.jit.registered_targets()
    # Host targets are present.
    assert any(t in targets for t in ("aarch64", "x86", "x86-64", "arm64"))
    # The GPU backends were dropped from the default build.
    assert "amdgcn" not in targets
    assert "r600" not in targets
    assert "nvptx" not in targets
    assert "nvptx64" not in targets
