#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import llvm


def test_linked_targets():
    # TargetRegistry reports the short target names (lowercase), not the LLVM
    # library names (AArch64/X86).
    targets = llvm.jit.registered_targets()
    # Host targets are present.
    assert any(t in targets for t in ("aarch64", "x86", "x86-64", "arm64"))
    # AMDGPU is linked when the LLVM provides it: it is the one target with
    # sub-register liveness, which mir.RAGreedy's tryInstructionSplit test needs
    # (see CMakeLists.txt). It is in the default distribution.
    assert "amdgcn" in targets
    # NVPTX is still not linked (its target-init lacks an AsmParser).
    assert "nvptx" not in targets
    assert "nvptx64" not in targets
