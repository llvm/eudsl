#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""SPIR-V -> WGSL, so kernels compiled in the browser can run on WebGPU.

Only built when the wheel is configured with MLIR_PYTHON_BINDINGS_WGSL=ON, since
it pulls in Tint.
"""

from ._mlir_libs import _mlirSPIRVToWGSL

__all__ = ["spirv_to_wgsl"]

SPIRV_MAGIC = 0x07230203


def spirv_to_wgsl(spirv: bytes) -> str:
    """Translate a SPIR-V binary to WGSL source.

    Raises RuntimeError carrying Tint's diagnostics if translation fails.
    """
    if len(spirv) >= 4:
        magic = int.from_bytes(spirv[:4], "little")
        if magic != SPIRV_MAGIC:
            raise ValueError(
                f"not a SPIR-V binary: magic is {magic:#010x}, expected {SPIRV_MAGIC:#010x}"
            )
    return _mlirSPIRVToWGSL.spirv_to_wgsl(spirv)
