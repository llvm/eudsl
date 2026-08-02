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

    Raises ValueError if the input is not a SPIR-V binary, RuntimeError carrying
    Tint's diagnostics if translation fails, and MemoryError if the translator
    runs out of memory.
    """
    # A SPIR-V module is at least a 5-word header. Check the length before the
    # magic so a short buffer gets this message rather than being handed to the
    # C++ side, which would report a confusing Tint-flavoured error instead.
    if len(spirv) < 20:
        raise ValueError(
            f"not a SPIR-V binary: {len(spirv)} bytes is shorter than the "
            "20-byte header"
        )
    magic = int.from_bytes(spirv[:4], "little")
    if magic != SPIRV_MAGIC:
        raise ValueError(
            f"not a SPIR-V binary: magic is {magic:#010x}, expected {SPIRV_MAGIC:#010x}"
        )
    return _mlirSPIRVToWGSL.spirv_to_wgsl(spirv)
