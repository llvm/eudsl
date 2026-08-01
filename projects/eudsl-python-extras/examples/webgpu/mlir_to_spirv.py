"""MLIR -> SPIR-V for WebGPU, using only passes present in the shipped wasm wheel.

Emits a Vulkan-flavored (Logical/GLSL450, [Shader]) SPIR-V compute module whose
storage buffers are decorated @group(0) @binding(i) for kernel argument i --
the layout the WebGPU host code binds against.

Why the module surgery in nest_spirv_module(): `convert-gpu-to-spirv` hoists the
generated `spirv.module` to the top level, as a *sibling* of the now-empty
`gpu.module`. But `gpu-module-to-binary` serializes via
SPIRVTargetAttrImpl::serializeToObject, which looks for a `spirv.module` *inside*
a `gpu.module` and returns nullopt otherwise. Upstream's Vulkan runner pipeline
avoids this by using `test-convert-to-spirv{nest-in-gpu-module=true}`, but that
pass is test-only and is NOT in the shipped wheel (verified against
libMLIRPythonWasmCAPI.so). So we do the nesting ourselves.
"""

import re
import struct

SPIRV_MAGIC = 0x07230203

TARGET_ENV = (
    "spirv-attach-target{ver=v1.0 caps=Shader "
    "exts=SPV_KHR_storage_buffer_storage_class client_api=Vulkan}"
)

# Runs before nesting: convert-gpu-to-spirv leaves spirv.module at the top level.
LOWER_PIPELINE = (
    "builtin.module("
    + TARGET_ENV
    + ",convert-gpu-to-spirv"
    + ",spirv.module(spirv-lower-abi-attrs,spirv-update-vce,spirv-webgpu-prepare)"
    + ")"
)

# Runs after nesting.
SERIALIZE_PIPELINE = "builtin.module(gpu-module-to-binary)"


def nest_spirv_module(asm: str) -> str:
    """Move a top-level `spirv.module` inside the `gpu.module` beside it."""
    lines = asm.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("  spirv.module"))
    depth = 0
    end = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            end = i
            break
    if end is None:
        raise ValueError("unterminated spirv.module")
    spv = lines[start : end + 1]
    rest = lines[:start] + lines[end + 1 :]
    g = next(i for i, l in enumerate(rest) if l.startswith("  gpu.module"))
    return "\n".join(rest[: g + 1] + ["  " + l for l in spv] + rest[g + 1 :])


def extract_binary(asm: str) -> bytes:
    """Pull the object blob out of a `gpu.binary` op. MLIR escapes bytes as \\XX."""
    i = asm.find("gpu.binary")
    if i < 0:
        raise ValueError("no gpu.binary op -- did gpu-module-to-binary run?")
    j = asm.index('"', i) + 1
    out = bytearray()
    while asm[j] != '"':
        if asm[j] == "\\":
            out.append(int(asm[j + 1 : j + 3], 16))
            j += 3
        else:
            out.append(ord(asm[j]))
            j += 1
    return bytes(out)


def describe(spv: bytes) -> dict:
    """Decode the header plus the parts the WebGPU host contract depends on."""
    w = struct.unpack("<%dI" % (len(spv) // 4), spv)
    if w[0] != SPIRV_MAGIC:
        raise ValueError(f"bad magic {w[0]:#x}, expected {SPIRV_MAGIC:#x}")

    def string_at(a, b):
        return b"".join(struct.pack("<I", x) for x in w[a:b]).split(b"\0")[0].decode()

    info = {
        "bytes": len(spv),
        "version": f"{(w[1] >> 16) & 0xFF}.{(w[1] >> 8) & 0xFF}",
        "bound": w[3],
        "entry_point": None,
        "workgroup_size": None,
        "bindings": [],
    }
    names, decos = {}, {}
    i = 5
    while i < len(w):
        wc, op = w[i] >> 16, w[i] & 0xFFFF
        if wc == 0:
            break
        if op == 5:  # OpName
            names[w[i + 1]] = string_at(i + 2, i + wc)
        elif op == 15:  # OpEntryPoint
            info["entry_point"] = string_at(i + 3, i + wc)
        elif op == 16:  # OpExecutionMode
            info["workgroup_size"] = list(w[i + 3 : i + wc])
        elif op == 71 and w[i + 2] in (33, 34):  # OpDecorate Binding/DescriptorSet
            decos.setdefault(w[i + 1], {})[w[i + 2]] = w[i + 3]
        i += wc
    info["bindings"] = sorted(
        (d.get(34), d.get(33), names.get(v, "")) for v, d in decos.items()
    )
    return info
