#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile MLIR to WGSL and run it on the GPU, from the browser.

    from mlir.webgpu import compile_to_wgsl, dispatch

    wgsl = compile_to_wgsl(my_gpu_module_mlir)
    c = await dispatch(wgsl, inputs=[a, b], out_shape=(32, 32),
                       entry_point="matmul")

The compile half (`compile_to_spirv`, `compile_to_wgsl`) is pure MLIR and works
anywhere the wheel does. The dispatch half needs `navigator.gpu`, so it only
works inside Pyodide in a browser -- `js` and `pyodide.ffi` are imported lazily
so importing this module elsewhere does not fail.

Only built when the wheel is configured with MLIR_PYTHON_BINDINGS_WGSL=ON.
"""

from .wgsl import spirv_to_wgsl

__all__ = [
    "LOWER_PIPELINE",
    "SERIALIZE_PIPELINE",
    "compile_to_spirv",
    "compile_to_wgsl",
    "dispatch",
    "extract_binary",
    "nest_spirv_module",
]

# Vulkan-flavoured: the Shader capability gives a Logical/GLSL450 module, which
# is what WebGPU accepts. The Kernel/OpenCL flavour used for CPU and Level Zero
# targets will not translate.
LOWER_PIPELINE = (
    "builtin.module("
    "spirv-attach-target{ver=v1.0 caps=Shader "
    "exts=SPV_KHR_storage_buffer_storage_class client_api=Vulkan},"
    "convert-gpu-to-spirv,"
    "spirv.module(spirv-lower-abi-attrs,spirv-update-vce,spirv-webgpu-prepare))"
)
SERIALIZE_PIPELINE = "builtin.module(gpu-module-to-binary)"

# GPUBufferUsage flags. Plain ints in JS, so no need to reach for the enum.
_STORAGE = 0x80
_COPY_DST = 0x8
_COPY_SRC = 0x4
_MAP_READ = 0x1


def nest_spirv_module(asm: str) -> str:
    """Move a top-level `spirv.module` inside the `gpu.module` beside it.

    convert-gpu-to-spirv hoists the generated spirv.module to the top level as a
    *sibling* of the now-empty gpu.module, but gpu-module-to-binary serializes
    via SPIRVTargetAttrImpl::serializeToObject, which only looks for a
    spirv.module *inside* a gpu.module. Upstream's Vulkan runner pipeline avoids
    this with test-convert-to-spirv{nest-in-gpu-module=true}, but that pass is
    test-only and is not in the shipped wheel.
    """
    lines = asm.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("  spirv.module"))
    depth = 0
    end = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        # A single-line spirv.module balances at i == start, so close there too.
        if depth == 0:
            end = i
            break
    if end is None:
        raise ValueError("unterminated spirv.module")
    spv = lines[start : end + 1]
    rest = lines[:start] + lines[end + 1 :]
    g = next(i for i, l in enumerate(rest) if l.startswith("  gpu.module"))
    return "\n".join(rest[: g + 1] + ["  " + l for l in spv] + rest[g + 1 :])


def extract_binary(asm: str) -> bytes:
    """Pull the object blob out of a `gpu.binary` op.

    MLIR escapes the bytes as \\XX hex pairs, except for `"` and `\\` which it
    escapes as themselves.
    """
    i = asm.find("gpu.binary")
    if i < 0:
        raise ValueError("no gpu.binary op -- did gpu-module-to-binary run?")
    if asm.find("gpu.binary", i + 1) >= 0:
        raise ValueError("multiple gpu.binary ops; expected one")
    j = asm.index('"', i) + 1
    out = bytearray()
    while True:
        if j >= len(asm):
            raise ValueError("unterminated gpu.binary blob")
        c = asm[j]
        if c == '"':
            break
        if c == "\\":
            nxt = asm[j + 1]
            if nxt in ('"', "\\"):
                out.append(ord(nxt))
                j += 2
            else:
                out.append(int(asm[j + 1 : j + 3], 16))
                j += 3
        else:
            out.append(ord(c))
            j += 1
    return bytes(out)


def compile_to_spirv(asm: str) -> bytes:
    """Lower a `gpu.module` of kernels to a SPIR-V binary."""
    from .ir import Context, Module
    from .passmanager import PassManager

    with Context():
        module = Module.parse(asm)
        PassManager.parse(LOWER_PIPELINE).run(module.operation)
        lowered = str(module)

    # Nesting is textual, so re-parse the result.
    with Context():
        module = Module.parse(nest_spirv_module(lowered))
        PassManager.parse(SERIALIZE_PIPELINE).run(module.operation)
        return extract_binary(str(module))


def compile_to_wgsl(asm: str) -> str:
    """Lower a `gpu.module` of kernels all the way to WGSL source."""
    return spirv_to_wgsl(compile_to_spirv(asm))


async def get_device():
    """Acquire a WebGPU device, with errors that say what is wrong."""
    from js import navigator

    if not hasattr(navigator, "gpu"):
        raise RuntimeError(
            "navigator.gpu is missing -- this browser has no WebGPU support"
        )
    adapter = await navigator.gpu.requestAdapter()
    if adapter is None:
        raise RuntimeError("requestAdapter() returned null -- no WebGPU adapter")
    return await adapter.requestDevice()


async def dispatch(wgsl, inputs, out_shape, entry_point, workgroup=(8, 8, 1),
                   device=None):
    """Run `wgsl` over `inputs`, returning an out_shape float32 array.

    Buffers bind at @group(0) @binding(i): the inputs in order, then the output
    last. That is what spirv-lower-abi-attrs emits for a gpu.func's memref
    arguments, so it matches a kernel whose output is its final parameter.
    """
    import numpy as np
    from js import Float32Array, Object
    from pyodide.ffi import to_js

    def js_obj(d):
        return to_js(d, dict_converter=Object.fromEntries)

    if device is None:
        device = await get_device()

    def upload(arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        buf = device.createBuffer(
            js_obj({"size": arr.nbytes, "usage": _STORAGE | _COPY_DST,
                    "mappedAtCreation": True})
        )
        Float32Array.new(buf.getMappedRange()).set(to_js(arr.ravel().tolist()))
        buf.unmap()
        return buf

    in_bufs = [upload(a) for a in inputs]
    rows, cols = out_shape
    out_bytes = rows * cols * 4
    out_buf = device.createBuffer(
        js_obj({"size": out_bytes, "usage": _STORAGE | _COPY_SRC})
    )
    read_buf = device.createBuffer(
        js_obj({"size": out_bytes, "usage": _COPY_DST | _MAP_READ})
    )

    module = device.createShaderModule(js_obj({"code": wgsl}))
    pipeline = device.createComputePipeline(
        js_obj({"layout": "auto",
                "compute": js_obj({"module": module, "entryPoint": entry_point})})
    )
    bind_group = device.createBindGroup(
        js_obj({
            "layout": pipeline.getBindGroupLayout(0),
            "entries": to_js([
                js_obj({"binding": i, "resource": js_obj({"buffer": buf})})
                for i, buf in enumerate([*in_bufs, out_buf])
            ]),
        })
    )

    wg_x, wg_y, _ = workgroup
    encoder = device.createCommandEncoder()
    compute = encoder.beginComputePass()
    compute.setPipeline(pipeline)
    compute.setBindGroup(0, bind_group)
    compute.dispatchWorkgroups(
        (rows + wg_x - 1) // wg_x, (cols + wg_y - 1) // wg_y, 1
    )
    compute.end()
    encoder.copyBufferToBuffer(out_buf, 0, read_buf, 0, out_bytes)
    device.queue.submit(to_js([encoder.finish()]))

    await read_buf.mapAsync(_MAP_READ)
    got = np.asarray(
        Float32Array.new(read_buf.getMappedRange()).to_py(), dtype=np.float32
    )
    got = got.reshape(rows, cols).copy()
    read_buf.unmap()
    return got
