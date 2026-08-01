"""Compile an MLIR kernel to WGSL and run it on the GPU, from the browser.

    >>> from mlir.webgpu import demo
    >>> await demo()

The whole chain runs client-side: gpu.func MLIR -> SPIR-V (mlir passes) ->
WGSL (tint) -> dispatch on navigator.gpu. Nothing is precompiled and there is no
Dawn runtime -- the browser already implements WebGPU, so this reaches it
straight through Pyodide's JS interop.

Only importable inside Pyodide; `js` and `pyodide.ffi` do not exist elsewhere.
"""

from .wgsl import spirv_to_wgsl

__all__ = ["compile_to_wgsl", "dispatch", "demo", "MATMUL"]

M = N = K = 32
WG = 8

MATMUL = """
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @matmul(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [8, 8, 1]>} {
      %row = gpu.global_id x
      %col = gpu.global_id y
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %zero = arith.constant 0.0 : f32
      %sum = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %zero) -> (f32) {
        %a = memref.load %A[%row, %k] : memref<32x32xf32>
        %b = memref.load %B[%k, %col] : memref<32x32xf32>
        %m = arith.mulf %a, %b : f32
        %n = arith.addf %acc, %m : f32
        scf.yield %n : f32
      }
      memref.store %sum, %C[%row, %col] : memref<32x32xf32>
      gpu.return
    }
  }
}
"""

_LOWER = (
    "builtin.module("
    "spirv-attach-target{ver=v1.0 caps=Shader "
    "exts=SPV_KHR_storage_buffer_storage_class client_api=Vulkan},"
    "convert-gpu-to-spirv,"
    "spirv.module(spirv-lower-abi-attrs,spirv-update-vce,spirv-webgpu-prepare))"
)
_SERIALIZE = "builtin.module(gpu-module-to-binary)"

# GPUBufferUsage flags. Plain ints in JS, so no need to reach for the enum.
_STORAGE = 0x80
_COPY_DST = 0x8
_COPY_SRC = 0x4
_MAP_READ = 0x1


def _nest_spirv_module(asm):
    """Move the top-level spirv.module inside the gpu.module beside it.

    convert-gpu-to-spirv hoists the generated spirv.module to the top level as a
    *sibling* of the now-empty gpu.module, but gpu-module-to-binary serializes
    via SPIRVTargetAttrImpl::serializeToObject, which only looks for a
    spirv.module *inside* a gpu.module. Upstream's vulkan runner pipeline avoids
    this with test-convert-to-spirv{nest-in-gpu-module=true}, but that pass is
    test-only and is not in the shipped wheel.
    """
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


def _extract_binary(asm):
    """Pull the object blob out of gpu.binary. MLIR escapes bytes as \\XX."""
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


def compile_to_spirv(asm=MATMUL):
    """gpu.func MLIR -> SPIR-V binary."""
    from .ir import Context, Module
    from .passmanager import PassManager

    with Context():
        module = Module.parse(asm)
        PassManager.parse(_LOWER).run(module.operation)
        lowered = str(module)

    with Context():
        module = Module.parse(_nest_spirv_module(lowered))
        PassManager.parse(_SERIALIZE).run(module.operation)
        return _extract_binary(str(module))


def compile_to_wgsl(asm=MATMUL):
    """gpu.func MLIR -> WGSL source."""
    return spirv_to_wgsl(compile_to_spirv(asm))


async def dispatch(wgsl, a, b, entry_point="matmul"):
    """Run `wgsl` on the GPU against two numpy arrays, returning the result.

    Buffers are bound at @group(0) @binding(0..2) in argument order, which is
    what spirv-lower-abi-attrs emits for a gpu.func's memref arguments.
    """
    import numpy as np
    from js import Float32Array, Object, navigator
    from pyodide.ffi import to_js

    def js_obj(d):
        return to_js(d, dict_converter=Object.fromEntries)

    if not hasattr(navigator, "gpu"):
        raise RuntimeError("navigator.gpu missing -- WebGPU unavailable in this browser")
    adapter = await navigator.gpu.requestAdapter()
    if adapter is None:
        raise RuntimeError("requestAdapter() returned null -- no WebGPU adapter")
    device = await adapter.requestDevice()

    def upload(arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        buf = device.createBuffer(
            js_obj({"size": arr.nbytes, "usage": _STORAGE | _COPY_DST,
                    "mappedAtCreation": True})
        )
        Float32Array.new(buf.getMappedRange()).set(to_js(arr.ravel().tolist()))
        buf.unmap()
        return buf

    rows, cols = a.shape[0], b.shape[1]
    out_bytes = rows * cols * 4
    a_buf, b_buf = upload(a), upload(b)
    c_buf = device.createBuffer(js_obj({"size": out_bytes, "usage": _STORAGE | _COPY_SRC}))
    read_buf = device.createBuffer(js_obj({"size": out_bytes, "usage": _COPY_DST | _MAP_READ}))

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
                for i, buf in enumerate((a_buf, b_buf, c_buf))
            ]),
        })
    )

    encoder = device.createCommandEncoder()
    compute = encoder.beginComputePass()
    compute.setPipeline(pipeline)
    compute.setBindGroup(0, bind_group)
    compute.dispatchWorkgroups((rows + WG - 1) // WG, (cols + WG - 1) // WG, 1)
    compute.end()
    encoder.copyBufferToBuffer(c_buf, 0, read_buf, 0, out_bytes)
    device.queue.submit(to_js([encoder.finish()]))

    await read_buf.mapAsync(_MAP_READ)
    got = np.asarray(Float32Array.new(read_buf.getMappedRange()).to_py(), dtype=np.float32)
    got = got.reshape(rows, cols).copy()
    read_buf.unmap()
    return got


async def demo(verbose=True):
    """Compile MATMUL, dispatch it, and check the result against numpy."""
    import numpy as np

    spv = compile_to_spirv()
    wgsl = spirv_to_wgsl(spv)
    if verbose:
        print(f"SPIR-V: {len(spv)} bytes (magic {int.from_bytes(spv[:4], 'little'):#010x})")
        print(f"WGSL:   {len(wgsl)} bytes")
        print(wgsl)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K), dtype=np.float32)
    b = rng.standard_normal((K, N), dtype=np.float32)

    got = await dispatch(wgsl, a, b)
    expected = a @ b
    err = float(np.max(np.abs(got - expected)))
    ok = bool(np.allclose(got, expected, rtol=1e-4, atol=1e-4))
    if verbose:
        print(f"max_abs_err = {err}")
        print("MATCH: GPU result == A @ B" if ok else "MISMATCH")
    return ok
