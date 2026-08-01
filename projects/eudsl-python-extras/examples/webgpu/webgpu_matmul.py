"""Dispatch a WGSL matmul from Python, in the browser, via navigator.gpu.

This is the runtime half of the MLIR-in-the-browser GPU story: it proves Pyodide
can reach WebGPU through JS interop, upload buffers, dispatch a compute shader,
and read results back -- with no Dawn, no emdawnwebgpu, and no new C++ in the
wheel. The shader here is hand-written WGSL standing in for what an MLIR ->
SPIR-V -> Tint (or MLIR -> WGSL) pipeline would eventually emit.
"""

import numpy as np
from js import Float32Array, Object, Uint32Array, console, navigator
from pyodide.ffi import to_js


def js_obj(d):
    """Python dict -> plain JS object (WebGPU descriptors are plain objects)."""
    return to_js(d, dict_converter=Object.fromEntries)

M = N = K = 32
WG = 8

WGSL = """
struct Dims { M: u32, N: u32, K: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform>             d : Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= d.M || col >= d.N) { return; }
  var acc = 0.0;
  for (var k: u32 = 0u; k < d.K; k = k + 1u) {
    acc = acc + A[row * d.K + k] * B[k * d.N + col];
  }
  C[row * d.N + col] = acc;
}
"""

# WebGPU enum values are plain strings / ints in JS. Usage flags are a bitmask.
STORAGE = 0x80
COPY_DST = 0x8
COPY_SRC = 0x4
MAP_READ = 0x1
UNIFORM = 0x40


async def get_device():
    if not hasattr(navigator, "gpu"):
        raise RuntimeError("navigator.gpu missing -- WebGPU unavailable in this browser")
    adapter = await navigator.gpu.requestAdapter()
    if adapter is None:
        raise RuntimeError("requestAdapter() returned null -- no WebGPU adapter")
    return await adapter.requestDevice()


def make_buffer(device, data, usage):
    """Create a mapped-at-creation buffer and fill it from a numpy array."""
    nbytes = data.nbytes
    buf = device.createBuffer(js_obj({"size": nbytes, "usage": usage, "mappedAtCreation": True}))
    ctor = Uint32Array if data.dtype == np.uint32 else Float32Array
    ctor.new(buf.getMappedRange()).set(to_js(data.tolist()))
    buf.unmap()
    return buf


async def main():
    device = await get_device()
    console.log("adapter/device acquired")

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)

    a_buf = make_buffer(device, A.ravel(), STORAGE | COPY_DST)
    b_buf = make_buffer(device, B.ravel(), STORAGE | COPY_DST)
    dims = np.array([M, N, K, 0], dtype=np.uint32)
    d_buf = make_buffer(device, dims, UNIFORM | COPY_DST)

    out_bytes = M * N * 4
    c_buf = device.createBuffer(js_obj({"size": out_bytes, "usage": STORAGE | COPY_SRC}))
    read_buf = device.createBuffer(js_obj({"size": out_bytes, "usage": COPY_DST | MAP_READ}))

    module = device.createShaderModule(js_obj({"code": WGSL}))
    pipeline = device.createComputePipeline(
        js_obj({"layout": "auto", "compute": js_obj({"module": module, "entryPoint": "main"})})
    )

    entries = [
        js_obj({"binding": i, "resource": js_obj({"buffer": b})})
        for i, b in enumerate((a_buf, b_buf, c_buf, d_buf))
    ]
    bind_group = device.createBindGroup(
        js_obj({"layout": pipeline.getBindGroupLayout(0), "entries": to_js(entries)})
    )

    encoder = device.createCommandEncoder()
    pass_ = encoder.beginComputePass()
    pass_.setPipeline(pipeline)
    pass_.setBindGroup(0, bind_group)
    pass_.dispatchWorkgroups((M + WG - 1) // WG, (N + WG - 1) // WG, 1)
    pass_.end()
    encoder.copyBufferToBuffer(c_buf, 0, read_buf, 0, out_bytes)
    device.queue.submit(to_js([encoder.finish()]))

    await read_buf.mapAsync(MAP_READ)
    got = np.asarray(Float32Array.new(read_buf.getMappedRange()).to_py(), dtype=np.float32)
    got = got.reshape(M, N)
    read_buf.unmap()

    expected = A @ B
    max_err = float(np.max(np.abs(got - expected)))
    ok = bool(np.allclose(got, expected, rtol=1e-4, atol=1e-4))
    return ok, max_err, got, expected
