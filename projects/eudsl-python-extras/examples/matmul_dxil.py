"""
Matmul kernel lowered via MLIR -> LLVM -> DXIL -> MetalIR -> Metal dispatch.

The MLIR kernel is written with eudsl-python-extras. After lowering to LLVM IR
the module is rewritten into DXIL-shaped IR by ``lower_mlir_to_dxil``, the
DirectX backend emits a DXContainer, and ``libmetalirconverter`` produces a
metallib the Metal device can execute.
"""

import struct
from pathlib import Path

import numpy as np

try:
    import Metal
    import Foundation
except ImportError:
    print("Metal / Foundation (pyobjc) not available; skipping.")
    raise exit(0)

from mlir.dxil import (
    IRShaderStage,
    LLVMContext,
    add_dxil_module_metadata,
    lower_mlir_to_dxil,
    mark_as_dxil_compute_kernel,
    translate_dxil_to_metallib,
    translate_llvm_to_dxil,
    translate_mlir_to_llvm,
)
from mlir.extras import types as T
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.dialects import arith, func, memref, scf
from mlir.extras.runtime.passes import Pipeline, run_pipeline
from mlir.ir import (
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MemRefType,
    Module,
    StridedLayoutAttr,
    VectorType,
)


DEVICE = 1  # device (global) memory address space
CONSTANT = 2  # constant memory address space

KERNEL_NAME = "matmul_f32"
M, N, K = 32, 32, 32
THREADS_PER_GROUP = (8, 8, 1)


def lowerToLLVM(module):
    return run_pipeline(
        module,
        Pipeline()
        .finalize_memref_to_llvm(index_bitwidth=32)
        .convert_scf_to_cf()
        .convert_func_to_llvm(index_bitwidth=32, use_bare_ptr_memref_call_conv=True)
        .convert_arith_to_llvm(index_bitwidth=32)
        .convert_index_to_llvm(index_bitwidth=32)
        .convert_cf_to_llvm(index_bitwidth=32)
        .convert_vector_to_llvm()
        .reconcile_unrealized_casts(),
    )


@func.func
@canonicalize(using=[scf.canonicalizer])
def matmul_f32(
    # fmt: off
    A: MemRefType[[0], F32Type, StridedLayoutAttr[0, [1]], IntegerAttr[IntegerType[64], DEVICE]],
    B: MemRefType[[0], F32Type, StridedLayoutAttr[0, [1]], IntegerAttr[IntegerType[64], DEVICE]],
    C: MemRefType[[0], F32Type, StridedLayoutAttr[0, [1]], IntegerAttr[IntegerType[64], DEVICE]],
    dims: MemRefType[[3], IndexType, StridedLayoutAttr[0, [1]], IntegerAttr[IntegerType[64], CONSTANT]],
    # fmt: on
    gid: VectorType[[3], IntegerType[32]],
):
    """C[m, n] = sum_k A[m, k] * B[k, n], with one thread per output element.

    gid = (col, row, 0), so gid.x iterates N and gid.y iterates M.
    """
    m_dim = dims[0]
    n_dim = dims[1]
    k_dim = dims[2]
    m_i32 = arith.index_cast(m_dim, T.i32())
    n_i32 = arith.index_cast(n_dim, T.i32())

    col = gid[0]
    row = gid[1]

    # Reinterpret the bare-ptr memrefs as 1D dynamic buffers.
    mk = arith.index_cast(m_i32 * arith.index_cast(k_dim, T.i32()), T.index())
    kn = arith.index_cast(arith.index_cast(k_dim, T.i32()) * n_i32, T.index())
    mn = arith.index_cast(m_i32 * n_i32, T.index())
    A = memref.reinterpret_cast(A, offsets=[0], sizes=[mk], strides=[1])
    B = memref.reinterpret_cast(B, offsets=[0], sizes=[kn], strides=[1])
    C = memref.reinterpret_cast(C, offsets=[0], sizes=[mn], strides=[1])

    zero = arith.constant(0.0, T.f32())
    acc = memref.alloca([1], T.f32())
    acc[0] = zero

    for k in scf.range_(0, arith.index_cast(k_dim, T.i32()), 1):
        k_idx = arith.index_cast(k, T.index())
        row_idx = arith.index_cast(row, T.index())
        col_idx = arith.index_cast(col, T.index())

        a_off = row_idx * k_dim + k_idx
        b_off = k_idx * arith.index_cast(n_i32, T.index()) + col_idx

        acc[0] = acc[0] + A[a_off] * B[b_off]

    c_off = arith.index_cast(row, T.index()) * arith.index_cast(
        n_i32, T.index()
    ) + arith.index_cast(col, T.index())
    C[c_off] = acc[0]


# ── Build MLIR module ─────────────────────────────────────────────────────────

with Context() as ctx, Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
        matmul_f32.emit()

    module.operation.verify()
    lowered = lowerToLLVM(module)
    print(lowered)

    ll_ctx = LLVMContext()
    llvm_module = translate_mlir_to_llvm(lowered.operation, ll_ctx)

# ── Attach DXIL container metadata + rewrite to DXIL-shaped IR ───────────────

add_dxil_module_metadata(llvm_module, sm_major=6, sm_minor=0)
kernel_fn = next(
    fn for fn in llvm_module.get_functions() if fn.name.startswith(KERNEL_NAME)
)
mark_as_dxil_compute_kernel(kernel_fn, *THREADS_PER_GROUP)
# Rewrite bare-ptr memref loads/stores + vector<3xi32> gid into dx.RawBuffer
# handles + rawbuffer load/store + dx.thread.id, which is what the DirectX
# backend actually consumes.
lower_mlir_to_dxil(llvm_module)
print(llvm_module)

# ── LLVM -> DXIL -> metallib ──────────────────────────────────────────────────

dxil_bytes = translate_llvm_to_dxil(llvm_module)
print(f"DXIL: {len(dxil_bytes)} bytes")

out_dir = Path(__file__).parent
(out_dir / f"{KERNEL_NAME}.dxil").write_bytes(dxil_bytes)

metallib_bytes, reflection = translate_dxil_to_metallib(
    dxil_bytes, IRShaderStage.Compute, KERNEL_NAME
)
print(f"metallib: {len(metallib_bytes)} bytes")
print("top-level argument buffer layout:")
for r in reflection:
    print(
        f"  type={r.resource_type} space={r.space} slot={r.slot} "
        f"offset={r.top_level_offset} size={r.size_bytes} name={r.name!r}"
    )

metallib_fp = out_dir / f"{KERNEL_NAME}.metallib"
metallib_fp.write_bytes(metallib_bytes)

# ── Dispatch ──────────────────────────────────────────────────────────────────

rng = np.random.default_rng(0)
A_host = rng.random((M, K), dtype=np.float32)
B_host = rng.random((K, N), dtype=np.float32)

device = Metal.MTLCreateSystemDefaultDevice()
assert device is not None, "no Metal device available"

url = Foundation.NSURL.fileURLWithPath_(str(metallib_fp))
library, err = device.newLibraryWithURL_error_(url, None)
assert err is None, f"failed to load metallib: {err}"

fn = library.newFunctionWithName_(KERNEL_NAME)
assert fn is not None, f"kernel '{KERNEL_NAME}' not found in library"

pipeline, err = device.newComputePipelineStateWithFunction_error_(fn, None)
assert err is None, f"compute pipeline creation failed: {err}"


def make_buffer_from_numpy(arr):
    arr = np.ascontiguousarray(arr)
    buf = device.newBufferWithBytes_length_options_(
        arr, arr.nbytes, Metal.MTLResourceStorageModeShared
    )
    assert buf is not None
    return buf


def make_buffer(nbytes):
    buf = device.newBufferWithLength_options_(
        nbytes, Metal.MTLResourceStorageModeShared
    )
    assert buf is not None
    return buf


buf_A = make_buffer_from_numpy(A_host)
buf_B = make_buffer_from_numpy(B_host)
buf_C = make_buffer(M * N * np.dtype(np.float32).itemsize)
buf_dims = make_buffer_from_numpy(np.array([M, N, K], dtype=np.int32))

# Top-level Argument Buffer the compiled kernel reads resources from. Each
# entry is 24 bytes: gpuAddress, resourceID (0 for buffers), flags (buffer
# length in the low 32 bits). Offsets come from the reflection above.
BIND_POINT = 2  # kIRArgumentBufferBindPoint
slot_to_buf = {0: buf_A, 1: buf_B, 2: buf_C, 3: buf_dims}
table_size = max(r.top_level_offset + r.size_bytes for r in reflection)
top_level = bytearray(table_size)
for r in reflection:
    b = slot_to_buf[r.slot]
    struct.pack_into(
        "<QQQ", top_level, r.top_level_offset, b.gpuAddress(), 0, b.length()
    )

queue = device.newCommandQueue()
cmd_buf = queue.commandBuffer()
enc = cmd_buf.computeCommandEncoder()

enc.setComputePipelineState_(pipeline)
enc.setBytes_length_atIndex_(bytes(top_level), len(top_level), BIND_POINT)
for b in slot_to_buf.values():
    enc.useResource_usage_(b, Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)

enc.dispatchThreads_threadsPerThreadgroup_(
    Metal.MTLSizeMake(N, M, 1),
    Metal.MTLSizeMake(*THREADS_PER_GROUP),
)
enc.endEncoding()

cmd_buf.commit()
cmd_buf.waitUntilCompleted()

# ── Verify ────────────────────────────────────────────────────────────────────

C_host = np.frombuffer(
    buf_C.contents().as_buffer(buf_C.length()), dtype=np.float32
).reshape(M, N)
expected = A_host @ B_host
np.testing.assert_allclose(C_host, expected, rtol=1e-4, atol=1e-4)
print("PASS")
