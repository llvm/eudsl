"""
Matmul kernel via MLIR -> LLVM IR -> DXIL -> MetalIR.

Uses only the upstream MLIR python bindings (no extras framework). The kernel
is a simple per-thread matmul: one thread computes one output element, with
the inner reduction on K.
"""

import struct
from pathlib import Path

import numpy as np

try:
    import Metal
    import Foundation
except ImportError:
    print("Metal / Foundation (pyobjc) not available; skipping.")
    raise SystemExit(0)

from mlir.ir import (
    Context,
    F32Type,
    FloatAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MemRefType,
    Module,
    VectorType,
)
from mlir.dialects import arith, func, memref, scf, vector
from mlir.passmanager import PassManager

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


KERNEL_NAME = "matmul_f32"
M, N, K = 32, 32, 32
THREADS_PER_GROUP = (8, 8, 1)
DEVICE = 1  # address space for GPU-visible buffers


def build_kernel_module():
    ctx = Context()
    with ctx, Location.unknown():
        i32 = IntegerType.get_signless(32)
        i64 = IntegerType.get_signless(64)
        f32 = F32Type.get()
        idx_ty = IndexType.get()
        addr_dev = IntegerAttr.get(i64, DEVICE)

        A_ty = MemRefType.get([M, K], f32, memory_space=addr_dev)
        B_ty = MemRefType.get([K, N], f32, memory_space=addr_dev)
        C_ty = MemRefType.get([M, N], f32, memory_space=addr_dev)
        gid_ty = VectorType.get([3], i32)
        acc_ty = MemRefType.get([1], f32)

        module = Module.create()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(A_ty, B_ty, C_ty, gid_ty, name=KERNEL_NAME)
            def matmul(A, B, C, gid):
                c0 = arith.constant(idx_ty, IntegerAttr.get(idx_ty, 0))
                c1 = arith.constant(idx_ty, IntegerAttr.get(idx_ty, 1))
                cK = arith.constant(idx_ty, IntegerAttr.get(idx_ty, K))

                zero_f = arith.constant(f32, FloatAttr.get(f32, 0.0))
                acc = memref.alloca(acc_ty, [], [])
                memref.store(zero_f, acc, [c0])

                col_i32 = vector.extract(gid, [], [0])
                row_i32 = vector.extract(gid, [], [1])
                col = arith.index_cast(idx_ty, col_i32)
                row = arith.index_cast(idx_ty, row_i32)

                for k in scf.for_(c0, cK, c1):
                    a = memref.load(A, [row, k])
                    b = memref.load(B, [k, col])
                    old = memref.load(acc, [c0])
                    prod = arith.mulf(a, b)
                    new = arith.addf(old, prod)
                    memref.store(new, acc, [c0])
                    scf.yield_([])

                final = memref.load(acc, [c0])
                memref.store(final, C, [row, col])

        module.operation.verify()
        return ctx, module


def lower_to_llvm_dialect(module):
    pm = PassManager.parse(
        "builtin.module("
        "finalize-memref-to-llvm{index-bitwidth=32},"
        "convert-scf-to-cf,"
        "convert-func-to-llvm{index-bitwidth=32 use-bare-ptr-memref-call-conv=1},"
        "convert-arith-to-llvm{index-bitwidth=32},"
        "convert-index-to-llvm{index-bitwidth=32},"
        "convert-cf-to-llvm{index-bitwidth=32},"
        "convert-vector-to-llvm,"
        "reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


# ── Build and lower ───────────────────────────────────────────────────────────

ctx, module = build_kernel_module()
with ctx:
    print(module)
    lowered = lower_to_llvm_dialect(module)
    print(lowered)

    # ── MLIR -> LLVM IR -> rewrite to DXIL-shaped IR -> DXIL bytes ───────────

    ll_ctx = LLVMContext()
    llvm_module = translate_mlir_to_llvm(lowered.operation, ll_ctx)

add_dxil_module_metadata(llvm_module, sm_major=6, sm_minor=0)
kernel_fn = next(
    fn for fn in llvm_module.get_functions() if fn.name.startswith(KERNEL_NAME)
)
mark_as_dxil_compute_kernel(kernel_fn, *THREADS_PER_GROUP)

# Rewrite MLIR-shaped LLVM IR (bare-ptr memref loads/stores, vector<3xi32>
# gid) into DXIL-shaped IR (dx.RawBuffer handles + rawbuffer load/store,
# dx.thread.id) that the DirectX backend can codegen.
lower_mlir_to_dxil(llvm_module)
print(llvm_module)

dxil_bytes = translate_llvm_to_dxil(llvm_module)
print(f"DXIL: {len(dxil_bytes)} bytes")

out_dir = Path(__file__).parent
(out_dir / f"{KERNEL_NAME}.dxil").write_bytes(dxil_bytes)

# ── DXIL -> metallib + resource reflection ────────────────────────────────────

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

# Build the top-level Argument Buffer the compiled kernel reads resources
# from. Each entry is 24 bytes:
#   uint64 gpuAddress
#   uint64 resourceID   (0 for buffers)
#   uint64 flags        (low 32 bits = buffer.length)
# The offset of each slot comes from the reflection returned above.
BIND_POINT = 2  # kIRArgumentBufferBindPoint from the runtime header
slot_to_buf = {0: buf_A, 1: buf_B, 2: buf_C}
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
