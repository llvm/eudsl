import contextlib
import math

import mlir.extras.types as T
import numpy as np
from mlir.dialects import builtin

from util import cuda_bindings_not_installed
from mlir.extras.ast.canonicalize import canonicalize
from mlir.extras.context import (
    mlir_mod_ctx,
    MLIRContext,
)
from mlir.extras.dialects import arith, memref, gpu, scf, linalg, vector, nvgpu
from mlir.extras.dialects.gpu import (
    block_idx,
    thread_idx,
    block_dim,
    get_compile_object_bytes,
    smem_space,
)
from mlir.extras.dialects.llvm import llvm_ptr_t
from mlir.extras.dialects.memref import S
from mlir.extras.dialects.scf import range_
from mlir.extras.runtime.passes import Pipeline, run_pipeline

# noinspection PyUnresolvedReferences
from mlir.extras.util import find_ops, enable_debug as enable_debug

# just so it doesn't get DCE'd by black/reformat
_ = memref


def build_cuda_func(compiled_module, kernel_name="naive"):
    from cupy.cuda import Module

    ptx = get_compile_object_bytes(compiled_module)
    mod = Module()
    mod.load(ptx)
    return mod.get_function(kernel_name)


def print_ptx(compiled_module):
    ptx = get_compile_object_bytes(compiled_module)
    print(ptx.decode())


def compile_module(
    module,
    chip="sm_80",
    features="+ptx83",
    opt_level=2,
    enable_ir_printing=False,
    print_ptx_=False,
    full_pipeline=True,
):
    if enable_ir_printing:
        print_ptx_ = True
    if full_pipeline:
        p = (
            Pipeline()
            .convert_linalg_to_loops()
            .convert_nvgpu_to_nvvm()
            .gpu_kernel_outlining()
            .convert_vector_to_scf()
            .convert_scf_to_cf()
            .convert_nvvm_to_llvm()
            .convert_func_to_llvm()
            .expand_strided_metadata()
            .add_pass(
                "nvvm-attach-target",
                **{
                    "chip": chip,
                    "features": features,
                    "O": str(opt_level),
                },
            )
            .lower_affine()
            .convert_arith_to_llvm()
            .convert_index_to_llvm()
            .canonicalize()
            .cse()
            .Gpu(
                Pipeline()
                .strip_debuginfo()
                # TODO(max): upstream this (add to gpu pipeline)
                # vector.transfer
                .convert_vector_to_llvm()
                .convert_gpu_to_nvvm(use_bare_ptr_memref_call_conv=True)
                .canonicalize()
                .cse()
                .reconcile_unrealized_casts()
            )
            .gpu_to_llvm(use_bare_pointers_for_kernels=True)
            .gpu_module_to_binary(format="isa")
            .canonicalize()
            .cse()
            .reconcile_unrealized_casts()
        )
    else:
        p = Pipeline().add_pass(
            "gpu-lower-to-nvvm-pipeline",
            # https://github.com/llvm/llvm-project/blob/ace69e6b942b8fa7e610d70be2a92e801ceea481/mlir/include/mlir/Dialect/GPU/Pipelines/Passes.h#L18
            **{
                "cubin-chip": chip,
                "cubin-features": features,
                "cubin-format": "isa",
                "kernel-bare-ptr-calling-convention": "1",
                "opt-level": str(opt_level),
                # "cubin-format": "fatbin",
                # "cubin-format": "bin",
            },
        )
    mod = run_pipeline(module, p, enable_ir_printing=enable_ir_printing)

    if print_ptx_:
        print_ptx(mod)

    return mod


@contextlib.contextmanager
def time_cuda():
    import cupy as cp

    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    start_gpu.record()
    yield start_gpu, end_gpu
    end_gpu.record()
    end_gpu.synchronize()


@gpu.func
@canonicalize(using=(arith.canonicalizer, scf.canonicalizer))
def sgemm_shared_mem_2d_block_tiling[
    # fmt: off
    M,
    K,
    N,
    dtype,
    BM,
    BN,
    BK,
    TM,
    TN,
    A_t = T.memref(M, K, dtype),
    B_t = T.memref(K, N, dtype),
    C_t = T.memref(M, N, dtype),
    # fmt: on
](
    A: A_t, B: B_t, C: C_t
):
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BM, BK), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    total_results_blocktile = BM * BN
    num_threads_blocktile = total_results_blocktile // (TM * TN)

    tid = gpu.thread_id()
    # BN/TN are the number of threads to span a column
    thread_col = tid % (BN // TN)
    thread_row = tid / (BN // TN)

    inner_col_A = tid % BK  # warp-level GMEM coalescing
    inner_row_A = tid / BK
    stride_A = num_threads_blocktile // BK

    inner_col_B = tid % BN  # warp-level GMEM coalescing
    inner_row_B = tid / BN
    stride_B = num_threads_blocktile // BN

    thread_results = memref.alloca((TM, TN), dtype)
    linalg.fill(0.0, thread_results)

    reg_M = memref.alloca((TM,), dtype)
    linalg.fill(0.0, reg_M)

    reg_N = memref.alloca((TN,), dtype)
    linalg.fill(0.0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_ = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_ = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        for load_offset in range_(0, BM, stride_A):
            A_shared[inner_row_A + load_offset, inner_col_A] = A_[
                inner_row_A + load_offset, inner_col_A
            ]
        for load_offset in range_(0, BK, stride_B):
            B_shared[inner_row_B + load_offset, inner_col_B] = B_[
                inner_row_B + load_offset, inner_col_B
            ]

        gpu.barrier()

        for dot_idx in range_(BK):
            for i in range_(TM):
                reg_M[i] = A_shared[thread_row * TM + i, dot_idx]
            for i in range_(TN):
                reg_N[i] = B_shared[dot_idx, thread_col * TN + i]

            for res_idx_m in range_(TM):
                for res_idx_n in range_(TN):
                    thread_results[res_idx_m, res_idx_n] += (
                        reg_M[res_idx_m] * reg_N[res_idx_n]
                    )

        gpu.barrier()

    one = arith.constant(1.0, type=dtype)
    C_ = C[c_row : c_row + BM, c_col : c_col + BN]

    for res_idx_m in range_(TM):
        for res_idx_n in range_(TN):
            C_[thread_row * TM + res_idx_m, thread_col * TN + res_idx_n] = (
                thread_results[res_idx_m, res_idx_n] + one
            )


def tile_iterator(A, B, C, BM, BK, BN, dtype, TM, TN, K):
    base = gpu.dynamic_shared_memory()
    A_shared = memref.view(base, (BM, BK), dtype=dtype)
    B_shared = memref.view(base, (BK, BN), dtype=dtype, shift=BM * BK)

    c_row = block_idx.y * BM
    c_col = block_idx.x * BN

    total_results_blocktile = BM * BN
    num_threads_blocktile = total_results_blocktile // (TM * TN)

    tid = gpu.thread_id()
    # BN/TN are the number of threads to span a column
    thread_col = tid % (BN // TN)
    thread_row = tid / (BN // TN)

    inner_col_A = tid % BK  # warp-level GMEM coalescing
    inner_row_A = tid / BK
    stride_A = num_threads_blocktile // BK

    inner_col_B = tid % BN  # warp-level GMEM coalescing
    inner_row_B = tid / BN
    stride_B = num_threads_blocktile // BN

    thread_results = memref.alloca((TM, TN), dtype)
    linalg.fill(0.0, thread_results)

    reg_M = memref.alloca((TM,), dtype)
    linalg.fill(0.0, reg_M)

    reg_N = memref.alloca((TN,), dtype)
    linalg.fill(0.0, reg_N)

    for bk_idx in range_(0, K, BK):
        A_local = A[c_row : c_row + BM, bk_idx : bk_idx + BK]
        B_local = B[bk_idx : bk_idx + BK, c_col : c_col + BN]

        for load_offset in range_(0, BM, stride_A):
            # fmt: off
            A_shared[inner_row_A + load_offset, inner_col_A] = \
                A_local[inner_row_A + load_offset, inner_col_A]
            # fmt: on
            scf.yield_()

        for load_offset in range_(0, BK, stride_B):
            # fmt: off
            B_shared[inner_row_B + load_offset, inner_col_B] = \
                B_local[inner_row_B + load_offset, inner_col_B]
            # fmt: on
            scf.yield_()

        gpu.barrier()

        for dot_idx in range_(BK):
            for i in range_(TM):
                reg_M[i] = A_shared[thread_row * TM + i, dot_idx]
                scf.yield_()

            for i in range_(TN):
                reg_N[i] = B_shared[dot_idx, thread_col * TN + i]
                scf.yield_()

            yield reg_M, reg_N, thread_results

            scf.yield_()

        gpu.barrier()

        scf.yield_()

    one = arith.constant(1.0, type=dtype)
    C_local = C[c_row : c_row + BM, c_col : c_col + BN]

    for res_idx_m in range_(TM):
        for res_idx_n in range_(TN):
            C_local[thread_row * TM + res_idx_m, thread_col * TN + res_idx_n] = (
                thread_results[res_idx_m, res_idx_n] + one
            )
            scf.yield_()
        scf.yield_()


@gpu.func
def sgemm_shared_mem_2d_block_tiling_iterator[
    # fmt: off
    M,
    K,
    N,
    dtype,
    BM,
    BN,
    BK,
    TM,
    TN,
    A_t = T.memref(M, K, dtype),
    B_t = T.memref(K, N, dtype),
    C_t = T.memref(M, N, dtype),
    # fmt: on
](
    A: A_t, B: B_t, C: C_t
):
    for reg_M, reg_N, thread_results in tile_iterator(
        A, B, C, BM, BK, BN, dtype, TM, TN, K
    ):
        for res_idx_m in range_(TM):
            for res_idx_n in range_(TN):
                thread_results[res_idx_m, res_idx_n] += (
                    reg_M[res_idx_m] * reg_N[res_idx_n]
                )
                scf.yield_()
            scf.yield_()
        # TODO(max): thread_results[:, :] += reg_M[:] @ reg_N[:]


class CUDABindingsNotInstalled(Exception):
    pass


def prepare_non_tiled_kernel(ctx: MLIRContext, kernel, M, K, N, BLOCK_SIZE=32):
    dtype = T.f32()
    npy_dtype = np.float32

    gpu.set_container_module(ctx.module)

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype].emit()

    assert ctx.module.operation.verify()

    p = Pipeline().cse()
    matmul_mod = run_pipeline(matmul_mod, p)

    print(matmul_mod)

    if cuda_bindings_not_installed():
        raise CUDABindingsNotInstalled()

    kernel_name = kernel.__name__
    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(M / BLOCK_SIZE), math.ceil(N / BLOCK_SIZE))
    block_dims = (BLOCK_SIZE, BLOCK_SIZE)

    if "shared" in kernel_name:
        shared_mem = 2 * BLOCK_SIZE * BLOCK_SIZE * npy_dtype().nbytes
    else:
        shared_mem = 0

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        "transpose_B" in kernel_name,
    )


def prepare_tiled_kernel(ctx: MLIRContext, kernel, M, K, N):
    dtype = T.f32()
    npy_dtype = np.float32
    kernel_name = kernel.__name__

    gpu.set_container_module(ctx.module)

    BK = 8
    TM = 8
    TN = 8
    if "2d" in kernel_name and M >= 128 and N >= 128:
        BM = 128
        BN = 128
    else:
        BM = 64
        BN = 64

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype, BM, BN, BK, TM, TN].emit()

    assert ctx.module.operation.verify()

    p = Pipeline().cse()
    matmul_mod = run_pipeline(matmul_mod, p)
    print(matmul_mod)

    if cuda_bindings_not_installed():
        raise CUDABindingsNotInstalled()

    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    if "2d" in kernel_name:
        block_dims = (BM // TM, BN // TN)
    else:
        block_dims = (BM // TM, BN)

    if "shared" in kernel_name:
        shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes
    else:
        shared_mem = 0

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def prepare_warp_tiled_kernel(ctx: MLIRContext, kernel, M, K, N):
    dtype = T.f32()
    npy_dtype = np.float32
    kernel_name = kernel.__name__

    gpu.set_container_module(ctx.module)

    # Settings for A100 (looks like it works for 3070 too?)
    NUM_THREADS = 128
    BN = 128
    BM = 64
    BK = 16
    WN = 64
    WM = 32
    WNITER = 1
    TN = 4
    TM = 4

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS].emit()

    # print(ctx.module)
    assert ctx.module.operation.verify()

    if cuda_bindings_not_installed():
        raise CUDABindingsNotInstalled()

    compiled_module = compile_module(ctx.module)
    cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    block_dims = (NUM_THREADS,)
    shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes

    return (
        cuda_func,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def prepare_tensor_core_kernel(ctx: MLIRContext, kernel, M, K, N):
    dtype = T.f16()
    npy_dtype = np.float16
    kernel_name = kernel.__name__

    gpu.set_container_module(ctx.module)

    # Settings for A100 (looks like it works for 3070 too?)
    NUM_THREADS = 128
    BN = 128
    BM = 64
    BK = 16
    WN = 64
    WM = 32
    WNITER = 1
    TN = 4
    TM = 4

    @gpu.module("matmul", ["#nvvm.target"])
    def matmul_mod():
        kernel[M, K, N, dtype].emit()

    assert ctx.module.operation.verify()

    if cuda_bindings_not_installed():
        raise CUDABindingsNotInstalled()

    compiled_module = compile_module(
        ctx.module, chip="sm_90a", opt_level=3, full_pipeline=False
    )
    # cuda_func = build_cuda_func(compiled_module, kernel_name)
    # print_ptx(compiled_module)

    grid_dims = (math.ceil(N / BN), math.ceil(M / BM))
    block_dims = (NUM_THREADS,)
    shared_mem = ((BM * BK) + (BK * BN)) * npy_dtype().nbytes

    return (
        # cuda_func,
        None,
        grid_dims,
        block_dims,
        shared_mem,
        npy_dtype,
        False,
    )


def run_eval(
    M,
    K,
    N,
    cuda_func,
    grid_dims,
    block_dims,
    shared_mem,
    npy_dtype,
    transpose_B,
    repeat_times=None,
):
    import cupy as cp

    if repeat_times is None:
        repeat_times = 50

    A = np.random.randint(0, 10, (M, K)).astype(npy_dtype)
    B = np.random.randint(0, 10, (K, N)).astype(npy_dtype)
    C = np.zeros((M, N)).astype(npy_dtype)

    dA = cp.asarray(A)
    if transpose_B:
        dB = cp.asarray(np.ascontiguousarray(B.T))
    else:
        dB = cp.asarray(B)
    dC = cp.asarray(C)

    cuda_func(
        grid_dims,
        block_dims,
        (dA.data.ptr, dB.data.ptr, dC.data.ptr),
        shared_mem=shared_mem,
    )
    C = cp.asnumpy(dC)
    if not np.array_equal(C, A @ B + 1):
        print(A @ B + 1)
        print(C)
        assert False
    if repeat_times < 1:
        return

    with time_cuda() as (start_gpu, end_gpu):
        for _ in range(repeat_times):
            cuda_func(
                grid_dims,
                block_dims,
                (dA.data.ptr, dB.data.ptr, dC.data.ptr),
                shared_mem=shared_mem,
            )

    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    print(f"t={t_gpu / repeat_times:.6f} ms")


sizes = [1024]
for k in [
    sgemm_shared_mem_2d_block_tiling,
    sgemm_shared_mem_2d_block_tiling_iterator,
]:
    for s in sizes:
        with (
            mlir_mod_ctx() as ctx,
            # enable_debug()
        ):
            try:
                cuda_func, grid_dims, block_dims, shared_mem, npy_dtype, transpose_B = (
                    prepare_tiled_kernel(ctx, k, s, s, s)
                )
                run_eval(
                    s,
                    s,
                    s,
                    cuda_func,
                    grid_dims,
                    block_dims,
                    shared_mem,
                    npy_dtype,
                    transpose_B,
                )
            except CUDABindingsNotInstalled:
                continue
