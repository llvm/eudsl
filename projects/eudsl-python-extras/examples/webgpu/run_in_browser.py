"""M1c: run the MLIR -> SPIR-V pipeline inside Pyodide, against the shipped wheel.

Same passes as the native run, but driven through mlir.passmanager instead of
mlir-opt, to prove the shipped wasm wheel can do this with no new C++.
"""

from mlir.ir import Context, Module
from mlir.passmanager import PassManager

from mlir_to_spirv import (
    LOWER_PIPELINE,
    SERIALIZE_PIPELINE,
    describe,
    extract_binary,
    nest_spirv_module,
)

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


def compile_to_spirv(asm: str = MATMUL) -> bytes:
    with Context():
        module = Module.parse(asm)
        PassManager.parse(LOWER_PIPELINE).run(module.operation)
        lowered = str(module)

    # Nesting is textual, so re-parse the result.
    with Context():
        module = Module.parse(nest_spirv_module(lowered))
        PassManager.parse(SERIALIZE_PIPELINE).run(module.operation)
        return extract_binary(str(module))


def main():
    spv = compile_to_spirv()
    return describe(spv), spv
