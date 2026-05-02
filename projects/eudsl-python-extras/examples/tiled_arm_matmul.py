# NB: this only works on aarch64/arm64 which supports SME

import mlir.extras.types as T
import numpy as np
from mlir.dialects import builtin
from mlir.dialects.transform import any_op_t
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.dialects.transform.structured import MatchInterfaceEnum, VectorizeOp
from mlir.dialects.transform.vector import (
    VectorContractLowering,
)
from mlir.ir import StringAttr, UnitAttr, Attribute

# you need this to register the memref value caster
# noinspection PyUnresolvedReferences
import mlir.extras.dialects.memref
from mlir.extras.context import RAIIMLIRContext, ExplicitlyManagedModule
from mlir.extras.dialects import linalg
from mlir.extras.dialects import transform, llvm
from mlir.extras.dialects.func import func
from mlir.extras.dialects.transform import (
    match,
    get_parent_op,
)
from mlir.extras.runtime.passes import Pipeline, run_pipeline
from mlir.extras.runtime.refbackend import LLVMJITBackend
from mlir.extras.util import find_ops

ctx = RAIIMLIRContext()
backend = LLVMJITBackend()
module = ExplicitlyManagedModule()

M, K, N = 7, 13, 7


@func
def matmul_armsme(
    A: T.tensor(M, K, T.f32()),
    B: T.tensor(K, N, T.f32()),
    C: T.tensor(M, N, T.f32()),
):
    return linalg.matmul(A, B, C)


@builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
def payload():
    matmul_armsme.emit(force=True)


# based on https://github.com/llvm/llvm-project/blob/ad656d3a1954dd6157ba689b3003b6fbb97a0833/mlir/test/Integration/Dialect/Linalg/CPU/ArmSME/matmul.mlir
@builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
def mod_transform():
    @named_sequence("main", [any_op_t()], [])
    def main(module_op: any_op_t()):
        # Step 1: Match the linalg.matmul operation
        matmul_op = match(module_op, ops=["linalg.matmul"])

        # Step 2: Tile for size [4] x [4], which corresponds to SVLs x SVLs
        tiled_linalg_op, loops = transform.tile_to_scf_for(
            matmul_op, sizes=[[4], [4], 1]
        )

        # Step 3: Vectorize
        VectorizeOp(tiled_linalg_op, vector_sizes=[[4], [4], 1])

        # Step 4: Bufferize ahead of TransferReadDropUnitDimsPattern
        bufferize = transform.bufferization.one_shot_bufferize(
            module_op, bufferize_function_boundaries=True
        )

        # Step 5: Match func.func operations
        func_op = match(bufferize, ops=["func.func"])

        # Step 6: Lower vector.multi_reduction to vector.contract (+ some helpful patterns)
        @apply_patterns(func_op)
        def patterns1():
            transform.apply_patterns.vector.lower_masked_transfers()
            transform.apply_patterns.vector.transfer_permutation_patterns()
            transform.apply_patterns.vector.reduction_to_contract()
            transform.apply_patterns.vector.sink_ops()

        # Step 7: Lower vector.contract to vector.outerproduct
        @apply_patterns(func_op)
        def patterns2():
            transform.apply_patterns.vector.lower_contraction(
                lowering_strategy=VectorContractLowering.OuterProduct
            )
            transform.apply_patterns.vector.lower_masks()
            transform.apply_patterns.vector.rank_reducing_subview_patterns()
            transform.apply_patterns.canonicalization()

        # # Step 8 (optional optimization): Hoist accumulator load/store
        func_h = transform.structured.hoist_redundant_vector_transfers(
            any_op_t(), func_op
        )

        all_loops = match(bufferize, interface=MatchInterfaceEnum.LoopLikeInterface)

        transform.apply_licm(all_loops)
        transform.loop.hoist_loop_invariant_subsets(all_loops)


module = module.finish()

vectorized_module = run_pipeline(
    module,
    pipeline=Pipeline()
    .transform_interpreter(entry_point="main", debug_payload_root_tag="payload")
    .canonicalize()
    .cse(),
)

# print(vectorized_module)

kernel_funcs = find_ops(
    vectorized_module.operation, lambda o: isinstance(o.opview, llvm.LLVMFuncOp)
)
for k in kernel_funcs:
    k.attributes["target_features"] = Attribute.parse(
        '#llvm.target_features<["+sme", "+sve"]>'
    )


lower_to_llvm = (
    Pipeline()
    # https://github.com/llvm/llvm-project/blob/9146ef5df0543f08a86686cfeb3bd1ea7338f4c6/mlir/test/lib/Dialect/ArmSME/TestLowerToArmSME.cpp#L45
    # Legalize vector operations so they can be converted to ArmSME.
    .arm_sme_vector_legalization()
    # Sprinkle some cleanups.
    .canonicalize()
    .cse()
    # Passes that convert operations on vectors to ArmSME operations.
    # Convert Arith to ArmSME.
    .convert_arith_to_arm_sme()
    # Convert Vector to ArmSME.
    .convert_vector_to_arm_sme()
    # Convert operations on high-level vectors to loops.
    # Convert ArmSME to SCF.
    .convert_arm_sme_to_scf()
    # Convert Vector to SCF (with full unroll enabled).
    .convert_vector_to_scf(full_unroll=True)
    # Enable streaming-mode and ZA.
    .Func(
        Pipeline().enable_arm_streaming(
            streaming_mode="streaming-locally",
            za_mode="new-za",
            if_required_by_ops=True,
        )
    )
    # Convert SCF to CF (required for ArmSME tile allocation).
    .convert_scf_to_cf()
    # Convert ArmSME to LLVM.
    .Func(Pipeline().convert_arm_sme_to_llvm())
    # Sprinkle some cleanups.
    .canonicalize()
    .cse()
    # https://github.com/makslevental/llvm-project/blob/f6643263631bcb0d191ef923963ac1a5ca9ac5fd/mlir/test/lib/Dialect/LLVM/TestLowerToLLVM.cpp#L44
    .Func(
        Pipeline()
        # Blanket-convert any remaining high-level vector ops to loops if any remain.
        .convert_vector_to_scf()
        # Blanket-convert any remaining linalg ops to loops if any remain.
        .convert_linalg_to_loops()
    )
    # Blanket-convert any remaining affine ops if any remain.
    .lower_affine()
    # Convert SCF to CF (always needed).
    .convert_scf_to_cf()
    # Sprinkle some cleanups.
    .canonicalize()
    .cse()
    # Convert vector to LLVM (always needed).
    .convert_vector_to_llvm()
    # Convert Math to LLVM (always needed).
    .Func(Pipeline().convert_math_to_llvm())
    # Expand complicated MemRef operations before lowering them.
    .expand_strided_metadata()
    # The expansion may create affine expressions. Get rid of them.
    .lower_affine()
    # Convert MemRef to LLVM (always needed).
    .finalize_memref_to_llvm()
    # Convert Func to LLVM (always needed).
    .convert_func_to_llvm()
    .convert_arith_to_llvm()
    .convert_cf_to_llvm()
    # Convert Index to LLVM (always needed).
    .convert_index_to_llvm()
    # Convert UB to LLVM (always needed).
    .convert_ub_to_llvm()
    # Convert remaining unrealized_casts (always needed).
    .reconcile_unrealized_casts()
)

compiled_module = backend.compile(
    find_ops(
        vectorized_module.operation,
        lambda x: "transform.target_tag" in x.attributes
        and x.attributes["transform.target_tag"].value == "payload",
        single=True,
    ),
    kernel_name=matmul_armsme.__name__,
    pipeline=lower_to_llvm,
)

# print(compiled_module)

A = np.random.randint(0, 10, (M, K)).astype(np.float32)
B = np.random.randint(0, 10, (K, N)).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

backend.load(compiled_module).matmul_armsme_capi_wrapper(A, B, C)
assert np.allclose(A @ B, C)
