# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

####################################################
# NOTE: This file is auto-generated using utils/generate_pass_pipeline.py.
# DO NOT add functionality here (instead add to _passes_base.py)
####################################################

from ._passes_base import *


class Pipeline(Pipeline):
    def acc_implicit_data(self, enable_implicit_reduction_copy: bool = None):
        """Generate implicit data attributes for OpenACC compute constructs

        This pass implements the OpenACC specification for "Variables with
        Implicitly Determined Data Attributes" (OpenACC 3.4 spec, section 2.6.2).

        The pass automatically generates data clause operations for variables used
        within OpenACC compute constructs (parallel, kernels, serial) that do not
        already have explicit data clauses. The semantics follow these rules:

        1. If there is a default(none) clause visible, no implicit data actions
           apply.

        2. An aggregate variable (arrays, derived types, etc.) will be treated as:
           - In a present clause when default(present) is visible.
           - In a copy clause otherwise.

        3. A scalar variable will be treated as if it appears in:
           - A copy clause if the compute construct is a kernels construct.
           - A firstprivate clause otherwise (parallel, serial).

        Args:
            enable_implicit_reduction_copy: Enable applying implicit copy in lieu of implicit firstprivate for reduction variables. This allows uniform treatment of reduction variables between combined constructs (e.g., 'parallel loop') and separate constructs (e.g., 'parallel' followed by 'loop'), where the OpenACC spec requires copy semantics for the former but firstprivate would normally apply for the latter.
        """
        self.add_pass(
            "acc-implicit-data",
            **{"enable-implicit-reduction-copy": enable_implicit_reduction_copy},
        )
        return self

    def acc_implicit_routine(self, device_type: "mlir::acc::DeviceType" = None):
        """Generate implicit acc routine for functions in acc regions

        This pass implements the implicit rules described in OpenACC specification
        for `Routine Directive` (OpenACC 3.4 spec, section 2.15.1).

        "If no explicit routine directive applies to a procedure whose definition
        appears in the program unit being compiled, then the implementation applies
        an implicit routine directive to that procedure if any of the following
        conditions holds:
        - The procedure is called or its address is accessed in a compute region."

        The specification further states:
        "When the implementation applies an implicit routine directive to a procedure,
        it must recursively apply implicit routine directives to other procedures for
        which the above rules specify relevant dependencies. Such dependencies can
        form a cycle, so the implementation must take care to avoid infinite recursion."

        This pass implements these requirements by:
        1. Walking through all OpenACC compute constructs and functions already
           marked with `acc routine` in the module and identifying function calls
           within these regions.
        2. Creating implicit `acc.routine` operations for functions that don't already
           have routine declarations.
        3. Recursively walking through all existing `acc routine` and creating
           implicit routine operations for function calls within these routines,
           while avoiding infinite recursion through proper tracking.

        Args:
            device_type: Target device type for implicit routine generation. Ensures that `acc routine` device_type clauses are properly considered not just default clauses.
        """
        self.add_pass("acc-implicit-routine", **{"device-type": device_type})
        return self

    def affine_data_copy_generate(
        self,
        fast_mem_capacity: int = None,
        fast_mem_space: int = None,
        generate_dma: bool = None,
        min_dma_transfer: int = None,
        slow_mem_space: int = None,
        skip_non_unit_stride_loops: bool = None,
        tag_mem_space: int = None,
    ):
        """Generate explicit copying for affine memory operations
        Args:
            fast_mem_capacity: Set fast memory space capacity in KiB (default: unlimited)
            fast_mem_space: Fast memory space identifier for copy generation (default: 1)
            generate_dma: Generate DMA instead of point-wise copy
            min_dma_transfer: Minimum DMA transfer size supported by the target in bytes
            slow_mem_space: Slow memory space identifier for copy generation (default: 0)
            skip_non_unit_stride_loops: Testing purposes: avoid non-unit stride loop choice depths for copy placement
            tag_mem_space: Tag memory space identifier for copy generation (default: 0)
        """
        self.add_pass(
            "affine-data-copy-generate",
            **{
                "fast-mem-capacity": fast_mem_capacity,
                "fast-mem-space": fast_mem_space,
                "generate-dma": generate_dma,
                "min-dma-transfer": min_dma_transfer,
                "slow-mem-space": slow_mem_space,
                "skip-non-unit-stride-loops": skip_non_unit_stride_loops,
                "tag-mem-space": tag_mem_space,
            },
        )
        return self

    def affine_expand_index_ops(self):
        """Lower affine operations operating on indices into more fundamental operations"""
        self.add_pass("affine-expand-index-ops")
        return self

    def affine_expand_index_ops_as_affine(self):
        """Lower affine operations operating on indices into affine.apply operations"""
        self.add_pass("affine-expand-index-ops-as-affine")
        return self

    def affine_loop_coalescing(self):
        """Coalesce nested loops with independent bounds into a single loop"""
        self.add_pass("affine-loop-coalescing")
        return self

    def affine_loop_fusion(
        self,
        compute_tolerance: float = None,
        fast_mem_space: int = None,
        local_buf_threshold: int = None,
        maximal: bool = None,
        mode: "FusionMode" = None,
    ):
        """Fuse affine loop nests

        This pass performs fusion of loop nests using a slicing-based approach. The
        transformation works on an MLIR `Block` granularity and applies to all
        blocks of the pass is run on. It combines two fusion strategies:
        producer-consumer fusion and sibling fusion. Producer-consumer fusion is
        aimed at fusing pairs of loops where the first one writes to a memref that
        the second reads. Sibling fusion targets pairs of loops that share no
        dependences between them but that load from the same memref. The fused loop
        nests, when possible, are rewritten to access significantly smaller local
        buffers instead of the original memref's, and the latter are often either
        completely optimized away or contracted. This transformation leads to
        enhanced locality and lower memory footprint through the elimination or
        contraction of temporaries/intermediate memref's. These benefits are
        sometimes achieved at the expense of redundant computation through a cost
        model that evaluates available choices such as the depth at which a source
        slice should be materialized in the designation slice.

        Example 1: Producer-consumer fusion.
        Input:
        ```mlir
        func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
          %0 = memref.alloc() : memref<10xf32>
          %1 = memref.alloc() : memref<10xf32>
          %cst = arith.constant 0.000000e+00 : f32
          affine.for %arg2 = 0 to 10 {
            affine.store %cst, %0[%arg2] : memref<10xf32>
            affine.store %cst, %1[%arg2] : memref<10xf32>
          }
          affine.for %arg2 = 0 to 10 {
            %2 = affine.load %0[%arg2] : memref<10xf32>
            %3 = arith.addf %2, %2 : f32
            affine.store %3, %arg0[%arg2] : memref<10xf32>
          }
          affine.for %arg2 = 0 to 10 {
            %2 = affine.load %1[%arg2] : memref<10xf32>
            %3 = arith.mulf %2, %2 : f32
            affine.store %3, %arg1[%arg2] : memref<10xf32>
          }
          return
        }
        ```
        Output:
        ```mlir
        func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
          %0 = memref.alloc() : memref<1xf32>
          %1 = memref.alloc() : memref<1xf32>
          %cst = arith.constant 0.000000e+00 : f32
          affine.for %arg2 = 0 to 10 {
            affine.store %cst, %0[0] : memref<1xf32>
            affine.store %cst, %1[0] : memref<1xf32>
            %2 = affine.load %1[0] : memref<1xf32>
            %3 = arith.mulf %2, %2 : f32
            affine.store %3, %arg1[%arg2] : memref<10xf32>
            %4 = affine.load %0[0] : memref<1xf32>
            %5 = arith.addf %4, %4 : f32
            affine.store %5, %arg0[%arg2] : memref<10xf32>
          }
          return
        }
        ```

        Example 2: Sibling fusion.
        Input:
        ```mlir
        func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                             %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                             %arg4: memref<10x10xf32>) {
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
              %2 = arith.mulf %0, %1 : f32
              affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
            }
          }
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %1 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
              %2 = arith.addf %0, %1 : f32
              affine.store %2, %arg4[%arg5, %arg6] : memref<10x10xf32>
            }
          }
          return
        }
        ```
        Output:
        ```mlir
        func.func @sibling_fusion(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>,
                             %arg2: memref<10x10xf32>, %arg3: memref<10x10xf32>,
                             %arg4: memref<10x10xf32>) {
          affine.for %arg5 = 0 to 3 {
            affine.for %arg6 = 0 to 3 {
              %0 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %1 = affine.load %arg1[%arg5, %arg6] : memref<10x10xf32>
              %2 = arith.mulf %0, %1 : f32
              affine.store %2, %arg3[%arg5, %arg6] : memref<10x10xf32>
              %3 = affine.load %arg0[%arg5, %arg6] : memref<10x10xf32>
              %4 = affine.load %arg2[%arg5, %arg6] : memref<10x10xf32>
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %arg4[%arg5, %arg6] : memref<10x10xf32>
            }
          }
          return
        }
        ```

        Args:
            compute_tolerance: Fractional increase in additional computation tolerated while fusing
            fast_mem_space: Faster memory space number to promote fusion buffers to
            local_buf_threshold: Threshold size (KiB) for promoting local buffers to fast memory space
            maximal: Enables maximal loop fusion
            mode: fusion mode to attempt
        """
        self.add_pass(
            "affine-loop-fusion",
            **{
                "compute-tolerance": compute_tolerance,
                "fast-mem-space": fast_mem_space,
                "local-buf-threshold": local_buf_threshold,
                "maximal": maximal,
                "mode": mode,
            },
        )
        return self

    def affine_loop_invariant_code_motion(self):
        """Hoist loop invariant instructions outside of affine loops"""
        self.add_pass("affine-loop-invariant-code-motion")
        return self

    def affine_loop_normalize(self, promote_single_iter: bool = None):
        """Apply normalization transformations to affine loop-like ops
        Args:
            promote_single_iter: Promote single iteration loops
        """
        self.add_pass(
            "affine-loop-normalize", **{"promote-single-iter": promote_single_iter}
        )
        return self

    def affine_loop_tile(
        self,
        cache_size: int = None,
        separate: bool = None,
        tile_size: int = None,
        tile_sizes: List[int] = None,
    ):
        """Tile affine loop nests
        Args:
            cache_size: Set size of cache to tile for in KiB (default: 512)
            separate: Separate full and partial tiles (default: false)
            tile_size: Use this tile size for all loops
            tile_sizes: List of tile sizes for each perfect nest (overridden by -tile-size)
        """
        self.add_pass(
            "affine-loop-tile",
            **{
                "cache-size": cache_size,
                "separate": separate,
                "tile-size": tile_size,
                "tile-sizes": tile_sizes,
            },
        )
        return self

    def affine_loop_unroll(
        self,
        unroll_factor: int = None,
        unroll_up_to_factor: bool = None,
        unroll_num_reps: int = None,
        unroll_full_threshold: int = None,
        cleanup_unroll: bool = None,
    ):
        """Unroll affine loops
        Args:
            unroll_factor: Use this unroll factor for all loops being unrolled
            unroll_up_to_factor: Allow unrolling up to the factor specified
            unroll_num_reps: Unroll innermost loops repeatedly this many times
            unroll_full_threshold: Unroll all loops with trip count less than or equal to this
            cleanup_unroll: Fully unroll the cleanup loop when possible.
        """
        self.add_pass(
            "affine-loop-unroll",
            **{
                "unroll-factor": unroll_factor,
                "unroll-up-to-factor": unroll_up_to_factor,
                "unroll-num-reps": unroll_num_reps,
                "unroll-full-threshold": unroll_full_threshold,
                "cleanup-unroll": cleanup_unroll,
            },
        )
        return self

    def affine_loop_unroll_jam(self, unroll_jam_factor: int = None):
        """Unroll and jam affine loops
        Args:
            unroll_jam_factor: Use this unroll jam factor for all loops (default 4)
        """
        self.add_pass(
            "affine-loop-unroll-jam", **{"unroll-jam-factor": unroll_jam_factor}
        )
        return self

    def affine_parallelize(
        self, max_nested: int = None, parallel_reductions: bool = None
    ):
        """Convert affine.for ops into 1-D affine.parallel
        Args:
            max_nested: Maximum number of nested parallel loops to produce. Defaults to unlimited (UINT_MAX).
            parallel_reductions: Whether to parallelize reduction loops. Defaults to false.
        """
        self.add_pass(
            "affine-parallelize",
            **{"max-nested": max_nested, "parallel-reductions": parallel_reductions},
        )
        return self

    def affine_pipeline_data_transfer(self):
        """Pipeline non-blocking data transfers between explicitly managed levels of the memory hierarchy

        This pass performs a transformation to overlap non-blocking DMA operations
        in a loop with computations through double buffering. This is achieved by
        advancing dma_start operations with respect to other operations.

        Input

        ```mlir
        func.func @pipelinedatatransfer() {
          %0 = memref.alloc() : memref<256xf32>
          %1 = memref.alloc() : memref<32xf32, 1>
          %2 = memref.alloc() : memref<1xf32>
          %c0 = arith.constant 0 : index
          %c128 = arith.constant 128 : index
          affine.for %i0 = 0 to 8 {
            affine.dma_start %0[%i0], %1[%i0], %2[%c0], %c128 : memref<256xf32>, memref<32xf32, 1>, memref<1xf32>
            affine.dma_wait %2[%c0], %c128 : memref<1xf32>
            %3 = affine.load %1[%i0] : memref<32xf32, 1>
            %4 = "compute"(%3) : (f32) -> f32
            affine.store %4, %1[%i0] : memref<32xf32, 1>
          }
          return
        }
        ```

        Output

        ```mlir
        module {
          func.func @pipelinedatatransfer() {
            %c8 = arith.constant 8 : index
            %c0 = arith.constant 0 : index
            %0 = memref.alloc() : memref<256xf32>
            %c0_0 = arith.constant 0 : index
            %c128 = arith.constant 128 : index
            %1 = memref.alloc() : memref<2x32xf32, 1>
            %2 = memref.alloc() : memref<2x1xf32>
            affine.dma_start %0[%c0], %1[%c0 mod 2, %c0], %2[%c0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
            affine.for %arg0 = 1 to 8 {
              affine.dma_start %0[%arg0], %1[%arg0 mod 2, %arg0], %2[%arg0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
              %8 = affine.apply #map3(%arg0)
              %9 = affine.apply #map4(%8)
              %10 = affine.apply #map4(%8)
              affine.dma_wait %2[%8 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
              %11 = affine.load %1[%8 mod 2, %8] : memref<2x32xf32, 1>
              %12 = "compute"(%11) : (f32) -> f32
              affine.store %12, %1[%8 mod 2, %8] : memref<2x32xf32, 1>
            }
            %3 = affine.apply #map3(%c8)
            %4 = affine.apply #map4(%3)
            %5 = affine.apply #map4(%3)
            affine.dma_wait %2[%3 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
            %6 = affine.load %1[%3 mod 2, %3] : memref<2x32xf32, 1>
            %7 = "compute"(%6) : (f32) -> f32
            affine.store %7, %1[%3 mod 2, %3] : memref<2x32xf32, 1>
            memref.dealloc %2 : memref<2x1xf32>
            memref.dealloc %1 : memref<2x32xf32, 1>
            return
          }
        }
        ```

        """
        self.add_pass("affine-pipeline-data-transfer")
        return self

    def affine_raise_from_memref(self):
        """Turn some memref operators to affine operators where supported

        Raise memref.load and memref.store to affine.store and affine.load, inferring
        the affine map of those operators if needed. This allows passes like --affine-scalrep
        to optimize those loads and stores (forwarding them or eliminating them).
        They can be turned back to memref dialect ops with --lower-affine.

        """
        self.add_pass("affine-raise-from-memref")
        return self

    def affine_scalrep(self):
        """Replace affine memref accesses by scalars by forwarding stores to loads and eliminating redundant loads

        This pass performs store to load forwarding and redundant load elimination
        for affine memref accesses and potentially eliminates the entire memref
        if all its accesses are forwarded.

        Input

        ```mlir
        func.func @store_load_affine_apply() -> memref<10x10xf32> {
          %cf7 = arith.constant 7.0 : f32
          %m = memref.alloc() : memref<10x10xf32>
          affine.for %i0 = 0 to 10 {
            affine.for %i1 = 0 to 10 {
              affine.store %cf7, %m[%i0, %i1] : memref<10x10xf32>
              %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
              %v1 = arith.addf %v0, %v0 : f32
            }
          }
          return %m : memref<10x10xf32>
        }
        ```

        Output

        ```mlir
        module {
          func.func @store_load_affine_apply() -> memref<10x10xf32> {
            %cst = arith.constant 7.000000e+00 : f32
            %0 = memref.alloc() : memref<10x10xf32>
            affine.for %arg0 = 0 to 10 {
              affine.for %arg1 = 0 to 10 {
                affine.store %cst, %0[%arg0, %arg1] : memref<10x10xf32>
                %1 = arith.addf %cst, %cst : f32
              }
            }
            return %0 : memref<10x10xf32>
          }
        }
        ```

        """
        self.add_pass("affine-scalrep")
        return self

    def affine_simplify_min_max(self):
        """Simplify affine min/max/apply

        Apply the SimplifyAffineMaxOp, SimplifyAffineMinOp and SimplifyAffineApplyOp
        patterns in addition to AffineMin/Max canonicalization patterns until a
        fixed point is reached.
        These patterns apply ValueBoundsOp interface on AffineMin/Max ops and
        additional simplifications such as:
        ```
           min(x, y, cst) / cst -> 1
        ```
        when x, y, cst are all >= 0.
        This is typically useful to extract more static informationfrom IR after
        tiling but can also come at a cost due to Presburger-style analysis.

        """
        self.add_pass("affine-simplify-min-max")
        return self

    def affine_simplify_structures(self):
        """Simplify affine expressions in maps/sets and normalize memrefs"""
        self.add_pass("affine-simplify-structures")
        return self

    def affine_super_vectorize(
        self,
        virtual_vector_size: List[int] = None,
        test_fastest_varying: List[int] = None,
        vectorize_reductions: bool = None,
    ):
        """Vectorize to a target independent n-D vector abstraction
        Args:
            virtual_vector_size: Specify an n-D virtual vector size for vectorization. This must be greater than zero.
            test_fastest_varying: Specify a 1-D, 2-D or 3-D pattern of fastest varying memory dimensions to match. See defaultPatterns in Vectorize.cpp for a description and examples. This is used for testing purposes
            vectorize_reductions: Vectorize known reductions expressed via iter_args. Switched off by default.
        """
        self.add_pass(
            "affine-super-vectorize",
            **{
                "virtual-vector-size": virtual_vector_size,
                "test-fastest-varying": test_fastest_varying,
                "vectorize-reductions": vectorize_reductions,
            },
        )
        return self

    def amdgpu_emulate_atomics(self, chipset: str = None):
        """Emulate atomic operations on chipsets that do not support them

        This pass rewrites any AMDGPU-specific atomic operation that is not supported
        on the given `chipset` into a compare-and-swap loop.

        Args:
            chipset: Chipset that these operations will run on
        """
        self.add_pass("amdgpu-emulate-atomics", **{"chipset": chipset})
        return self

    def amdgpu_fold_memrefs_ops(self):
        """Fold memref operations into their parent operations

        This pass identifies memref operations (subview, expand_shape, collapse_shape)
        that are sources of `GatherToLDSOp` and attempts to fold the source ops,
        potentially simplifying the overall operation and improving performance.

        """
        self.add_pass("amdgpu-fold-memrefs-ops")
        return self

    def amdgpu_maskedload_to_load(self):
        """Lower the operations from the vector maskedload to vector load

        This pass creates a transfer read op lowering optimization. The lowering
        will produce a conditional check at runtime. If within bounds, a vector
        trasfer read op will be lowered to a combination of vector.load, arith.select
        and vector.broadcast. If not, it will fallback to the default lowering
        of the transfer_read op.

        This pattern will make it possible for masked transfer_read to be lowered
        towards buffer load with bounds check, allowing a more optimized global
        load accessing pattern compared with existing implementation of
        llvm.intr.masked.load on vectors.

        """
        self.add_pass("amdgpu-maskedload-to-load")
        return self

    def amdgpu_resolve_strided_metadata(self):
        """Resolve memref.extract_strided_metadata on AMDGPU ops

        This pass rrewrites `memref.extract_strided_metadata` operations
        targeting the AMDGPU dialect casts.

        The patterns in this pass should normally be run alongside those in
        -expand-strided-metadata, and creating a pass that combines those two
        sets of patterns is the recommended way to use this functionality.
        However, this pass (which will likely need a second -expand-strided-metadata
        after it) is provided so that simple usecases do not need to create custom passes.
        These patterns have not been added to -expnad-strided-metadata to
        prevent the memref dialect from depending on platform-specific code.

        """
        self.add_pass("amdgpu-resolve-strided-metadata")
        return self

    def arith_emulate_unsupported_floats(
        self, source_types: List[str] = None, target_type: str = None
    ):
        """Emulate operations on unsupported floats with extf/truncf

        Emulate arith and vector floating point operations that use float types
        which are unspported on a target by inserting extf/truncf pairs around all
        such operations in order to produce arithmetic that can be performed while
        preserving the original rounding behavior.

        This pass does not attempt to reason about the operations being performed
        to determine when type conversions can be elided.

        Args:
            source_types: MLIR types without arithmetic support on a given target
            target_type: MLIR type to convert the unsupported source types to
        """
        self.add_pass(
            "arith-emulate-unsupported-floats",
            **{"source-types": source_types, "target-type": target_type},
        )
        return self

    def arith_emulate_wide_int(self, widest_int_supported: int = None):
        """Emulate 2*N-bit integer operations using N-bit operations

        Emulate arith integer operations that use too wide integer types with
        equivalent operations on supported narrow integer types. This is done by
        splitting original integer values into two halves.

        This pass is intended preserve semantics but not necessarily provide the
        most efficient implementation.
        TODO: Optimize op emulation.

        Currently, only power-of-two integer bitwidths are supported.

        Args:
            widest_int_supported: Widest integer type supported by the target
        """
        self.add_pass(
            "arith-emulate-wide-int", **{"widest-int-supported": widest_int_supported}
        )
        return self

    def arith_expand(
        self,
        include_bf16: bool = None,
        include_f8e8m0: bool = None,
        include_f4e2m1: bool = None,
    ):
        """Legalize Arith ops to be convertible to LLVM.
        Args:
            include_bf16: Enable the BF16 expansion patterns
            include_f8e8m0: Enable the F8E8M0 expansion patterns
            include_f4e2m1: Enable the F4E2M1 expansion patterns
        """
        self.add_pass(
            "arith-expand",
            **{
                "include-bf16": include_bf16,
                "include-f8e8m0": include_f8e8m0,
                "include-f4e2m1": include_f4e2m1,
            },
        )
        return self

    def arith_int_range_narrowing(self, int_bitwidths_supported: List[int] = None):
        """Reduce integer operations bitwidth based on integer range analysis

        This pass runs integer range analysis and tries to narrow arith ops to the
        specified bitwidth based on its results.

        `bitwidthsSupported` assumed to be not wider than `index` type.
        TODO: get index width from DLTI.

        Args:
            int_bitwidths_supported: Integer bitwidths supported
        """
        self.add_pass(
            "arith-int-range-narrowing",
            **{"int-bitwidths-supported": int_bitwidths_supported},
        )
        return self

    def arith_unsigned_when_equivalent(self):
        """Replace signed ops with unsigned ones where they are proven equivalent

        Replace signed ops with their unsigned equivalents when integer range analysis
        determines that their arguments and results are all guaranteed to be
        non-negative when interpreted as signed integers. When this occurs,
        we know that the semantics of the signed and unsigned operations are the same,
        since they share the same behavior when their operands and results  are in the
        range [0, signed_max(type)].

        The affect ops include division, remainder, shifts, min, max, and integer
        comparisons.

        """
        self.add_pass("arith-unsigned-when-equivalent")
        return self

    def arm_neon_2d_to_intr(self):
        """Convert Arm NEON structured ops to intrinsics

        Creates a pass to lower Arm NEON 2D ops to intrinsics, i.e.
        equivalent ops operating on flattened 1D vectors and mapping more
        directly to the corresponding Arm NEON instruction.

        """
        self.add_pass("arm-neon-2d-to-intr")
        return self

    def arm_sme_outer_product_fusion(self):
        """Fuse 'arm_sme.outerproduct' operations into 2-way or 4-way widening variants

        This pass fuses 'arm_sme.outerproduct' operations that are chained via the
        accumulator into 2-way or 4-way ArmSME outer product operations.

        For example:
        ```mlir
        %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
        %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
        %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
        %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>

        %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>, vector<[4]xf32>
        %1 = arm_sme.outerproduct %a1_ext, %b1_ext acc(%0) : vector<[4]xf32>, vector<[4]xf32>
        ```

        Becomes:

        ```mlir
        %a_packed = vector.interleave %a0, %a1 : vector<[4]xf16> -> vector<[8]xf16>
        %b_packed = vector.interleave %b0, %b1 : vector<[4]xf16> -> vector<[8]xf16>
        %0 = arm_sme.fmopa_2way %a_packed, %b_packed : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
        ```

        For further information on the 2-way or 4-way widening ops see:
        https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smefmopa_2way-arm_smefmopa_2wayop
        https://mlir.llvm.org/docs/Dialects/ArmSME/#arm_smesmopa_4way-arm_smesmopa_4wayop

        """
        self.add_pass("arm-sme-outer-product-fusion")
        return self

    def arm_sme_vector_legalization(self):
        """Legalize vectors for ArmSME

        This pass legalizes vector operations so that they can be lowered to ArmSME.
        This includes decomposing operations that operate on vector types larger
        than a single SME tile (e.g. `vector<[8]x[8]xf32>`) into multiple SME
        tile-sized operations, as well as rewrites needed to get operations into
        forms compatible with SME lowerings.

        Note: Decomposition is currently limited to vector types that are an exact
        multiple of SME tiles. That is scalable in two dimensions, with both the
        rows and columns divisible by the SVE vector length for the element type.

        """
        self.add_pass("arm-sme-vector-legalization")
        return self

    def arm_sve_legalize_vector_storage(self):
        """Ensures stores of SVE vector types will be legal

        This pass ensures that loads, stores, and allocations of SVE vector types
        will be legal in the LLVM backend. It does this at the memref level, so this
        pass must be applied before lowering all the way to LLVM.

        This pass currently addresses two issues.

        #### Loading and storing predicate types

        It is only legal to load/store predicate types equal to (or greater than) a
        full predicate register, which in MLIR is `vector<[16]xi1>`. Smaller
        predicate types (`vector<[1|2|4|8]xi1>`) must be converted to/from a full
        predicate type (referred to as a `svbool`) before and after storing and
        loading respectively. This pass does this by widening allocations and
        inserting conversion intrinsics. Note: Non-powers-of-two masks (e.g.
        `vector<[7]xi1>`), which are not SVE predicates, are ignored.

        For example:

        ```mlir
        %alloca = memref.alloca() : memref<vector<[4]xi1>>
        %mask = vector.constant_mask [4] : vector<[4]xi1>
        memref.store %mask, %alloca[] : memref<vector<[4]xi1>>
        %reload = memref.load %alloca[] : memref<vector<[4]xi1>>
        ```
        Becomes:
        ```mlir
        %alloca = memref.alloca() {alignment = 1 : i64} : memref<vector<[16]xi1>>
        %mask = vector.constant_mask [4] : vector<[4]xi1>
        %svbool = arm_sve.convert_to_svbool %mask : vector<[4]xi1>
        memref.store %svbool, %alloca[] : memref<vector<[16]xi1>>
        %reload_svbool = memref.load %alloca[] : memref<vector<[16]xi1>>
        %reload = arm_sve.convert_from_svbool %reload_svbool : vector<[4]xi1>
        ```

        #### Relax alignments for SVE vector allocas

        The storage for SVE vector types only needs to have an alignment that
        matches the element type (for example 4 byte alignment for `f32`s). However,
        the LLVM backend currently defaults to aligning to `base size` x
        `element size` bytes. For non-legal vector types like `vector<[8]xf32>` this
        results in 8 x 4 = 32-byte alignment, but the backend only supports up to
        16-byte alignment for SVE vectors on the stack. Explicitly setting a smaller
        alignment prevents this issue.

        """
        self.add_pass("arm-sve-legalize-vector-storage")
        return self

    def async_func_to_async_runtime(self):
        """Lower async.func operations to the explicit async.runtime andasync.coro operations"""
        self.add_pass("async-func-to-async-runtime")
        return self

    def async_parallel_for(
        self,
        async_dispatch: bool = None,
        num_workers: int = None,
        min_task_size: int = None,
    ):
        """Convert scf.parallel operations to multiple async compute ops executed concurrently for non-overlapping iteration ranges
        Args:
            async_dispatch: Dispatch async compute tasks using recursive work splitting. If `false` async compute tasks will be launched using simple for loop in the caller thread.
            num_workers: The number of available workers to execute async operations. If `-1` the value will be retrieved from the runtime.
            min_task_size: The minimum task size for sharding parallel operation.
        """
        self.add_pass(
            "async-parallel-for",
            **{
                "async-dispatch": async_dispatch,
                "num-workers": num_workers,
                "min-task-size": min_task_size,
            },
        )
        return self

    def async_runtime_policy_based_ref_counting(self):
        """Policy based reference counting for Async runtime operations

        This pass works at the async runtime abtraction level, after all
        `async.execute` and `async.await` operations are lowered to the async
        runtime API calls, and async coroutine operations.

        This pass doesn't rely on reference counted values liveness analysis, and
        instead uses simple policy to create reference counting operations. If the
        program violates any of the assumptions, then this pass might lead to
        memory leaks or runtime errors.

        The default reference counting policy assumptions:
          1. Async token can be awaited or added to the group only once.
          2. Async value or group can be awaited only once.

        Under these assumptions reference counting only needs to drop reference:
          1. After `async.runtime.await` operation for async tokens and groups
             (until error handling is not implemented for the sync await).
          2. After `async.runtime.is_error` operation for async tokens and groups
             (this is the last operation in the coroutine resume function).
          3. After `async.runtime.load` operation for async values.

        This pass introduces significanly less runtime overhead compared to the
        automatic reference counting.

        """
        self.add_pass("async-runtime-policy-based-ref-counting")
        return self

    def async_runtime_ref_counting(self):
        """Automatic reference counting for Async runtime operations

        This pass works at the async runtime abtraction level, after all
        `async.execute` and `async.await` operations are lowered to the async
        runtime API calls, and async coroutine operations.

        It relies on the LLVM coroutines switched-resume lowering semantics for
        the correct placing of the reference counting operations.

        See: https://llvm.org/docs/Coroutines.html#switched-resume-lowering

        """
        self.add_pass("async-runtime-ref-counting")
        return self

    def async_runtime_ref_counting_opt(self):
        """Optimize automatic reference counting operations for theAsync runtime by removing redundant operations"""
        self.add_pass("async-runtime-ref-counting-opt")
        return self

    def async_to_async_runtime(self):
        """Lower all high level async operations (e.g. async.execute) tothe explicit async.runtime and async.coro operations"""
        self.add_pass("async-to-async-runtime")
        return self

    def bubble_down_memory_space_casts(self):
        """Bubbles down memory-space cast operations.

        This pass tries to iteratively bubble down all possible memory-space cast
        operations. It is important to note that the determination of which casts
        are bubbled down is based on the interfaces
        `MemorySpaceCastConsumerOpInterface`, and `MemorySpaceCastOpInterface`, and
        not the pass. The pass only looks for operations implementing the
        `MemorySpaceCastConsumerOpInterface` interface, and invoking the interface
        methods to perform the bubbling down.

        Example:

        ```mlir
        func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
          %memspacecast = memref.memory_space_cast %arg0 : memref<4x4xf32, 1> to memref<4x4xf32>
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %expanded = memref.expand_shape %memspacecast [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32> into memref<4x2x2xf32>
          %collapsed = memref.collapse_shape %expanded [[0, 1, 2]] : memref<4x2x2xf32> into memref<16xf32>
          %loaded = memref.load %collapsed[%c0] : memref<16xf32>
          %added = arith.addf %loaded, %arg2 : f32
          memref.store %added, %collapsed[%c0] : memref<16xf32>
          %atomic_result = memref.atomic_rmw addf %arg2, %collapsed[%c4] : (f32, memref<16xf32>) -> f32
          return %collapsed : memref<16xf32>
        }
        // mlir-opt --bubble-down-memory-space-casts
        func.func @op_with_cast_sequence(%arg0: memref<4x4xf32, 1>, %arg1: index, %arg2: f32) -> memref<16xf32> {
          %c4 = arith.constant 4 : index
          %c0 = arith.constant 0 : index
          %expand_shape = memref.expand_shape %arg0 [[0], [1, 2]] output_shape [4, 2, 2] : memref<4x4xf32, 1> into memref<4x2x2xf32, 1>
          %collapse_shape = memref.collapse_shape %expand_shape [[0, 1, 2]] : memref<4x2x2xf32, 1> into memref<16xf32, 1>
          %memspacecast = memref.memory_space_cast %collapse_shape : memref<16xf32, 1> to memref<16xf32>
          %0 = memref.load %collapse_shape[%c0] : memref<16xf32, 1>
          %1 = arith.addf %0, %arg2 : f32
          memref.store %1, %collapse_shape[%c0] : memref<16xf32, 1>
          %2 = memref.atomic_rmw addf %arg2, %collapse_shape[%c4] : (f32, memref<16xf32, 1>) -> f32
          return %memspacecast : memref<16xf32>
        }
        ```

        """
        self.add_pass("bubble-down-memory-space-casts")
        return self

    def buffer_deallocation_simplification(self):
        """Optimizes `bufferization.dealloc` operation for more efficient codegen

        This pass uses static alias analysis to reduce the number of alias checks
        required at runtime. Such checks are sometimes necessary to make sure that
        memrefs aren't deallocated before their last usage (use after free) or that
        some memref isn't deallocated twice (double free).

        """
        self.add_pass("buffer-deallocation-simplification")
        return self

    def buffer_hoisting(self):
        """Optimizes placement of allocation operations by moving them into common dominators and out of nested regions

        This pass implements an approach to aggressively move allocations upwards
        into common dominators and out of nested regions.

        """
        self.add_pass("buffer-hoisting")
        return self

    def buffer_loop_hoisting(self):
        """Optimizes placement of allocation operations by moving them out of loop nests

        This pass implements an approach to aggressively move allocations upwards
        out of loop nests. It does not move allocations into common dominators.

        """
        self.add_pass("buffer-loop-hoisting")
        return self

    def buffer_results_to_out_params(
        self,
        add_result_attr: bool = None,
        hoist_static_allocs: bool = None,
        hoist_dynamic_allocs: bool = None,
        modify_public_functions: bool = None,
    ):
        """Converts memref-typed function results to out-params

        Some calling conventions prefer to pass output memrefs as "out params". The
        conversion to this calling convention must be done as an atomic
        transformation of the entire program (hence this is a module pass).

        For example, if a call is rewritten, the callee needs to be rewritten
        otherwise the IR will end up invalid. Thus, this transformation
        require an atomic change to the entire program (e.g. the whole module).

        This pass is expected to run immediately after bufferization is finished.
        At that point, tensor-typed results will have been converted to memref-typed
        results, and can be consistently converted to out params.

        All memref-typed results are appended to the function argument list.

        The main issue with this pass (and the out-param calling convention) is that
        buffers for results need to be allocated in the caller. This currently only
        works for static shaped memrefs.

        If the hoist-static-allocs option is on, the pass tries to eliminate the
        allocation for the returned memref and avoid the memory-copy if possible.
        This optimization applies on the returned memref which has static shape and
        is allocated by memref.alloc in the function. It will use the memref given
        in function argument to replace the allocated memref.

        Args:
            add_result_attr: Add the attribute 'bufferize.result' to all output parameters.
            hoist_static_allocs: Hoist static allocations to call sites.
            hoist_dynamic_allocs: Hoist dynamic allocations to call sites.
            modify_public_functions: Modify function signatures of public functions.
        """
        self.add_pass(
            "buffer-results-to-out-params",
            **{
                "add-result-attr": add_result_attr,
                "hoist-static-allocs": hoist_static_allocs,
                "hoist-dynamic-allocs": hoist_dynamic_allocs,
                "modify-public-functions": modify_public_functions,
            },
        )
        return self

    def bufferization_lower_deallocations(self):
        """Lowers `bufferization.dealloc` operations to `memref.dealloc`operations

        This pass lowers `bufferization.dealloc` operations to the `memref` dialect.
        It can be applied to a `builtin.module` or operations implementing the
        `FunctionOpInterface`. For the latter, only simple `dealloc` operations can
        be lowered because the library function necessary for the fully generic
        lowering cannot be inserted. In this case, an error will be emitted.
        Next to `memref.dealloc` operations, it may also emit operations from the
        `arith`, `scf`, and `func` dialects to build conditional deallocations and
        library functions to avoid code-size blow-up.

        """
        self.add_pass("bufferization-lower-deallocations")
        return self

    def canonicalize(
        self,
        top_down: bool = None,
        region_simplify: GreedySimplifyRegionLevel = None,
        max_iterations: int = None,
        max_num_rewrites: int = None,
        test_convergence: bool = None,
        disable_patterns: List[str] = None,
        enable_patterns: List[str] = None,
    ):
        """Canonicalize operations

        This pass performs various types of canonicalizations over a set of
        operations by iteratively applying the canonicalization patterns of all
        loaded dialects until either a fixpoint is reached or the maximum number of
        iterations/rewrites is exhausted. Canonicalization is best-effort and does
        not guarantee that the entire IR is in a canonical form after running this
        pass. See [Operation Canonicalization](Canonicalization.md) for more
        details.

        Args:
            top_down: Seed the worklist in general top-down order
            region_simplify: Perform control flow optimizations to the region tree
            max_iterations: Max. iterations between applying patterns / simplifying regions
            max_num_rewrites: Max. number of pattern rewrites within an iteration
            test_convergence: Test only: Fail pass on non-convergence to detect cyclic pattern
            disable_patterns: Labels of patterns that should be filtered out during application
            enable_patterns: Labels of patterns that should be used during application, all other patterns are filtered out
        """
        self.add_pass(
            "canonicalize",
            **{
                "top-down": top_down,
                "region-simplify": region_simplify,
                "max-iterations": max_iterations,
                "max-num-rewrites": max_num_rewrites,
                "test-convergence": test_convergence,
                "disable-patterns": disable_patterns,
                "enable-patterns": enable_patterns,
            },
        )
        return self

    def composite_fixed_point_pass(
        self, name: str = None, pipeline: str = None, max_iterations: int = None
    ):
        """Composite fixed point pass

        Composite pass runs provided set of passes until fixed point or maximum
        number of iterations reached.

        Args:
            name: Composite pass display name
            pipeline: Composite pass inner pipeline
            max_iterations: Maximum number of iterations if inner pipeline
        """
        self.add_pass(
            "composite-fixed-point-pass",
            **{"name": name, "pipeline": pipeline, "max-iterations": max_iterations},
        )
        return self

    def control_flow_sink(self):
        """Sink operations into conditional blocks

        This pass implements control-flow sink on operations that implement
        `RegionBranchOpInterface` by moving dominating operations whose only uses
        are in a conditionally-executed regions into those regions so that
        executions paths where their results are not needed do not perform
        unnecessary computations.

        This is similar (but opposite) to loop-invariant code motion, which hoists
        operations out of regions executed more than once. The implementation of
        control-flow sink uses a simple and conversative cost model: operations are
        never duplicated and are only moved into singly-executed regions.

        It is recommended to run canonicalization first to remove unreachable
        blocks: ops in unreachable blocks may prevent other operations from being
        sunk as they may contain uses of their results

        """
        self.add_pass("control-flow-sink")
        return self

    def convert_affine_for_to_gpu(
        self, gpu_block_dims: int = None, gpu_thread_dims: int = None
    ):
        """Convert top-level AffineFor Ops to GPU kernels
        Args:
            gpu_block_dims: Number of GPU block dimensions for mapping
            gpu_thread_dims: Number of GPU thread dimensions for mapping
        """
        self.add_pass(
            "convert-affine-for-to-gpu",
            **{"gpu-block-dims": gpu_block_dims, "gpu-thread-dims": gpu_thread_dims},
        )
        return self

    def convert_amdgpu_to_rocdl(self, chipset: str = None):
        """Convert AMDGPU dialect to ROCDL dialect

        This pass converts supported AMDGPU ops to ROCDL dialect intrinsics.

        Args:
            chipset: Chipset that these operations will run on
        """
        self.add_pass("convert-amdgpu-to-rocdl", **{"chipset": chipset})
        return self

    def convert_arith_to_amdgpu(
        self,
        chipset: str = None,
        saturate_fp8_truncf: bool = None,
        allow_packed_f16_round_to_zero: bool = None,
    ):
        """Convert Arith operations to AMDGPU-specific implementations

        Convert `arith` operations (currently extf and truncf on 8-bit floats)
        to operations in the `amdgpu` dialect. This pass is done in two steps
        in order to avoid running a notional arith-to-rocdl and arith-to-llvm
        simultaniously.

        Args:
            chipset: Chipset that these operations will run on
            saturate_fp8_truncf: Use saturating truncation for 8-bit float types
            allow_packed_f16_round_to_zero: Whether we should allow f32->f16 packed round-to-zero conversion
        """
        self.add_pass(
            "convert-arith-to-amdgpu",
            **{
                "chipset": chipset,
                "saturate-fp8-truncf": saturate_fp8_truncf,
                "allow-packed-f16-round-to-zero": allow_packed_f16_round_to_zero,
            },
        )
        return self

    def convert_arith_to_apfloat(self):
        """Convert Arith ops to APFloat runtime library calls

        This pass converts supported Arith ops to APFloat-based runtime library
        calls (APFloatWrappers.cpp). APFloat is a software implementation of
        floating-point arithmetic operations.

        """
        self.add_pass("convert-arith-to-apfloat")
        return self

    def convert_arith_to_arm_sme(self):
        """Convert Arith dialect to ArmSME dialect"""
        self.add_pass("convert-arith-to-arm-sme")
        return self

    def convert_arith_to_emitc(self):
        """Convert Arith dialect to EmitC dialect"""
        self.add_pass("convert-arith-to-emitc")
        return self

    def convert_arith_to_llvm(self, index_bitwidth: int = None):
        """Convert Arith dialect to LLVM dialect

        This pass converts supported Arith ops to LLVM dialect instructions.

        Args:
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass("convert-arith-to-llvm", **{"index-bitwidth": index_bitwidth})
        return self

    def convert_arith_to_spirv(
        self,
        emulate_lt_32_bit_scalar_types: bool = None,
        emulate_unsupported_float_types: bool = None,
    ):
        """Convert Arith dialect to SPIR-V dialect
        Args:
            emulate_lt_32_bit_scalar_types: Emulate narrower scalar types with 32-bit ones if not supported by the target
            emulate_unsupported_float_types: Emulate unsupported float types by representing them with integer types of same bit width
        """
        self.add_pass(
            "convert-arith-to-spirv",
            **{
                "emulate-lt-32-bit-scalar-types": emulate_lt_32_bit_scalar_types,
                "emulate-unsupported-float-types": emulate_unsupported_float_types,
            },
        )
        return self

    def convert_arm_sme_to_llvm(self, dump_tile_live_ranges: bool = None):
        """Lower the operations from the ArmSME dialect into the LLVM dialect
        Args:
            dump_tile_live_ranges: Dump the live ranges of SME tiles (for debugging)
        """
        self.add_pass(
            "convert-arm-sme-to-llvm",
            **{"dump-tile-live-ranges": dump_tile_live_ranges},
        )
        return self

    def convert_arm_sme_to_scf(self):
        """Lower the operations from the ArmSME dialect into the SCF dialect"""
        self.add_pass("convert-arm-sme-to-scf")
        return self

    def convert_async_to_llvm(self):
        """Convert the operations from the async dialect into the LLVM dialect

        Convert `async.execute` operations to LLVM coroutines and use async runtime
        API to execute them.

        """
        self.add_pass("convert-async-to-llvm")
        return self

    def convert_bufferization_to_memref(self):
        """Convert operations from the Bufferization dialect to the MemRef dialect


        This pass converts bufferization operations into memref operations.

        In the current state, this pass only transforms a `bufferization.clone`
        operation into `memref.alloc` and `memref.copy` operations and
        `bufferization.dealloc` operations (the same way as the
        `-bufferization-lower-deallocations` pass). The conversion of `clone`
        operations is needed, since some clone operations could remain after
        applying several transformation processes. Currently, only `canonicalize`
        transforms clone operations or even eliminates them. This can lead to errors
        if any clone op survived after all conversion passes (starting from the
        bufferization dialect) are performed.

        See:
        https://llvm.discourse.group/t/bufferization-error-related-to-memref-clone/4665

        To avoid these errors, this pass can be performed as a last clean-up pass to
        transform remaining operations and to proceed in other dialects (memref
        e.g.).

        Note that this pass only transforms the operation without any further
        analyses. This pass does not consider any memory analysis or optimization
        and hence does not resolve any memory leaks.


        """
        self.add_pass("convert-bufferization-to-memref")
        return self

    def convert_cf_to_llvm(self, index_bitwidth: int = None):
        """Convert ControlFlow operations to the LLVM dialect

        Convert ControlFlow operations into LLVM IR dialect operations.

        If other operations are present and their results are required by the LLVM
        IR dialect operations, the pass will fail.  Any LLVM IR operations or types
        already present in the IR will be kept as is.

        Args:
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass("convert-cf-to-llvm", **{"index-bitwidth": index_bitwidth})
        return self

    def convert_cf_to_spirv(
        self,
        emulate_lt_32_bit_scalar_types: bool = None,
        emulate_unsupported_float_types: bool = None,
    ):
        """Convert ControlFlow dialect to SPIR-V dialect
        Args:
            emulate_lt_32_bit_scalar_types: Emulate narrower scalar types with 32-bit ones if not supported by the target
            emulate_unsupported_float_types: Emulate unsupported float types by representing them with integer types of same bit width
        """
        self.add_pass(
            "convert-cf-to-spirv",
            **{
                "emulate-lt-32-bit-scalar-types": emulate_lt_32_bit_scalar_types,
                "emulate-unsupported-float-types": emulate_unsupported_float_types,
            },
        )
        return self

    def convert_complex_to_libm(self):
        """Convert Complex dialect to libm calls

        This pass converts supported Complex ops to libm calls.

        """
        self.add_pass("convert-complex-to-libm")
        return self

    def convert_complex_to_llvm(
        self, complex_range: "::mlir::complex::ComplexRangeFlags" = None
    ):
        """Convert Complex dialect to LLVM dialect
        Args:
            complex_range: Control the intermediate calculation of complex number division
        """
        self.add_pass("convert-complex-to-llvm", **{"complex-range": complex_range})
        return self

    def convert_complex_to_rocdl_library_calls(self):
        """Convert Complex dialect to ROCDL library calls

        This pass converts supported Complex ops to calls to the AMD device library.

        """
        self.add_pass("convert-complex-to-rocdl-library-calls")
        return self

    def convert_complex_to_spirv(self):
        """Convert Complex dialect to SPIRV dialect"""
        self.add_pass("convert-complex-to-spirv")
        return self

    def convert_complex_to_standard(
        self, complex_range: "::mlir::complex::ComplexRangeFlags" = None
    ):
        """Convert Complex dialect to standard dialect
        Args:
            complex_range: Control the intermediate calculation of complex number division
        """
        self.add_pass("convert-complex-to-standard", **{"complex-range": complex_range})
        return self

    def convert_elementwise_to_linalg(self):
        """Convert ElementwiseMappable ops to linalg

        Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.

        This pass only converts ops that operate on ranked tensors. It can be
        run on op which contains linalg ops (most commonly a
        FunctionOpInterface op).

        """
        self.add_pass("convert-elementwise-to-linalg")
        return self

    def convert_func_to_emitc(self):
        """Convert Func dialect to EmitC dialect"""
        self.add_pass("convert-func-to-emitc")
        return self

    def convert_func_to_llvm(
        self, use_bare_ptr_memref_call_conv: bool = None, index_bitwidth: int = None
    ):
        """Convert from the Func dialect to the LLVM dialect

        Convert Func dialect operations into the LLVM IR dialect operations.

        #### Input invariant

        -   no `tensor` types;
        -   all `vector` are one-dimensional;
        -   all blocks are reachable by following the successors of the first basic
            block;

        If other operations are present and their results are required by the LLVM
        IR dialect operations, the pass will fail.  Any LLVM IR operations or types
        already present in the IR will be kept as is.

        An LLVM datalayout string can be attached as an attribute to the module on
        which the pass anchors. Such an attribute is attached by calling the
        set-module-datalayout pass. If present, an llvm::DataLayout object is
        created from this attribute and used in the conversion to LLVM.

        #### Output IR

        Functions converted to LLVM IR. Function arguments types are converted
        one-to-one. Function results are converted one-to-one and, in case more than
        1 value is returned, packed into an LLVM IR struct type. Function calls and
        returns are updated accordingly. Block argument types are updated to use
        LLVM IR types.

        Args:
            use_bare_ptr_memref_call_conv: Replace FuncOp's MemRef arguments with bare pointers to the MemRef element types
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass(
            "convert-func-to-llvm",
            **{
                "use-bare-ptr-memref-call-conv": use_bare_ptr_memref_call_conv,
                "index-bitwidth": index_bitwidth,
            },
        )
        return self

    def convert_func_to_spirv(
        self,
        emulate_lt_32_bit_scalar_types: bool = None,
        emulate_unsupported_float_types: bool = None,
    ):
        """Convert Func dialect to SPIR-V dialect
        Args:
            emulate_lt_32_bit_scalar_types: Emulate narrower scalar types with 32-bit ones if not supported by the target
            emulate_unsupported_float_types: Emulate unsupported float types by representing them with integer types of same bit width
        """
        self.add_pass(
            "convert-func-to-spirv",
            **{
                "emulate-lt-32-bit-scalar-types": emulate_lt_32_bit_scalar_types,
                "emulate-unsupported-float-types": emulate_unsupported_float_types,
            },
        )
        return self

    def convert_gpu_to_llvm_spv(self, use_64bit_index: bool = None):
        """Generate LLVM operations to be ingested by a SPIR-V backend for gpu operations
        Args:
            use_64bit_index: Use 64-bit integers to convert index types
        """
        self.add_pass("convert-gpu-to-llvm-spv", **{"use-64bit-index": use_64bit_index})
        return self

    def convert_gpu_to_nvvm(
        self,
        index_bitwidth: int = None,
        has_redux: bool = None,
        use_bare_ptr_memref_call_conv: bool = None,
        allowed_dialects: List[str] = None,
    ):
        """Generate NVVM operations for gpu operations
        Args:
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
            has_redux: Target gpu supports redux
            use_bare_ptr_memref_call_conv: Replace memref arguments in GPU functions with bare pointers. All memrefs must have static shape.
            allowed_dialects: Run conversion patterns of only the specified dialects
        """
        self.add_pass(
            "convert-gpu-to-nvvm",
            **{
                "index-bitwidth": index_bitwidth,
                "has-redux": has_redux,
                "use-bare-ptr-memref-call-conv": use_bare_ptr_memref_call_conv,
                "allowed-dialects": allowed_dialects,
            },
        )
        return self

    def convert_gpu_to_rocdl(
        self,
        chipset: str = None,
        index_bitwidth: int = None,
        use_bare_ptr_memref_call_conv: bool = None,
        runtime: "gpu::amd::Runtime" = None,
        allowed_dialects: List[str] = None,
    ):
        """Generate ROCDL operations for gpu operations
        Args:
            chipset: Chipset that these operations will run on
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
            use_bare_ptr_memref_call_conv: Replace memref arguments in GPU functions with bare pointers.All memrefs must have static shape
            runtime: Runtime code will be run on (default is Unknown, can also use HIP or OpenCL)
            allowed_dialects: Run conversion patterns of only the specified dialects
        """
        self.add_pass(
            "convert-gpu-to-rocdl",
            **{
                "chipset": chipset,
                "index-bitwidth": index_bitwidth,
                "use-bare-ptr-memref-call-conv": use_bare_ptr_memref_call_conv,
                "runtime": runtime,
                "allowed-dialects": allowed_dialects,
            },
        )
        return self

    def convert_gpu_to_spirv(self, use_64bit_index: bool = None):
        """Convert GPU dialect to SPIR-V dialect

        This pass converts supported GPU device ops to SPIR-V ops. It does not
        handle GPU host ops.

        A `gpu.func` op can have parameters to pass in resources. But in SPIR-V
        entry functions cannot take parameters; they use descriptors to access
        resources. By default, parameters to a `gpu.func` op will be converted to
        global variables. These global variables will be assigned sequential binding
        numbers following their order in the original `gpu.func` op, starting from
        0, in set 0. One can attach `spirv.interface_var_abi` to those parameters
        to control the set and binding if wanted.

        Args:
            use_64bit_index: Use 64-bit integers to convert index types
        """
        self.add_pass("convert-gpu-to-spirv", **{"use-64bit-index": use_64bit_index})
        return self

    def convert_index_to_llvm(self, index_bitwidth: int = None):
        """Lower the `index` dialect to the `llvm` dialect.

        This pass lowers Index dialect operations to LLVM dialect operations.
        Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`,
        `ceildivu`, and `floordivs`, which expand to series of LLVM operations.
        Importantly, the index bitwidth should be correctly set to the target
        pointer width via `index-bitwidth`.

        Args:
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass("convert-index-to-llvm", **{"index-bitwidth": index_bitwidth})
        return self

    def convert_index_to_spirv(self, use_64bit_index: bool = None):
        """Lower the `index` dialect to the `spirv` dialect.

        This pass lowers Index dialect operations to SPIR-V dialect operations.
        Operation conversions are 1-to-1 except for the exotic divides: `ceildivs`,
        `ceildivu`, and `floordivs`. The index bitwidth will be 32 or 64 as
        specified by use-64bit-index.

        Args:
            use_64bit_index: Use 64-bit integers to convert index types
        """
        self.add_pass("convert-index-to-spirv", **{"use-64bit-index": use_64bit_index})
        return self

    def convert_linalg_to_affine_loops(self):
        """Lower the operations from the linalg dialect into affine loops"""
        self.add_pass("convert-linalg-to-affine-loops")
        return self

    def convert_linalg_to_loops(self):
        """Lower the operations from the linalg dialect into loops

        Lowers the `linalg` ops to loop nests using `scf.for`.

        Pre-condition: the operands used by the `linalg` ops have buffer semantics,
        i.e., tensor operands and results must be converted to memrefs via
        bufferization.

        """
        self.add_pass("convert-linalg-to-loops")
        return self

    def convert_linalg_to_parallel_loops(self):
        """Lower the operations from the linalg dialect into parallel loops"""
        self.add_pass("convert-linalg-to-parallel-loops")
        return self

    def convert_linalg_to_std(self):
        """Convert the operations from the linalg dialect into the Standard dialect"""
        self.add_pass("convert-linalg-to-std")
        return self

    def convert_math_to_emitc(
        self, language_target: "::mlir::emitc::LanguageTarget" = None
    ):
        """Convert some Math operations to EmitC call_opaque operations

        This pass converts supported Math ops to `call_opaque` ops targeting libc/libm
        functions. Unlike convert-math-to-funcs pass, converting to `call_opaque` ops
        allows to overload the same function with different argument types.

        Args:
            language_target: Select the language standard target for callees (c99 or cpp11).
        """
        self.add_pass("convert-math-to-emitc", **{"language-target": language_target})
        return self

    def convert_math_to_funcs(
        self, min_width_of_fpowi_exponent: int = None, convert_ctlz: bool = None
    ):
        """Convert Math operations to calls of outlined implementations.

        This pass converts supported Math ops to calls of compiler generated
        functions implementing these operations in software.
        The LLVM dialect is used for LinkonceODR linkage of the generated functions.

        Args:
            min_width_of_fpowi_exponent: Convert FPowI only if the width of its exponent's integer type is greater than or equal to this value
            convert_ctlz: Convert math.ctlz to a software implementation. Enable for targets that do not natively support ctlz.
        """
        self.add_pass(
            "convert-math-to-funcs",
            **{
                "min-width-of-fpowi-exponent": min_width_of_fpowi_exponent,
                "convert-ctlz": convert_ctlz,
            },
        )
        return self

    def convert_math_to_libm(self):
        """Convert Math dialect to libm calls

        This pass converts supported Math ops to libm calls.

        """
        self.add_pass("convert-math-to-libm")
        return self

    def convert_math_to_llvm(self, approximate_log1p: bool = None):
        """Convert Math dialect to LLVM dialect
        Args:
            approximate_log1p: Enable approximation of Log1p.
        """
        self.add_pass(
            "convert-math-to-llvm", **{"approximate-log1p": approximate_log1p}
        )
        return self

    def convert_math_to_rocdl(self, chipset: str = None):
        """Convert Math dialect to ROCDL library calls

        This pass converts supported Math ops to ROCDL library calls.

        The chipset option specifies the target AMDGPU architecture. If the chipset
        is empty, none of the chipset-dependent patterns are added, and the pass
        will not attempt to parse the chipset.

        Args:
            chipset: Chipset that these operations will run on
        """
        self.add_pass("convert-math-to-rocdl", **{"chipset": chipset})
        return self

    def convert_math_to_spirv(self):
        """Convert Math dialect to SPIR-V dialect"""
        self.add_pass("convert-math-to-spirv")
        return self

    def convert_math_to_xevm(self, convert_arith: bool = None):
        """Convert (fast) math operations to native XeVM/SPIRV equivalents

        This pass converts supported math ops marked with the `afn` fastmath flag
        to function calls for OpenCL `native_` math intrinsics: These intrinsics
        are typically mapped directly to native device instructions, often resulting
        in better performance. However, the precision/error of these intrinsics
        are implementation-defined, and thus math ops are only converted when they
        have the `afn` fastmath flag enabled.

        Args:
            convert_arith: Convert supported Arith ops (e.g. arith.divf) as well.
        """
        self.add_pass("convert-math-to-xevm", **{"convert-arith": convert_arith})
        return self

    def convert_memref_to_emitc(self, lower_to_cpp: bool = None):
        """Convert MemRef dialect to EmitC dialect
        Args:
            lower_to_cpp: Target C++ (true) instead of C (false)
        """
        self.add_pass("convert-memref-to-emitc", **{"lower-to-cpp": lower_to_cpp})
        return self

    def convert_memref_to_spirv(
        self, bool_num_bits: int = None, use_64bit_index: bool = None
    ):
        """Convert MemRef dialect to SPIR-V dialect
        Args:
            bool_num_bits: The number of bits to store a boolean value
            use_64bit_index: Use 64-bit integers to convert index types
        """
        self.add_pass(
            "convert-memref-to-spirv",
            **{"bool-num-bits": bool_num_bits, "use-64bit-index": use_64bit_index},
        )
        return self

    def convert_nvgpu_to_nvvm(self):
        """Convert NVGPU dialect to NVVM dialect

        This pass converts supported NVGPU ops to NVVM dialect intrinsics.

        """
        self.add_pass("convert-nvgpu-to-nvvm")
        return self

    def convert_nvvm_to_llvm(self):
        """Convert NVVM to PTX with Inline Assembly in LLVM dialect

        This pass generates PTX instructions using inline assembly for NVVM
        operations implements `BasicPtxBuilderInterface`.

        """
        self.add_pass("convert-nvvm-to-llvm")
        return self

    def convert_openacc_to_scf(self):
        """Convert the OpenACC ops to OpenACC with SCF dialect"""
        self.add_pass("convert-openacc-to-scf")
        return self

    def convert_openmp_to_llvm(self):
        """Convert the OpenMP ops to OpenMP ops with LLVM dialect"""
        self.add_pass("convert-openmp-to-llvm")
        return self

    def convert_parallel_loops_to_gpu(self):
        """Convert mapped scf.parallel ops to gpu launch operations

        Creates a pass that converts scf.parallel operations into a gpu.launch
        operation. The mapping of loop dimensions to launch dimensions is derived
        from mapping attributes. See ParallelToGpuLaunchLowering::matchAndRewrite
        for a description of the used attributes.

        """
        self.add_pass("convert-parallel-loops-to-gpu")
        return self

    def convert_pdl_to_pdl_interp(self):
        """Convert PDL ops to PDL interpreter ops"""
        self.add_pass("convert-pdl-to-pdl-interp")
        return self

    def convert_scf_to_cf(self):
        """Convert SCF dialect to ControlFlow dialect, replacing structured control flow with a CFG"""
        self.add_pass("convert-scf-to-cf")
        return self

    def convert_scf_to_emitc(self):
        """Convert SCF dialect to EmitC dialect, maintaining structured control flow"""
        self.add_pass("convert-scf-to-emitc")
        return self

    def convert_scf_to_openmp(self, num_threads: int = None):
        """Convert SCF parallel loop to OpenMP parallel + workshare constructs.
        Args:
            num_threads: Number of threads to use
        """
        self.add_pass("convert-scf-to-openmp", **{"num-threads": num_threads})
        return self

    def convert_scf_to_spirv(self):
        """Convert SCF dialect to SPIR-V dialect.

        Converts SCF ops into SPIR-V structured control flow ops.
        SPIR-V structured control flow ops do not support yielding values.
        So for SCF ops yielding values, SPIR-V variables are created for
        holding the values and load/store operations are emitted for updating
        them.

        """
        self.add_pass("convert-scf-to-spirv")
        return self

    def convert_shape_constraints(self):
        """Convert shape constraint operations to the standard dialect

        This pass eliminates shape constraints from the program, converting them to
        eager (side-effecting) error handling code.

        This pass is separate from the regular convert-shape-to-standard, despite
        converting between the same dialects, because converting shape constraints
        can happen at a different part of the program than general shape
        computation lowering.

        """
        self.add_pass("convert-shape-constraints")
        return self

    def convert_shape_to_std(self):
        """Convert operations from the shape dialect into the standard dialect"""
        self.add_pass("convert-shape-to-std")
        return self

    def convert_shard_to_mpi(self):
        """Convert Shard dialect to MPI dialect.

        This pass converts communication operations from the Shard dialect to the
        MPI dialect.
        If it finds the DLTI attribute "MPI:comm_world-rank" on the module it will
        use that integer value instead of calling MPI_Comm_rank. This allows
        optimizations like constant shape propagation and fusion because
        shard/partition sizes depend on the rank.

        """
        self.add_pass("convert-shard-to-mpi")
        return self

    def convert_spirv_to_llvm(self, client_api: "::mlir::spirv::ClientAPI" = None):
        """Convert SPIR-V dialect to LLVM dialect

        See https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/
        for more details.

        Args:
            client_api: Derive StorageClass to address space mapping from the client API
        """
        self.add_pass("convert-spirv-to-llvm", **{"client-api": client_api})
        return self

    def convert_tensor_to_linalg(self):
        """Convert some Tensor dialect ops to Linalg dialect"""
        self.add_pass("convert-tensor-to-linalg")
        return self

    def convert_tensor_to_spirv(
        self,
        emulate_lt_32_bit_scalar_types: bool = None,
        emulate_unsupported_float_types: bool = None,
    ):
        """Convert Tensor dialect to SPIR-V dialect
        Args:
            emulate_lt_32_bit_scalar_types: Emulate narrower scalar types with 32-bit ones if not supported by the target
            emulate_unsupported_float_types: Emulate unsupported float types by representing them with integer types of same bit width
        """
        self.add_pass(
            "convert-tensor-to-spirv",
            **{
                "emulate-lt-32-bit-scalar-types": emulate_lt_32_bit_scalar_types,
                "emulate-unsupported-float-types": emulate_unsupported_float_types,
            },
        )
        return self

    def convert_to_emitc(self, filter_dialects: List[str] = None):
        """Convert to EmitC dialect via dialect interfaces

        This is a generic pass to convert to the EmitC dialect, it uses the
        `ConvertToEmitCPatternInterface` dialect interface to delegate to dialects
        the injection of conversion patterns.

        Args:
            filter_dialects: Test conversion patterns of only the specified dialects
        """
        self.add_pass("convert-to-emitc", **{"filter-dialects": filter_dialects})
        return self

    def convert_to_llvm(
        self,
        filter_dialects: List[str] = None,
        dynamic: bool = None,
        allow_pattern_rollback: bool = None,
    ):
        """Convert to LLVM via dialect interfaces found in the input IR

        This is a generic pass to convert to LLVM, it uses the
        `ConvertToLLVMPatternInterface` dialect interface to delegate to dialects
        the injection of conversion patterns.

        If `dynamic` is set to `true`, the pass will look for
        `ConvertToLLVMAttrInterface` attributes and use them to further configure
        the conversion process. This option also uses the `DataLayoutAnalysis`
        analysis to configure the type converter. Enabling this option incurs in
        extra overhead.

        Args:
            filter_dialects: Test conversion patterns of only the specified dialects
            dynamic: Use op conversion attributes to configure the conversion
            allow_pattern_rollback: Experimental performance flag to disallow pattern rollback
        """
        self.add_pass(
            "convert-to-llvm",
            **{
                "filter-dialects": filter_dialects,
                "dynamic": dynamic,
                "allow-pattern-rollback": allow_pattern_rollback,
            },
        )
        return self

    def convert_ub_to_llvm(self, index_bitwidth: int = None):
        """Convert UB dialect to LLVM dialect

        This pass converts supported UB ops to LLVM dialect instructions.

        Args:
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
        """
        self.add_pass("convert-ub-to-llvm", **{"index-bitwidth": index_bitwidth})
        return self

    def convert_ub_to_spirv(self):
        """Convert UB dialect to SPIR-V dialect

        This pass converts supported UB ops to SPIR-V dialect ops.

        """
        self.add_pass("convert-ub-to-spirv")
        return self

    def convert_vector_to_amx(self):
        """Lower the operations from the vector dialect into the AMX dialect"""
        self.add_pass("convert-vector-to-amx")
        return self

    def convert_vector_to_arm_sme(self):
        """Lower the operations from the vector dialect into the ArmSME dialect

        Pass that converts vector dialect operations into equivalent ArmSME dialect
        operations.

        """
        self.add_pass("convert-vector-to-arm-sme")
        return self

    def convert_vector_to_gpu(self, use_nvgpu: bool = None):
        """Lower the operations from the vector dialect into the GPU dialect
        Args:
            use_nvgpu: convert to NvGPU ops instead of GPU dialect ops
        """
        self.add_pass("convert-vector-to-gpu", **{"use-nvgpu": use_nvgpu})
        return self

    def convert_vector_to_llvm(
        self,
        reassociate_fp_reductions: bool = None,
        force_32bit_vector_indices: bool = None,
        use_vector_alignment: bool = None,
        enable_amx: bool = None,
        enable_arm_neon: bool = None,
        enable_arm_sve: bool = None,
        enable_arm_i8mm: bool = None,
        enable_arm_bf16: bool = None,
        enable_x86vector: bool = None,
        vector_contract_lowering: "vector::VectorContractLowering" = None,
        vector_transpose_lowering: "vector::VectorTransposeLowering" = None,
    ):
        """Lower the operations from the vector dialect into the LLVM dialect


        Convert operations from the vector dialect into the LLVM IR dialect
        operations. The lowering pass provides several options to control
        the kinds of optimizations that are allowed. It also provides options
        that enable the use of one or more architectural-specific dialects
        (AMX, X86Vector, ArmNeon, ArmSVE, etc.) in combination with the
        architectural-neutral vector dialect lowering.


        Args:
            reassociate_fp_reductions: Allows llvm to reassociate floating-point reductions for speed
            force_32bit_vector_indices: Allows compiler to assume vector indices fit in 32-bit if that yields faster code
            use_vector_alignment: Use the preferred alignment of a vector type in load/store operations instead of the alignment of the element type of the memref. This flag is intended for use with hardware which requiresvector alignment, or in application contexts where it is known all vector access are naturally aligned. If operations have an alignment attribute set, the alignment attribute takes priority over this option
            enable_amx: Enables the use of AMX dialect while lowering the vector dialect.
            enable_arm_neon: Enables the use of ArmNeon dialect while lowering the vector dialect.
            enable_arm_sve: Enables the use of ArmSVE dialect while lowering the vector dialect.
            enable_arm_i8mm: Enables the use of Arm FEAT_I8MM instructions while lowering the vector dialect.
            enable_arm_bf16: Enables the use of Arm FEAT_BF16 instructions while lowering the vector dialect.
            enable_x86vector: Enables the use of X86Vector dialect while lowering the vector dialect.
            vector_contract_lowering: control the lowering of `vector.contract` operations.
            vector_transpose_lowering: control the lowering of `vector.transpose` operations.
        """
        self.add_pass(
            "convert-vector-to-llvm",
            **{
                "reassociate-fp-reductions": reassociate_fp_reductions,
                "force-32bit-vector-indices": force_32bit_vector_indices,
                "use-vector-alignment": use_vector_alignment,
                "enable-amx": enable_amx,
                "enable-arm-neon": enable_arm_neon,
                "enable-arm-sve": enable_arm_sve,
                "enable-arm-i8mm": enable_arm_i8mm,
                "enable-arm-bf16": enable_arm_bf16,
                "enable-x86vector": enable_x86vector,
                "vector-contract-lowering": vector_contract_lowering,
                "vector-transpose-lowering": vector_transpose_lowering,
            },
        )
        return self

    def convert_vector_to_scf(
        self,
        full_unroll: bool = None,
        target_rank: int = None,
        lower_tensors: bool = None,
        lower_scalable: bool = None,
    ):
        """Lower the operations from the vector dialect into the SCF dialect
        Args:
            full_unroll: Perform full unrolling when converting vector transfers to SCF
            target_rank: Target vector rank to which transfer ops should be lowered
            lower_tensors: Lower transfer ops that operate on tensors
            lower_scalable: Add scalable vector specific lowerings (that introduce loops)
        """
        self.add_pass(
            "convert-vector-to-scf",
            **{
                "full-unroll": full_unroll,
                "target-rank": target_rank,
                "lower-tensors": lower_tensors,
                "lower-scalable": lower_scalable,
            },
        )
        return self

    def convert_vector_to_spirv(self):
        """Convert Vector dialect to SPIR-V dialect"""
        self.add_pass("convert-vector-to-spirv")
        return self

    def convert_vector_to_xegpu(self):
        """Lower the operations from the vector dialect into the XeGPU dialect"""
        self.add_pass("convert-vector-to-xegpu")
        return self

    def convert_xegpu_to_xevm(self):
        """Convert XeGPU to XeVM dialect"""
        self.add_pass("convert-xegpu-to-xevm")
        return self

    def convert_xevm_to_llvm(self):
        """Convert XeVM to LLVM dialect"""
        self.add_pass("convert-xevm-to-llvm")
        return self

    def cse(self):
        """Eliminate common sub-expressions

        This pass implements a generalized algorithm for common sub-expression
        elimination. This pass relies on information provided by the
        `Memory SideEffect` interface to identify when it is safe to eliminate
        operations. See [Common subexpression elimination](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
        for more general details on this optimization.

        """
        self.add_pass("cse")
        return self

    def decorate_spirv_composite_type_layout(self):
        """Decorate SPIR-V composite type with layout info

        Module pass that converts composite types used by objects in the
        StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant storage
        classes to attatch layout information.
        Right now this pass only supports Vulkan layout rules.

        """
        self.add_pass("decorate-spirv-composite-type-layout")
        return self

    def drop_equivalent_buffer_results(self):
        """Remove MemRef return values that are equivalent to a bbArg

        This pass removes MemRef return values from functions if they are equivalent
        to a function bbArg. In that case, the return value is redundant and the
        respective CallOp operand can be used at the call site.

        Note: If a bbArg buffer is not returned directly but casted to beforehand,
        the buffer is still considered equivalent.

        """
        self.add_pass("drop-equivalent-buffer-results")
        return self

    def duplicate_function_elimination(self):
        """Deduplicate functions

        Deduplicate functions that are equivalent in all aspects but their symbol
        name. The pass chooses one representative per equivalence class, erases
        the remainder, and updates function calls accordingly.

        """
        self.add_pass("duplicate-function-elimination")
        return self

    def eliminate_empty_tensors(self):
        """Try to eliminate all tensor.empty ops.

        Try to eliminate "tensor.empty" ops inside `op`. This transformation looks
        for subset ops that insert a tensor that originates from a "tensor.empty"
        (as per the reverse use-def chain). Such "tensor.empty" ops are replaced
        with the destination subset.

        E.g.:
        ```
        %0 = tensor.empty() : tensor<10xf32>
        %1 = linalg.fill ... outs(%0 : tensor<10xf32>)
        %2 = tensor.insert_slice %1 into %t ...
        ```

        In the above example, the subset op is "tensor.insert_slice". When tracing
        back the reverse use-def chain of a the source, we end up at a
        "tensor.empty" op. The "tensor.empty" op is replaced with a
        "tensor.extract_slice" op.

        """
        self.add_pass("eliminate-empty-tensors")
        return self

    def empty_tensor_to_alloc_tensor(self):
        """Replace all empty ops by alloc_tensor ops.

        tensor.empty ops return a tensor of unspecified contents who's only purpose
        is to carry the tensor shape. This pass converts such ops to
        bufferization.alloc_tensor ops, which bufferize to buffer allocations.

        """
        self.add_pass("empty-tensor-to-alloc-tensor")
        return self

    def enable_arm_streaming(
        self,
        streaming_mode: "mlir::arm_sme::ArmStreamingMode" = None,
        za_mode: "mlir::arm_sme::ArmZaMode" = None,
        if_required_by_ops: bool = None,
        if_scalable_and_supported: bool = None,
    ):
        """Enable Armv9 Streaming SVE mode

        Enables the Armv9 Streaming SVE mode [1] for func.func ops by annotating
        them with attributes. See options for more details.

        [1] https://developer.arm.com/documentation/ddi0616/aa

        Args:
            streaming_mode: Select how streaming-mode is managed at the function-level.
            za_mode: Select how ZA-storage is managed at the function-level.
            if_required_by_ops: Only apply the selected streaming/ZA modes if the function contains ops that implement the ArmSMETileOpInterface.
            if_scalable_and_supported: Only apply the selected streaming/ZA modes if the function contains supported scalable vector operations.
        """
        self.add_pass(
            "enable-arm-streaming",
            **{
                "streaming-mode": streaming_mode,
                "za-mode": za_mode,
                "if-required-by-ops": if_required_by_ops,
                "if-scalable-and-supported": if_scalable_and_supported,
            },
        )
        return self

    def ensure_debug_info_scope_on_llvm_func(
        self, emission_kind: "mlir::LLVM::DIEmissionKind" = None
    ):
        """Materialize LLVM debug info subprogram attribute on every LLVMFuncOp

        Having a debug info subprogram attribute on a function is required for
        emitting line tables from MLIR FileLocCol locations.

        This is not intended to be a proper replacement for frontends to emit
        complete debug informations, however it is a convenient way to get line
        tables for debugging purposes. This allow to step trough in a debugger
        line-by-line or get a backtrace with line numbers.

        Args:
            emission_kind: Emission kind to generate debug info.
        """
        self.add_pass(
            "ensure-debug-info-scope-on-llvm-func", **{"emission-kind": emission_kind}
        )
        return self

    def expand_realloc(self, emit_deallocs: bool = None):
        """Expand memref.realloc operations into its components

        The `memref.realloc` operation performs a conditional allocation and copy to
        increase the size of a buffer if necessary. This pass converts a `realloc`
        operation into this sequence of simpler operations such that other passes
        at a later stage in the compilation pipeline do not have to consider the
        `realloc` operation anymore (e.g., the buffer deallocation pass and the
        conversion pass to LLVM).

        Example of an expansion:
        ```mlir
        %realloc = memref.realloc %alloc (%size) : memref<?xf32> to memref<?xf32>
        ```
        is expanded to
        ```mlir
        %c0 = arith.constant 0 : index
        %dim = memref.dim %alloc, %c0 : memref<?xf32>
        %is_old_smaller = arith.cmpi ult, %dim, %arg1
        %realloc = scf.if %is_old_smaller -> (memref<?xf32>) {
          %new_alloc = memref.alloc(%size) : memref<?xf32>
          %subview = memref.subview %new_alloc[0] [%dim] [1]
          memref.copy %alloc, %subview
          memref.dealloc %alloc
          scf.yield %alloc_0 : memref<?xf32>
        } else {
          %reinterpret_cast = memref.reinterpret_cast %alloc to
            offset: [0], sizes: [%size], strides: [1]
          scf.yield %reinterpret_cast : memref<?xf32>
        }
        ```

        Args:
            emit_deallocs: Emit deallocation operations for the original MemRef
        """
        self.add_pass("expand-realloc", **{"emit-deallocs": emit_deallocs})
        return self

    def expand_strided_metadata(self):
        """Expand memref operations into easier to analyze constructs

        The pass expands memref operations that modify the metadata of a memref
        (sizes, offset, strides) into a sequence of easier to analyze constructs.
        In particular, this pass transforms operations into explicit sequence of
        operations that model the effect of this operation on the different metadata.
        This pass uses affine constructs to materialize these effects.

        Supported ops include:

        - `memref.collapse_shape`
        - `memref.expand_shape`
        - `memref.extract_aligned_pointer_as_index`
        - `memref.extract_strided_metadata`
        - `memref.subview`

        """
        self.add_pass("expand-strided-metadata")
        return self

    def finalize_memref_to_llvm(
        self,
        use_aligned_alloc: bool = None,
        index_bitwidth: int = None,
        use_generic_functions: bool = None,
    ):
        """Finalize MemRef dialect to LLVM dialect conversion

        Finalize the conversion of the operations from the MemRef
        dialect to the LLVM dialect.
        This conversion will not convert some complex MemRef
        operations. Make sure to run `expand-strided-metadata`
        beforehand for these.

        Args:
            use_aligned_alloc: Use aligned_alloc in place of malloc for heap allocations
            index_bitwidth: Bitwidth of the index type, 0 to use size of machine word
            use_generic_functions: Use generic allocation and deallocation functions instead of the classic 'malloc', 'aligned_alloc' and 'free' functions
        """
        self.add_pass(
            "finalize-memref-to-llvm",
            **{
                "use-aligned-alloc": use_aligned_alloc,
                "index-bitwidth": index_bitwidth,
                "use-generic-functions": use_generic_functions,
            },
        )
        return self

    def flatten_memref(self):
        """Flatten a multiple dimensional memref to 1-dimensional"""
        self.add_pass("flatten-memref")
        return self

    def fold_memref_alias_ops(self):
        """Fold memref alias ops into consumer load/store ops

        The pass folds loading/storing from/to memref aliasing ops to loading/storing
        from/to the original memref.

        """
        self.add_pass("fold-memref-alias-ops")
        return self

    def fold_tensor_subset_ops(self):
        """Fold tensor subset ops into producer/consumer ops

        The pass folds tensor subset ops into producer/consumer ops.

        At the moment, the following foldings occur when possible:
          - tensor.extract_slice into vector.transfer_read
          - vector.transfer_write into tensor.insert_slice


        """
        self.add_pass("fold-tensor-subset-ops")
        return self

    def form_expressions(self):
        """Form C-style expressions from C-operator ops

        The pass wraps emitc ops modelling C operators in emitc.expression ops and
        then folds single-use expressions into their users where possible.

        """
        self.add_pass("form-expressions")
        return self

    def generate_runtime_verification(self, verbose_level: int = None):
        """Generate additional runtime op verification checks

        This pass generates op-specific runtime checks using the
        `RuntimeVerifiableOpInterface`. It can be run for debugging purposes after
        passes that are suspected to introduce faulty IR.

        Args:
            verbose_level: Verbosity level for runtime verification messages: 0 = Minimum (only source location), 1 = Detailed (include full operation details, names, types, shapes, etc.)
        """
        self.add_pass(
            "generate-runtime-verification", **{"verbose-level": verbose_level}
        )
        return self

    def gpu_async_region(self):
        """Make GPU ops async"""
        self.add_pass("gpu-async-region")
        return self

    def gpu_decompose_memrefs(self):
        """Decomposes memref index computation into explicit ops.

        This pass decomposes memref index computation into explicit computations on
        sizes/strides, obtained from `memref.extract_memref_metadata` which it tries
        to place outside of `gpu.launch` body. Memrefs are then reconstructed using
        `memref.reinterpret_cast`.
        This is needed for as some targets (SPIR-V) lower memrefs to bare pointers
        and sizes/strides for dynamically-sized memrefs are not available inside
        `gpu.launch`.

        """
        self.add_pass("gpu-decompose-memrefs")
        return self

    def gpu_eliminate_barriers(self):
        """Erase unnecessary barriers

        Barrier elimination pass. If a barrier does not enforce any conflicting
        pair of memory effects, including a pair that is enforced by another
        barrier, it is unnecessary and can be removed. Adapted from
        "High-Performance GPU-to-CPU Transpilation and Optimization via High-Level
        Parallel Constructs" by Moses, Ivanov, Domke, Endo, Doerfert, and Zinenko in
        PPoPP 2023 and implementation in Polygeist.

        """
        self.add_pass("gpu-eliminate-barriers")
        return self

    def gpu_kernel_outlining(self, data_layout_str: str = None):
        """Outline gpu.launch bodies to kernel functions
        Args:
            data_layout_str: String description of the data layout
        """
        self.add_pass("gpu-kernel-outlining", **{"data-layout-str": data_layout_str})
        return self

    def gpu_launch_sink_index_computations(self):
        """Sink index computations into gpu.launch body"""
        self.add_pass("gpu-launch-sink-index-computations")
        return self

    def gpu_map_parallel_loops(self, mapping_policy: str = None):
        """Greedily maps loops to GPU hardware dimensions.

        Maps the parallel loops found in the given function to workgroups. The first
        loop encountered will be mapped to the global workgroup and the second loop
        encountered to the local workgroup. Within each mapping, the first three
        dimensions are mapped to x/y/z hardware ids and all following dimensions are
        mapped to sequential loops.

        Ordering of the loop mapping against the different dimensions is controlled
        by the `mapping-policy` option.
        Two policies are supported:
           1. `outermost-first` (default): the outermost loop maps to X, then Y
              and finally Z.
           2. `innermost-first`: the innermost loop maps to X, then Y and finally Z.

        Args:
            mapping_policy: Policy outlining how to assign loops to GPU dimensions.Supported values are `outermost-first` and `innermost-first`.
        """
        self.add_pass("gpu-map-parallel-loops", **{"mapping-policy": mapping_policy})
        return self

    def gpu_module_to_binary(
        self,
        toolkit: str = None,
        l: List[str] = None,
        opts: str = None,
        format: str = None,
        section: str = None,
    ):
        """Transforms a GPU module into a GPU binary.

        This pass searches for all nested GPU modules and serializes the module
        using the target attributes attached to the module, producing a GPU binary
        with an object for every target.

        The `format` argument can have the following values:
        1. `offloading`, `llvm`: produces an offloading representation.
        2. `assembly`, `isa`: produces assembly code.
        3. `binary`, `bin`: produces binaries.
        4. `fatbinary`, `fatbin`: produces fatbinaries.

        Args:
            toolkit: Toolkit path.
            l: Extra files to link to.
            opts: Command line options to pass to the tools.
            format: The target representation of the compilation process.
            section: ELF section where binary is to be located.
        """
        self.add_pass(
            "gpu-module-to-binary",
            **{
                "toolkit": toolkit,
                "l": l,
                "opts": opts,
                "format": format,
                "section": section,
            },
        )
        return self

    def gpu_to_llvm(
        self,
        use_bare_pointers_for_host: bool = None,
        use_bare_pointers_for_kernels: bool = None,
        intersperse_sizes_for_kernels: bool = None,
    ):
        """Convert GPU dialect to LLVM dialect with GPU runtime calls

        Creates a pass to convert a GPU operations into a sequence of GPU runtime
        calls.

        This pass does not generate code to call GPU runtime APIs directly but
        instead uses a small wrapper library that exports a stable and conveniently
        typed ABI on top of GPU runtimes such as CUDA or ROCm (HIP).

        Args:
            use_bare_pointers_for_host: Use bare pointers to pass memref arguments to host functions. All memrefs must have static shape.
            use_bare_pointers_for_kernels: Use bare pointers to pass memref arguments to kernels. The kernel must use the same setting for this option.
            intersperse_sizes_for_kernels: Inserts a size_t argument following each memref argument, containing the static size in bytes of the buffer. Incompatible arguments are rejected. This is intended for use by the Vulkan runtime with the kernel bare pointer calling convention, to enable dynamic binding of buffers as arguments without static type info.
        """
        self.add_pass(
            "gpu-to-llvm",
            **{
                "use-bare-pointers-for-host": use_bare_pointers_for_host,
                "use-bare-pointers-for-kernels": use_bare_pointers_for_kernels,
                "intersperse-sizes-for-kernels": intersperse_sizes_for_kernels,
            },
        )
        return self

    def inline(
        self,
        default_pipeline: str = None,
        op_pipelines: List["OpPassManager"] = None,
        max_iterations: int = None,
        inlining_threshold: int = None,
    ):
        """Inline function calls
        Args:
            default_pipeline: The optimizer pipeline used for callables that do not have a dedicated optimizer pipeline in opPipelineList
            op_pipelines: Callable operation specific optimizer pipelines (in the form of `dialect.op(pipeline)`)
            max_iterations: Maximum number of iterations when inlining within an SCC
            inlining_threshold: If the ratio between the number of the operations in the callee and the number of the operations in the caller exceeds this value (in percentage), then the callee is not inlined even if it is legal to inline it
        """
        self.add_pass(
            "inline",
            **{
                "default-pipeline": default_pipeline,
                "op-pipelines": op_pipelines,
                "max-iterations": max_iterations,
                "inlining-threshold": inlining_threshold,
            },
        )
        return self

    def int_range_optimizations(self):
        """Do optimizations based on integer range analysis

        This pass runs integer range analysis and apllies optimizations based on its
        results. It replaces operations with known-constant results with said constants,
        rewrites `(0 <= %x < D) mod D` to `%x`.

        """
        self.add_pass("int-range-optimizations")
        return self

    def lift_cf_to_scf(self):
        """Lift ControlFlow dialect to SCF dialect

        Lifts ControlFlow operations to SCF dialect operations.

        This pass is prefixed with "lift" instead of "convert" as it is not always
        guaranteed to replace all ControlFlow ops.
        If a region contains only a single kind of return-like operation, all
        ControlFlow operations will be replaced successfully.
        Otherwise a single ControlFlow switch branching to one block per return-like
        operation kind remains.

        This pass may need to create unreachable terminators in case of infinite
        loops, which is only supported for 'func.func' for now. If you potentially
        have infinite loops inside CFG regions not belonging to 'func.func',
        consider using `transformCFGToSCF` function directly with corresponding
        `CFGToSCFInterface::createUnreachableTerminator` implementation.

        """
        self.add_pass("lift-cf-to-scf")
        return self

    def linalg_block_pack_matmul(
        self,
        block_factors: List[int] = None,
        allow_padding: bool = None,
        mnk_padded_multiples: List[int] = None,
        mnk_order: List[int] = None,
        lhs_transpose_outer_blocks: bool = None,
        lhs_transpose_inner_blocks: bool = None,
        rhs_transpose_outer_blocks: bool = None,
        rhs_transpose_inner_blocks: bool = None,
    ):
        """Convert linalg matmul ops to block layout and back

        Pack a matmul operation into blocked layout with two levels of subdivision:
        - major 2D blocks - outer dimensions, consist of minor blocks
        - minor 2D blocks - inner dimensions, consist of scalar elements

        A 2D matmul MxNxK gets reshaped into blocked 4D representation
        as: [MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][nb][kb]
        where the (MB, NB, KB) dimensions represent the major blocks,
        and the (mb, nb, kb) are the minor blocks of their respective
        original 2D dimensions (M, N, K).

        Depending on the initial operands' data layout and the specified
        packing options, the major blocks dimensions might get transposed
        e.g., [MB][KB] -> [KB][MB]. The minor blocks can also be transposed
        e.g., [mb][kb] -> [kb][mb].
        Any present batch dimensions remain unchanged.
        The final result is unpacked back to the original shape.

        For example, given a matmul operation:
        ```mlir
          %res = linalg.matmul ins(%A, %B) outs(%C)
        ```
        the default transformation result can be represented as:
        ```mlir
          %A_packed = pack %A : 2D <MxK> -> 4D <MBxKBxmbxkb>
          %B_packed = pack %B : 2D <KxN> -> 4D <NBxKBxnbxkb>
          %C_packed = pack %C : 2D <MxN> -> 4D <MBxNBxmbxnb>
          %res_packed = linalg.mmt4d ins(%A_packed, %B_packed) outs(%C_packed)
          %res = unpack %res_packed : 4D <MBxNBxmbxnb> -> 2D <MxN>
        ```

        Args:
            block_factors: Block factors (mb, nb, kb) for relayout
            allow_padding: Allow packing padding
            mnk_padded_multiples: Next multiples of the packing sizes
            mnk_order: Permutation of matmul (M, N, K) dimensions order
            lhs_transpose_outer_blocks: Transpose LHS outer block layout [MB][KB] -> [KB][MB]
            lhs_transpose_inner_blocks: Transpose LHS inner block layout [mb][kb] -> [kb][mb]
            rhs_transpose_outer_blocks: Transpose RHS outer block layout [KB][NB] -> [NB][KB]
            rhs_transpose_inner_blocks: Transpose RHS inner block layout [kb][nb] -> [nb][kb]
        """
        self.add_pass(
            "linalg-block-pack-matmul",
            **{
                "block-factors": block_factors,
                "allow-padding": allow_padding,
                "mnk-padded-multiples": mnk_padded_multiples,
                "mnk-order": mnk_order,
                "lhs-transpose-outer-blocks": lhs_transpose_outer_blocks,
                "lhs-transpose-inner-blocks": lhs_transpose_inner_blocks,
                "rhs-transpose-outer-blocks": rhs_transpose_outer_blocks,
                "rhs-transpose-inner-blocks": rhs_transpose_inner_blocks,
            },
        )
        return self

    def linalg_detensorize(self, aggressive_mode: bool = None):
        """Detensorize linalg ops

        Detensoring is the process through which a tensor value is converted to one
        or potentially more primitive value(s). During this process, operations with
        such detensored operands are also converted to an equivalent form that works
        on primitives.

        The detensoring process is driven by linalg-on-tensor ops. In particular, a
        linalg-on-tensor op is checked to see whether *all* its operands can be
        detensored. If so, those operands are converted to their primitive
        counterparts and the linalg op is replaced by an equivalent op that takes
        those new primitive values as operands. Therefore, detensoring an op can be
        divided into 2 main logical phases:

        1. Detect/match an op that can be detensored.
        2. Detensor the operands of the op and replace it with a primitive
           equivalent.

        In addition to detensoring individual ops, this pass detensors internal
        control flow inside a function. All blocks except for the entry block are
        detensored by converting their arguments whenever possible.

        This can be run on any FunctionOpInterface op and must not be
        run on others. This is because it performs specific legalization of the
        blocks that make up the body, which it assumes has is a FunctionOpInterface.

        Args:
            aggressive_mode: Detensorize all ops that qualify for detensoring along with branch operands and basic-block arguments.
        """
        self.add_pass("linalg-detensorize", **{"aggressive-mode": aggressive_mode})
        return self

    def linalg_fold_into_elementwise(self):
        """Fold transform, broadcast and other ops into elementwise"""
        self.add_pass("linalg-fold-into-elementwise")
        return self

    def linalg_fold_unit_extent_dims(self, use_rank_reducing_slices: bool = None):
        """Remove unit-extent dimension in Linalg ops on tensors
        Args:
            use_rank_reducing_slices: Generate rank-reducing slices instead of reassociative reshapes
        """
        self.add_pass(
            "linalg-fold-unit-extent-dims",
            **{"use-rank-reducing-slices": use_rank_reducing_slices},
        )
        return self

    def linalg_fuse_elementwise_ops(self):
        """Fuse elementwise operations on tensors"""
        self.add_pass("linalg-fuse-elementwise-ops")
        return self

    def linalg_generalize_named_ops(self):
        """Convert named ops into generic ops"""
        self.add_pass("linalg-generalize-named-ops")
        return self

    def linalg_inline_scalar_operands(self):
        """Inline scalar operands into linalg generic ops"""
        self.add_pass("linalg-inline-scalar-operands")
        return self

    def linalg_morph_ops(
        self,
        named_to_category: bool = None,
        category_to_generic: bool = None,
        named_to_generic: bool = None,
        generic_to_named: bool = None,
    ):
        """Convert linalg ops between forms

        Convert a linalg op from one representation to another equivalent.
        For example, a linalg named op `linalg.add` can also be written as an
        category op `linalg.elementwise`, and can also be re-written as
        a `linalg.generic`, giving the morphism:

          named-op <--> category_op (elementwise, contraction, ..) <--> generic

        Note that the set of `linalg.generic` subsumes named and category ops
        and therefore not all `linalg.genric` can be converted to  named or
        category op. Similarly, catgory ops subsume named ops.

        Note:
         Legacy converters:
         `--linalg-generalize-named-ops` is the path `named-op --> generic-op`
         `--linalg-specialize-generic-ops` is the path `named-op <-- generic-op`

        Args:
            named_to_category: convert named ops to category op e.g. `linalg.elementwise`
            category_to_generic: convert category ops e.g. `linalg.elementwise` to `linalg.generic`
            named_to_generic: convert named ops e.g. `linalg.add` to `linalg.generic`
            generic_to_named: convert linalg.generic to equivalent named ops
        """
        self.add_pass(
            "linalg-morph-ops",
            **{
                "named-to-category": named_to_category,
                "category-to-generic": category_to_generic,
                "named-to-generic": named_to_generic,
                "generic-to-named": generic_to_named,
            },
        )
        return self

    def linalg_specialize_generic_ops(self):
        """Convert generic ops back to named ops"""
        self.add_pass("linalg-specialize-generic-ops")
        return self

    def llvm_add_comdats(self):
        """Add comdats to linkonce and linkonce_odr functions

        Add an any COMDAT to every linkonce and linkonce_odr function.
        This is necessary on Windows to link these functions as the system
        linker won't link weak symbols without a COMDAT. It also provides better
        behavior than standard weak symbols on ELF-based platforms.
        This pass will still add COMDATs on platforms that do not support them,
        for example macOS, so should only be run when the target platform supports
        COMDATs.

        """
        self.add_pass("llvm-add-comdats")
        return self

    def llvm_legalize_for_export(self):
        """Legalize LLVM dialect to be convertible to LLVM IR

        Creates a pass that legalizes the LLVM dialect operations so that they can
        be translated to LLVM IR.

        """
        self.add_pass("llvm-legalize-for-export")
        return self

    def llvm_optimize_for_nvvm_target(self):
        """Optimize NVVM IR"""
        self.add_pass("llvm-optimize-for-nvvm-target")
        return self

    def llvm_request_c_wrappers(self):
        """Request C wrapper emission for all functions

        Annotate every builtin function in the module with the LLVM dialect
        attribute that instructs the conversion to LLVM to emit the C wrapper for
        the function. This pass is expected to be applied immediately before the
        conversion of builtin functions to LLVM to avoid the attribute being
        dropped by other passes.

        """
        self.add_pass("llvm-request-c-wrappers")
        return self

    def llvm_target_to_data_layout(self, initialize_llvm_targets: bool = None):
        """Derive data layout attributes from LLVM target attributes

        Derive a `DataLayoutSpecInterface`-implementing data layout attribute from
        the LLVM-backend target specified by the `TargetAttrInterface`-implementing
        attribute attached to the target op at the name `llvm.target`.

        Args:
            initialize_llvm_targets: Whether to pre-load all available target machines, that LLVM is configured to support, into the TargetRegistry.
        """
        self.add_pass(
            "llvm-target-to-data-layout",
            **{"initialize-llvm-targets": initialize_llvm_targets},
        )
        return self

    def llvm_target_to_target_features(self, initialize_llvm_targets: bool = None):
        """Update attached #llvm.target's features per the described target

        Obtain the TargetMachine specified by the attached #llvm.target's attributes
        and obtain from it the full list of features of the selected target. Updates
        the attached #llvm.target so that its features reflect the full list of
        features.

        Args:
            initialize_llvm_targets: Whether to pre-load all available target machines, that LLVM is configured to support, into the TargetRegistry.
        """
        self.add_pass(
            "llvm-target-to-target-features",
            **{"initialize-llvm-targets": initialize_llvm_targets},
        )
        return self

    def loop_invariant_code_motion(self):
        """Hoist loop invariant instructions outside of the loop"""
        self.add_pass("loop-invariant-code-motion")
        return self

    def loop_invariant_subset_hoisting(self):
        """Hoist loop invariant subset ops outside of the loop"""
        self.add_pass("loop-invariant-subset-hoisting")
        return self

    def lower_affine(self):
        """Lower Affine operations to a combination of Arith and SCF operations


        Convert operations from the affine dialect into operations from the SCF and
        standard dialects.

        `affine.for` operations are converted to `scf.for` operations that are free
        of certain structural restrictions (on their bounds and step). `affine.if`
        is similarly converted to the `scf.if` operation. `affine.apply` operations
        are converted into sequences of primitive arithmetic operations from the
        arith dialect that have the same effect, using operands of the `index`
        type. Consequently, named maps and sets thare are no longer in use may be
        removed from the module.

        For example, `%r = affine.apply affine_map<(d0, d1)[s0] -> (d0 + 2*d1 +
        s0)>(%d0, %d1)[%s0]`
        can be converted into:

        ```mlir
        %d0 = <...>
        %d1 = <...>
        %s0 = <...>
        %0 = arith.constant 2 : index
        %1 = arith.muli %0, %d1
        %2 = arith.addi %d0, %1
        %r = arith.addi %2, %s0
        ```

        #### Input invariant

        -   no `Tensor` types;

        These restrictions may be lifted in the future.

        #### Output IR

        Functions with `affine.for` and `affine.if` operations eliminated. These
        functions may contain operations from the Standard dialect in addition to
        those already present before the pass.

        #### Invariants

        -   Functions without a body are not modified.
        -   The semantics of the other functions is preserved.
        -   Individual operations other than those mentioned above are not modified
            if they do not depend on the loop iterator value or on the result of
            `affine.apply`.

        """
        self.add_pass("lower-affine")
        return self

    def lower_host_to_llvm(self):
        """Lowers the host module code and `gpu.launch_func` to LLVM

        Creates a pass to emulate `gpu.launch_func` call in LLVM dialect and lower
        the host module code to LLVM.

        This transformation creates a sequence of global variables that are later
        linked to the variables in the kernel module, and a series of copies to/from
        them to emulate the memory transfer from the host or to the device sides. It
        also converts the remaining Arithmetic, Func, and MemRef dialects into LLVM
        dialect, emitting C wrappers.

        """
        self.add_pass("lower-host-to-llvm")
        return self

    def lower_quant_ops(self):
        """Lower quant.dcast and quant.qcast ops

        Lower quantization (`quant.qcast`) and dequantization (`quant.dcast`) ops
        into other core dialects.

        The lowering process generates storage type casts in the form of
        `quant.scast` ops to act as an interface between the original quantized
        types of operands and results and their corresponding storage types used in
        the generated arithmetic computations.

        """
        self.add_pass("lower-quant-ops")
        return self

    def lower_sparse_foreach_to_scf(self):
        """Decompose a complex sparse operation into multiple stages

        A pass that lowers sparse_tensor.foreach operation to scf dialect.

        """
        self.add_pass("lower-sparse-foreach-to-scf")
        return self

    def lower_sparse_iteration_to_scf(self):
        """lower sparse_tensor.iterate/coiterate into scf loops

        This pass lowers `sparse_tensor.iterate` operations into `scf.for/while` operations.
        The pass is not yet stabilized.

        """
        self.add_pass("lower-sparse-iteration-to-scf")
        return self

    def lower_sparse_ops_to_foreach(
        self, enable_runtime_library: bool = None, enable_convert: bool = None
    ):
        """Applies sparse tensor rewriting rules after sparsification

        A pass that lowers high-level sparse operations to sparse_tensor.foreach.

        Args:
            enable_runtime_library: Enable runtime library for manipulating sparse tensors
            enable_convert: Enable rewriting rules for the convert operator
        """
        self.add_pass(
            "lower-sparse-ops-to-foreach",
            **{
                "enable-runtime-library": enable_runtime_library,
                "enable-convert": enable_convert,
            },
        )
        return self

    def lower_vector_mask(self):
        """Lower 'vector.mask' operations"""
        self.add_pass("lower-vector-mask")
        return self

    def lower_vector_multi_reduction(
        self, lowering_strategy: "mlir::vector::VectorMultiReductionLowering" = None
    ):
        """Lower 'vector.multi_reduction' operations
        Args:
            lowering_strategy: Select the strategy to control how multi_reduction is lowered.
        """
        self.add_pass(
            "lower-vector-multi-reduction", **{"lowering-strategy": lowering_strategy}
        )
        return self

    def lower_vector_to_from_elements_to_shuffle_tree(self):
        """Lower `vector.to_elements` and `vector.from_elements` to a tree of `vector.shuffle` operations"""
        self.add_pass("lower-vector-to-from-elements-to-shuffle-tree")
        return self

    def map_memref_spirv_storage_class(self, client_api: str = None):
        """Map numeric MemRef memory spaces to SPIR-V storage classes
        Args:
            client_api: The client API to use for populating mappings
        """
        self.add_pass("map-memref-spirv-storage-class", **{"client-api": client_api})
        return self

    def math_expand_ops(self, ops: List[str] = None):
        """Expand math operations.

        Expands some math operations into more fundamental operations, allowing them
        to be subsequently lowered through these. For example, hyperbolic functions
        are transformed into their expanded form containing only `exp` functions.

        The `ops` parameter can be used to apply only a subset of all the
        available expansions, these must correspond to the operation mnemonic.
        For example, `ops=sinh,acosh` will expand only `math.sinh` and
        `math.acosh` operations. If the list is empty, then all expansions are
        applied.

        Args:
            ops: Operations to expand.
        """
        self.add_pass("math-expand-ops", **{"ops": ops})
        return self

    def math_extend_to_supported_types(
        self, extra_types: List[str] = None, target_type: str = None
    ):
        """Legalize floating-point math ops on low-precision floats

        On many targets, the math functions are not implemented for floating-point
        types less precise than IEEE single-precision (aka f32), such as half-floats,
        bfloat16, or 8-bit floats.

        This pass explicitly legalizes these math functions by inserting
        `arith.extf` and `arith.truncf` pairs around said op, which preserves
        the original semantics while enabling lowering. The extra supported floating-point
        types for the target are passed as arguments. Types f64 and f32 are implicitly
        supported.

        As an exception, this pass does not legalize `math.fma`, because
        that is an operation frequently implemented at low precisions.

        Args:
            extra_types: MLIR types with arithmetic support on a given target (f64 and f32 are implicitly supported)
            target_type: MLIR type to convert the unsupported source types to
        """
        self.add_pass(
            "math-extend-to-supported-types",
            **{"extra-types": extra_types, "target-type": target_type},
        )
        return self

    def math_sincos_fusion(self):
        """Fuse sin and cos operations.

        Fuse sin and cos operations into a sincos operation.

        """
        self.add_pass("math-sincos-fusion")
        return self

    def math_uplift_to_fma(self):
        """Uplift arith ops to math.fma.

        Uplift sequence of addf and mulf ops to math.fma if fastmath flags allows it.

        """
        self.add_pass("math-uplift-to-fma")
        return self

    def mem2reg(self, region_simplify: bool = None):
        """Promotes memory slots into values.

        This pass removes loads out of and stores into a memory slot, and turns
        them into direct uses of SSA values. This is done generically using the
        `PromotableAllocationOpInterface`, `PromotableOpInterface` and
        `PromotableMemOpInterface` interfaces.

        This pass will attempt to compute which definitions of the content of
        the memory slot reach operations that use the memory slot pointer. It
        will rewire or remove operations that use the slot pointer so they no
        longer use it. If any of this is not possible, the IR will be left
        without mutation.

        This pass only supports unstructured control-flow. Promotion of operations
        within subregions will not happen.

        Args:
            region_simplify: Perform control flow optimizations to the region tree
        """
        self.add_pass("mem2reg", **{"region-simplify": region_simplify})
        return self

    def memref_emulate_wide_int(self, widest_int_supported: int = None):
        """Emulate 2*N-bit integer operations using N-bit operations

        Emulate memref integer operations that use too wide integer types with
        equivalent operations on supported narrow integer types. This is done by
        splitting original integer values into two halves.

        Currently, only power-of-two integer bitwidths are supported.

        Args:
            widest_int_supported: Widest integer type supported by the target
        """
        self.add_pass(
            "memref-emulate-wide-int", **{"widest-int-supported": widest_int_supported}
        )
        return self

    def memref_expand(self):
        """Legalize memref operations to be convertible to LLVM."""
        self.add_pass("memref-expand")
        return self

    def mlprogram_pipeline_globals(self):
        """Optimize `ml_program` global operations for read and store

        `ml_program`'s load and store operations can be optimized for
        write-write or write-read sets of operations. This allows known
        tensors to not be re-read when the value is already known in IR.

        The pass is designed to handle both nested regions and function calls
        safely.

        """
        self.add_pass("mlprogram-pipeline-globals")
        return self

    def normalize_memrefs(self):
        """Normalize memrefs

          This pass transforms memref types with a non-trivial
          [layout map](https://mlir.llvm.org/docs/Dialects/Builtin/#affine-map-layout)
          into memref types with an identity layout map, e.g. (i, j) -> (i, j). This
          pass is inter-procedural, in the sense that it can modify function
          interfaces and call sites that pass memref types. In order to modify
          memref types while preserving the original behavior, users of those
          memref types are also modified to incorporate the resulting layout map.
          For instance, an [AffineLoadOp](https://mlir.llvm.org/docs/Dialects/Affine/#affineload-mliraffineloadop)
          will be updated to compose the layout map with with the affine expression
          contained in the op. Operations marked with the
          [MemRefsNormalizable](https://mlir.llvm.org/docs/Traits/#memrefsnormalizable)
          trait are expected to be normalizable. Supported operations include affine
          operations, memref.alloc, memref.dealloc, and func.return.

          Given an appropriate layout map specified in the code, this transformation
          can express tiled or linearized access to multi-dimensional data
          structures, but will not modify memref types without an explicit layout
          map.

          Currently this pass is limited to only modify
          functions where all memref types can be normalized. If a function
          contains any operations that are not MemRefNormalizable, then the function
          and any functions that call or call it will not be modified.

          Input

          ```mlir
          #tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
          func.func @matmul(%A: memref<16xf64, #tile>,
                       %B: index, %C: memref<16xf64>) -> (memref<16xf64, #tile>) {
            affine.for %arg3 = 0 to 16 {
                  %a = affine.load %A[%arg3] : memref<16xf64, #tile>
                  %p = arith.mulf %a, %a : f64
                  affine.store %p, %A[%arg3] : memref<16xf64, #tile>
            }
            %c = memref.alloc() : memref<16xf64, #tile>
            %d = affine.load %c[0] : memref<16xf64, #tile>
            return %A: memref<16xf64, #tile>
          }
          ```

          Output

          ```mlir
          func.func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>)
            -> memref<4x4xf64> {
            affine.for %arg3 = 0 to 16 {
              %3 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
              %4 = arith.mulf %3, %3 : f64
              affine.store %4, %arg0[%arg3 floordiv 4, %arg3 mod 4]: memref<4x4xf64>
            }
            %0 = memref.alloc() : memref<4x4xf64>
            %1 = affine.apply #map1()
            %2 = affine.load %0[0, 0] : memref<4x4xf64>
            return %arg0 : memref<4x4xf64>
          }
          ```

          Input

          ```
          #linear8 = affine_map<(i, j) -> (i * 8 + j)>
          func.func @linearize(%arg0: memref<8x8xi32, #linear8>,
                          %arg1: memref<8x8xi32, #linear8>,
                          %arg2: memref<8x8xi32, #linear8>) {
            %c8 = arith.constant 8 : index
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            affine.for %arg3 = %c0 to %c8  {
            affine.for %arg4 = %c0 to %c8  {
              affine.for %arg5 = %c0 to %c8 {
                %0 = affine.load %arg0[%arg3, %arg5] : memref<8x8xi32, #linear8>
                %1 = affine.load %arg1[%arg5, %arg4] : memref<8x8xi32, #linear8>
                %2 = affine.load %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
                %3 = arith.muli %0, %1 : i32
                %4 = arith.addi %2, %3 : i32
                affine.store %4, %arg2[%arg3, %arg4] : memref<8x8xi32, #linear8>
              }
            }
            }
            return
          }
          ```

          Output

          ```mlir
          func.func @linearize(%arg0: memref<64xi32>,
                          %arg1: memref<64xi32>,
                          %arg2: memref<64xi32>) {
          %c8 = arith.constant 8 : index
          %c0 = arith.constant 0 : index
          affine.for %arg3 = %c0 to %c8 {
            affine.for %arg4 = %c0 to %c8 {
              affine.for %arg5 = %c0 to %c8 {
                %0 = affine.load %arg0[%arg3 * 8 + %arg5] : memref<64xi32>
                %1 = affine.load %arg1[%arg5 * 8 + %arg4] : memref<64xi32>
                %2 = affine.load %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
                %3 = arith.muli %0, %1 : i32
                %4 = arith.addi %2, %3 : i32
                affine.store %4, %arg2[%arg3 * 8 + %arg4] : memref<64xi32>
              }
            }
          }
          return
        }
        ```

        """
        self.add_pass("normalize-memrefs")
        return self

    def normalize_quant_types(self):
        """Normalize generic quantized types to specific quantized types

        This pass converts generic quantized types in the `quant` dialect to more
        specific types when possible.

        The following conversions are performed:

        1. Sub-channel to per-axis: If the shape of the scales tensor of sub-channel
           quantized type has all but one non-one value, it is converted to a
           per-axis quantized type.

           For example:

           * `!quant.uniform<i8:f32:{0:1}, {{2.0}, {3.0}}>`
              -> `!quant.uniform<i8:f32:0, {2.0, 3.0}>`
           * `tensor<?x?x!quant.uniform<i8:f32:{0:1,1:4}, {{2.0}, {3.0}}>>`
              -> `tensor<?x?x!quant.uniform<i8:f32:0, {2.0, 3.0}>>`

        2. Sub-channel to per-tensor: If a sub-channel quantized type has only
           one scale or zero-point, it is converted to a per-tensor
           quantized type.

           For example:

           * `!quant.uniform<i8:f32:{}, {{2.0}}>`
              -> `!quant.uniform<i8:f32, 2.0>`
           * `tensor<?x?x!quant.uniform<i8:f32:{0:1, 0:4}, {{2.0}}>>`
              -> `tensor<?x?x!quant.uniform<i8:f32, 2.0>>`

        The rationale for these conversions is that the decompositions / handling of
        more precise quantized types tends to be more efficient than treating
        everything as subchannel type.


        """
        self.add_pass("normalize-quant-types")
        return self

    def nvgpu_optimize_shared_memory(self):
        """Optimizes accesses to shard memory memrefs in order to reduce bank conflicts."""
        self.add_pass("nvgpu-optimize-shared-memory")
        return self

    def nvvm_attach_target(
        self,
        module: str = None,
        triple: str = None,
        chip: str = None,
        features: str = None,
        O: int = None,
        fast: bool = None,
        ftz: bool = None,
        l: List[str] = None,
        ptxas_cmd_options: str = None,
    ):
        """Attaches an NVVM target attribute to a GPU Module.

        This pass searches for all GPU Modules in the immediate regions and attaches
        an NVVM target if the module matches the name specified by the `module` argument.

        Example:
        ```
        // File: in.mlir:
        gpu.module @nvvm_module_1 {...}
        gpu.module @nvvm_module_2 {...}
        gpu.module @rocdl_module_1 {...}
        // mlir-opt --nvvm-attach-target="module=nvvm.* chip=sm_90" in.mlir
        gpu.module @nvvm_module_1 [#nvvm.target<chip = "sm_90">] {...}
        gpu.module @nvvm_module_2 [#nvvm.target<chip = "sm_90">] {...}
        gpu.module @rocdl_module_1 {...}
        ```

        Args:
            module: Regex used to identify the modules to attach the target to.
            triple: Target triple.
            chip: Target chip.
            features: Target features.
            O: Optimization level.
            fast: Enable fast math mode.
            ftz: Enable flush to zero for denormals.
            l: Extra bitcode libraries paths to link to.
            ptxas_cmd_options: Command line options passed to downstream compiler
        """
        self.add_pass(
            "nvvm-attach-target",
            **{
                "module": module,
                "triple": triple,
                "chip": chip,
                "features": features,
                "O": O,
                "fast": fast,
                "ftz": ftz,
                "l": l,
                "ptxas-cmd-options": ptxas_cmd_options,
            },
        )
        return self

    def omp_offload_privatization_prepare(self):
        """Prepare OpenMP maps for privatization for deferred target tasks

        When generating LLVMIR for privatized variables in an OpenMP offloading directive (eg. omp::TargetOp)
        that creates a deferred target task (when the nowait clause is used), we need to copy the privatized
        variable out of the stack of the generating task and into the heap so that the deferred target task
        can still access it. However, if such a privatized variable is also mapped, typically the case for
        allocatables, then the corresponding `omp::MapInfoOp` needs to be fixed up to map the new heap-allocated
        variable and not the original variable.

        """
        self.add_pass("omp-offload-privatization-prepare")
        return self

    def one_shot_bufferize(
        self,
        allow_return_allocs_from_loops: bool = None,
        allow_unknown_ops: bool = None,
        analysis_fuzzer_seed: int = None,
        analysis_heuristic: str = None,
        bufferize_function_boundaries: bool = None,
        check_parallel_regions: bool = None,
        copy_before_write: bool = None,
        dialect_filter: List[str] = None,
        dump_alias_sets: bool = None,
        no_analysis_func_filter: List[str] = None,
        function_boundary_type_conversion: "LayoutMapOption" = None,
        must_infer_memory_space: bool = None,
        use_encoding_for_memory_space: bool = None,
        test_analysis_only: bool = None,
        print_conflicts: bool = None,
        unknown_type_conversion: "LayoutMapOption" = None,
        buffer_alignment: int = None,
    ):
        """One-Shot Bufferize

        This pass bufferizes all ops that implement `BufferizableOpInterface`. It
        first performs an inplacability analysis on SSA use-def chains of tensor
        values to determine which OpOperands may bufferize in-place, i.e., without
        inserting a buffer copy. It then rewrites the IR, inserting a buffer
        allocation and copy for each OpOperand that was decided to bufferize
        out-of-place.

        One-Shot Bufferize (and `BufferizableOpInterface`) was designed for ops that
        are in destination-passing style. When bufferizing such ops, it is possible
        to reuse the buffer of a tensor OpOperand for a tensor OpResult. In essence,
        a possible destination of an operation is already passed as an SSA value.

        `tensor.insert` is an example for an op in destination-passing style. E.g.,
        when bufferizing `%t0 = tensor.insert %f into %dest[%idx]`, `buffer(%t0)` is
        identical to `buffer(%dest)` in the absence of RaW conflicts. As a counter
        example, `tensor.generate` is not in destination-passing style and always
        results in a new buffer allocation.

        One-Shot Bufferize does not deallocate any buffers that it allocates. The
        `-buffer-deallocation-pipeline` pipeline should be run after One-Shot
        Bufferize to insert the deallocation operations necessary to eliminate
        memory leaks.

        One-Shot Bufferize will by default reject IR that contains non-bufferizable
        op, i.e., ops that do not implemement BufferizableOpInterface. Such IR can
        be allowed with `allow-unknown-ops=1`. In that case, to_buffer and to_tensor
        ops will be generated at the bufferization boundary. This is useful for
        compatibility with existing partial bufferization passes: These can
        bufferize the remaining IR after running One-Shot Bufferize.

        Note: Running One-Shot Bufferize after a partial bufferization pass is
        currently not supported. Running partial bufferization passes after running
        One-Shot Bufferize is supported and the recommended way to gradually
        migrate from partial bufferization to One-Shot Bufferize.

        With `dialect-filter`, bufferization can be restricted to a set of dialects.
        If no filter is specified, all ops that implement `BufferizableOpInterface`
        are bufferized. Ops from the `std` dialect are an exception: These ops are
        always ignored, even if no filter is specified. When specifying a dialect
        filter and `allow-unknown-ops` is not turned on, bufferization would fail
        when encountering an op that is not included in the filter (even if it is
        bufferizable).

        One-Shot Bufferize will by default assume memref types with fully dynamic
        layout maps when a precise layout cannot be inferred. E.g., this is the case
        when wrapping a non-bufferizable op in to_buffer/to_tensor ops. This
        behavior can be overridden with `unknown-type-conversion`. Valid values are
        `fully-dynamic-layout-map` and `identity-layout-map`.

        For testing/debugging purposes, `test-analysis-only=1 print-conflicts=1`
        prints analysis results and explains why an OpOperand was decided to
        bufferize out-of-place. This is useful for understanding why One-Shot
        Bufferize chose to insert a certain buffer copy.

        `bufferize-function-boundaries` is an experimental flag for bufferizing
        `FuncOp`, `ReturnOp` and `CallOp`. This feature is still under development
        and supports only simple cases at the moment. In particular:

        * Recursive or circular function call graphs are not supported.
        * External functions (without bodies) that return a tensor are not
          supported.
        * Function with multiple blocks or multiple ReturnOps are not supported.
        * Layout maps on function signatures can be controlled with a separate
          `function-boundary-type-conversion` option, which is similar to
          `unknown-type-conversion` but supports an additional `infer-layout-map`
          option. `fully-dynamic-layout-map` and `identity-layout-map` ensure that
          function signatures bufferize to easily predictable types, potentially at
          the cost of additional casts and copies, respectively. When layout maps
          are inferred, function return types may be more precise, but less
          predictable. Function argument types cannot be inferred and always have
          fully dynamic layout maps with `infer-layout-map`.

        One-Shot Bufferize implements the following contract around function calls:
        The buffer of function arguments is always writable (unless annotated with
        `bufferization.writable = false`). A buffer copy may be inserted at the call
        site where necessary. Alias sets and equivalence info is propagated through
        function calls. Whenever a function is bufferized, all other functions that
        are being called were already analyzed and bufferized, so exact alias and
        equivalence information is available. This is why recursive function calls
        are not yet supported.

        One-Shot Bufferize gathers additional information during the analysis phase
        when function boundary bufferization is activated. E.g., whether a function
        argument is read/written and which returned values are aliasing/equivalent.
        For debugging purposes, such information can be printed with
        `test-analysis-only`.

        The order in which ops are analyzed is important. The analysis is greedy and
        ops that are analyzed earlier are more likely to bufferize in-place. The
        heuristic can be set with `analysis-heuristic`. At the moment, the following
        heuristics are available:

        * `bottom-up` (default): Analyze ops from bottom to top.
        * `top-down`: Analyze ops from top to bottom.
        * `fuzzer`: Randomize the ordering of ops with `analysis-fuzzer-seed`.
        * `bottom-up-from-terminators`: Traverse the reverse use-def chains of
          tensor IR, starting from region branch terminators (bottom-up). Nested
          regions are traversed before enclosing regions. Analyze the traversed ops
          first, then analyze the remaining ops bottom-up. This heuristic is useful
          for bufferizing loop constructs. One-Shot Bufferize currently supports
          only such IR where yielded tensor values bufferize to equivalent region
          iter_args, and first analyzing all ops on the path from the "yielding" op
          to the beginning of the loop body makes it more likely for the region
          iter_args and yielded values to bufferize to equivalent buffers.

        Args:
            allow_return_allocs_from_loops: Allows returning/yielding new allocations from a loop.
            allow_unknown_ops: Allows unknown (not bufferizable) ops in the input IR.
            analysis_fuzzer_seed: Test only: Analyze ops in random order with a given seed (fuzzer)
            analysis_heuristic: Heuristic that control the IR traversal during analysis
            bufferize_function_boundaries: Bufferize function boundaries (experimental).
            check_parallel_regions: Account for parallel regions in RaW analysis.
            copy_before_write: Skip the analysis. Make a buffer copy on every write.
            dialect_filter: Restrict bufferization to ops from these dialects.
            dump_alias_sets: Test only: Annotate tensor IR with alias sets
            no_analysis_func_filter: Skip analysis of functions with these symbol names.Set copyBeforeWrite to true when bufferizing them.
            function_boundary_type_conversion: Controls layout maps when bufferizing function signatures.
            must_infer_memory_space: The memory space of an memref types must always be inferred. If unset, a default memory space of 0 is used otherwise.
            use_encoding_for_memory_space: Use the Tensor encoding attribute for the memory space. Exclusive to the 'must-infer-memory-space' option
            test_analysis_only: Test only: Only run inplaceability analysis and annotate IR
            print_conflicts: Test only: Annotate IR with RaW conflicts. Requires test-analysis-only.
            unknown_type_conversion: Controls layout maps for non-inferrable memref types.
            buffer_alignment: Sets the alignment of newly allocated buffers.
        """
        self.add_pass(
            "one-shot-bufferize",
            **{
                "allow-return-allocs-from-loops": allow_return_allocs_from_loops,
                "allow-unknown-ops": allow_unknown_ops,
                "analysis-fuzzer-seed": analysis_fuzzer_seed,
                "analysis-heuristic": analysis_heuristic,
                "bufferize-function-boundaries": bufferize_function_boundaries,
                "check-parallel-regions": check_parallel_regions,
                "copy-before-write": copy_before_write,
                "dialect-filter": dialect_filter,
                "dump-alias-sets": dump_alias_sets,
                "no-analysis-func-filter": no_analysis_func_filter,
                "function-boundary-type-conversion": function_boundary_type_conversion,
                "must-infer-memory-space": must_infer_memory_space,
                "use-encoding-for-memory-space": use_encoding_for_memory_space,
                "test-analysis-only": test_analysis_only,
                "print-conflicts": print_conflicts,
                "unknown-type-conversion": unknown_type_conversion,
                "buffer-alignment": buffer_alignment,
            },
        )
        return self

    def openacc_legalize_data_values(
        self, host_to_device: bool = None, apply_to_acc_data_construct: bool = None
    ):
        """Legalizes SSA values in compute regions with results from data clause operations

        This pass replace uses of the `varPtr` in compute regions (kernels,
        parallel, serial) with the result of data clause operations (`accPtr`).

        Args:
            host_to_device: Replace varPtr uses with accPtr if true. Replace accPtr uses with varPtr if false
            apply_to_acc_data_construct: Replaces varPtr uses with accPtr for acc compute regions contained within acc.data or acc.declare region.
        """
        self.add_pass(
            "openacc-legalize-data-values",
            **{
                "host-to-device": host_to_device,
                "apply-to-acc-data-construct": apply_to_acc_data_construct,
            },
        )
        return self

    def opt_reduction_pass(
        self, opt_pass: str = None, test: str = None, test_arg: List[str] = None
    ):
        """A wrapper pass that reduces the file with optimization passes
        Args:
            opt_pass: The optimization passes used for reduction, e.g., symbol-dce
            test: The location of the tester which tests the file interestingness
            test_arg: arguments of the tester
        """
        self.add_pass(
            "opt-reduction-pass",
            **{"opt-pass": opt_pass, "test": test, "test-arg": test_arg},
        )
        return self

    def optimize_allocation_liveness(self):
        """This pass optimizes the liveness of temp allocations in the input function

        This pass will find all operations that have a memory allocation effect.
        It will search for the corresponding deallocation and move it right after
        the last user of the allocation.
        This will optimize the liveness of the allocations.

        The pass is expected to run after the deallocation pipeline.

        """
        self.add_pass("optimize-allocation-liveness")
        return self

    def outline_shape_computation(self):
        """Using shape.func to preserve shape computation

        This pass outlines the shape computation part in high level IR by adding
        shape.func and populate corresponding mapping information into
        ShapeMappingAnalysis. The shape computation part is usually introduced by
        shape reification, and each single dynamic shape is denoted by shape.with_shape.

        There're two main reasons this shape-outline pass is needed:
        1. Many passes don't take shape reification part into consideration.
           Therefore we need to "remove" the shape reification part temporarily for
           these passes.
        2. Sometimes we cannot redo shape reification after converting from dialect
           A to dialect B. Because op-level shape reification is only implemented
           on A.

        Input:

        ```mlir
        func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
          tensor<?x4x?xf32> {
          %c2 = arith.constant 2 : index
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %0 = shape.shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
          %1 = shape.get_extent %0, %c2 : tensor<3xindex>, index -> index
          %2 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
          %3 = shape.with_shape %2, %0 : tensor<?x4x?xf32>, tensor<3xindex>
          %4 = shape.value_of %3 : tensor<?x4x?xf32>
          %5 = "test.concat"(%4, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
                tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
          %6 = shape.get_extent %0, %c0 : tensor<3xindex>, index -> index
          %7 = arith.addi %6, %c2 : index
          %8 = shape.from_extents %7, %c4, %1 : index, index, index
          %9 = shape.with_shape %5, %8 : tensor<?x4x?xf32>, !shape.shape
          %10 = shape.value_of %9 : tensor<?x4x?xf32>
          return %10 : tensor<?x4x?xf32>
        }
        ```

        Output
        ```mlir
        func.func @main(%arg0: tensor<?x4x?xf32>, %arg1: tensor<2x4x?xf32>) ->
          tensor<?x4x?xf32> {
          %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32>
          %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>,
                tensor<2x4x?xf32>) -> tensor<?x4x?xf32>
          return %1 : tensor<?x4x?xf32>
        }
        shape.func private @shape_cal_1(%arg0: tensor<?x4x?xf32>) -> !shape.shape {
          %c2 = arith.constant 2 : index
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
          %1 = get_extent %0, %c2 : tensor<3xindex>, index -> index
          %2 = get_extent %0, %c0 : tensor<3xindex>, index -> index
          %3 = arith.addi %2, %c2 : index
          %4 = from_extents %3, %c4, %1 : index, index, index
          return %4 : !shape.shape
        }
        shape.func private @shape_cal_0(%arg0: tensor<?x4x?xf32>) -> tensor<3xindex> {
          %0 = shape_of %arg0 : tensor<?x4x?xf32> -> tensor<3xindex>
          return %0 : tensor<3xindex>
        }
        ```

        For the above example, the shape computation is inlined in the input IR,
        which is used for two values' (test.abs and test.concat) shape. And the shape
        computation part is outlined in the output IR.

        And the shape mapping information will be:

        ```
        // ---- Shape Mapping Information -----
        // - Shape for: %0 = "test.abs"(%arg0) : (tensor<?x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_0(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
        // - Shape for: %1 = "test.concat"(%0, %arg1) {axis = 0 : i64} : (tensor<?x4x?xf32>, tensor<2x4x?xf32>) -> tensor<?x4x?xf32> :: @shape_cal_1(<block argument> of type 'tensor<?x4x?xf32>' at index: 0)
        ```

        """
        self.add_pass("outline-shape-computation")
        return self

    def ownership_based_buffer_deallocation(
        self, private_function_dynamic_ownership: bool = None
    ):
        """Adds all required dealloc operations for all allocations in the input program

        This pass implements an algorithm to automatically introduce all required
        deallocation operations for all buffers in the input program. This ensures
        that the resulting program does not have any memory leaks.

        The Buffer Deallocation pass operates on the level of operations
        implementing the FunctionOpInterface. Such operations can take MemRefs as
        arguments, but also return them. To ensure compatibility among all functions
        (including external ones), some rules have to be enforced. They are just
        assumed to hold for all external functions. Functions for which the
        definition is available ideally also already adhere to the ABI.
        Otherwise, all MemRef write operations in the input IR must dominate all
        MemRef read operations in the input IR. Then, the pass may modify the input
        IR by inserting `bufferization.clone` operations such that the output IR
        adheres to the function boundary ABI:
        * When a MemRef is passed as a function argument, ownership is never
          acquired. It is always the caller's responsibility to deallocate such
          MemRefs.
        * Returning a MemRef from a function always passes ownership to the caller,
          i.e., it is also the caller's responsibility to deallocate MemRefs
          returned from a called function.
        * A function must not return a MemRef with the same allocated base buffer as
          one of its arguments (in this case a copy has to be created). Note that in
          this context two subviews of the same buffer that don't overlap are also
          considered an alias.

        It is recommended to bufferize all operations first such that no tensor
        values remain in the IR once this pass is applied. That way all allocated
        MemRefs will be properly deallocated without any additional manual work.
        Otherwise, the pass that bufferizes the remaining tensors is responsible to
        add the corresponding deallocation operations. Note that this pass does not
        consider any values of tensor type and assumes that MemRef values defined by
        `bufferization.to_buffer` do not return ownership and do not have to be
        deallocated. `bufferization.to_tensor` operations are handled similarly to
        `bufferization.clone` operations with the exception that the result value is
        not handled because it's a tensor (not a MemRef).

        Input

        ```mlir
        #map0 = affine_map<(d0) -> (d0)>
        module {
          func.func @condBranch(%arg0: i1,
                                %arg1: memref<2xf32>,
                                %arg2: memref<2xf32>) {
            cf.cond_br %arg0, ^bb1, ^bb2
          ^bb1:
            cf.br ^bb3(%arg1 : memref<2xf32>)
          ^bb2:
            %0 = memref.alloc() : memref<2xf32>
            linalg.generic {
              indexing_maps = [#map0, #map0],
              iterator_types = ["parallel"]}
            outs(%arg1, %0 : memref<2xf32>, memref<2xf32>) {
            ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
              %tmp1 = exp %gen1_arg0 : f32
              linalg.yield %tmp1 : f32
            }
            cf.br ^bb3(%0 : memref<2xf32>)
          ^bb3(%1: memref<2xf32>):
            "memref.copy"(%1, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
            return
          }
        }
        ```

        Output

        ```mlir
        #map = affine_map<(d0) -> (d0)>
        module {
          func.func @condBranch(%arg0: i1,
                                %arg1: memref<2xf32>,
                                %arg2: memref<2xf32>) {
            %false = arith.constant false
            %true = arith.constant true
            cf.cond_br %arg0, ^bb1, ^bb2
          ^bb1:  // pred: ^bb0
            cf.br ^bb3(%arg1, %false : memref<2xf32>, i1)
          ^bb2:  // pred: ^bb0
            %alloc = memref.alloc() : memref<2xf32>
            linalg.generic {
              indexing_maps = [#map, #map],
              iterator_types = ["parallel"]}
            outs(%arg1, %alloc : memref<2xf32>, memref<2xf32>)
            ^bb0(%out: f32, %out_0: f32):
              %2 = math.exp %out : f32
              linalg.yield %2, %out_0 : f32, f32
            }
            cf.br ^bb3(%alloc, %true : memref<2xf32>, i1)
          ^bb3(%0: memref<2xf32>, %1: i1):  // 2 preds: ^bb1, ^bb2
            memref.copy %0, %arg2 : memref<2xf32> to memref<2xf32>
            %base_buffer, %offset, %sizes, %strides =
              memref.extract_strided_metadata %0 :
              memref<2xf32> -> memref<f32>, index, index, index
            bufferization.dealloc (%base_buffer : memref<f32>) if (%1)
            return
          }
        }
        ```

        The `private-function-dynamic-ownership` pass option allows the pass to add
        additional arguments to private functions to dynamically give ownership of
        MemRefs to callees. This can enable earlier deallocations and allows the
        pass to by-pass the function boundary ABI and thus potentially leading to
        fewer MemRef clones being inserted. For example, the private function
        ```mlir
        func.func private @passthrough(%memref: memref<2xi32>) -> memref<2xi32> {
          return %memref : memref<2xi32>
        }
        ```
        would be converted to
        ```mlir
        func.func private @passthrough(%memref: memref<2xi32>,
                                       %ownership: i1) -> (memref<2xi32>, i1) {
          return %memref, %ownership : memref<2xi32>, i1
        }
        ```
        and thus allows the returned MemRef to alias with the MemRef passed as
        argument (which would otherwise be forbidden according to the function
        boundary ABI).

        Args:
            private_function_dynamic_ownership: Allows to add additional arguments to private functions to dynamically pass ownership of memrefs to callees. This can enable earlier deallocations.
        """
        self.add_pass(
            "ownership-based-buffer-deallocation",
            **{
                "private-function-dynamic-ownership": private_function_dynamic_ownership
            },
        )
        return self

    def pre_sparsification_rewrite(self):
        """Applies sparse tensor rewriting rules prior to sparsification

        A pass that applies rewriting rules to sparse tensor operations prior
        to running the actual sparsification pass.

        """
        self.add_pass("pre-sparsification-rewrite")
        return self

    def print_ir(self, label: str = None):
        """Print IR on the debug stream

        Print the entire IR on the debug stream. This is meant for debugging
        purposes to inspect the IR at a specific point in the pipeline.

        Args:
            label: Label
        """
        self.add_pass("print-ir", **{"label": label})
        return self

    def print_op_stats(self, json: bool = None):
        """Print statistics of operations
        Args:
            json: print the stats as JSON
        """
        self.add_pass("print-op-stats", **{"json": json})
        return self

    def promote_buffers_to_stack(
        self,
        max_alloc_size_in_bytes: int = None,
        max_rank_of_allocated_memref: int = None,
    ):
        """Promotes heap-based allocations to automatically managed stack-based allocations

        This pass implements a simple algorithm to convert heap-based memory
        allocations to stack-based ones. It uses a built-in heuristic to decide
        whether it makes sense to convert an allocation. Furthermore, dynamic
        shaped buffers that are limited by the rank of the tensor can be
        converted. They are only transformed if they are considered to be small.

        Args:
            max_alloc_size_in_bytes: Maximal size in bytes to promote allocations to stack.
            max_rank_of_allocated_memref: Maximal memref rank to promote dynamic buffers.
        """
        self.add_pass(
            "promote-buffers-to-stack",
            **{
                "max-alloc-size-in-bytes": max_alloc_size_in_bytes,
                "max-rank-of-allocated-memref": max_rank_of_allocated_memref,
            },
        )
        return self

    def reconcile_unrealized_casts(self):
        """Simplify and eliminate unrealized conversion casts

        Eliminate `unrealized_conversion_cast` operations, commonly introduced by
        partial dialect conversions, that transitively convert a value to another
        value of the same type, that is:

        ```
        %0 = "producer.op"() : () -> !type.A
        %1 = unrealized_conversion_cast %0 : !type.A to !type.B
        %2 = unrealized_conversion_cast %1 : !type.B to !type.C
        %3 = unrealized_conversion_cast %2 : !type.C to !type.A
        "consumer.op"(%3) : (!type.A) -> ()
        ```

        Such situations appear when the consumer operation is converted by one pass
        and the producer operation is converted by another pass, each of which
        produces an unrealized cast. This pass can be used to clean up the IR.

        """
        self.add_pass("reconcile-unrealized-casts")
        return self

    def reduction_tree(
        self, traversal_mode: int = None, test: str = None, test_arg: List[str] = None
    ):
        """Reduce the input with reduction-tree algorithm
        Args:
            traversal_mode: The graph traversal mode, the default is single-path mode
            test: The location of the tester which tests the file interestingness
            test_arg: arguments of the tester
        """
        self.add_pass(
            "reduction-tree",
            **{"traversal-mode": traversal_mode, "test": test, "test-arg": test_arg},
        )
        return self

    def reify_result_shapes(self):
        """Reifies the results of `tensor::PadOp` and `tensor::ConcatOp`.

        This pass reifies the shapes of a subset of `ReifyRankedShapedTypeOpInterface`
        ops with `tensor` results.

        The pass currently only supports result shape type reification for:
          - tensor::PadOp
          - tensor::ConcatOp
        It addresses a representation gap where implicit op semantics are needed to
        infer static result types from dynamic operands.
        But it does so by using `ReifyRankedShapedTypeOpInterface` as the source of
        truth rather than the op itself. As a consequence, this cannot generalize
        today.

        TODO: in the future, we should consider coupling this information with op
        "transfer functions" (e.g. `IndexingMapOpInterface`) to provide a source of
        truth that can work across result shape inference, canonicalization and op
        verifiers.

        The pass replaces the operations with their reified versions, when more
        static information can be derived, and inserts casts when results shapes
        are updated.

        Example:
        ```mlir
        #map = affine_map<(d0) -> (-d0 + 256)>
        func.func @func(%arg0: f32, %arg1: index, %arg2: tensor<64x?x64xf32>)
            -> tensor<1x?x64xf32>
        {
          %0 = affine.apply #map(%arg1)
          %extracted_slice = tensor.extract_slice %arg2[0, 0, 0] [1, %arg1, 64] [1, 1, 1]
            : tensor<64x?x64xf32> to tensor<1x?x64xf32>
          %padded = tensor.pad %extracted_slice low[0, 0, 0] high[0, %0, 0] {
          ^bb0(%arg3: index, %arg4: index, %arg5: index):
            tensor.yield %arg0 : f32
          } : tensor<1x?x64xf32> to tensor<1x?x64xf32>
          return %padded : tensor<1x?x64xf32>
        }

        // mlir-opt --reify-result-shapes
        #map = affine_map<()[s0] -> (-s0 + 256)>
        func.func @func(%arg0: f32, %arg1: index, %arg2: tensor<64x?x64xf32>)
            -> tensor<1x?x64xf32>
        {
          %0 = affine.apply #map()[%arg1]
          %extracted_slice = tensor.extract_slice %arg2[0, 0, 0] [1, %arg1, 64] [1, 1, 1]
            : tensor<64x?x64xf32> to tensor<1x?x64xf32>
          %padded = tensor.pad %extracted_slice low[0, 0, 0] high[0, %0, 0] {
          ^bb0(%arg3: index, %arg4: index, %arg5: index):
            tensor.yield %arg0 : f32
          } : tensor<1x?x64xf32> to tensor<1x256x64xf32>
          %cast = tensor.cast %padded : tensor<1x256x64xf32> to tensor<1x?x64xf32>
          return %cast : tensor<1x?x64xf32>
        }
        ```

        """
        self.add_pass("reify-result-shapes")
        return self

    def remove_dead_values(self):
        """Remove dead values

        The goal of this pass is optimization (reducing runtime) by removing
        unnecessary instructions. Unlike other passes that rely on local information
        gathered from patterns to accomplish optimization, this pass uses a full
        analysis of the IR, specifically, liveness analysis, and is thus more
        powerful.

        Currently, this pass performs the following optimizations:
        (A) Removes function arguments that are not live,
        (B) Removes function return values that are not live across all callers of
        the function,
        (C) Removes unneccesary operands, results, region arguments, and region
        terminator operands of region branch ops, and,
        (D) Removes simple and region branch ops that have all non-live results and
        don't affect memory in any way,

        iff

        the IR doesn't have any non-function symbol ops, non-call symbol user ops
        and branch ops.

        Here, a "simple op" refers to an op that isn't a symbol op, symbol-user op,
        region branch op, branch op, region branch terminator op, or return-like.

        It is noteworthy that we do not refer to non-live values as "dead" in this
        file to avoid confusing it with dead code analysis's "dead", which refers to
        unreachable code (code that never executes on hardware) while "non-live"
        refers to code that executes on hardware but is unnecessary. Thus, while the
        removal of dead code helps little in reducing runtime, removing non-live
        values should theoretically have significant impact (depending on the amount
        removed).

        It is also important to note that unlike other passes (like `canonicalize`)
        that apply op-specific optimizations through patterns, this pass uses
        different interfaces to handle various types of ops and tries to cover all
        existing ops through these interfaces.

        It is because of its reliance on (a) liveness analysis and (b) interfaces
        that makes it so powerful that it can optimize ops that don't have a
        canonicalizer and even when an op does have a canonicalizer, it can perform
        more aggressive optimizations, as observed in the test files associated with
        this pass.

        Example of optimization (A):-

        ```
        int add_2_to_y(int x, int y) {
          return 2 + y
        }

        print(add_2_to_y(3, 4))
        print(add_2_to_y(5, 6))
        ```

        becomes

        ```
        int add_2_to_y(int y) {
          return 2 + y
        }

        print(add_2_to_y(4))
        print(add_2_to_y(6))
        ```

        Example of optimization (B):-

        ```
        int, int get_incremented_values(int y) {
          store y somewhere in memory
          return y + 1, y + 2
        }

        y1, y2 = get_incremented_values(4)
        y3, y4 = get_incremented_values(6)
        print(y2)
        ```

        becomes

        ```
        int get_incremented_values(int y) {
          store y somewhere in memory
          return y + 2
        }

        y2 = get_incremented_values(4)
        y4 = get_incremented_values(6)
        print(y2)
        ```

        Example of optimization (C):-

        Assume only `%result1` is live here. Then,

        ```
        %result1, %result2, %result3 = scf.while (%arg1 = %operand1, %arg2 = %operand2) {
          %terminator_operand2 = add %arg2, %arg2
          %terminator_operand3 = mul %arg2, %arg2
          %terminator_operand4 = add %arg1, %arg1
          scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3, %terminator_operand4
        } do {
        ^bb0(%arg3, %arg4, %arg5):
          %terminator_operand6 = add %arg4, %arg4
          %terminator_operand5 = add %arg5, %arg5
          scf.yield %terminator_operand5, %terminator_operand6
        }
        ```

        becomes

        ```
        %result1, %result2 = scf.while (%arg2 = %operand2) {
          %terminator_operand2 = add %arg2, %arg2
          %terminator_operand3 = mul %arg2, %arg2
          scf.condition(%terminator_operand1) %terminator_operand2, %terminator_operand3
        } do {
        ^bb0(%arg3, %arg4):
          %terminator_operand6 = add %arg4, %arg4
          scf.yield %terminator_operand6
        }
        ```

        It is interesting to see that `%result2` won't be removed even though it is
        not live because `%terminator_operand3` forwards to it and cannot be
        removed. And, that is because it also forwards to `%arg4`, which is live.

        Example of optimization (D):-

        ```
        int square_and_double_of_y(int y) {
          square = y ^ 2
          double = y * 2
          return square, double
        }

        sq, do = square_and_double_of_y(5)
        print(do)
        ```

        becomes

        ```
        int square_and_double_of_y(int y) {
          double = y * 2
          return double
        }

        do = square_and_double_of_y(5)
        print(do)
        ```

        """
        self.add_pass("remove-dead-values")
        return self

    def remove_shape_constraints(self):
        """Replace all cstr_ ops with a true witness"""
        self.add_pass("remove-shape-constraints")
        return self

    def resolve_ranked_shaped_type_result_dims(
        self, error_on_pattern_iteration_limit: bool = None
    ):
        """Resolve memref.dim of result values of ranked shape type

        The pass resolves memref.dim of result of operations that
        implement the `ReifyRankedShapedTypeOpInterface` in terms of
        shapes of its operands.

        Args:
            error_on_pattern_iteration_limit: Throw an error when pattern rewriter hits iteration limit
        """
        self.add_pass(
            "resolve-ranked-shaped-type-result-dims",
            **{"error-on-pattern-iteration-limit": error_on_pattern_iteration_limit},
        )
        return self

    def resolve_shaped_type_result_dims(
        self, error_on_pattern_iteration_limit: bool = None
    ):
        """Resolve memref.dim of result values

        The pass resolves memref.dim of result of operations that
        implement the `InferShapedTypeOpInterface` or
        `ReifyRankedShapedTypeOpInterface` in terms of shapes of its
        operands.

        Args:
            error_on_pattern_iteration_limit: Throw an error when pattern rewriter hits iteration limit
        """
        self.add_pass(
            "resolve-shaped-type-result-dims",
            **{"error-on-pattern-iteration-limit": error_on_pattern_iteration_limit},
        )
        return self

    def rocdl_attach_target(
        self,
        module: str = None,
        triple: str = None,
        chip: str = None,
        features: str = None,
        abi: str = None,
        O: int = None,
        wave64: bool = None,
        fast: bool = None,
        daz: bool = None,
        finite_only: bool = None,
        unsafe_math: bool = None,
        correct_sqrt: bool = None,
        l: List[str] = None,
    ):
        """Attaches a ROCDL target attribute to a GPU Module.

        This pass searches for all GPU Modules in the immediate regions and attaches
        a ROCDL target if the module matches the name specified by the `module` argument.

        Example:
        ```
        // File: in.mlir:
        gpu.module @nvvm_module_1 {...}
        gpu.module @nvvm_module_2 {...}
        gpu.module @rocdl_module_1 {...}
        // mlir-opt --nvvm-attach-target="module=rocdl.* chip=gfx90a" in.mlir
        gpu.module @nvvm_module_1 {...}
        gpu.module @nvvm_module_2 {...}
        gpu.module @rocdl_module_1 [#rocdl.target<chip = "gfx90a">] {...}
        ```

        Args:
            module: Regex used to identify the modules to attach the target to.
            triple: Target triple.
            chip: Target chip.
            features: Target features.
            abi: ABI version.
            O: Optimization level.
            wave64: Use Wave64 mode.
            fast: Enable fast relaxed math opt.
            daz: Enable denormals are zero opt.
            finite_only: Enable finite only opt.
            unsafe_math: Enable unsafe math opt.
            correct_sqrt: Enable correct rounded sqrt.
            l: Extra bitcode libraries paths to link to.
        """
        self.add_pass(
            "rocdl-attach-target",
            **{
                "module": module,
                "triple": triple,
                "chip": chip,
                "features": features,
                "abi": abi,
                "O": O,
                "wave64": wave64,
                "fast": fast,
                "daz": daz,
                "finite-only": finite_only,
                "unsafe-math": unsafe_math,
                "correct-sqrt": correct_sqrt,
                "l": l,
            },
        )
        return self

    def sccp(self):
        """Sparse Conditional Constant Propagation

        This pass implements a general algorithm for sparse conditional constant
        propagation. This algorithm detects values that are known to be constant and
        optimistically propagates this throughout the IR. Any values proven to be
        constant are replaced, and removed if possible.

        This implementation is based on the algorithm described by Wegman and Zadeck
        in [“Constant Propagation with Conditional Branches”](https://dl.acm.org/doi/10.1145/103135.103136) (1991).

        """
        self.add_pass("sccp")
        return self

    def scf_for_loop_canonicalization(self):
        """Canonicalize operations within scf.for loop bodies"""
        self.add_pass("scf-for-loop-canonicalization")
        return self

    def scf_for_loop_peeling(self, peel_front: bool = None, skip_partial: bool = None):
        """Peel `for` loops at their upper bounds.
        Args:
            peel_front: Peel the first iteration out of the loop.
            skip_partial: Do not peel loops inside of the last, partial iteration of another already peeled loop.
        """
        self.add_pass(
            "scf-for-loop-peeling",
            **{"peel-front": peel_front, "skip-partial": skip_partial},
        )
        return self

    def scf_for_loop_range_folding(self):
        """Fold add/mul ops into loop range"""
        self.add_pass("scf-for-loop-range-folding")
        return self

    def scf_for_loop_specialization(self):
        """Specialize `for` loops for vectorization"""
        self.add_pass("scf-for-loop-specialization")
        return self

    def scf_for_to_while(self):
        """Convert SCF for loops to SCF while loops

        This pass transforms SCF.ForOp operations to SCF.WhileOp. The For loop
        condition is placed in the 'before' region of the while operation, and the
        induction variable incrementation and loop body in the 'after' region.
        The loop carried values of the while op are the induction variable (IV) of
        the for-loop + any iter_args specified for the for-loop.
        Any 'yield' ops in the for-loop are rewritten to additionally yield the
        (incremented) induction variable.

        ```mlir
        # Before:
          scf.for %i = %c0 to %arg1 step %c1 {
            %0 = arith.addi %arg2, %arg2 : i32
            memref.store %0, %arg0[%i] : memref<?xi32>
          }

        # After:
          %0 = scf.while (%i = %c0) : (index) -> index {
            %1 = arith.cmpi slt, %i, %arg1 : index
            scf.condition(%1) %i : index
          } do {
          ^bb0(%i: index):
            %1 = arith.addi %i, %c1 : index
            %2 = arith.addi %arg2, %arg2 : i32
            memref.store %2, %arg0[%i] : memref<?xi32>
            scf.yield %1 : index
          }
        ```

        """
        self.add_pass("scf-for-to-while")
        return self

    def scf_forall_to_for(self):
        """Convert SCF forall loops to SCF for loops"""
        self.add_pass("scf-forall-to-for")
        return self

    def scf_forall_to_parallel(self):
        """Convert SCF forall loops to SCF parallel loops"""
        self.add_pass("scf-forall-to-parallel")
        return self

    def scf_parallel_for_to_nested_fors(self):
        """Convert SCF parallel for loops to nested SCF for loops

        This pass transforms SCF::ParallelOp operations into a nest of SCF::ForOp
        operations. The transformation is useful for cases where the parallel loop
        can be expressed as a series of sequential iterations, allowing for more
        fine-grained control over the loop execution.

        """
        self.add_pass("scf-parallel-for-to-nested-fors")
        return self

    def scf_parallel_loop_fusion(self):
        """Fuse adjacent parallel loops"""
        self.add_pass("scf-parallel-loop-fusion")
        return self

    def scf_parallel_loop_specialization(self):
        """Specialize parallel loops for vectorization"""
        self.add_pass("scf-parallel-loop-specialization")
        return self

    def scf_parallel_loop_tiling(
        self, parallel_loop_tile_sizes: List[int] = None, no_min_max_bounds: bool = None
    ):
        """Tile parallel loops
        Args:
            parallel_loop_tile_sizes: Factors to tile parallel loops by
            no_min_max_bounds: Perform tiling with fixed upper bound with inbound check inside the internal loops
        """
        self.add_pass(
            "scf-parallel-loop-tiling",
            **{
                "parallel-loop-tile-sizes": parallel_loop_tile_sizes,
                "no-min-max-bounds": no_min_max_bounds,
            },
        )
        return self

    def set_llvm_module_datalayout(self, data_layout: str = None):
        """Attach a datalayout string as a module attribute

        Verify that the dataLayout string is a valid LLVM datalayout string and
        attach it as an attribute `LLVMDialect::getDataLayoutAttrName()` to the
        module, overriding the existing one.

        Args:
            data_layout: String description (LLVM format) of the data layout that is expected on the produced module
        """
        self.add_pass("set-llvm-module-datalayout", **{"data-layout": data_layout})
        return self

    def shape_to_shape_lowering(self):
        """Legalize Shape dialect to be convertible to Arith"""
        self.add_pass("shape-to-shape-lowering")
        return self

    def simplify_depthwise_conv(self):
        """Simplify depthwise convolution."""
        self.add_pass("simplify-depthwise-conv")
        return self

    def snapshot_op_locations(
        self,
        filename: str = None,
        tag: str = None,
        print_debuginfo: bool = None,
        print_op_generic: bool = None,
        print_local_scope: bool = None,
        pretty_debuginfo: bool = None,
    ):
        """Generate new locations from the current IR

        This pass allows for generating new locations from the IR during any stage
        of compilation, by snapshotting the IR to a file and using that file to
        generate new locations for the operations.

        Depending on the value of the `tag` option, different resulting locations
        may be generated:

        * If unset, the original location of the operation is replaced.

        Example:

        ```mlir
        // old:
        ... loc("original_source.cpp":1:1)

        // new:
        ... loc("snapshot_source.mlir":10:10)
        ```

        * If set, the new location is fused with the original location in the form
        of a [`Name Location`](Dialects/Builtin.md/#nameloc) with the specified tag.

        Example:

        ```mlir
        // old:
        ... loc("original_source.cpp":1:1)

        // new:
        ... loc(fused["original_source.cpp":1:1, "snapshot"("snapshot_source.mlir":10:10)])
        ```

        Args:
            filename: The filename to print the generated IR
            tag: A tag to use when fusing the new locations with the original. If unset, the locations are replaced.
            print_debuginfo: Print debug info in MLIR output
            print_op_generic: Print the generic op form
            print_local_scope: Print with local scope and inline information (eliding aliases for attributes, types, and locations
            pretty_debuginfo: Print pretty debug info in MLIR output
        """
        self.add_pass(
            "snapshot-op-locations",
            **{
                "filename": filename,
                "tag": tag,
                "print-debuginfo": print_debuginfo,
                "print-op-generic": print_op_generic,
                "print-local-scope": print_local_scope,
                "pretty-debuginfo": pretty_debuginfo,
            },
        )
        return self

    def sparse_assembler(self, direct_out: bool = None):
        """Add [dis]assemble operations on external sparse tensors

        Unlike dense tensors, MLIR does **not** provide a direct `_mlir_ciface_`
        ABI for passing sparse tensors as arguments from and to external methods
        (within MLIR-generated methods, sparse tensors can be freely passed
        around, but this eventually uses a bespoke parameter passing format
        that is subject to change; like opaque pointers when the sparse runtime
        support library is used or the constituent arrays and structs for
        direct IR codegen). The sparse assembler pass, however, can be used
        to obtain a stable `_mlir_ciface_` API for passing sparse tensors
        from and to an external environment, such as Python, PyTorch, or JAX.

        The pass converts public entry methods that use sparse tensors as
        input parameters and/or output return values into wrapper methods
        that [dis]assemble the individual tensors that constitute the actual
        storage used externally into MLIR sparse tensors. This pass can be used
        to prepare the public entry methods of a program that is compiled by the
        MLIR sparsifier to interface with an external runtime, e.g., when passing
        sparse tensors as numpy arrays from and to Python. Note that eventual
        bufferization decisions (e.g. who [de]allocates the underlying memory)
        should be resolved in agreement with the external runtime.

        By default, the pass uses the [dis]assemble operations to input and output
        sparse tensors. When the direct-out option is set, however, the output
        directly returns the MLIR allocated buffers to the external runtime.

        The pass should always run before the actual sparsification passes.

        Args:
            direct_out: Directly returns buffers externally
        """
        self.add_pass("sparse-assembler", **{"direct-out": direct_out})
        return self

    def sparse_buffer_rewrite(self, enable_buffer_initialization: bool = None):
        """Rewrite sparse primitives on buffers to actual code

        A pass that rewrites sparse primitives on buffers to the MLIR implementation
        of the primitives. For example, sparse_tensor.sort operator is implemented
        in this pass.

        Args:
            enable_buffer_initialization: Enable zero-initialization of the memory buffers
        """
        self.add_pass(
            "sparse-buffer-rewrite",
            **{"enable-buffer-initialization": enable_buffer_initialization},
        )
        return self

    def sparse_gpu_codegen(
        self, num_threads: int = None, enable_runtime_library: bool = None
    ):
        """Generates GPU code during sparsification

        Enables the sparsifier to use GPU acceleration. When the number of GPU
        threads is set to zero, the pass tries to enable GPU acceleration by
        means of direct library calls (like cuSPARSE).

        Args:
            num_threads: Sets the number of GPU threads
            enable_runtime_library: Enable runtime library for manipulating sparse tensors
        """
        self.add_pass(
            "sparse-gpu-codegen",
            **{
                "num-threads": num_threads,
                "enable-runtime-library": enable_runtime_library,
            },
        )
        return self

    def sparse_reinterpret_map(
        self,
        scope: "mlir::ReinterpretMapScope" = None,
        loop_ordering_strategy: "mlir::sparse_tensor::LoopOrderingStrategy" = None,
    ):
        """Reinterprets sparse tensor type mappings

        A pass that reinterprets the mappings in all sparse tensor types in a
        way that enables subsequent sparsification. This involves expressing all
        `linalg.generic` operations in terms of level coordinates (rather than
        the dimension coordinates of the input tensors) to align the iteration
        space with the potentially remapped level space as well as resolving cycles
        in the resulting iteration graphs with explicit sparse tensor conversions
        where needed.

        Args:
            scope: Set the reiterpretation scope
            loop_ordering_strategy: Set the loop ordering strategy for sparse code generation
        """
        self.add_pass(
            "sparse-reinterpret-map",
            **{"scope": scope, "loop-ordering-strategy": loop_ordering_strategy},
        )
        return self

    def sparse_space_collapse(self):
        """sparse space collapsing pass

        This pass collapses consecutive sparse spaces (extracted from the same tensor)
        into one multi-dimensional space. The pass is not yet stabilized.

        """
        self.add_pass("sparse-space-collapse")
        return self

    def sparse_storage_specifier_to_llvm(self):
        """Lower sparse storage specifer to llvm structure

        This pass rewrites sparse tensor storage specifier-related operations into
        LLVMDialect, and converts sparse tensor storage specifier into an llvm.struct.

        Example of the conversion:
        ```mlir
        Before:
          %0 = sparse_tensor.storage_specifier.get %arg0 dim_sz at 0
          : !sparse_tensor.storage_specifier<#CSR> to i64

        After:
          %0 = llvm.extractvalue %arg0[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)>
        ```

        """
        self.add_pass("sparse-storage-specifier-to-llvm")
        return self

    def sparse_tensor_codegen(
        self,
        enable_buffer_initialization: bool = None,
        create_sparse_deallocs: bool = None,
    ):
        """Convert sparse tensors and primitives to actual code

        A pass that converts sparse tensor types and primitives to actual
        compiler visible buffers and compiler IR that implements these
        primitives on the selected sparse tensor storage schemes.

        This pass provides an alternative to the SparseTensorConversion pass,
        eliminating the dependence on a runtime support library, and providing
        much more opportunities for subsequent compiler optimization of the
        generated code.

        Example of the conversion:

        ```mlir
          Before:
            func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
              %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
                 : tensor<8x8xf32, #CSR> to memref<?xindex>
              return %0 : memref<?xindex>
            }

          After:
            func.func @foo(%arg0: memref<2xindex>,
                           %arg1: memref<3xindex>,
                           %arg2: memref<?xindex>,
                           %arg3: memref<?xindex>,
                           %arg4: memref<?xf32>) -> memref<?xindex> {
              return %arg2 : memref<?xindex>
            }
        ```

        Args:
            enable_buffer_initialization: Enable zero-initialization of the memory buffers
            create_sparse_deallocs: Specify if the temporary buffers created by the sparse compiler should be deallocated. For compatibility with core bufferization passes. This option is only used when enable-runtime-library=false. See also create-deallocs for BufferizationOption.
        """
        self.add_pass(
            "sparse-tensor-codegen",
            **{
                "enable-buffer-initialization": enable_buffer_initialization,
                "create-sparse-deallocs": create_sparse_deallocs,
            },
        )
        return self

    def sparse_tensor_conversion(self):
        """Convert sparse tensors and primitives to library calls

        A pass that converts sparse tensor primitives into calls into a runtime
        support library. Sparse tensor types are converted into opaque pointers
        to the underlying sparse storage schemes.

        The use of opaque pointers together with runtime support library keeps
        the conversion relatively simple, but at the expense of IR opacity,
        which obscures opportunities for subsequent optimization of the IR.
        An alternative is provided by the SparseTensorCodegen pass.

        Example of the conversion:

        ```mlir
          Before:
            func.func @foo(%arg0: tensor<8x8xf32, #CSR>) -> memref<?xindex> {
              %0 = sparse_tensor.pointers %arg0 {dimension = 1 : index}
                 : tensor<8x8xf32, #CSR> to memref<?xindex>
              return %0 : memref<?xindex>
            }

          After:
            func.func @foo(%arg0: !llvm.ptr) -> memref<?xindex> {
              %c1 = arith.constant 1 : index
              %0 = call @sparsePointers0(%arg0, %c1)
                 : (!llvm.ptr, index) -> memref<?xindex>
              return %0 : memref<?xindex>
            }
        ```

        """
        self.add_pass("sparse-tensor-conversion")
        return self

    def sparse_vectorization(
        self,
        vl: int = None,
        enable_vla_vectorization: bool = None,
        enable_simd_index32: bool = None,
    ):
        """Vectorizes loops after sparsification

        A pass that converts loops after sparsification into vector loops.
        The vector dialect is used as target to provide an architectural
        neutral way of exploiting any platform that supports SIMD instructions.

        The vector length (viz. `vl`) describes the number of packed data elements
        (e.g. both vector<16xf32> and vector<16xf64> have a vector length of 16 even
        though the actual bitwidths differ). A small multiple of the actual lengths
        supported in hardware typically results in efficient SIMD code, since the
        backend will map longer vectors to multiple vector registers, thereby
        effectively unrolling an addition level within the generated for-loop.

        Example of the conversion:

        ```mlir
          Before:
            %3 = memref.load %2[] : memref<f32>
            %4 = scf.for %arg3 = %c0 to %c1024 step %c1 iter_args(%arg4 = %3) -> (f32) {
              %6 = memref.load %0[%arg3] : memref<?xf32>
              %7 = memref.load %1[%arg3] : memref<1024xf32>
              %8 = arith.mulf %6, %7 : f32
              %9 = arith.addf %arg4, %8 : f32
              scf.yield %9 : f32
            }
            memref.store %4, %2[] : memref<f32>

          After:
            %3 = memref.load %2[] : memref<f32>
            %4 = vector.insert %3, %cst [0] : f32 into vector<32xf32>
            %5 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %4) -> (vector<32xf32>) {
              %8 = vector.load %0[%arg3] : memref<?xf32>, vector<32xf32>
              %9 = vector.load %1[%arg3] : memref<1024xf32>, vector<32xf32>
              %10 = arith.mulf %8, %9 : vector<32xf32>
              %11 = arith.addf %arg4, %10 : vector<32xf32>
              scf.yield %11 : vector<32xf32>
            }
            %6 = vector.reduction <add>, %5 : vector<32xf32> into f32
            memref.store %6, %2[] : memref<f32>
        ```

        Args:
            vl: Set the vector length (use 0 to disable vectorization)
            enable_vla_vectorization: Enable vector length agnostic vectorization
            enable_simd_index32: Enable i32 indexing into vectors (for efficient gather/scatter)
        """
        self.add_pass(
            "sparse-vectorization",
            **{
                "vl": vl,
                "enable-vla-vectorization": enable_vla_vectorization,
                "enable-simd-index32": enable_simd_index32,
            },
        )
        return self

    def sparsification(
        self,
        parallelization_strategy: SparseParallelizationStrategy = None,
        sparse_emit_strategy: "mlir::SparseEmitStrategy" = None,
        enable_runtime_library: bool = None,
    ):
        """Automatically generate sparse tensor code from sparse tensor types

        A pass that implements the core functionality of a **sparsifier**.
        Each Linalg operation (MLIR's tensor index notation) that operates on
        sparse tensor types is converted into code in which the sparsity is
        explicit both in terms of co-iterating looping logic as well as
        selected sparse storage schemes.

        See the `SparseTensor` dialect documentation for more background.

        Example input:

        ```mlir
        #matvec = {
          indexing_maps = [
            affine_map<(i,j) -> (i,j)>, // A
            affine_map<(i,j) -> (j)>,   // b
            affine_map<(i,j) -> (i)>    // x (out)
          ],
          iterator_types = ["parallel", "reduction"],
          doc = "X(i) += A(i,j) * B(j)"
        }

        // Multiply a sparse matrix A with a dense vector b into a dense vector x.
        func.func @kernel_matvec(%arga: tensor<?x?xf64, #SparseMatrix>,
                                 %argb: tensor<?xf64>,
                                 %argx: tensor<?xf64>) -> tensor<?xf64> {
          %0 = linalg.generic #matvec
            ins(%arga, %argb: tensor<?x?xf64, #SparseMatrix>, tensor<?xf64>)
            outs(%argx: tensor<?xf64>) {
            ^bb(%a: f64, %b: f64, %x: f64):
              %0 = arith.mulf %a, %b : f64
              %1 = arith.addf %x, %0 : f64
              linalg.yield %1 : f64
          } -> tensor<?xf64>
          return %0 : tensor<?xf64>
        }
        ```

        Args:
            parallelization_strategy: Set the parallelization strategy
            sparse_emit_strategy: Emit functional code or interfaces (to debug) for sparse loops
            enable_runtime_library: Enable runtime library for manipulating sparse tensors
        """
        self.add_pass(
            "sparsification",
            **{
                "parallelization-strategy": parallelization_strategy,
                "sparse-emit-strategy": sparse_emit_strategy,
                "enable-runtime-library": enable_runtime_library,
            },
        )
        return self

    def sparsification_and_bufferization(
        self,
        vl: int = None,
        enable_vla_vectorization: bool = None,
        enable_simd_index32: bool = None,
        enable_gpu_libgen: bool = None,
        sparse_emit_strategy: "mlir::SparseEmitStrategy" = None,
        parallelization_strategy: SparseParallelizationStrategy = None,
    ):
        """Mini-pipeline that combines bufferization and sparsifiation

         This pass forms a mini-pipeline that combines bufferization and sparsifiation.

        Args:
            vl: Set the vector length (use 0 to disable vectorization)
            enable_vla_vectorization: Enable vector length agnostic vectorization
            enable_simd_index32: Enable i32 indexing into vectors (for efficient gather/scatter)
            enable_gpu_libgen: Enable GPU acceleration by means of direct library calls
            sparse_emit_strategy: Emit functional code or interfaces (to debug) for sparse loops
            parallelization_strategy: Set the parallelization strategy
        """
        self.add_pass(
            "sparsification-and-bufferization",
            **{
                "vl": vl,
                "enable-vla-vectorization": enable_vla_vectorization,
                "enable-simd-index32": enable_simd_index32,
                "enable-gpu-libgen": enable_gpu_libgen,
                "sparse-emit-strategy": sparse_emit_strategy,
                "parallelization-strategy": parallelization_strategy,
            },
        )
        return self

    def spirv_attach_target(
        self,
        module: str = None,
        ver: str = None,
        caps: List[str] = None,
        exts: List[str] = None,
        client_api: str = None,
        vendor: str = None,
        device_type: str = None,
        device_id: "uint32_t" = None,
    ):
        """Attaches an SPIR-V target attribute to a GPU Module.

        This pass searches for all GPU Modules in the immediate regions and attaches
        an SPIR-V target if the module matches the name specified by the `module` argument.

        Example:
        ```
        // Given the following file: in1.mlir:
        gpu.module @nvvm_module_1 {...}
        gpu.module @spirv_module_1 {...}
        // With
        // mlir-opt --spirv-attach-target="module=spirv.* ver=v1.0 caps=Kernel" in1.mlir
        // it will generate,
        gpu.module @nvvm_module_1 {...}
        gpu.module @spirv_module_1 [#spirv.target<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>] {...}
        ```

        Args:
            module: Regex used to identify the modules to attach the target to.
            ver: SPIR-V Version.
            caps: List of supported SPIR-V Capabilities
            exts: List of supported SPIR-V Extensions
            client_api: Client API
            vendor: Device Vendor
            device_type: Device Type
            device_id: Device ID
        """
        self.add_pass(
            "spirv-attach-target",
            **{
                "module": module,
                "ver": ver,
                "caps": caps,
                "exts": exts,
                "client_api": client_api,
                "vendor": vendor,
                "device_type": device_type,
                "device_id": device_id,
            },
        )
        return self

    def spirv_canonicalize_gl(self):
        """Canonicalize GLSL ops

        Pass to run canoncalization patterns that involve GL ops.
        These patterns cannot be run in default canonicalization because GL ops
        aren't always available. So they should be involed specifically when needed.

        """
        self.add_pass("spirv-canonicalize-gl")
        return self

    def spirv_lower_abi_attrs(self):
        """Decorate SPIR-V composite type with layout info

        Operation pass that lowers the ABI attributes specified during
        SPIR-V Lowering. Specifically:
        1. Creates the global variables for arguments of entry point function using
          the specification in the `spirv.interface_var_abi` attribute for each
          argument.
        2. Inserts the EntryPointOp and the ExecutionModeOp for entry point
          functions using the specification in the `spirv.entry_point_abi`
          attribute.

        """
        self.add_pass("spirv-lower-abi-attrs")
        return self

    def spirv_promote_to_replicated_constants(self):
        """Convert splat composite constants and spec constants to corresponding replicated constant composite ops defined by SPV_EXT_replicated_composites"""
        self.add_pass("spirv-promote-to-replicated-constants")
        return self

    def spirv_rewrite_inserts(self):
        """Rewrite sequential chains of `spirv.CompositeInsert` operations into `spirv.CompositeConstruct` operations"""
        self.add_pass("spirv-rewrite-inserts")
        return self

    def spirv_unify_aliased_resource(self):
        """Unify access of multiple aliased resources into access of one single resource"""
        self.add_pass("spirv-unify-aliased-resource")
        return self

    def spirv_update_vce(self):
        """Deduce and attach minimal (version, capabilities, extensions) requirements to spirv.module ops

        Operation pass that deduces and attaches the minimal version/
        capabilities/extensions requirements for spirv.module ops.
        For each spirv.module op, this pass requires a `spirv.target_env` attribute
        on it or an enclosing module-like op to drive the deduction. The reason is
        that an op can be enabled by multiple extensions/capabilities. So we need
        to know which one to pick. `spirv.target_env` gives the hard limit as for
        what the target environment can support; this pass deduces what are
        actually needed for a specific spirv.module op.

        """
        self.add_pass("spirv-update-vce")
        return self

    def spirv_webgpu_prepare(self):
        """Prepare SPIR-V to target WebGPU by expanding unsupported ops and replacing with supported ones"""
        self.add_pass("spirv-webgpu-prepare")
        return self

    def sroa(self):
        """Scalar Replacement of Aggregates

        Scalar Replacement of Aggregates. Replaces allocations of aggregates into
        independant allocations of its elements.

        Allocators must implement `DestructurableAllocationOpInterface` to provide
        the list of memory slots for which destructuring should be attempted.

        This pass will only be applied if all accessors of the aggregate implement
        the `DestructurableAccessorOpInterface`. If the accessors provide a view
        into the struct, users of the view must ensure it is used in a type-safe
        manner and within bounds by implementing `TypeSafeOpInterface`.

        """
        self.add_pass("sroa")
        return self

    def stage_sparse_ops(self):
        """Decompose a complex sparse operation into multiple stages

        A pass that decomposes a complex sparse operation into multiple stages.
        E.g., CSR -> CSC is staged into CSR -> COO (unordered) -> sort -> CSC.

        """
        self.add_pass("stage-sparse-ops")
        return self

    def strip_debuginfo(self):
        """Strip debug info from all operations

        This pass strips the IR of any location information, by replacing all
        operation locations with [`unknown`](Dialects/Builtin.md/#unknownloc).

        """
        self.add_pass("strip-debuginfo")
        return self

    def strip_func_quant_types(self):
        """Strip quantized types from function headers

        Identify occurrences of function arguments using a quantized type and
        replace them with a new value of the corresponding storage (signless
        integer) type. For each converted argument, a `quant.scast` op is introduced
        at the head of the function's entry block converting the new integer
        argument into the original quantized value.

        """
        self.add_pass("strip-func-quant-types")
        return self

    def symbol_dce(self):
        """Eliminate dead symbols

        This pass deletes all symbols that are found to be unreachable. This is done
        by computing the set of operations that are known to be live, propagating
        that liveness to other symbols, and then deleting all symbols that are not
        within this live set. Live symbols are those that have a
        [visibility](SymbolsAndSymbolTables.md/#symbol-visibility) that extends
        beyond the IR, e.g. `public`, or those that are referenced by live symbols
        or other non-Symbol operations.

        For example, consider the following input:

        ```mlir
        func.func private @dead_private_function()
        func.func private @live_private_function()

        // Note: The `public` isn't necessary here, as this is the default.
        func.func public @public_function() {
          "foo.return"() {uses = [@live_private_function]} : () -> ()
        }
        ```

        A known live function, `public_function`, contains a reference to an
        otherwise non-live function `live_private_function`. After running
        `symbol-dce`, only these two symbols should remain, as the final symbol
        `dead_private_function` is not visible outside of the current IR and there
        are no links to known-live operations. After running, we get the expected:

        ```mlir
        func.func private @live_private_function()

        func.func public @public_function() {
          "foo.return"() {uses = [@live_private_function]} : () -> ()
        }
        ```

        See [Symbols and SymbolTables](SymbolsAndSymbolTables.md) for more
        information on `Symbols`.

        """
        self.add_pass("symbol-dce")
        return self

    def symbol_privatize(self, exclude: List[str] = None):
        """Mark symbols private

        This pass marks all top-level symbols of the operation run as `private`
        except if listed in `exclude` pass option.

        Args:
            exclude: Comma separated list of symbols that should not be marked private
        """
        self.add_pass("symbol-privatize", **{"exclude": exclude})
        return self

    def test_arm_sme_tile_allocation(
        self, dump_tile_live_ranges: bool = None, preprocess_only: bool = None
    ):
        """Tests SME 'virtual tile' allocation

        This pass does tile allocation for SME "virtual tiles". It is run at the
        'func.func' op level, and assigns tile IDs (via an attribute) to all ops
        that implement the `ArmSMETileOpInterface`. Note: This pass is only intended
        to be used for testing, tile allocation is done as part of the ArmSME to
        LLVM conversion (`convert-arm-sme-to-llvm`).

        Args:
            dump_tile_live_ranges: Dump the live ranges of SME tiles (for debugging)
            preprocess_only: Only preprocess IR so it is ready for tile allocation (but do not allocate any tiles)
        """
        self.add_pass(
            "test-arm-sme-tile-allocation",
            **{
                "dump-tile-live-ranges": dump_tile_live_ranges,
                "preprocess-only": preprocess_only,
            },
        )
        return self

    def test_scf_parallel_loop_collapsing(
        self,
        collapsed_indices_0: List[int] = None,
        collapsed_indices_1: List[int] = None,
        collapsed_indices_2: List[int] = None,
    ):
        """Test parallel loops collapsing transformation

          This pass is purely for testing the scf::collapseParallelLoops
          transformation. The transformation does not have opinions on how a
          parallel loop should be collapsed, so this pass is structured for the
          common case on GPUs of collapsing to a 3d parallel loop. 3 lists can be
          provided to collapsed-indices-{0,1,2} to represent how the loop should be
          collapsed and must reference evrey iterator in the original parallel loop.

        ```mlir
        # Before:
        scf.parallel (%arg0, %arg1)
                     = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
          "test.sink"(%5, %3) : (index, index) -> ()
          scf.yield
        }

        # After:
        scf.parallel (%arg0) = (%c0) to (%c4) step (%c1) {
          %0 = arith.remsi %arg0, %c2 : index
          %1 = arith.divsi %arg0, %c2 : index
          %2 = arith.muli %0, %c7 : index
          %3 = arith.addi %2, %c3 : index
          %4 = arith.muli %1, %c7 : index
          %5 = arith.addi %4, %c3 : index
          "test.sink"(%5, %3) : (index, index) -> ()
        }
        ```

        Args:
            collapsed_indices_0: Which loop indices to combine 0th loop index
            collapsed_indices_1: Which loop indices to combine into the position 1 loop index
            collapsed_indices_2: Which loop indices to combine into the position 2 loop index
        """
        self.add_pass(
            "test-scf-parallel-loop-collapsing",
            **{
                "collapsed-indices-0": collapsed_indices_0,
                "collapsed-indices-1": collapsed_indices_1,
                "collapsed-indices-2": collapsed_indices_2,
            },
        )
        return self

    def topological_sort(self):
        """Sort regions without SSA dominance in topological order

        Recursively sorts all nested regions without SSA dominance in topological
        order. The main purpose is readability, as well as potentially processing of
        certain transformations and analyses. The function sorts the operations in
        all nested regions such that, as much as possible, all users appear after
        their producers.

        This sort is stable. If the block is already topologically sorted, the IR
        is not changed. Operations that form a cycle are moved to the end of the
        regions in a stable order.

        """
        self.add_pass("topological-sort")
        return self

    def tosa_attach_target(
        self,
        specification_version: "mlir::tosa::SpecificationVersion" = None,
        level: "mlir::tosa::Level" = None,
        profiles: List[str] = None,
        extensions: List[str] = None,
    ):
        """Attach tosa.target_env information to the given module.

        This pass allows the user to specify a TOSA target environment consisting of
        the following components: level, profiles and extensions.

        The target environment is attached to the module as an attribute, allowing other
        transformations to query the selected target and adapt their behaviour based on
        this information.

        Args:
            specification_version: The specification version that TOSA operators should conform to.
            level: The TOSA level that operators should conform to. A TOSA level defines operator argument ranges that an implementation shall support.
            profiles: The TOSA profile(s) that operators should conform to. TOSA profiles enable efficient implementation on different classes of device. Each profile is an independent set of operations and data type combinations.
            extensions: The TOSA extension(s) that operators should conform to. TOSA profile extensions define optional operation and data type combinations.
        """
        self.add_pass(
            "tosa-attach-target",
            **{
                "specification_version": specification_version,
                "level": level,
                "profiles": profiles,
                "extensions": extensions,
            },
        )
        return self

    def tosa_convert_integer_type_to_signless(self):
        """Convert integer types to signless

        This pass converts signed or unsigned integer types to signless. It
        currently does this greedily for all operators and can also change the
        signature of the function. Should the signature of the entrypoint
        function change, it will be the responsibility of the user to carry
        signedness information of the inputs and outputs independently.

        This can be a useful transformation for conversion to other formats
        that require strict adherence to the TOSA specification.

        """
        self.add_pass("tosa-convert-integer-type-to-signless")
        return self

    def tosa_infer_shapes(self):
        """Propagate shapes across TOSA operations

        Pass that uses operand types and propagates shapes to TOSA operations.
        This includes legalizing rankless and dynamic shapes towards static.

        """
        self.add_pass("tosa-infer-shapes")
        return self

    def tosa_layerwise_constant_fold(self, aggressive_reduce_constant: bool = None):
        """Fold layerwise operations on constant tensors

        Pass that enables folding of full-layer operations on constant tensors.

        Args:
            aggressive_reduce_constant: Always perform the reduce constant optimizationMay add more tosa.const but would reduce runtime calculations
        """
        self.add_pass(
            "tosa-layerwise-constant-fold",
            **{"aggressive-reduce-constant": aggressive_reduce_constant},
        )
        return self

    def tosa_make_broadcastable(self):
        """TOSA rank Reshape to enable Broadcasting

        Pass that enables broadcast by making all input arrays have the same
        number of dimensions. Insert RESHAPE operations to prepend dimensions
        of size one until the number of dimensions is equal. Implements
        approach similar to step 1 of Numpy 4-step broadcasting:
        https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting

        """
        self.add_pass("tosa-make-broadcastable")
        return self

    def tosa_optional_decompositions(self):
        """Applies Tosa operations optional decompositions

        Pass to apply the Tosa operations decompositions
        exposed as populate functions in include/mlir/Dialect/Tosa/Transforms/Passes.h

        """
        self.add_pass("tosa-optional-decompositions")
        return self

    def tosa_reduce_transposes(self):
        """Reduce transposes through other operators

        Pass that identifies and reduces tosa.TRANSPOSE operations through chains
        of operators.

        The pass traverses dependencies of tosa.TRANSPOSE operations until they
        terminate in either a tosa.RESHAPE that we can fold the hoisted
        tosa.TRANSPOSE into, a tosa.TRANSPOSE that forms the identity with the
        hoisted one, or a tosa.CONST with a dense elements attribute. It then
        propagates the hoisted transform upward through the intervening operators
        if the support is implemented. Finally, it observes that no duplication
        will occur of both the chain that was hoisted through and the new chain
        that results, and if so, it replaces the hoisted tosa.TRANSPOSE.

        The pass has an important use-case in cleaning up the results of frameworks
        that introduce a lot of data-layout transformations when legalizing to TOSA,
        a common one being transformations between NHWC and NCHW layouts.

        """
        self.add_pass("tosa-reduce-transposes")
        return self

    def tosa_to_arith(
        self, include_apply_rescale: bool = None, use_32_bit: bool = None
    ):
        """Lower TOSA to the Arith dialect

        Pass that converts TOSA operations to the equivalent operations using the
        operations in the Arith dialect. The ApplyScale operator is optionally
        included as it is often preserved until the final invocation.

        Args:
            include_apply_rescale: Whether to include the lowering for tosa.apply_rescale to arith
            use_32_bit: Whether to prioritze lowering to 32-bit operations
        """
        self.add_pass(
            "tosa-to-arith",
            **{
                "include-apply-rescale": include_apply_rescale,
                "use-32-bit": use_32_bit,
            },
        )
        return self

    def tosa_to_linalg(
        self,
        disable_tosa_decompositions: bool = None,
        aggressive_reduce_constant: bool = None,
    ):
        """Lower TOSA to LinAlg on tensors

        Pass that converts TOSA operations to the equivalent operations using the
        tensor operations in LinAlg.

        Args:
            disable_tosa_decompositions: Disable tosa decompositions pass
            aggressive_reduce_constant: Always perform the reduce constant optimization
        """
        self.add_pass(
            "tosa-to-linalg",
            **{
                "disable-tosa-decompositions": disable_tosa_decompositions,
                "aggressive-reduce-constant": aggressive_reduce_constant,
            },
        )
        return self

    def tosa_to_linalg_named(self, prefer_conv2d_kernel_layout_hwcf: bool = None):
        """Lower TOSA to LinAlg named operations

        Pass that converts TOSA operations to the equivalent operations using the
        Linalg named operations.

        Args:
            prefer_conv2d_kernel_layout_hwcf: Prefer generating linalg.conv_2d_nhwc_hwcf over linalg.conv_2d_nhwc_fhwc
        """
        self.add_pass(
            "tosa-to-linalg-named",
            **{"prefer-conv2d-kernel-layout-hwcf": prefer_conv2d_kernel_layout_hwcf},
        )
        return self

    def tosa_to_mlprogram(self):
        """Lower TOSA to the MLProgram dialect

        Pass that converts TOSA's variable operator operations to the equivalent
        MLProgram operations.

        """
        self.add_pass("tosa-to-mlprogram")
        return self

    def tosa_to_scf(self):
        """Lower TOSA to the SCF dialect

        Pass that converts TOSA's control flow operations to the equivalent SCF
        operations.

        """
        self.add_pass("tosa-to-scf")
        return self

    def tosa_to_tensor(self):
        """Lower TOSA to the Tensor dialect

        Pass that converts TOSA operations to the equivalent operations using the
        operations in the Tensor dialect.

        """
        self.add_pass("tosa-to-tensor")
        return self

    def tosa_validate(
        self,
        strict_op_spec_alignment: bool = None,
        allow_invalid_op_datatype_combinations: bool = None,
    ):
        """Validates TOSA dialect

        This pass validates if input TOSA operations match the specification for given
        criteria, e.g. TOSA profile.

        Args:
            strict_op_spec_alignment: Verify if the properties of certain operations align the spec requirement
            allow_invalid_op_datatype_combinations: Disable checks for operations that are determined to be invalid due to their operand/result datatypes not aligning with the 'Supported Data Types' sections of the specifciation
        """
        self.add_pass(
            "tosa-validate",
            **{
                "strict-op-spec-alignment": strict_op_spec_alignment,
                "allow-invalid-op-datatype-combinations": allow_invalid_op_datatype_combinations,
            },
        )
        return self

    def transform_dialect_check_uses(self):
        """warn about potential use-after-free in the transform dialect

        This pass analyzes operations from the transform dialect and its extensions
        and warns if a transform IR value may be used by an operation after it was
        "freed" by some other operation, as described by side effects on the
        `TransformMappingResource`. This statically detects situations that lead to
        errors when interpreting the Transform IR.

        The pass is capable of handling branching control flow and reports all
        _potential_ use-after-free situations, e.g., a may-use-after-free is
        reported if at least one of the control flow paths between the definition of
        a value and its use contains an operation with a "free" effect on the
        `TransformMappingResource`. It does not currently perform an SCCP-style data
        flow analysis to prove that some branches are not taken, however, SCCP and
        other control flow simplifications can be performed on the transform IR
        prior to this pass provided that transform ops implement the relevant
        control flow interfaces.

        """
        self.add_pass("transform-dialect-check-uses")
        return self

    def transform_infer_effects(self):
        """infer transform side effects for symbols

        This pass analyzes the definitions of transform dialect callable symbol
        operations, such as `transform.named_sequence`, and annotates the symbol
        arguments with attributes indicating the side effects that the nested
        operations have on them.

        """
        self.add_pass("transform-infer-effects")
        return self

    def transform_interpreter(
        self,
        debug_payload_root_tag: str = None,
        debug_bind_trailing_args: List[str] = None,
        disable_expensive_checks: bool = None,
        entry_point: str = None,
    ):
        """transform dialect interpreter

        This pass runs the transform dialect interpreter and applies the named
        sequence transformation specified by the provided name (defaults to
        `TransformDialect::kTransformEntryPointSymbolName`,
        i.e. `__transform_main`).

        Additional options can be used to narrow down the pass applicability for
        debugging purposes:
          * `debugPayloadRootTag` makes the transform script apply to the payload
            operation that has a `transform.target_tag` string attribute with the
            given value, rather than to the anchor operation of the pass.
          * `debugBindTrailingArgs` allows one to bind values to trailing arguments
            of the transform entry point as follows:
            * arguments of `TransformHandleTypeInterface` type can be bound to all
              payload operations with the name provided as a simple string;
            * arguments of `TransformValueHandleTypeInterface` type can be bound to
              a flattened list of results of all operations with the name provided
              as a string prefixed with `^`;
            * arguments of `TransformParamTypeInterface` type can be bound to
              integer constants provided as `;`-separated list prefixed with `#`.
          * `entryPoint` specifies the name of the transform symbol to serve as the
            entry point.

        Args:
            debug_payload_root_tag: Select the operation with 'transform.target_tag' attribute having the given value as payload IR root. If empty select the pass anchor operation as the payload IR root.
            debug_bind_trailing_args: Binds trailing arguments of the entry point to the payload operations with specified names.
            disable_expensive_checks: Disable expensive checks in the interpreter for a faster run.
            entry_point: Entry point of the pass pipeline.
        """
        self.add_pass(
            "transform-interpreter",
            **{
                "debug-payload-root-tag": debug_payload_root_tag,
                "debug-bind-trailing-args": debug_bind_trailing_args,
                "disable-expensive-checks": disable_expensive_checks,
                "entry-point": entry_point,
            },
        )
        return self

    def transform_preload_library(self, transform_library_paths: List[str] = None):
        """preload transform dialect library

        This pass preloads a transform library and makes it available to subsequent
        transform interpreter passes. The preloading occurs into the Transform
        dialect and thus provides very limited functionality that does not scale.

        Warning: Only a single such pass should exist for a given MLIR context.
        This is a temporary solution until a resource-based solution is available.

        Args:
            transform_library_paths: Optional paths to files with modules that should be merged into the transform module to provide the definitions of external named sequences.
        """
        self.add_pass(
            "transform-preload-library",
            **{"transform-library-paths": transform_library_paths},
        )
        return self

    def view_op_graph(
        self,
        max_label_len: int = None,
        print_attrs: bool = None,
        print_control_flow_edges: bool = None,
        print_data_flow_edges: bool = None,
        print_result_types: bool = None,
    ):
        """Print Graphviz visualization of an operation

        This pass prints a Graphviz graph of a module.

        - Operations are represented as nodes;
        - Uses (data flow) as edges;
        - Control flow as dashed edges;
        - Regions/blocks as subgraphs.

        By default, only data flow edges are printed.

        Note: See https://www.graphviz.org/doc/info/lang.html for more information
        about the Graphviz DOT language.

        Args:
            max_label_len: Limit attribute/type length to number of chars
            print_attrs: Print attributes of operations
            print_control_flow_edges: Print control flow edges
            print_data_flow_edges: Print data flow edges
            print_result_types: Print result types of operations
        """
        self.add_pass(
            "view-op-graph",
            **{
                "max-label-len": max_label_len,
                "print-attrs": print_attrs,
                "print-control-flow-edges": print_control_flow_edges,
                "print-data-flow-edges": print_data_flow_edges,
                "print-result-types": print_result_types,
            },
        )
        return self

    def wrap_emitc_func_in_class(self):
        """Wrap functions in classes, using arguments as fields.

        This pass transforms `emitc.func` operations into `emitc.class` operations.
        Function arguments become fields of the class, and the function body is moved
        to a new `execute` method within the class.
        If the corresponding function argument has attributes (accessed via `argAttrs`),
        these attributes are attached to the field operation.
        Otherwise, the field is created without additional attributes.

        Example:

        ```mlir
        emitc.func @model(%input_data : !emitc.array<1xf32> {emitc.opaque = "input_tensor"}) attributes { } {
          %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
          %1 = subscript %input_data[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
          return
        }
        // becomes
        emitc.class @modelClass {
          emitc.field @input_tensor : !emitc.array<1xf32> {emitc.opaque = "input_tensor"}
          emitc.func @execute() {
            %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
            %1 = get_field @input_tensor : !emitc.array<1xf32>
            %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
            return
          }
        }
        ```

        """
        self.add_pass("wrap-emitc-func-in-class")
        return self

    def xegpu_blocking(self):
        """Block XeGPU ops into smaller size.

        This pass partitions operations that process large shapes into multiple
        operations on smaller shapes, as specified by the inst_data in the layout
        attribute. This enables each resulting operation to be efficiently mapped
        to a hardware instruction.

        """
        self.add_pass("xegpu-blocking")
        return self

    def xegpu_fold_alias_ops(self):
        """Fold alias ops into XeGPU ops

        The pass folds aliasing ops into XeGPU ops that they operate on the original
        source references.

        """
        self.add_pass("xegpu-fold-alias-ops")
        return self

    def xegpu_optimize_block_loads(self):
        """Optimize XeGPU block load operations

        This pass rewrites XeGPU loadNd operations into more optimal forms
        to improve performance. This includes,
        - Rewriting transpose B loads into more optimal forms to use HW block
          transpose instructions for better performance.

        """
        self.add_pass("xegpu-optimize-block-loads")
        return self

    def xegpu_propagate_layout(
        self, print_analysis_only: bool = None, layout_kind: str = None
    ):
        """Propagate and assign XeGPU layout information

        This pass propagates the XeGPU layout information accross ops. Starting
        from a set of anchor operations (e.g. `dpas`, `store_nd`), this will
        propagate the layouts required for their operands to the producers. With
        this propagated layout information, pass will then update op result type
        with the layout information.

        Args:
            print_analysis_only: Print the result of layout propagation analysis and exit.
            layout_kind: Propagate `inst` / `lane` level of xegpu layouts.
        """
        self.add_pass(
            "xegpu-propagate-layout",
            **{"print-analysis-only": print_analysis_only, "layout-kind": layout_kind},
        )
        return self

    def xegpu_subgroup_distribute(self):
        """Distribute XeGPU ops to work items

        The pass distributes subgroup level (SIMD) XeGPU ops to work items.

        """
        self.add_pass("xegpu-subgroup-distribute")
        return self

    def xegpu_vector_linearize(self):
        """Linearize n-D vectors to 1-D vectors

        This pass linearizes n-D vectors to 1-D vectors for lowering to XeVM.

        """
        self.add_pass("xegpu-vector-linearize")
        return self

    def xegpu_wg_to_sg_distribute(self):
        """Transform WorkGroup level XeGPU code to SubGroup level

        This transform pass distributes the workgroup level computation to
        multiple subgroups based on the sg_layout and sg_data attributes.

        """
        self.add_pass("xegpu-wg-to-sg-distribute")
        return self

    def xevm_attach_target(
        self,
        module: str = None,
        triple: str = None,
        chip: str = None,
        O: int = None,
        l: List[str] = None,
        cmd_options: str = None,
    ):
        """Attaches a XeVM target attribute to a GPU Module.

        This pass searches for all GPU Modules in the immediate regions and attaches
        a XeVM target if the module matches the name specified by the `module` argument.

        Example:
        ```
        // File: in.mlir:
        gpu.module @nvvm_module_1 {...}
        gpu.module @rocdl_module_2 {...}
        gpu.module @xevm_module_3 {...}
        // mlir-opt --xevm-attach-target="module=xevm.* chip=pvc" in.mlir
        gpu.module @nvvm_module_1 {...}
        gpu.module @rocdl_module_2 {...}
        gpu.module @xevm_module_3 [#xevm.target<chip = "pvc">] {...}
        ```

        Args:
            module: Regex used to identify the modules to attach the target to.
            triple: Target triple.
            chip: Target chip.
            O: Optimization level.
            l: Extra bitcode libraries paths to link to.
            cmd_options: Command line options passed to downstream compiler
        """
        self.add_pass(
            "xevm-attach-target",
            **{
                "module": module,
                "triple": triple,
                "chip": chip,
                "O": O,
                "l": l,
                "cmd-options": cmd_options,
            },
        )
        return self
