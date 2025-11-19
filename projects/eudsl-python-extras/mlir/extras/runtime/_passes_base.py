# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import logging
import os
import sys
import tempfile
from contextlib import ExitStack
from enum import StrEnum
from io import StringIO
from typing import List, Optional, Union

from ..context import disable_multithreading
from ...ir import Module, StringAttr
from ...passmanager import PassManager

logger = logging.getLogger(__name__)


class MlirCompilerError(Exception):
    pass


def get_module_name_for_debug_dump(module):
    if "debug_module_name" not in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["debug_module_name"]).value


def run_pipeline(
    module,
    pipeline: Union[str, "Pipeline"],
    description: Optional[str] = None,
    enable_ir_printing=False,
    print_pipeline=False,
    verify=True,
):
    module = Module.parse(module.operation.get_asm(enable_debug_info=True))

    if isinstance(pipeline, Pipeline):
        pipeline = str(pipeline)
    """Runs `pipeline` on `module`, with a nice repro report if it fails."""
    module_name = get_module_name_for_debug_dump(module)
    try:
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        # Lower module in place to make it ready for compiler backends.
        with ExitStack() as stack:
            stack.enter_context(module.context)
            asm_for_error_report = module.operation.get_asm(
                large_elements_limit=10,
                enable_debug_info=True,
            )
            pm = PassManager.parse(pipeline)
            pm.enable_verifier(verify)
            if print_pipeline:
                print(pm)
            if enable_ir_printing:
                stack.enter_context(disable_multithreading())
                pm.enable_ir_printing()

            pm.run(module.operation)
    except Exception as e:
        print(e, file=sys.stderr)
        filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
        with open(filename, "w") as f:
            f.write(asm_for_error_report)
        debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
        description = description or f"{module_name} compile"

        message = f"""\
            {description} failed with the following diagnostics:

            {"*" * 80}
            {sys.stderr.getvalue().strip()}
            {"*" * 80}

            For developers, the error can be reproduced with:
            $ mlir-opt {debug_options} -pass-pipeline='{pipeline}' {filename}
            """
        trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
        raise MlirCompilerError(trimmed_message)
    finally:
        sys.stderr = original_stderr

    return module


class Pipeline:
    _pipeline: List[str] = []

    def __init__(self, pipeline=None, wrapper=None):
        if pipeline is None:
            pipeline = []
        self._pipeline = pipeline

    def Nested(self, context, p: "Pipeline"):
        self._pipeline.append(f"{context}({p.materialize(module=False)})")
        return self

    def Func(self, p: "Pipeline"):
        return self.Nested("func.func", p)

    def Spirv(self, p: "Pipeline"):
        return self.Nested("spirv.module", p)

    def Gpu(self, p: "Pipeline"):
        assert isinstance(p, Pipeline)
        return self.Nested("gpu.module", p)

    def materialize(self, module=True):
        pipeline_str = ",".join(self._pipeline)
        if module:
            pipeline_str = f"builtin.module({pipeline_str})"
        logger.debug(f"{pipeline_str}")
        return pipeline_str

    def __str__(self):
        return self.materialize()

    def __iadd__(self, other: "Pipeline"):
        self._pipeline.extend(other._pipeline)
        return self

    def __add__(self, other: "Pipeline"):
        return Pipeline(self._pipeline + other._pipeline)

    def add_pass(self, pass_name, **kwargs):
        kwargs = {
            k.replace("_", "-"): int(v) if isinstance(v, bool) else v
            for k, v in kwargs.items()
            if v is not None
        }
        if kwargs:
            args_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            pass_str = f"{pass_name}{{ {args_str} }}"
        else:
            pass_str = f"{pass_name}"
        self._pipeline.append(pass_str)
        return self

    def lower_to_llvm(self, use_bare_ptr_memref_call_conv=False):
        # https://github.com/makslevental/llvm-project/blob/f6643263631bcb0d191ef923963ac1a5ca9ac5fd/mlir/test/lib/Dialect/LLVM/TestLowerToLLVM.cpp#L44
        return (
            self.Func(
                self.__class__()
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
            .convert_vector_to_llvm(force_32bit_vector_indices=True)
            # Convert Math to LLVM (always needed).
            .Func(self.__class__().convert_math_to_llvm())
            # Expand complicated MemRef operations before lowering them.
            .expand_strided_metadata()
            # The expansion may create affine expressions. Get rid of them.
            .lower_affine()
            # Convert MemRef to LLVM (always needed).
            .finalize_memref_to_llvm()
            # Convert Func to LLVM (always needed).
            .convert_func_to_llvm(
                use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv
            )
            .convert_arith_to_llvm()
            .convert_cf_to_llvm()
            # Convert Index to LLVM (always needed).
            .convert_index_to_llvm()
            # Convert remaining unrealized_casts (always needed).
            .reconcile_unrealized_casts()
        )

    def bufferize(self):
        return (
            self.Func(self.__class__().empty_tensor_to_alloc_tensor())
            .one_shot_bufferize()
            .Func(self.__class__().buffer_deallocation_simplification())
        )

    def lower_to_openmp(self):
        return self.convert_scf_to_openmp().Func(self.__class__().lower_affine())

    def sparse_compiler(
        self,
        parallelization_strategy=None,
        enable_runtime_library=None,
        enable_buffer_initialization=None,
        vl=None,
        s2s_strategy=None,
        reassociate_fp_reductions=None,
        enable_index_optimizations=None,
        enable_amx=None,
        enable_arm_neon=None,
        enable_arm_sve=None,
        enable_x86vector=None,
    ):
        self.add_pass(
            "sparse-compiler",
            parallelization_strategy=parallelization_strategy,
            enable_runtime_library=enable_runtime_library,
            enable_buffer_initialization=enable_buffer_initialization,
            vl=vl,
            s2s_strategy=s2s_strategy,
            reassociate_fp_reductions=reassociate_fp_reductions,
            enable_index_optimizations=enable_index_optimizations,
            enable_amx=enable_amx,
            enable_arm_neon=enable_arm_neon,
            enable_arm_sve=enable_arm_sve,
            enable_x86vector=enable_x86vector,
        )
        return self

    def lower_to_vulkan(self, index_bitwidth=None):
        return (
            self.gpu_kernel_outlining()
            .fold_memref_alias_ops()
            .convert_gpu_to_spirv()
            .Spirv(self.__class__().spirv_lower_abi_attrs().spirv_update_vce())
            .convert_gpu_launch_to_vulkan_launch()
            .finalize_memref_to_llvm()
            .Func(self.__class__().llvm_request_c_wrappers())
            .convert_func_to_llvm(index_bitwidth=index_bitwidth)
            .reconcile_unrealized_casts()
            .launch_func_to_vulkan()
        )


class GreedySimplifyRegionLevel(StrEnum):
    DISABLED = "disabled"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


class SparseParallelizationStrategy(StrEnum):
    NONE = "none"
    DENSE_OUTER_LOOP = "dense-outer-loop"
    ANY_STORAGE_OUTER_LOOP = "any-storage-outer-loop"
    DENSE_ANY_LOOP = "dense-any-loop"
    ANY_STORAGE_ANY_LOOP = "any-storage-any-loop"
