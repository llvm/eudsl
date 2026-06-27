import pytest

from mlir.extras.runtime._passes_base import (
    run_pipeline,
    get_module_name_for_debug_dump,
    MlirCompilerError,
)
from mlir.extras.runtime.passes import Pipeline as pipe
from mlir.extras.testing import MLIRContext, mlir_ctx as ctx

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_basic():
    p = (
        pipe()
        .cse()
        .Func(pipe().lower_affine().arith_expand().convert_math_to_llvm())
        .convert_math_to_libm()
        .expand_strided_metadata()
        .finalize_memref_to_llvm()
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
        .cse()
        .lower_affine()
        .Func(pipe().convert_arith_to_llvm())
        .convert_func_to_llvm()
        .canonicalize()
        .convert_openmp_to_llvm()
        .cse()
        .reconcile_unrealized_casts()
    )
    assert (
        p.materialize()
        == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,cse,lower-affine,func.func(convert-arith-to-llvm),convert-func-to-llvm,canonicalize,convert-openmp-to-llvm,cse,reconcile-unrealized-casts)"
    )

    p1 = (
        pipe()
        .cse()
        .Func(pipe().lower_affine().arith_expand().convert_math_to_llvm())
        .convert_math_to_libm()
        .expand_strided_metadata()
        .finalize_memref_to_llvm()
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
    )

    p2 = (
        pipe()
        .cse()
        .lower_affine()
        .Func(pipe().convert_arith_to_llvm())
        .convert_func_to_llvm()
        .canonicalize()
        .convert_openmp_to_llvm()
        .cse()
        .reconcile_unrealized_casts()
    )

    assert (
        p1 + p2
    ).materialize() == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,cse,lower-affine,func.func(convert-arith-to-llvm),convert-func-to-llvm,canonicalize,convert-openmp-to-llvm,cse,reconcile-unrealized-casts)"

    p1 = (
        pipe()
        .cse()
        .Func(pipe().lower_affine().arith_expand().convert_math_to_llvm())
        .convert_math_to_libm()
        .expand_strided_metadata()
        .finalize_memref_to_llvm()
        .convert_scf_to_cf()
        .convert_cf_to_llvm()
    )
    assert (
        str(p1)
        == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm)"
    )

    p1 += p2
    assert (
        str(p1)
        == "builtin.module(cse,func.func(lower-affine,arith-expand,convert-math-to-llvm),convert-math-to-libm,expand-strided-metadata,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,cse,lower-affine,func.func(convert-arith-to-llvm),convert-func-to-llvm,canonicalize,convert-openmp-to-llvm,cse,reconcile-unrealized-casts)"
    )


def test_context():
    p = pipe().Nested(
        "aie.device",
        pipe().add_pass("aie-localize-locks").add_pass("aie-normalize-address-spaces"),
    )
    assert (
        str(p)
        == "builtin.module(aie.device(aie-localize-locks,aie-normalize-address-spaces))"
    )


def test_spirv_nested():
    p = pipe().Spirv(pipe().spirv_lower_abi_attrs().spirv_update_vce())
    assert "spirv.module" in str(p)


def test_gpu_nested():
    p = pipe().Gpu(pipe().strip_debuginfo())
    assert "gpu.module" in str(p)


def test_lower_to_llvm():
    p = pipe().lower_to_llvm()
    s = str(p)
    assert "convert-func-to-llvm" in s
    assert "convert-vector-to-llvm" in s


def test_lower_to_llvm_bare_ptr():
    p = pipe().lower_to_llvm(use_bare_ptr_memref_call_conv=True)
    assert "use-bare-ptr-memref-call-conv=1" in str(p)


def test_lower_to_openmp():
    p = pipe().lower_to_openmp()
    assert "convert-scf-to-openmp" in str(p)


def test_sparse_compiler():
    p = pipe().sparse_compiler(parallelization_strategy="dense-outer-loop", vl=4)
    s = str(p)
    assert "sparse-compiler" in s
    assert "parallelization_strategy=dense-outer-loop" in s
    assert "vl=4" in s


def test_bufferize():
    p = pipe().bufferize()
    s = str(p)
    assert "one-shot-bufferize" in s


def test_get_module_name_for_debug_dump(ctx: MLIRContext):
    from mlir.ir import Module, StringAttr

    module = Module.create()
    assert get_module_name_for_debug_dump(module) == "UnnammedModule"
    module.operation.attributes["debug_module_name"] = StringAttr.get("TestModule")
    assert get_module_name_for_debug_dump(module) == "TestModule"


def test_run_pipeline_success(ctx: MLIRContext):
    from mlir.ir import Module

    module = Module.parse("module {}")
    result = run_pipeline(module, "builtin.module(canonicalize)")
    assert result is not None


def test_run_pipeline_with_pipeline_object(ctx: MLIRContext):
    from mlir.ir import Module

    module = Module.parse("module {}")
    p = pipe().canonicalize().cse()
    result = run_pipeline(module, p)
    assert result is not None


def test_run_pipeline_failure(ctx: MLIRContext):
    from mlir.ir import Module

    module = Module.parse("module {}")
    with pytest.raises(MlirCompilerError, match="compile failed"):
        run_pipeline(module, "builtin.module(not-a-real-pass)", description="compile")


def test_run_pipeline_print_pipeline(ctx: MLIRContext, capsys):
    from mlir.ir import Module

    module = Module.parse("module {}")
    run_pipeline(module, "builtin.module(canonicalize)", print_pipeline=True)
    captured = capsys.readouterr()
    assert "canonicalize" in captured.out


def test_run_pipeline_enable_ir_printing(ctx: MLIRContext):
    from mlir.ir import Module

    module = Module.parse("module {}")
    result = run_pipeline(
        module, "builtin.module(canonicalize)", enable_ir_printing=True
    )
    assert result is not None
