# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import inspect
import sys
from typing import TypeVar

import pytest

import mlir.extras.types as T
from mlir.extras.context import mlir_mod_ctx, RAIIMLIRContextModule
from mlir.extras.dialects import arith
from mlir.extras.dialects.arith import constant
from mlir.extras.dialects.func import (
    func,
    call,
    evaluate_type_annotation,
    ReifiedTypeParam,
)

# noinspection PyUnresolvedReferences
from mlir.extras.testing import (
    mlir_ctx as ctx,
    filecheck,
    filecheck_with_comments,
    MLIRContext,
)
from mlir.ir import (
    Attribute,
    ComplexType,
    F32Type,
    F64Type,
    FlatSymbolRefAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    OpaqueType,
    RankedTensorType,
    UnrankedMemRefType,
    UnrankedTensorType,
    Value,
    VectorType,
)

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_emit(ctx: MLIRContext):
    @func
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()

    # CHECK:  func.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_declare_byte_rep(ctx: MLIRContext):
    def demo_fun1(): ...

    if sys.version_info.minor == 14:
        assert demo_fun1.__code__.co_code == b"\x80\x00R\x00#\x00"
    elif sys.version_info.minor == 13:
        assert demo_fun1.__code__.co_code == b"\x95\x00g\x00"
    elif sys.version_info.minor == 12:
        assert demo_fun1.__code__.co_code == b"\x97\x00y\x00"
    elif sys.version_info.minor == 11:
        assert demo_fun1.__code__.co_code == b"\x97\x00d\x00S\x00"
    elif sys.version_info.minor in {8, 9, 10}:
        assert demo_fun1.__code__.co_code == b"d\x00S\x00"
    else:
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")


def test_declare(ctx: MLIRContext):
    @func
    def demo_fun1() -> T.i32(): ...

    @func
    def demo_fun2() -> (T.i32(), T.i32()): ...

    @func
    def demo_fun3(x: T.i32()) -> (T.i32(), T.i32()): ...

    @func
    def demo_fun4(x: T.i32(), y: T.i32()) -> (T.i32(), T.i32()): ...

    demo_fun1()
    demo_fun2()
    one = constant(1)
    demo_fun3(one)
    demo_fun4(one, one)

    ctx.module.operation.verify()

    # CHECK:  func.func private @demo_fun1() -> i32
    # CHECK:  func.func private @demo_fun2() -> (i32, i32)
    # CHECK:  func.func private @demo_fun3(i32) -> (i32, i32)
    # CHECK:  func.func private @demo_fun4(i32, i32) -> (i32, i32)
    # CHECK:  %[[VAL_0:.*]] = func.call @demo_fun1() : () -> i32
    # CHECK:  %[[VAL_1:.*]]:2 = func.call @demo_fun2() : () -> (i32, i32)
    # CHECK:  %[[VAL_2:.*]] = arith.constant 1 : i32
    # CHECK:  %[[VAL_3:.*]]:2 = func.call @demo_fun3(%[[VAL_2]]) : (i32) -> (i32, i32)
    # CHECK:  %[[VAL_4:.*]]:2 = func.call @demo_fun4(%[[VAL_2]], %[[VAL_2]]) : (i32, i32) -> (i32, i32)

    filecheck_with_comments(ctx.module)


def test_func_base_meta(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1.emit()
    foo1()

    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_func_base_meta2(ctx: MLIRContext):
    @func
    def foo1():
        one = constant(1)
        return one

    foo1()

    # CHECK:  func.func @foo1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }
    # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

    filecheck_with_comments(ctx.module)


def test_func_no_context():
    @func
    def foo1():
        one = constant(1)
        return one

    with mlir_mod_ctx() as mod_ctx:
        foo1()

        # CHECK:  func.func @foo1() -> i32 {
        # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
        # CHECK:    return %[[VAL_0]] : i32
        # CHECK:  }
        # CHECK:  %[[VAL_1:.*]] = func.call @foo1() : () -> i32

        filecheck_with_comments(mod_ctx.module)


def test_raii_mlir_context_module():
    ctx = RAIIMLIRContextModule()

    @func
    def demo_fun1():
        one = constant(1)
        return one

    assert hasattr(demo_fun1, "emit")
    assert inspect.ismethod(demo_fun1.emit)
    demo_fun1.emit()

    # CHECK:  func.func @demo_fun1() -> i32 {
    # CHECK:    %[[VAL_0:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_0]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_explicit_function_type(ctx: MLIRContext):
    input_types = [T.i32(), T.i32()]
    result_types = [T.i32()]
    func_type = FunctionType.get(input_types, result_types)

    @func(function_type=func_type)
    def demo_fun1(a, b):
        one = constant(1)
        return one

    demo_fun1.emit()

    # CHECK:  func.func @demo_fun1(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i32 {
    # CHECK:    %[[VAL_2:.*]] = arith.constant 1 : i32
    # CHECK:    return %[[VAL_2]] : i32
    # CHECK:  }

    filecheck_with_comments(ctx.module)


def test_integer_index_type_annotations(ctx: MLIRContext):
    @func
    def f_i32(x: IntegerType[32]) -> IntegerType[32]: ...

    @func
    def f_i64(x: IntegerType[64]) -> IntegerType[64]: ...

    @func
    def f_index(x: IndexType) -> IndexType: ...

    ctx.module.operation.verify()

    # CHECK: func.func private @f_i32(i32) -> i32
    # CHECK: func.func private @f_i64(i64) -> i64
    # CHECK: func.func private @f_index(index) -> index

    filecheck_with_comments(ctx.module)


def test_complex_type_annotations(ctx: MLIRContext):
    @func
    def f_complex_f32(x: ComplexType[F32Type]) -> ComplexType[F32Type]: ...

    @func
    def f_complex_f64(x: ComplexType[F64Type]) -> ComplexType[F64Type]: ...

    @func
    def f_complex_i32(
        x: ComplexType[IntegerType[32]],
    ) -> ComplexType[IntegerType[32]]: ...

    ctx.module.operation.verify()

    # CHECK: func.func private @f_complex_f32(complex<f32>) -> complex<f32>
    # CHECK: func.func private @f_complex_f64(complex<f64>) -> complex<f64>
    # CHECK: func.func private @f_complex_i32(complex<i32>) -> complex<i32>

    filecheck_with_comments(ctx.module)


def test_vector_type_annotations(ctx: MLIRContext):
    @func
    def f_vec_1d(x: VectorType[[4], F32Type]) -> VectorType[[4], F32Type]: ...

    @func
    def f_vec_2d(x: VectorType[[2, 3], F32Type]) -> VectorType[[2, 3], F32Type]: ...

    @func
    def f_vec_i32(
        x: VectorType[[8], IntegerType[32]],
    ) -> VectorType[[8], IntegerType[32]]: ...

    ctx.module.operation.verify()

    # CHECK: func.func private @f_vec_1d(vector<4xf32>) -> vector<4xf32>
    # CHECK: func.func private @f_vec_2d(vector<2x3xf32>) -> vector<2x3xf32>
    # CHECK: func.func private @f_vec_i32(vector<8xi32>) -> vector<8xi32>

    filecheck_with_comments(ctx.module)


def test_tensor_type_annotations(ctx: MLIRContext):
    @func
    def f_ranked_tensor(
        x: RankedTensorType[[2, 3], F32Type],
    ) -> RankedTensorType[[2, 3], F32Type]: ...

    @func
    def f_unranked_tensor(
        x: UnrankedTensorType[F32Type],
    ) -> UnrankedTensorType[F32Type]: ...

    ctx.module.operation.verify()

    # CHECK: func.func private @f_ranked_tensor(tensor<2x3xf32>) -> tensor<2x3xf32>
    # CHECK: func.func private @f_unranked_tensor(tensor<*xf32>) -> tensor<*xf32>

    filecheck_with_comments(ctx.module)


def test_memref_type_annotations(ctx: MLIRContext):
    @func
    def f_memref(
        x: MemRefType[[2, 3], F32Type],
    ) -> MemRefType[[2, 3], F32Type]: ...

    @func
    def f_unranked_memref(
        x: UnrankedMemRefType[F32Type, IntegerAttr[IntegerType[64], 2]],
    ) -> UnrankedMemRefType[F32Type, IntegerAttr[IntegerType[64], 2]]: ...

    ctx.module.operation.verify()

    # CHECK: func.func private @f_memref(memref<2x3xf32>) -> memref<2x3xf32>
    # CHECK: func.func private @f_unranked_memref(memref<*xf32, 2>) -> memref<*xf32, 2>

    filecheck_with_comments(ctx.module)


def test_opaque_type_annotation(ctx: MLIRContext):
    @func
    def f_opaque(x: OpaqueType["tensor", "bob"]) -> OpaqueType["tensor", "bob"]: ...

    ctx.module.operation.verify()

    # CHECK: func.func private @f_opaque(!tensor.bob) -> !tensor.bob

    filecheck_with_comments(ctx.module)


def test_type_annotations_with_body(ctx: MLIRContext):
    @func
    def f_f32(x: F32Type):
        return x

    @func
    def f_i32(x: IntegerType[32]):
        return x

    @func
    def f_index(x: IndexType):
        return x

    @func
    def f_complex(x: ComplexType[F32Type]):
        return x

    @func
    def f_vector(x: VectorType[[4], F32Type]):
        return x

    @func
    def f_ranked_tensor(x: RankedTensorType[[2, 3], F32Type]):
        return x

    @func
    def f_memref(x: MemRefType[[2, 3], F32Type]):
        return x

    f_f32.emit()
    f_i32.emit()
    f_index.emit()
    f_complex.emit()
    f_vector.emit()
    f_ranked_tensor.emit()
    f_memref.emit()

    ctx.module.operation.verify()

    # CHECK: func.func @f_f32(%[[V:.*]]: f32) -> f32 {
    # CHECK: func.func @f_i32(%[[V:.*]]: i32) -> i32 {
    # CHECK: func.func @f_index(%[[V:.*]]: index) -> index {
    # CHECK: func.func @f_complex(%[[V:.*]]: complex<f32>) -> complex<f32> {
    # CHECK: func.func @f_vector(%[[V:.*]]: vector<4xf32>) -> vector<4xf32> {
    # CHECK: func.func @f_ranked_tensor(%[[V:.*]]: tensor<2x3xf32>) -> tensor<2x3xf32> {
    # CHECK: func.func @f_memref(%[[V:.*]]: memref<2x3xf32>) -> memref<2x3xf32> {

    filecheck_with_comments(ctx.module)


def test_multiple_arg_type_annotations(ctx: MLIRContext):
    @func
    def f_two_args(x: IntegerType[32], y: F32Type) -> F32Type: ...

    @func
    def f_three_args(
        a: IntegerType[32],
        b: VectorType[[4], F32Type],
        c: MemRefType[[2, 3], F32Type],
    ) -> VectorType[[4], F32Type]: ...

    @func
    def f_same_type_args(
        x: RankedTensorType[[2, 3], F32Type],
        y: RankedTensorType[[2, 3], F32Type],
    ) -> RankedTensorType[[2, 3], F32Type]: ...

    ctx.module.operation.verify()

    # CHECK: func.func private @f_two_args(i32, f32) -> f32
    # CHECK: func.func private @f_three_args(i32, vector<4xf32>, memref<2x3xf32>) -> vector<4xf32>
    # CHECK: func.func private @f_same_type_args(tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>

    filecheck_with_comments(ctx.module)


def test_multiple_arg_with_body(ctx: MLIRContext):
    @func
    def f_return_first(x: IntegerType[32], y: IntegerType[64]):
        return x

    @func
    def f_mixed_args(
        a: F32Type,
        b: VectorType[[4], F32Type],
        c: MemRefType[[2, 3], F32Type],
    ):
        return b

    f_return_first.emit()
    f_mixed_args.emit()

    ctx.module.operation.verify()

    # CHECK: func.func @f_return_first(%[[A:.*]]: i32, %[[B:.*]]: i64) -> i32 {
    # CHECK: func.func @f_mixed_args(%[[A:.*]]: f32, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: memref<2x3xf32>) -> vector<4xf32> {

    filecheck_with_comments(ctx.module)


def test_multiple_arg_with_body_with_value(ctx: MLIRContext):

    @func
    def f_mixed_args(
        a: Value[F32Type],
        b: Value[VectorType[[4], F32Type]],
        c: Value[MemRefType[[2, 3], F32Type]],
    ):
        return b

    f_mixed_args.emit()

    ctx.module.operation.verify()

    # CHECK: func.func @f_mixed_args(%[[A:.*]]: f32, %[[B:.*]]: vector<4xf32>, %[[C:.*]]: memref<2x3xf32>) -> vector<4xf32> {

    filecheck_with_comments(ctx.module)


def test_call_func_second_arg_not_list(ctx: MLIRContext):
    """Line 50: ValueError when second arg is not a list when calling a FuncOp."""

    @func
    def demo() -> T.i32(): ...

    func_op = demo.emit()
    with pytest.raises(ValueError, match="expected.*second argument to be a list"):
        call(func_op, "not_a_list")


def test_call_func_third_arg_not_none(ctx: MLIRContext):
    """Line 56: ValueError when third arg is not None when calling a FuncOp."""

    @func
    def demo() -> T.i32(): ...

    func_op = demo.emit()
    with pytest.raises(ValueError, match="unexpected third argument"):
        call(func_op, [], "extra")


def test_call_func_args_not_values(ctx: MLIRContext):
    """Line 62: ValueError when args are not all Value/Operation/OpView."""

    @func
    def demo(x: T.i32()) -> T.i32(): ...

    func_op = demo.emit()
    with pytest.raises(ValueError, match="must all be Value, Operation, or OpView"):
        call(func_op, ["not_a_value"])


def test_call_by_name_with_list_as_second_arg(ctx: MLIRContext):
    """Line 76: ValueError when first arg is list of types and second arg is also a list."""
    with pytest.raises(ValueError, match="expected the second argument to be a string"):
        call([T.i32()], [T.i32()])


def test_call_by_flat_symbol_ref(ctx: MLIRContext):
    """Lines 83-88: call using FlatSymbolRefAttr."""

    @func
    def target(x: T.i32()) -> T.i32(): ...

    one = constant(1)
    ref = FlatSymbolRefAttr.get("target")
    result = call([T.i32()], ref, [one])

    ctx.module.operation.verify()

    # CHECK: func.call @target
    filecheck_with_comments(ctx.module)


def test_call_by_string_name(ctx: MLIRContext):
    """Lines 89-98: call using string name."""

    @func
    def target2(x: T.i32()) -> T.i32(): ...

    one = constant(1)
    result = call([T.i32()], "target2", [one])

    ctx.module.operation.verify()

    # CHECK: func.call @target2
    filecheck_with_comments(ctx.module)


def test_call_unexpected_type(ctx: MLIRContext):
    """Line 100: ValueError for unexpected type in call."""
    with pytest.raises(ValueError, match="unexpected type"):
        call([T.i32()], 12345, [])


def test_reified_type_param_no_default_no_concrete(ctx: MLIRContext):
    """Line 178: ValueError when no concrete_val and no default."""
    tvar = TypeVar("NoDefault")
    with pytest.raises(ValueError, match="either concrete_val must be provided"):
        ReifiedTypeParam(tvar, concrete_val=None)


def test_evaluate_type_annotation_lambda(ctx: MLIRContext):
    """Line 264: evaluate_type_annotation with a lambda."""
    t = evaluate_type_annotation(lambda: T.i32())
    assert t == T.i32()


def test_evaluate_type_annotation_unsupported(ctx: MLIRContext):
    """Line 278: NotImplementedError for unsupported type annotation."""
    with pytest.raises(NotImplementedError, match="unsupported type annotation"):
        evaluate_type_annotation(12345)


def test_func_base_generics_none_getitem(ctx: MLIRContext):
    """Line 480: RuntimeError when using __getitem__ on non-generic func."""

    @func
    def non_generic():
        one = constant(1)
        return one

    # The generics attribute might be [] (empty list) not None after FuncBase init.
    # Force it to None to test the guard.
    non_generic.generics = None
    with pytest.raises(RuntimeError, match="using a generic call requires"):
        non_generic[T.i32()]


def test_func_emit_with_call_args(ctx: MLIRContext):
    """Line 413: emit with call_args infers input types from call args."""

    @func
    def inferred(x: T.i32()):
        one = constant(1)
        return one

    one = constant(1)
    # Calling triggers emit(*call_args) but since there are type annotations,
    # it gets emitted via _build_input_types first. Reset to force re-emit path.
    inferred._func_op = None
    inferred.function_type = None
    inferred.emit(one, force=True)

    ctx.module.operation.verify()

    # CHECK: func.func @inferred(%[[ARG:.*]]: i32) -> i32
    filecheck_with_comments(ctx.module)


def test_func_with_func_attrs(ctx: MLIRContext):
    """Line 442: func_attrs iteration."""

    @func(func_attrs={"llvm.emit_c_interface": Attribute.parse("unit")})
    def with_attrs():
        one = constant(1)
        return one

    with_attrs.emit()

    ctx.module.operation.verify()

    # CHECK: func.func @with_attrs() -> i32 attributes {llvm.emit_c_interface}
    filecheck_with_comments(ctx.module)


def test_func_multiple_return_values(ctx: MLIRContext):
    """Line 459: return_types.extend for tuple/list returns."""

    @func
    def multi_return(x: T.i32(), y: T.f32()):
        return x, y

    multi_return.emit()

    ctx.module.operation.verify()

    # CHECK: func.func @multi_return(%[[A:.*]]: i32, %[[B:.*]]: f32) -> (i32, f32)
    filecheck_with_comments(ctx.module)


def test_func_emit_true(ctx: MLIRContext):
    """Line 563: emit=True causes immediate emission."""

    @func(emit=True)
    def emitted_immediately():
        one = constant(1)
        return one

    ctx.module.operation.verify()

    # CHECK: func.func @emitted_immediately() -> i32
    filecheck_with_comments(ctx.module)


def test_global_typevar_in_getitem(ctx: MLIRContext):
    """Line 497: RuntimeError for global typevars in __getitem__."""
    from mlir.extras.dialects.func import FuncBase
    from mlir.dialects.func import FuncOp, ReturnOp, CallOp

    tvar = TypeVar("my_global_tvar")

    def body():
        one = constant(1)
        return one

    # Put the typevar name in the function's globals to trigger the error
    body.__globals__["my_global_tvar"] = tvar

    fb = FuncBase(
        body,
        FuncOp.__base__,
        ReturnOp,
        CallOp.__base__,
        generics=[tvar],
    )

    with pytest.raises(
        RuntimeError, match="global typevars for generics are not supported"
    ):
        fb[T.i32()]

    # Clean up the global
    del body.__globals__["my_global_tvar"]


def test_evaluate_type_annotation_typevar(ctx: MLIRContext):
    """Lines 260, 262: evaluate_type_annotation with TypeVar uses eval."""
    tvar = TypeVar("my_type")
    # TypeVar gets converted to its __name__ (a string), then eval'd
    t = evaluate_type_annotation(
        tvar,
        globals_={"my_type": T.i32()},
        locals_=None,
    )
    assert str(t) == "i32"


def test_func_base_generics_none_default(ctx: MLIRContext):
    """Line 319: generics=None default assignment in FuncBase.__init__."""
    from mlir.extras.dialects.func import FuncBase
    from mlir.dialects.func import FuncOp, ReturnOp, CallOp

    def body():
        one = constant(1)
        return one

    # Directly pass generics=None to trigger line 319
    fb = FuncBase(
        body,
        FuncOp.__base__,
        ReturnOp,
        CallOp.__base__,
        generics=None,
    )
    assert fb.generics == []
    fb.emit()

    ctx.module.operation.verify()

    # CHECK: func.func @body() -> i32
    filecheck_with_comments(ctx.module)


def test_func_base_str(ctx: MLIRContext):
    """Line 362: __str__ method of FuncBase."""

    @func
    def demo_str():
        one = constant(1)
        return one

    s = str(demo_str)
    assert "FuncBase" in s or "class" in s


def test_build_input_types_invalid_generic(ctx: MLIRContext):
    """Line 377: ValueError when generics contains non-TypeVar, non-ReifiedTypeParam."""
    from mlir.extras.dialects.func import FuncBase
    from mlir.dialects.func import FuncOp, ReturnOp, CallOp

    def body(x):
        one = constant(1)
        return one

    body.__annotations__ = {"x": T.i32()}

    fb = FuncBase(
        body,
        FuncOp.__base__,
        ReturnOp,
        CallOp.__base__,
        generics=["not_a_typevar"],  # invalid
    )

    with pytest.raises(ValueError, match="expected .* to be a TypeVar"):
        fb._build_input_types()
