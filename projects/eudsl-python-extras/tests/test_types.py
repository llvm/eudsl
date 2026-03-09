# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import mlir.extras.types as T
from mlir.extras.types import tensor, memref, vector
from mlir.extras.dialects.func import evaluate_generic_alias_type

from mlir.extras.dialects.memref import alloc
from mlir.extras.dialects.tensor import S, empty
from mlir.ir import (
    IntegerType,
    IndexType,
    NoneType,
    F16Type,
    BF16Type,
    F32Type,
    F64Type,
    FloatTF32Type,
    Float4E2M1FNType,
    Float6E2M3FNType,
    Float6E3M2FNType,
    Float8E3M4Type,
    Float8E4M3Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    Float8E4M3FNUZType,
    Float8E4M3B11FNUZType,
    Float8E5M2FNUZType,
    Float8E8M0FNUType,
    ComplexType,
    VectorType,
    RankedTensorType,
    UnrankedTensorType,
    MemRefType,
    UnrankedMemRefType,
    FunctionType,
    OpaqueType,
    UnitAttr,
    BoolAttr,
    StringAttr,
    FlatSymbolRefAttr,
    SymbolRefAttr,
    FloatAttr,
    IntegerAttr,
    ArrayAttr,
    StridedLayoutAttr,
    DenseBoolArrayAttr,
    DenseI8ArrayAttr,
    DenseI16ArrayAttr,
    DenseI32ArrayAttr,
    DenseI64ArrayAttr,
    DenseF32ArrayAttr,
    DenseF64ArrayAttr,
)

# noinspection PyUnresolvedReferences
from mlir.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext


def test_shaped_types(ctx: MLIRContext):
    t = tensor(S, 3, S, T.f64())
    assert repr(t) == "RankedTensorType(tensor<?x3x?xf64>)"
    ut = tensor(T.f64())
    assert repr(ut) == "UnrankedTensorType(tensor<*xf64>)"
    t = tensor(S, 3, S, element_type=T.f64())
    assert repr(t) == "RankedTensorType(tensor<?x3x?xf64>)"
    ut = tensor(element_type=T.f64())
    assert repr(ut) == "UnrankedTensorType(tensor<*xf64>)"

    m = memref(S, 3, S, T.f64())
    assert repr(m) == "MemRefType(memref<?x3x?xf64>)"
    um = memref(T.f64())
    assert repr(um) == "UnrankedMemRefType(memref<*xf64>)"
    m = memref(S, 3, S, element_type=T.f64())
    assert repr(m) == "MemRefType(memref<?x3x?xf64>)"
    um = memref(element_type=T.f64())
    assert repr(um) == "UnrankedMemRefType(memref<*xf64>)"

    v = vector(3, 3, 3, T.f64())
    assert repr(v) == "VectorType(vector<3x3x3xf64>)"


def test_n_elements(ctx: MLIRContext):
    ten = empty(1, 2, 3, 4, T.i32())
    assert ten.n_elements == 1 * 2 * 3 * 4

    mem = alloc((1, 2, 3, 4), T.i32())
    assert mem.n_elements == 1 * 2 * 3 * 4


# evaluate_generic_type tests: subclass form (no args) and GenericAlias[args] -> .get(args)


def test_evaluate_generic_type_nullary_types(ctx: MLIRContext):
    assert evaluate_generic_alias_type(IndexType) == IndexType.get()
    assert evaluate_generic_alias_type(NoneType) == NoneType.get()
    assert evaluate_generic_alias_type(F16Type) == F16Type.get()
    assert evaluate_generic_alias_type(BF16Type) == BF16Type.get()
    assert evaluate_generic_alias_type(F32Type) == F32Type.get()
    assert evaluate_generic_alias_type(F64Type) == F64Type.get()
    assert evaluate_generic_alias_type(FloatTF32Type) == FloatTF32Type.get()
    assert evaluate_generic_alias_type(Float4E2M1FNType) == Float4E2M1FNType.get()
    assert evaluate_generic_alias_type(Float6E2M3FNType) == Float6E2M3FNType.get()
    assert evaluate_generic_alias_type(Float6E3M2FNType) == Float6E3M2FNType.get()
    assert evaluate_generic_alias_type(Float8E3M4Type) == Float8E3M4Type.get()
    assert evaluate_generic_alias_type(Float8E4M3Type) == Float8E4M3Type.get()
    assert evaluate_generic_alias_type(Float8E4M3FNType) == Float8E4M3FNType.get()
    assert evaluate_generic_alias_type(Float8E5M2Type) == Float8E5M2Type.get()
    assert evaluate_generic_alias_type(Float8E4M3FNUZType) == Float8E4M3FNUZType.get()
    assert (
        evaluate_generic_alias_type(Float8E4M3B11FNUZType)
        == Float8E4M3B11FNUZType.get()
    )
    assert evaluate_generic_alias_type(Float8E5M2FNUZType) == Float8E5M2FNUZType.get()
    assert evaluate_generic_alias_type(Float8E8M0FNUType) == Float8E8M0FNUType.get()


def test_evaluate_generic_type_integer_type(ctx: MLIRContext):
    assert evaluate_generic_alias_type(IntegerType[32]) == IntegerType.get(32)
    assert evaluate_generic_alias_type(IntegerType[64]) == IntegerType.get(64)
    assert evaluate_generic_alias_type(IntegerType[1]) == IntegerType.get(1)


def test_evaluate_generic_type_complex_type(ctx: MLIRContext):
    assert evaluate_generic_alias_type(ComplexType[F32Type]) == ComplexType.get(
        F32Type.get()
    )
    assert evaluate_generic_alias_type(ComplexType[F64Type]) == ComplexType.get(
        F64Type.get()
    )
    assert evaluate_generic_alias_type(ComplexType[IntegerType[32]]) == ComplexType.get(
        IntegerType.get(32)
    )


def test_evaluate_generic_type_vector_type(ctx: MLIRContext):
    assert evaluate_generic_alias_type(VectorType[[2, 3], F32Type]) == VectorType.get(
        [2, 3], F32Type.get()
    )
    assert evaluate_generic_alias_type(
        VectorType[[3, 3, 3], F64Type]
    ) == VectorType.get([3, 3, 3], F64Type.get())


def test_evaluate_generic_type_tensor_types(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        RankedTensorType[[2, 3], F32Type]
    ) == RankedTensorType.get([2, 3], F32Type.get())
    assert evaluate_generic_alias_type(
        UnrankedTensorType[F32Type]
    ) == UnrankedTensorType.get(F32Type.get())
    assert evaluate_generic_alias_type(
        UnrankedTensorType[F64Type]
    ) == UnrankedTensorType.get(F64Type.get())


def test_evaluate_generic_type_memref_types(ctx: MLIRContext):
    assert evaluate_generic_alias_type(MemRefType[[2, 3], F32Type]) == MemRefType.get(
        [2, 3], F32Type.get()
    )
    assert evaluate_generic_alias_type(
        UnrankedMemRefType[F32Type, IntegerAttr[IntegerType[64], 2]]
    ) == UnrankedMemRefType.get(F32Type.get(), IntegerAttr.get(IntegerType.get(64), 2))


def test_evaluate_generic_type_function_type(ctx: MLIRContext):
    assert evaluate_generic_alias_type(FunctionType[[], []]) == FunctionType.get([], [])
    assert evaluate_generic_alias_type(
        FunctionType[[F32Type.get(), F64Type.get()], [IndexType.get()]]
    ) == FunctionType.get([F32Type.get(), F64Type.get()], [IndexType.get()])


def test_evaluate_generic_type_opaque_type(ctx: MLIRContext):
    assert evaluate_generic_alias_type(OpaqueType["tensor", "bob"]) == OpaqueType.get(
        "tensor", "bob"
    )
    assert evaluate_generic_alias_type(
        OpaqueType["foobar", "mytype"]
    ) == OpaqueType.get("foobar", "mytype")


def test_evaluate_generic_type_unit_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(UnitAttr) == UnitAttr.get()


def test_evaluate_generic_type_bool_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(BoolAttr[True]) == BoolAttr.get(True)
    assert evaluate_generic_alias_type(BoolAttr[False]) == BoolAttr.get(False)


def test_evaluate_generic_type_string_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(StringAttr["hello"]) == StringAttr.get("hello")
    assert evaluate_generic_alias_type(StringAttr["foobar"]) == StringAttr.get("foobar")


def test_evaluate_generic_type_integer_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        IntegerAttr[IntegerType[32], 42]
    ) == IntegerAttr.get(IntegerType.get(32), 42)
    assert evaluate_generic_alias_type(
        IntegerAttr[IntegerType[64], 0]
    ) == IntegerAttr.get(IntegerType.get(64), 0)


def test_evaluate_generic_type_float_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(FloatAttr[F32Type, 42.0]) == FloatAttr.get(
        F32Type.get(), 42.0
    )
    assert evaluate_generic_alias_type(FloatAttr[F64Type, 1.5]) == FloatAttr.get(
        F64Type.get(), 1.5
    )


def test_evaluate_generic_type_flat_symbol_ref_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        FlatSymbolRefAttr["symbol"]
    ) == FlatSymbolRefAttr.get("symbol")
    assert evaluate_generic_alias_type(
        FlatSymbolRefAttr["foobar"]
    ) == FlatSymbolRefAttr.get("foobar")


def test_evaluate_generic_type_symbol_ref_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        SymbolRefAttr[["symbol1", "symbol2"]]
    ) == SymbolRefAttr.get(["symbol1", "symbol2"])


def test_evaluate_generic_type_strided_layout_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        StridedLayoutAttr[42, [5, 7, 13]]
    ) == StridedLayoutAttr.get(42, [5, 7, 13])
    assert evaluate_generic_alias_type(
        StridedLayoutAttr[0, [1, 2]]
    ) == StridedLayoutAttr.get(0, [1, 2])


def test_evaluate_generic_type_array_attr(ctx: MLIRContext):
    items = [StringAttr.get("a"), StringAttr.get("b")]
    assert evaluate_generic_alias_type(ArrayAttr[items]) == ArrayAttr.get(items)
    assert evaluate_generic_alias_type(ArrayAttr[[]]) == ArrayAttr.get([])


def test_evaluate_generic_type_dense_bool_array_attr(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        DenseBoolArrayAttr[[True, False, True]]
    ) == DenseBoolArrayAttr.get([True, False, True])


def test_evaluate_generic_type_dense_int_array_attrs(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        DenseI8ArrayAttr[[1, 2, 3]]
    ) == DenseI8ArrayAttr.get([1, 2, 3])
    assert evaluate_generic_alias_type(
        DenseI16ArrayAttr[[4, 5, 6]]
    ) == DenseI16ArrayAttr.get([4, 5, 6])
    assert evaluate_generic_alias_type(
        DenseI32ArrayAttr[[6, 7, 8]]
    ) == DenseI32ArrayAttr.get([6, 7, 8])
    assert evaluate_generic_alias_type(
        DenseI64ArrayAttr[[8, 9, 10]]
    ) == DenseI64ArrayAttr.get([8, 9, 10])


def test_evaluate_generic_type_dense_float_array_attrs(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        DenseF32ArrayAttr[[1.0, 2.0, 3.0]]
    ) == DenseF32ArrayAttr.get([1.0, 2.0, 3.0])
    assert evaluate_generic_alias_type(
        DenseF64ArrayAttr[[4.0, 5.0, 6.0]]
    ) == DenseF64ArrayAttr.get([4.0, 5.0, 6.0])


def test_evaluate_generic_type_dense_float_array_attrs_tuple(ctx: MLIRContext):
    assert evaluate_generic_alias_type(
        DenseF32ArrayAttr[(1.0, 2.0, 3.0),]
    ) == DenseF32ArrayAttr.get((1.0, 2.0, 3.0))
    assert evaluate_generic_alias_type(
        DenseF64ArrayAttr[(4.0, 5.0, 6.0),]
    ) == DenseF64ArrayAttr.get((4.0, 5.0, 6.0))
