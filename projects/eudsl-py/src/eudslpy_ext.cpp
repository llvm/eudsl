// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2024.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/typing.h>

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialOps.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Unit.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"

#include "eudsl/bind_vec_like.h"
#include "eudsl/helpers.h"
#include "eudsl/type_casters.h"

namespace nb = nanobind;
using namespace nb::literals;

class FakeDialect : public mlir::Dialect {
public:
  FakeDialect(llvm::StringRef name, mlir::MLIRContext *context, mlir::TypeID id)
      : Dialect(name, context, id) {}
};

nb::class_<_SmallVector> smallVector;
nb::class_<_ArrayRef> arrayRef;
nb::class_<_MutableArrayRef> mutableArrayRef;

static void bind_array_ref_smallvector(nb::handle scope) {
  scope.attr("T") = nb::type_var("T");
  arrayRef = nb::class_<_ArrayRef>(scope, "ArrayRef", nb::is_generic(),
                                   nb::sig("class ArrayRef[T]"));
  mutableArrayRef =
      nb::class_<_MutableArrayRef>(scope, "MutableArrayRef", nb::is_generic(),
                                   nb::sig("class MutableArrayRef[T]"));
  smallVector = nb::class_<_SmallVector>(scope, "SmallVector", nb::is_generic(),
                                         nb::sig("class SmallVector[T]"));
}

static void populateIRModule(nb::module_ &m) {
  using namespace mlir;
  auto mlir_DialectRegistry =
      non_copying_non_moving_class_<mlir::DialectRegistry>(m, "DialectRegistry")
          .def(nb::init<>())
          .def(
              "insert",
              [](mlir::DialectRegistry &self, mlir::TypeID typeID,
                 llvm::StringRef dialectName) {
                self.insert(
                    typeID, dialectName,
                    [=](mlir::MLIRContext *ctx) -> mlir::Dialect * {
                      return ctx->getOrLoadDialect(dialectName, typeID, [=]() {
                        return std::make_unique<FakeDialect>(dialectName, ctx,
                                                             typeID);
                      });
                    });
              },
              "type_id"_a, "name"_a)
          .def("insert_dynamic", &mlir::DialectRegistry::insertDynamic,
               "name"_a, "ctor"_a)
          .def("get_dialect_allocator",
               &mlir::DialectRegistry::getDialectAllocator, "name"_a)
          .def("append_to", &mlir::DialectRegistry::appendTo, "destination"_a)
          .def_prop_ro("dialect_names", &mlir::DialectRegistry::getDialectNames)
          .def(
              "apply_extensions",
              [](mlir::DialectRegistry &self, mlir::Dialect *dialect) {
                return self.applyExtensions(dialect);
              },
              "dialect"_a)
          .def(
              "apply_extensions",
              [](mlir::DialectRegistry &self, mlir::MLIRContext *ctx) {
                return self.applyExtensions(ctx);
              },
              "ctx"_a)
          .def(
              "add_extension",
              [](mlir::DialectRegistry &self, mlir::TypeID extensionID,
                 std::unique_ptr<mlir::DialectExtensionBase> extension) {
                return self.addExtension(extensionID, std::move(extension));
              },
              "extension_id"_a, "extension"_a)
          .def("is_subset_of", &mlir::DialectRegistry::isSubsetOf, "rhs"_a);

  auto mlir_OperationState =
      non_copying_non_moving_class_<mlir::OperationState>(m, "OperationState")
          .def(nb::init<mlir::Location, llvm::StringRef>(), "location"_a,
               "name"_a)
          .def(nb::init<mlir::Location, mlir::OperationName>(), "location"_a,
               "name"_a)
          .def(nb::init<mlir::Location, mlir::OperationName, mlir::ValueRange,
                        mlir::TypeRange, llvm::ArrayRef<mlir::NamedAttribute>,
                        mlir::BlockRange,
                        llvm::MutableArrayRef<std::unique_ptr<mlir::Region>>>(),
               "location"_a, "name"_a, "operands"_a, "types"_a, "attributes"_a,
               "successors"_a, "regions"_a)
          .def(nb::init<mlir::Location, llvm::StringRef, mlir::ValueRange,
                        mlir::TypeRange, llvm::ArrayRef<mlir::NamedAttribute>,
                        mlir::BlockRange,
                        llvm::MutableArrayRef<std::unique_ptr<mlir::Region>>>(),
               "location"_a, "name"_a, "operands"_a, "types"_a, "attributes"_a,
               "successors"_a, "regions"_a)
          .def_prop_ro("raw_properties",
                       &mlir::OperationState::getRawProperties)
          .def("set_properties", &mlir::OperationState::setProperties, "op"_a,
               "emit_error"_a)
          .def("add_operands", &mlir::OperationState::addOperands,
               "new_operands"_a)
          .def(
              "add_types",
              [](mlir::OperationState &self,
                 llvm::ArrayRef<mlir::Type> newTypes) {
                return self.addTypes(newTypes);
              },
              "new_types"_a)
          .def(
              "add_attribute",
              [](mlir::OperationState &self, llvm::StringRef name,
                 mlir::Attribute attr) {
                return self.addAttribute(name, attr);
              },
              "name"_a, "attr"_a)
          .def(
              "add_attribute",
              [](mlir::OperationState &self, mlir::StringAttr name,
                 mlir::Attribute attr) {
                return self.addAttribute(name, attr);
              },
              "name"_a, "attr"_a)
          .def("add_attributes", &mlir::OperationState::addAttributes,
               "new_attributes"_a)
          .def(
              "add_successors",
              [](mlir::OperationState &self, mlir::Block *successor) {
                return self.addSuccessors(successor);
              },
              "successor"_a)
          .def(
              "add_successors",
              [](mlir::OperationState &self, mlir::BlockRange newSuccessors) {
                return self.addSuccessors(newSuccessors);
              },
              "new_successors"_a)
          .def(
              "add_region",
              [](mlir::OperationState &self) { return self.addRegion(); },
              nb::rv_policy::reference_internal)
          .def(
              "add_region",
              [](mlir::OperationState &self,
                 std::unique_ptr<mlir::Region> &&region) {
                return self.addRegion(std::move(region));
              },
              "region"_a)
          .def("add_regions", &mlir::OperationState::addRegions, "regions"_a)
          .def_prop_ro("context", &mlir::OperationState::getContext);

#include "ir.cpp.inc"
}

// too big
// extern void populateEUDSLGen_accModule(nb::module_ &m);

// extern void populateEUDSLGen_affineModule(nb::module_ &m);

// extern void populateEUDSLGen_amdgpuModule(nb::module_ &m);

// extern void populateEUDSLGen_amxModule(nb::module_ &m);

extern void populateEUDSLGen_arithModule(nb::module_ &m);

// extern void populateEUDSLGen_arm_neonModule(nb::module_ &m);

// too big
// extern void populateEUDSLGen_arm_smeModule(nb::module_ &m);

// extern void populateEUDSLGen_arm_sveModule(nb::module_ &m);

// extern void populateEUDSLGen_asyncModule(nb::module_ &m);

// extern void populateEUDSLGen_bufferizationModule(nb::module_ &m);

// extern void populateEUDSLGen_cfModule(nb::module_ &m);

// extern void populateEUDSLGen_complexModule(nb::module_ &m);

// extern void populateEUDSLGen_DLTIDialectModule(nb::module_ &m);

// extern void populateEUDSLGen_emitcModule(nb::module_ &m);
//
// extern void populateEUDSLGen_funcModule(nb::module_ &m);
//
// extern void populateEUDSLGen_gpuModule(nb::module_ &m);
//
// extern void populateEUDSLGen_indexModule(nb::module_ &m);

// error: use of class template 'ArrayRef' requires template arguments; argument
// deduction not allowed in conversion function type void
// populateEUDSLGen_irdlModule(nb::module_ &m) {
//   using namespace llvm;
// #include "EUDSLGen_irdl.cpp.inc"
// }

// extern void populateEUDSLGen_linalgModule(nb::module_ &m);
//
// extern void populateEUDSLGen_LLVMModule(nb::module_ &m);
//
// extern void populateEUDSLGen_mathModule(nb::module_ &m);
//
// extern void populateEUDSLGen_memrefModule(nb::module_ &m);
//
// // extern void populateEUDSLGen_meshModule(nb::module_ &m);
//
// // extern void populateEUDSLGen_ml_programModule(nb::module_ &m);
//
// // extern void populateEUDSLGen_mpiModule(nb::module_ &m);
//
// extern void populateEUDSLGen_nvgpuModule(nb::module_ &m);
//
// extern void populateEUDSLGen_NVVMModule(nb::module_ &m);
//
// //
// mlir::omp::TaskloopOp::getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&)
// // void populateEUDSLGen_ompModule(nb::module_ &m) {
// // #include "EUDSLGen_omp.cpp.inc"
// // }
//
// extern void populateEUDSLGen_pdlModule(nb::module_ &m);
//
// extern void populateEUDSLGen_pdl_interpModule(nb::module_ &m);
//
// extern void populateEUDSLGen_polynomialModule(nb::module_ &m);
//
// // extern void populateEUDSLGen_ptrModule(nb::module_ &m);
//
// extern void populateEUDSLGen_quantModule(nb::module_ &m);
//
// extern void populateEUDSLGen_ROCDLModule(nb::module_ &m);
//
// extern void populateEUDSLGen_scfModule(nb::module_ &m);
//
// extern void populateEUDSLGen_shapeModule(nb::module_ &m);
//
// // extern void populateEUDSLGen_sparse_tensorModule(nb::module_ &m);
//
// // nb::detail::nb_func_new("get_vce_triple_attr_name"): mismatched
// // static/instance method flags in function overloads! extern void
// // populateEUDSLGen_spirvModule(nb::module_ &m);
//
// extern void populateEUDSLGen_tensorModule(nb::module_ &m);
//
// // extern void populateEUDSLGen_tosaModule(nb::module_ &m);
//
// extern void populateEUDSLGen_transformModule(nb::module_ &m);
//
// extern void populateEUDSLGen_ubModule(nb::module_ &m);

// can't cast std::pair<VectorDim, VectorDim>
// void populateEUDSLGen_vectorModule(nb::module_ &m) {
// #include "EUDSLGen_vector.cpp.inc"
// }

// extern void populateEUDSLGen_x86vectorModule(nb::module_ &m);

// extern void populateEUDSLGen_xegpuModule(nb::module_ &m);

NB_MODULE(eudslpy_ext, m) {
  bind_array_ref_smallvector(m);

  nb::class_<llvm::APFloat>(m, "APFloat");
  nb::class_<llvm::APInt>(m, "APInt");
  nb::class_<llvm::APSInt>(m, "APSInt");
  nb::class_<llvm::LogicalResult>(m, "LogicalResult");
  nb::class_<llvm::ParseResult>(m, "ParseResult");
  nb::class_<llvm::SourceMgr>(m, "SourceMgr");
  nb::class_<llvm::ThreadPoolInterface>(m, "ThreadPoolInterface");
  nb::class_<llvm::hash_code>(m, "hash_code");
  nb::class_<llvm::raw_ostream>(m, "raw_ostream");
  nb::class_<mlir::AsmParser>(m, "AsmParser");
  nb::class_<mlir::AsmResourcePrinter>(m, "AsmResourcePrinter");
  nb::class_<mlir::DataLayoutSpecInterface>(m, "DataLayoutSpecInterface");
  nb::class_<mlir::DialectBytecodeReader>(m, "DialectBytecodeReader");
  nb::class_<mlir::DialectBytecodeWriter>(m, "DialectBytecodeWriter");
  nb::class_<mlir::IntegerValueRange>(m, "IntegerValueRange");
  nb::class_<mlir::StorageUniquer>(m, "StorageUniquer");
  nb::class_<mlir::TargetSystemSpecInterface>(m, "TargetSystemSpecInterface");
  nb::class_<mlir::TypeID>(m, "TypeID");
  nb::class_<mlir::detail::InterfaceMap>(m, "InterfaceMap");

  auto irModule = m.def_submodule("ir");
  populateIRModule(irModule);

  nb::class_<llvm::FailureOr<bool>>(m, "FailureOr[bool]");
  nb::class_<llvm::FailureOr<mlir::StringAttr>>(m, "FailureOr[StringAttr]");
  nb::class_<llvm::FailureOr<mlir::AsmResourceBlob>>(
      m, "FailureOr[AsmResourceBlob]");
  nb::class_<llvm::FailureOr<mlir::AffineMap>>(m, "FailureOr[AffineMap]");
  nb::class_<llvm::FailureOr<mlir::detail::ElementsAttrIndexer>>(
      m, "FailureOr[ElementsAttrIndexer]");
  nb::class_<llvm::FailureOr<mlir::AsmDialectResourceHandle>>(
      m, "FailureOr[AsmDialectResourceHandle]");
  nb::class_<llvm::FailureOr<mlir::OperationName>>(m,
                                                   "FailureOr[OperationName]");

  nb::class_<mlir::IRObjectWithUseList<mlir::BlockOperand>>(
      m, "IRObjectWithUseList[BlockOperand]");
  nb::class_<mlir::IRObjectWithUseList<mlir::OpOperand>>(
      m, "IRObjectWithUseList[OpOperand]");

  nb::class_<mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>>(
      m, "DialectResourceBlobHandle[BuiltinDialect]");

  nb::class_<mlir::AttrTypeSubElementReplacements<mlir::Attribute>>(
      m, "AttrTypeSubElementReplacements[Attribute]");
  nb::class_<mlir::AttrTypeSubElementReplacements<mlir::Type>>(
      m, "AttrTypeSubElementReplacements[Type]");

  nb::class_<std::reverse_iterator<mlir::BlockArgument *>>(
      m, "reverse_iterator[BlockArgument]");

  nb::class_<llvm::SmallPtrSetImpl<mlir::Operation *>>(
      m, "SmallPtrSetImpl[Operation]");

  nb::class_<mlir::ValueUseIterator<mlir::OpOperand>>(
      m, "ValueUseIterator[OpOperand]");
  nb::class_<mlir::ValueUseIterator<mlir::BlockOperand>>(
      m, "ValueUseIterator[BlockOperand]");

  nb::class_<std::initializer_list<mlir::Type>>(m, "initializer_list[Type]");
  nb::class_<std::initializer_list<mlir::Value>>(m, "initializer_list[Value]");
  nb::class_<std::initializer_list<mlir::Block *>>(m,
                                                   "initializer_list[Block]");

  nb::class_<llvm::SmallBitVector>(m, "SmallBitVector");
  nb::class_<llvm::BitVector>(m, "BitVector");

  auto [smallVectorOfBool, arrayRefOfBool, mutableArrayRefOfBool] =
      bind_array_ref<bool>(m);
  auto [smallVectorOfFloat, arrayRefOfFloat, mutableArrayRefOfFloat] =
      bind_array_ref<float>(m);
  auto [smallVectorOfInt, arrayRefOfInt, mutableArrayRefOfInt] =
      bind_array_ref<int>(m);

  auto [smallVectorOfChar, arrayRefOfChar, mutableArrayRefOfChar] =
      bind_array_ref<char>(m);
  auto [smallVectorOfDouble, arrayRefOfDouble, mutableArrayRefOfDouble] =
      bind_array_ref<double>(m);

  auto [smallVectorOfInt16, arrayRefOfInt16, mutableArrayRefOfInt16] =
      bind_array_ref<int16_t>(m);
  auto [smallVectorOfInt32, arrayRefOfInt32, mutableArrayRefOfInt32] =
      bind_array_ref<int32_t>(m);
  auto [smallVectorOfInt64, arrayRefOfInt64, mutableArrayRefOfInt64] =
      bind_array_ref<int64_t>(m);

  auto [smallVectorOfUInt16, arrayRefOfUInt16, mutableArrayRefOfUInt16] =
      bind_array_ref<uint16_t>(m);
  auto [smallVectorOfUInt32, arrayRefOfUInt32, mutableArrayRefOfUInt32] =
      bind_array_ref<uint32_t>(m);
  auto [smallVectorOfUInt64, arrayRefOfUInt64, mutableArrayRefOfUInt64] =
      bind_array_ref<uint64_t>(m);

  // these have to precede...
  bind_array_ref<mlir::Type>(m);
  bind_array_ref<mlir::Location>(m);
  bind_array_ref<mlir::Attribute>(m);
  bind_array_ref<mlir::AffineExpr>(m);
  bind_array_ref<mlir::AffineMap>(m);
  bind_array_ref<mlir::IRUnit>(m);
  bind_array_ref<mlir::Dialect *>(m);

  bind_array_ref<mlir::RegisteredOperationName>(m);

  bind_array_ref<llvm::APInt>(m);
  bind_array_ref<llvm::APFloat>(m);
  bind_array_ref<mlir::Value>(m);
  bind_array_ref<mlir::StringAttr>(m);
  bind_array_ref<mlir::OperationName>(m);
  bind_array_ref<mlir::Region *>(m);
  bind_array_ref<mlir::SymbolTable *>(m);
  bind_array_ref<mlir::Operation *>(m);
  bind_array_ref<mlir::OpFoldResult>(m);
  bind_array_ref<mlir::NamedAttribute>(m);

  bind_array_ref<mlir::FlatSymbolRefAttr>(m);
  bind_array_ref<mlir::BlockArgument>(m);
  bind_array_ref<mlir::Block *>(m);

  bind_array_ref<llvm::StringRef>(m);
  bind_array_ref<mlir::DiagnosticArgument>(m);
  // bind_array_ref<mlir::PDLValue>(m);
  bind_array_ref<mlir::OpAsmParser::Argument>(m);
  bind_array_ref<mlir::OpAsmParser::UnresolvedOperand>(m);

  smallVector.def_static("__class_getitem__",
                         // https://stackoverflow.com/a/48103632
                         [smallVectorOfBool = smallVectorOfBool,
                          smallVectorOfInt16 = smallVectorOfInt16,
                          smallVectorOfInt32 = smallVectorOfInt32,
                          smallVectorOfInt64 = smallVectorOfInt64,
                          smallVectorOfUInt16 = smallVectorOfUInt16,
                          smallVectorOfUInt32 = smallVectorOfUInt32,
                          smallVectorOfUInt64 = smallVectorOfUInt64,
                          smallVectorOfChar = smallVectorOfChar,
                          smallVectorOfDouble = smallVectorOfDouble](
                             nb::type_object type) -> nb::object {
                           PyTypeObject *typeObj = (PyTypeObject *)type.ptr();
                           if (typeObj == &PyBool_Type)
                             return smallVectorOfBool;
                           if (typeObj == &PyLong_Type)
                             return smallVectorOfInt64;
                           if (typeObj == &PyFloat_Type)
                             return smallVectorOfDouble;

                           auto np = nb::module_::import_("numpy");
                           auto npCharDType = np.attr("char");
                           auto npDoubleDType = np.attr("double");
                           auto npInt16DType = np.attr("int16");
                           auto npInt32DType = np.attr("int32");
                           auto npInt64DType = np.attr("int64");
                           auto npUInt16DType = np.attr("uint16");
                           auto npUInt32DType = np.attr("uint32");
                           auto npUInt64DType = np.attr("uint64");

                           if (type.is(npCharDType))
                             return smallVectorOfChar;
                           if (type.is(npDoubleDType))
                             return smallVectorOfDouble;
                           if (type.is(npInt16DType))
                             return smallVectorOfInt16;
                           if (type.is(npInt32DType))
                             return smallVectorOfInt32;
                           if (type.is(npInt64DType))
                             return smallVectorOfInt64;
                           if (type.is(npUInt16DType))
                             return smallVectorOfUInt16;
                           if (type.is(npUInt32DType))
                             return smallVectorOfUInt32;
                           if (type.is(npUInt64DType))
                             return smallVectorOfUInt64;

                           std::string errMsg =
                               "unsupported type for SmallVector";
                           errMsg += nb::repr(type).c_str();
                           throw std::runtime_error(errMsg);
                         });

  smallVector.def_static("__class_getitem__",
                         [smallVectorOfFloat = smallVectorOfFloat,
                          smallVectorOfInt16 = smallVectorOfInt16,
                          smallVectorOfInt32 = smallVectorOfInt32,
                          smallVectorOfInt64 = smallVectorOfInt64,
                          smallVectorOfUInt16 = smallVectorOfUInt16,
                          smallVectorOfUInt32 = smallVectorOfUInt32,
                          smallVectorOfUInt64 = smallVectorOfUInt64,
                          smallVectorOfChar = smallVectorOfChar,
                          smallVectorOfDouble = smallVectorOfDouble](
                             std::string type) -> nb::object {
                           if (type == "char")
                             return smallVectorOfChar;
                           if (type == "float")
                             return smallVectorOfFloat;
                           if (type == "double")
                             return smallVectorOfDouble;
                           if (type == "int16")
                             return smallVectorOfInt16;
                           if (type == "int32")
                             return smallVectorOfInt32;
                           if (type == "int64")
                             return smallVectorOfInt64;
                           if (type == "uint16")
                             return smallVectorOfUInt16;
                           if (type == "uint32")
                             return smallVectorOfUInt32;
                           if (type == "uint64")
                             return smallVectorOfUInt64;

                           std::string errMsg =
                               "unsupported type for SmallVector: ";
                           errMsg += type;
                           throw std::runtime_error(errMsg);
                         });

  nb::class_<llvm::iterator_range<mlir::BlockArgument *>>(
      m, "iterator_range[BlockArgument]");
  nb::class_<llvm::iterator_range<mlir::PredecessorIterator>>(
      m, "iterator_range[PredecessorIterator]");
  nb::class_<llvm::iterator_range<mlir::Region::OpIterator>>(
      m, "iterator_range[Region.OpIterator]");
  nb::class_<llvm::iterator_range<mlir::Operation::dialect_attr_iterator>>(
      m, "iterator_range[Operation.dialect_attr_iterator]");
  nb::class_<llvm::iterator_range<mlir::ResultRange::UseIterator>>(
      m, "iterator_range[ResultRange.UseIterator]");

  bind_iter_range<mlir::ValueTypeRange<mlir::ValueRange>, mlir::Type>(
      m, "ValueTypeRange[ValueRange]");
  bind_iter_range<mlir::ValueTypeRange<mlir::OperandRange>, mlir::Type>(
      m, "ValueTypeRange[OperandRange]");
  bind_iter_range<mlir::ValueTypeRange<mlir::ResultRange>, mlir::Type>(
      m, "ValueTypeRange[ResultRange]");

  bind_iter_like<llvm::iplist<mlir::Block>, nb::rv_policy::reference_internal>(
      m, "iplist[Block]");
  bind_iter_like<llvm::iplist<mlir::Operation>,
                 nb::rv_policy::reference_internal>(m, "iplist[Operation]");

  auto dialectsModule = m.def_submodule("dialects");

  // auto accModule = dialectsModule.def_submodule("acc");
  // populateEUDSLGen_accModule(accModule);

  // auto affineModule = dialectsModule.def_submodule("affine");
  // populateEUDSLGen_affineModule(affineModule);
  //
  // auto amdgpuModule = dialectsModule.def_submodule("amdgpu");
  // populateEUDSLGen_amdgpuModule(amdgpuModule);

  // auto amxModule = dialectsModule.def_submodule("amx");
  // populateEUDSLGen_amxModule(amxModule);

  auto arithModule = dialectsModule.def_submodule("arith");
  populateEUDSLGen_arithModule(arithModule);

  // auto arm_neonModule = dialectsModule.def_submodule("arm_neon");
  // populateEUDSLGen_arm_neonModule(arm_neonModule);

  // auto arm_smeModule = dialectsModule.def_submodule("arm_sme");
  // populateEUDSLGen_arm_smeModule(arm_smeModule);

  // auto arm_sveModule = dialectsModule.def_submodule("arm_sve");
  // populateEUDSLGen_arm_sveModule(arm_sveModule);

  // auto asyncModule = dialectsModule.def_submodule("async");
  // populateEUDSLGen_asyncModule(asyncModule);
  //
  // auto bufferizationModule = dialectsModule.def_submodule("bufferization");
  // populateEUDSLGen_bufferizationModule(bufferizationModule);
  //
  // auto cfModule = dialectsModule.def_submodule("cf");
  // populateEUDSLGen_cfModule(cfModule);
  //
  // auto complexModule = dialectsModule.def_submodule("complex");
  // populateEUDSLGen_complexModule(complexModule);
  //
  // // auto DLTIDialectModule = dialectsModule.def_submodule("DLTIDialect");
  // // populateEUDSLGen_DLTIDialectModule(DLTIDialectModule);
  //
  // auto emitcModule = dialectsModule.def_submodule("emitc");
  // populateEUDSLGen_emitcModule(emitcModule);
  //
  // auto funcModule = dialectsModule.def_submodule("func");
  // populateEUDSLGen_funcModule(funcModule);
  //
  // auto gpuModule = dialectsModule.def_submodule("gpu");
  // populateEUDSLGen_gpuModule(gpuModule);
  //
  // auto indexModule = dialectsModule.def_submodule("index");
  // populateEUDSLGen_indexModule(indexModule);
  //
  // // auto irdlModule = dialectsModule.def_submodule("irdl");
  // // populateEUDSLGen_irdlModule(irdlModule);
  //
  // auto linalgModule = dialectsModule.def_submodule("linalg");
  // populateEUDSLGen_linalgModule(linalgModule);
  //
  // auto LLVMModule = dialectsModule.def_submodule("LLVM");
  // populateEUDSLGen_LLVMModule(LLVMModule);
  //
  // auto mathModule = dialectsModule.def_submodule("math");
  // populateEUDSLGen_mathModule(mathModule);
  //
  // auto memrefModule = dialectsModule.def_submodule("memref");
  // populateEUDSLGen_memrefModule(memrefModule);
  //
  // // auto meshModule = dialectsModule.def_submodule("mesh");
  // // populateEUDSLGen_meshModule(meshModule);
  //
  // // auto ml_programModule = dialectsModule.def_submodule("ml_program");
  // // populateEUDSLGen_ml_programModule(ml_programModule);
  //
  // // auto mpiModule = dialectsModule.def_submodule("mpi");
  // // populateEUDSLGen_mpiModule(mpiModule);
  //
  // auto nvgpuModule = dialectsModule.def_submodule("nvgpu");
  // populateEUDSLGen_nvgpuModule(nvgpuModule);
  //
  // auto NVVMModule = dialectsModule.def_submodule("NVVM");
  // populateEUDSLGen_NVVMModule(NVVMModule);
  //
  // // auto ompModule = dialectsModule.def_submodule("omp");
  // // populateEUDSLGen_ompModule(ompModule);
  //
  // auto pdlModule = dialectsModule.def_submodule("pdl");
  // populateEUDSLGen_pdlModule(pdlModule);
  //
  // auto pdl_interpModule = dialectsModule.def_submodule("pdl_interp");
  // populateEUDSLGen_pdl_interpModule(pdl_interpModule);
  //
  // auto polynomialModule = dialectsModule.def_submodule("polynomial");
  // populateEUDSLGen_polynomialModule(polynomialModule);
  //
  // // auto ptrModule = dialectsModule.def_submodule("ptr");
  // // populateEUDSLGen_ptrModule(ptrModule);
  //
  // // auto quantModule = dialectsModule.def_submodule("quant");
  // // populateEUDSLGen_quantModule(quantModule);
  //
  // auto ROCDLModule = dialectsModule.def_submodule("ROCDL");
  // populateEUDSLGen_ROCDLModule(ROCDLModule);
  //
  // auto scfModule = dialectsModule.def_submodule("scf");
  // populateEUDSLGen_scfModule(scfModule);
  //
  // auto shapeModule = dialectsModule.def_submodule("shape");
  // populateEUDSLGen_shapeModule(shapeModule);
  //
  // // auto sparse_tensorModule =
  // dialectsModule.def_submodule("sparse_tensor");
  // // populateEUDSLGen_sparse_tensorModule(sparse_tensorModule);
  //
  // // auto spirvModule = dialectsModule.def_submodule("spirv");
  // // populateEUDSLGen_spirvModule(spirvModule);
  //
  // auto tensorModule = dialectsModule.def_submodule("tensor");
  // populateEUDSLGen_tensorModule(tensorModule);

  // auto tosaModule = dialectsModule.def_submodule("tosa");
  // populateEUDSLGen_tosaModule(tosaModule);

  // auto transformModule = dialectsModule.def_submodule("transform");
  // populateEUDSLGen_transformModule(transformModule);

  // auto ubModule = dialectsModule.def_submodule("ub");
  // populateEUDSLGen_ubModule(ubModule);

  // auto vectorModule = dialectsModule.def_submodule("vector");
  // populateEUDSLGen_vectorModule(vectorModule);

  // auto x86vectorModule = dialectsModule.def_submodule("x86vector");
  // populateEUDSLGen_x86vectorModule(x86vectorModule);

  // auto xegpuModule = dialectsModule.def_submodule("xegpu");
  // populateEUDSLGen_xegpuModule(xegpuModule);
}
