#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
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

#include "bind_vec_like.h"
#include "type_casters.h"

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

void bind_array_ref_smallvector(nb::handle scope) {
  scope.attr("T") = nb::type_var("T");
  arrayRef = nb::class_<_ArrayRef>(scope, "ArrayRef", nb::is_generic(),
                                   nb::sig("class ArrayRef[T]"));
  mutableArrayRef =
      nb::class_<_MutableArrayRef>(scope, "MutableArrayRef", nb::is_generic(),
                                   nb::sig("class MutableArrayRef[T]"));
  smallVector = nb::class_<_SmallVector>(scope, "SmallVector", nb::is_generic(),
                                         nb::sig("class SmallVector[T]"));
}

template <typename T, typename... Ts>
struct non_copying_non_moving_class_ : nb::class_<T, Ts...> {
  template <typename... Extra>
  NB_INLINE non_copying_non_moving_class_(nb::handle scope, const char *name,
                                          const Extra &...extra) {
    nb::detail::type_init_data d;

    d.flags = 0;
    d.align = (uint8_t)alignof(typename nb::class_<T, Ts...>::Alias);
    d.size = (uint32_t)sizeof(typename nb::class_<T, Ts...>::Alias);
    d.name = name;
    d.scope = scope.ptr();
    d.type = &typeid(T);

    if constexpr (!std::is_same_v<typename nb::class_<T, Ts...>::Base, T>) {
      d.base = &typeid(typename nb::class_<T, Ts...>::Base);
      d.flags |= (uint32_t)nb::detail::type_init_flags::has_base;
    }

    if constexpr (std::is_destructible_v<T>) {
      d.flags |= (uint32_t)nb::detail::type_flags::is_destructible;

      if constexpr (!std::is_trivially_destructible_v<T>) {
        d.flags |= (uint32_t)nb::detail::type_flags::has_destruct;
        d.destruct = nb::detail::wrap_destruct<T>;
      }
    }

    if constexpr (nb::detail::has_shared_from_this_v<T>) {
      d.flags |= (uint32_t)nb::detail::type_flags::has_shared_from_this;
      d.keep_shared_from_this_alive = [](PyObject *self) noexcept {
        if (auto sp = nb::inst_ptr<T>(self)->weak_from_this().lock()) {
          nb::detail::keep_alive(
              self, new auto(std::move(sp)),
              [](void *p) noexcept { delete (decltype(sp) *)p; });
          return true;
        }
        return false;
      };
    }

    (nb::detail::type_extra_apply(d, extra), ...);

    this->m_ptr = nb::detail::nb_type_new(&d);
  }
};

void populateIRModule(nb::module_ &m) {
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

extern void populateaccModule(nb::module_ &m);

extern void populateaffineModule(nb::module_ &m);

extern void populateamdgpuModule(nb::module_ &m);

extern void populateamxModule(nb::module_ &m);

extern void populatearithModule(nb::module_ &m);

extern void populatearm_neonModule(nb::module_ &m);

extern void populatearm_smeModule(nb::module_ &m);

extern void populatearm_sveModule(nb::module_ &m);

extern void populateasyncModule(nb::module_ &m);

extern void populatebufferizationModule(nb::module_ &m);

extern void populatecfModule(nb::module_ &m);

extern void populatecomplexModule(nb::module_ &m);

extern void populateDLTIDialectModule(nb::module_ &m);

// mlir::emitc::IfOp::elseBlock()
// void populateemitcModule(nb::module_ &m) {
// #include "EUDSLGenemitc.cpp.inc"
// }

extern void populatefuncModule(nb::module_ &m);

extern void populategpuModule(nb::module_ &m);

extern void populateindexModule(nb::module_ &m);

// error: use of class template 'ArrayRef' requires template arguments; argument
// deduction not allowed in conversion function type void
// populateirdlModule(nb::module_ &m) {
//   using namespace llvm;
// #include "EUDSLGenirdl.cpp.inc"
// }

// __ZN4mlir6linalg11TransposeOp12createRegionERNS_9OpBuilderERNS_14OperationStateE
// void populatelinalgModule(nb::module_ &m) {
//   using namespace mlir;
// #include "EUDSLGenlinalg.cpp.inc"
// }

// missing LLVM_TargetFeaturesAttr builder
// void populateLLVMModule(nb::module_ &m) {
// #include "EUDSLGenLLVM.cpp.inc"
// }

extern void populatemathModule(nb::module_ &m);

extern void populatememrefModule(nb::module_ &m);

extern void populatemeshModule(nb::module_ &m);

extern void populateml_programModule(nb::module_ &m);

extern void populatempiModule(nb::module_ &m);

extern void populatenvgpuModule(nb::module_ &m);

extern void populateNVVMModule(nb::module_ &m);

// mlir::omp::TaskloopOp::getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&)
// void populateompModule(nb::module_ &m) {
// #include "EUDSLGenomp.cpp.inc"
// }

extern void populatepdlModule(nb::module_ &m);

extern void populatepdl_interpModule(nb::module_ &m);

extern void populatepolynomialModule(nb::module_ &m);

extern void populateptrModule(nb::module_ &m);

extern void populatequantModule(nb::module_ &m);

extern void populateROCDLModule(nb::module_ &m);

// missing ensureLoopTerminator
// extern void populatescfModule(nb::module_ &m);

// missing dim op builder
// extern void populateshapeModule(nb::module_ &m);

extern void populatesparse_tensorModule(nb::module_ &m);

// too big...
// extern void populatespirvModule(nb::module_ &m);

extern void populatetensorModule(nb::module_ &m);

extern void populatetosaModule(nb::module_ &m);

extern void populatetransformModule(nb::module_ &m);

extern void populateubModule(nb::module_ &m);

// can't cast std::pair<VectorDim, VectorDim>
// void populatevectorModule(nb::module_ &m) {
// #include "EUDSLGenvector.cpp.inc"
// }

extern void populatex86vectorModule(nb::module_ &m);

// missing mlir::xegpu::SGMapAttr::getChecked
// void populatexegpuModule(nb::module_ &m) {
// #include "EUDSLGenxegpu.cpp.inc"
// }

NB_MODULE(eudslpy_ext, m) {
  // nb::class_<mlir::SymbolTableCollection>(m, "SymbolTableCollection");
  // nb::class_<mlir::FallbackAsmResourceMap>(m, "FallbackAsmResourceMap");

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
  auto [smallVectorOfLong, arrayRefOfLong, mutableArrayRefOfLong] =
      bind_array_ref<long>(m);
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

  smallVector.def_static(
      "__class_getitem__",
      [smallVectorOfBool, smallVectorOfInt, smallVectorOfFloat,
       smallVectorOfInt16, smallVectorOfInt32, smallVectorOfInt64,
       smallVectorOfUInt16, smallVectorOfUInt32, smallVectorOfUInt64,
       smallVectorOfChar, smallVectorOfDouble,
       smallVectorOfLong](nb::type_object type) -> nb::object {
        PyTypeObject *typeObj = (PyTypeObject *)type.ptr();
        if (typeObj == &PyBool_Type)
          return smallVectorOfBool;
        if (typeObj == &PyLong_Type)
          return smallVectorOfInt;
        if (typeObj == &PyFloat_Type)
          return smallVectorOfFloat;

        auto np = nb::module_::import_("numpy");
        auto charDType = np.attr("char");
        auto doubleDType = np.attr("double");
        auto longDType = np.attr("long");
        auto int16DType = np.attr("int16");
        auto int32DType = np.attr("int32");
        auto int64DType = np.attr("int64");
        auto uint16DType = np.attr("uint16");
        auto uint32DType = np.attr("uint32");
        auto uint64DType = np.attr("uint64");

        if (type.is(charDType))
          return smallVectorOfChar;
        if (type.is(doubleDType))
          return smallVectorOfDouble;
        if (type.is(longDType))
          return smallVectorOfLong;
        if (type.is(int16DType))
          return smallVectorOfInt16;
        if (type.is(int32DType))
          return smallVectorOfInt32;
        if (type.is(int64DType))
          return smallVectorOfInt64;
        if (type.is(uint16DType))
          return smallVectorOfUInt16;
        if (type.is(uint32DType))
          return smallVectorOfUInt32;
        if (type.is(uint64DType))
          return smallVectorOfUInt64;
        std::string errMsg = "unsupported type for SmallVector";
        errMsg += nb::repr(type).c_str();
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

  auto irModule = m.def_submodule("ir");
  populateIRModule(irModule);
  auto dialectsModule = m.def_submodule("dialects");

  auto accModule = dialectsModule.def_submodule("acc");
  populateaccModule(accModule);

  auto affineModule = dialectsModule.def_submodule("affine");
  populateaffineModule(affineModule);

  auto amdgpuModule = dialectsModule.def_submodule("amdgpu");
  populateamdgpuModule(amdgpuModule);

  auto amxModule = dialectsModule.def_submodule("amx");
  populateamxModule(amxModule);

  auto arithModule = dialectsModule.def_submodule("arith");
  populatearithModule(arithModule);

  auto arm_neonModule = dialectsModule.def_submodule("arm_neon");
  populatearm_neonModule(arm_neonModule);

  auto arm_smeModule = dialectsModule.def_submodule("arm_sme");
  populatearm_smeModule(arm_smeModule);

  auto arm_sveModule = dialectsModule.def_submodule("arm_sve");
  populatearm_sveModule(arm_sveModule);

  auto asyncModule = dialectsModule.def_submodule("async");
  populateasyncModule(asyncModule);

  auto bufferizationModule = dialectsModule.def_submodule("bufferization");
  populatebufferizationModule(bufferizationModule);

  auto cfModule = dialectsModule.def_submodule("cf");
  populatecfModule(cfModule);

  auto complexModule = dialectsModule.def_submodule("complex");
  populatecomplexModule(complexModule);

  auto DLTIDialectModule = dialectsModule.def_submodule("DLTIDialect");
  populateDLTIDialectModule(DLTIDialectModule);

  // auto emitcModule = dialectsModule.def_submodule("emitc");
  // populateemitcModule(emitcModule);

  auto funcModule = dialectsModule.def_submodule("func");
  populatefuncModule(funcModule);

  auto gpuModule = dialectsModule.def_submodule("gpu");
  populategpuModule(gpuModule);

  auto indexModule = dialectsModule.def_submodule("index");
  populateindexModule(indexModule);

  // auto irdlModule = dialectsModule.def_submodule("irdl");
  // populateirdlModule(irdlModule);

  // auto linalgModule = dialectsModule.def_submodule("linalg");
  // populatelinalgModule(linalgModule);

  // auto LLVMModule = dialectsModule.def_submodule("LLVM");
  // populateLLVMModule(LLVMModule);

  auto mathModule = dialectsModule.def_submodule("math");
  populatemathModule(mathModule);

  auto memrefModule = dialectsModule.def_submodule("memref");
  populatememrefModule(memrefModule);

  auto meshModule = dialectsModule.def_submodule("mesh");
  populatemeshModule(meshModule);

  auto ml_programModule = dialectsModule.def_submodule("ml_program");
  populateml_programModule(ml_programModule);

  auto mpiModule = dialectsModule.def_submodule("mpi");
  populatempiModule(mpiModule);

  auto nvgpuModule = dialectsModule.def_submodule("nvgpu");
  populatenvgpuModule(nvgpuModule);

  auto NVVMModule = dialectsModule.def_submodule("NVVM");
  populateNVVMModule(NVVMModule);

  // auto ompModule = dialectsModule.def_submodule("omp");
  // populateompModule(ompModule);

  auto pdlModule = dialectsModule.def_submodule("pdl");
  populatepdlModule(pdlModule);

  auto pdl_interpModule = dialectsModule.def_submodule("pdl_interp");
  populatepdl_interpModule(pdl_interpModule);

  auto polynomialModule = dialectsModule.def_submodule("polynomial");
  populatepolynomialModule(polynomialModule);

  auto ptrModule = dialectsModule.def_submodule("ptr");
  populateptrModule(ptrModule);

  auto quantModule = dialectsModule.def_submodule("quant");
  populatequantModule(quantModule);

  auto ROCDLModule = dialectsModule.def_submodule("ROCDL");
  populateROCDLModule(ROCDLModule);

  // auto scfModule = dialectsModule.def_submodule("scf");
  // populatescfModule(scfModule);

  // auto shapeModule = dialectsModule.def_submodule("shape");
  // populateshapeModule(shapeModule);

  auto sparse_tensorModule = dialectsModule.def_submodule("sparse_tensor");
  populatesparse_tensorModule(sparse_tensorModule);

  // auto spirvModule = dialectsModule.def_submodule("spirv");
  // populatespirvModule(spirvModule);

  auto tensorModule = dialectsModule.def_submodule("tensor");
  populatetensorModule(tensorModule);

  auto tosaModule = dialectsModule.def_submodule("tosa");
  populatetosaModule(tosaModule);

  auto transformModule = dialectsModule.def_submodule("transform");
  populatetransformModule(transformModule);

  auto ubModule = dialectsModule.def_submodule("ub");
  populateubModule(ubModule);

  // auto vectorModule = dialectsModule.def_submodule("vector");
  // populatevectorModule(vectorModule);

  auto x86vectorModule = dialectsModule.def_submodule("x86vector");
  populatex86vectorModule(x86vectorModule);

  // auto xegpuModule = dialectsModule.def_submodule("xegpu");
  // populatexegpuModule(xegpuModule);
}
