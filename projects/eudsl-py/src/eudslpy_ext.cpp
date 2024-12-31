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

// too big
// extern void populateEUDSLGenaccModule(nb::module_ &m);

extern void populateEUDSLGenaffineModule(nb::module_ &m);

extern void populateEUDSLGenamdgpuModule(nb::module_ &m);

// extern void populateEUDSLGenamxModule(nb::module_ &m);

extern void populateEUDSLGenarithModule(nb::module_ &m);

// extern void populateEUDSLGenarm_neonModule(nb::module_ &m);

// too big
// extern void populateEUDSLGenarm_smeModule(nb::module_ &m);

// extern void populateEUDSLGenarm_sveModule(nb::module_ &m);

extern void populateEUDSLGenasyncModule(nb::module_ &m);

extern void populateEUDSLGenbufferizationModule(nb::module_ &m);

extern void populateEUDSLGencfModule(nb::module_ &m);

extern void populateEUDSLGencomplexModule(nb::module_ &m);

// extern void populateEUDSLGenDLTIDialectModule(nb::module_ &m);

extern void populateEUDSLGenemitcModule(nb::module_ &m);

extern void populateEUDSLGenfuncModule(nb::module_ &m);

extern void populateEUDSLGengpuModule(nb::module_ &m);

extern void populateEUDSLGenindexModule(nb::module_ &m);

// error: use of class template 'ArrayRef' requires template arguments; argument
// deduction not allowed in conversion function type void
// populateEUDSLGenirdlModule(nb::module_ &m) {
//   using namespace llvm;
// #include "EUDSLGenirdl.cpp.inc"
// }

extern void populateEUDSLGenlinalgModule(nb::module_ &m);

extern void populateEUDSLGenLLVMModule(nb::module_ &m);

extern void populateEUDSLGenmathModule(nb::module_ &m);

extern void populateEUDSLGenmemrefModule(nb::module_ &m);

// extern void populateEUDSLGenmeshModule(nb::module_ &m);

// extern void populateEUDSLGenml_programModule(nb::module_ &m);

// extern void populateEUDSLGenmpiModule(nb::module_ &m);

extern void populateEUDSLGennvgpuModule(nb::module_ &m);

extern void populateEUDSLGenNVVMModule(nb::module_ &m);

// mlir::omp::TaskloopOp::getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&)
// void populateEUDSLGenompModule(nb::module_ &m) {
// #include "EUDSLGenomp.cpp.inc"
// }

extern void populateEUDSLGenpdlModule(nb::module_ &m);

extern void populateEUDSLGenpdl_interpModule(nb::module_ &m);

extern void populateEUDSLGenpolynomialModule(nb::module_ &m);

// extern void populateEUDSLGenptrModule(nb::module_ &m);

extern void populateEUDSLGenquantModule(nb::module_ &m);

extern void populateEUDSLGenROCDLModule(nb::module_ &m);

extern void populateEUDSLGenscfModule(nb::module_ &m);

extern void populateEUDSLGenshapeModule(nb::module_ &m);

// extern void populateEUDSLGensparse_tensorModule(nb::module_ &m);

// nb::detail::nb_func_new("get_vce_triple_attr_name"): mismatched
// static/instance method flags in function overloads! extern void
// populateEUDSLGenspirvModule(nb::module_ &m);

extern void populateEUDSLGentensorModule(nb::module_ &m);

extern void populateEUDSLGentosaModule(nb::module_ &m);

extern void populateEUDSLGentransformModule(nb::module_ &m);

extern void populateEUDSLGenubModule(nb::module_ &m);

// can't cast std::pair<VectorDim, VectorDim>
// void populateEUDSLGenvectorModule(nb::module_ &m) {
// #include "EUDSLGenvector.cpp.inc"
// }

extern void populateEUDSLGenx86vectorModule(nb::module_ &m);

// extern void populateEUDSLGenxegpuModule(nb::module_ &m);

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
        nb::print(type);
        if (typeObj == &PyBool_Type)
          return smallVectorOfBool;
        if (typeObj == &PyLong_Type)
          return smallVectorOfInt64;
        if (typeObj == &PyFloat_Type)
          return smallVectorOfDouble;
        nb::print("Wtfbbq");

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

        std::string errMsg = "unsupported type for SmallVector";
        errMsg += nb::repr(type).c_str();
        throw std::runtime_error(errMsg);
      });

  smallVector.def_static(
      "__class_getitem__",
      [smallVectorOfFloat, smallVectorOfInt16, smallVectorOfInt32,
       smallVectorOfInt64, smallVectorOfUInt16, smallVectorOfUInt32,
       smallVectorOfUInt64, smallVectorOfChar, smallVectorOfDouble,
       smallVectorOfLong](std::string type) -> nb::object {
        if (type == "char")
          return smallVectorOfChar;
        if (type == "float")
          return smallVectorOfFloat;
        if (type == "double")
          return smallVectorOfDouble;
        if (type == "long")
          return smallVectorOfLong;
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

        std::string errMsg = "unsupported type for SmallVector: ";
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

  auto irModule = m.def_submodule("ir");
  populateIRModule(irModule);
  auto dialectsModule = m.def_submodule("dialects");

  // auto accModule = dialectsModule.def_submodule("acc");
  // populateEUDSLGenaccModule(accModule);

  auto affineModule = dialectsModule.def_submodule("affine");
  populateEUDSLGenaffineModule(affineModule);

  auto amdgpuModule = dialectsModule.def_submodule("amdgpu");
  populateEUDSLGenamdgpuModule(amdgpuModule);

  // auto amxModule = dialectsModule.def_submodule("amx");
  // populateEUDSLGenamxModule(amxModule);

  auto arithModule = dialectsModule.def_submodule("arith");
  populateEUDSLGenarithModule(arithModule);

  // auto arm_neonModule = dialectsModule.def_submodule("arm_neon");
  // populateEUDSLGenarm_neonModule(arm_neonModule);

  // auto arm_smeModule = dialectsModule.def_submodule("arm_sme");
  // populateEUDSLGenarm_smeModule(arm_smeModule);

  // auto arm_sveModule = dialectsModule.def_submodule("arm_sve");
  // populateEUDSLGenarm_sveModule(arm_sveModule);

  auto asyncModule = dialectsModule.def_submodule("async");
  populateEUDSLGenasyncModule(asyncModule);

  auto bufferizationModule = dialectsModule.def_submodule("bufferization");
  populateEUDSLGenbufferizationModule(bufferizationModule);

  auto cfModule = dialectsModule.def_submodule("cf");
  populateEUDSLGencfModule(cfModule);

  auto complexModule = dialectsModule.def_submodule("complex");
  populateEUDSLGencomplexModule(complexModule);

  // auto DLTIDialectModule = dialectsModule.def_submodule("DLTIDialect");
  // populateEUDSLGenDLTIDialectModule(DLTIDialectModule);

  auto emitcModule = dialectsModule.def_submodule("emitc");
  populateEUDSLGenemitcModule(emitcModule);

  auto funcModule = dialectsModule.def_submodule("func");
  populateEUDSLGenfuncModule(funcModule);

  auto gpuModule = dialectsModule.def_submodule("gpu");
  populateEUDSLGengpuModule(gpuModule);

  auto indexModule = dialectsModule.def_submodule("index");
  populateEUDSLGenindexModule(indexModule);

  // auto irdlModule = dialectsModule.def_submodule("irdl");
  // populateEUDSLGenirdlModule(irdlModule);

  auto linalgModule = dialectsModule.def_submodule("linalg");
  populateEUDSLGenlinalgModule(linalgModule);

  auto LLVMModule = dialectsModule.def_submodule("LLVM");
  populateEUDSLGenLLVMModule(LLVMModule);

  auto mathModule = dialectsModule.def_submodule("math");
  populateEUDSLGenmathModule(mathModule);

  auto memrefModule = dialectsModule.def_submodule("memref");
  populateEUDSLGenmemrefModule(memrefModule);

  // auto meshModule = dialectsModule.def_submodule("mesh");
  // populateEUDSLGenmeshModule(meshModule);

  // auto ml_programModule = dialectsModule.def_submodule("ml_program");
  // populateEUDSLGenml_programModule(ml_programModule);

  // auto mpiModule = dialectsModule.def_submodule("mpi");
  // populateEUDSLGenmpiModule(mpiModule);

  auto nvgpuModule = dialectsModule.def_submodule("nvgpu");
  populateEUDSLGennvgpuModule(nvgpuModule);

  auto NVVMModule = dialectsModule.def_submodule("NVVM");
  populateEUDSLGenNVVMModule(NVVMModule);

  // auto ompModule = dialectsModule.def_submodule("omp");
  // populateEUDSLGenompModule(ompModule);

  auto pdlModule = dialectsModule.def_submodule("pdl");
  populateEUDSLGenpdlModule(pdlModule);

  auto pdl_interpModule = dialectsModule.def_submodule("pdl_interp");
  populateEUDSLGenpdl_interpModule(pdl_interpModule);

  auto polynomialModule = dialectsModule.def_submodule("polynomial");
  populateEUDSLGenpolynomialModule(polynomialModule);

  // auto ptrModule = dialectsModule.def_submodule("ptr");
  // populateEUDSLGenptrModule(ptrModule);

  // auto quantModule = dialectsModule.def_submodule("quant");
  // populateEUDSLGenquantModule(quantModule);

  auto ROCDLModule = dialectsModule.def_submodule("ROCDL");
  populateEUDSLGenROCDLModule(ROCDLModule);

  auto scfModule = dialectsModule.def_submodule("scf");
  populateEUDSLGenscfModule(scfModule);

  auto shapeModule = dialectsModule.def_submodule("shape");
  populateEUDSLGenshapeModule(shapeModule);

  // auto sparse_tensorModule = dialectsModule.def_submodule("sparse_tensor");
  // populateEUDSLGensparse_tensorModule(sparse_tensorModule);

  // auto spirvModule = dialectsModule.def_submodule("spirv");
  // populateEUDSLGenspirvModule(spirvModule);

  auto tensorModule = dialectsModule.def_submodule("tensor");
  populateEUDSLGentensorModule(tensorModule);

  auto tosaModule = dialectsModule.def_submodule("tosa");
  populateEUDSLGentosaModule(tosaModule);

  // auto transformModule = dialectsModule.def_submodule("transform");
  // populateEUDSLGentransformModule(transformModule);

  // auto ubModule = dialectsModule.def_submodule("ub");
  // populateEUDSLGenubModule(ubModule);

  // auto vectorModule = dialectsModule.def_submodule("vector");
  // populateEUDSLGenvectorModule(vectorModule);

  // auto x86vectorModule = dialectsModule.def_submodule("x86vector");
  // populateEUDSLGenx86vectorModule(x86vectorModule);

  // auto xegpuModule = dialectsModule.def_submodule("xegpu");
  // populateEUDSLGenxegpuModule(xegpuModule);
}
