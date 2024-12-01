#include <iostream>
#include <nanobind/nanobind.h>

#include "bind_vec_like.h"
#include "ir.h"

namespace nb = nanobind;
using namespace nb::literals;

class FakeDialect : public mlir::Dialect {
public:
  FakeDialect(llvm::StringRef name, mlir::MLIRContext *context, mlir::TypeID id)
      : Dialect(name, context, id) {}
};

NB_MODULE(eudsl_ext, m) {

  nb::class_<mlir::TypeID>(m, "TypeID");
  nb::class_<llvm::LogicalResult>(m, "LogicalResult");
  nb::class_<llvm::APSInt>(m, "APSInt");
  nb::class_<llvm::APInt>(m, "APInt");
  nb::class_<llvm::APFloat>(m, "APFloat");
  nb::class_<llvm::raw_ostream>(m, "raw_ostream");
  nb::class_<mlir::detail::InterfaceMap>(m, "InterfaceMap");
  nb::class_<llvm::SmallBitVector>(m, "SmallBitVector");
  nb::class_<llvm::SmallVectorImpl<mlir::Attribute>>(
      m, "SmallVectorImpl[Attribute]");
  nb::class_<llvm::SmallVectorImpl<long>>(m, "SmallVectorImpl[int]");
  nb::class_<llvm::FailureOr<bool>>(m, "FailureOr[bool]");
  nb::class_<llvm::FailureOr<mlir::StringAttr>>(m, "FailureOr[StringAttr]");
  nb::class_<llvm::FailureOr<mlir::AsmResourceBlob>>(
      m, "FailureOr[AsmResourceBlob]");
  // nb::class_<mlir::FallbackAsmResourceMap>(m, "FallbackAsmResourceMap");
  nb::class_<mlir::AsmResourcePrinter>(m, "AsmResourcePrinter");
  nb::class_<std::reverse_iterator<mlir::BlockArgument *>>(
      m, "reverse_iterator[BlockArgument]");
  nb::class_<llvm::iterator_range<mlir::BlockArgument *>>(
      m, "iterator_range[BlockArgument]");
  nb::class_<llvm::iterator_range<mlir::PredecessorIterator>>(
      m, "iterator_range[PredecessorIterator]");
  nb::class_<llvm::iterator_range<mlir::Region::OpIterator>>(
      m, "iterator_range[Region.OpIterator]");
  nb::class_<llvm::BitVector>(m, "BitVector");
  nb::class_<mlir::IRObjectWithUseList<mlir::BlockOperand>>(
      m, "IRObjectWithUseList[BlockOperand]");
  nb::class_<mlir::IRObjectWithUseList<mlir::OpOperand>>(
      m, "IRObjectWithUseList[OpOperand]");
  nb::class_<llvm::ArrayRef<mlir::Block *>>(m, "ArrayRef[Block]");
  nb::class_<mlir::AsmParser>(m, "AsmParser");
  nb::class_<mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>>(
      m, "DialectResourceBlobHandle[BuiltinDialect]");
  nb::class_<llvm::MutableArrayRef<mlir::DiagnosticArgument>>(
      m, "MutableArrayRef[DiagnosticArgument]");
  nb::class_<llvm::SmallVectorImpl<mlir::DiagnosticArgument>>(
      m, "SmallVectorImpl[DiagnosticArgument]");
  nb::class_<llvm::MutableArrayRef<mlir::Dialect *>>(
      m, "MutableArrayRef[Dialect]");
  nb::class_<llvm::SmallVectorImpl<mlir::NamedAttribute>>(
      m, "SmallVectorImpl[NamedAttribute]");
  nb::class_<llvm::ParseResult>(m, "ParseResult");
  nb::class_<llvm::SmallVectorImpl<mlir::OpFoldResult>>(
      m, "SmallVectorImpl[OpFoldResult]");
  nb::class_<llvm::hash_code>(m, "hash_code");
  nb::class_<mlir::AttrTypeSubElementReplacements<mlir::Attribute>>(
      m, "AttrTypeSubElementReplacements[Attribute]");
  nb::class_<mlir::AttrTypeSubElementReplacements<mlir::Type>>(
      m, "AttrTypeSubElementReplacements[Type]");
  nb::class_<llvm::FailureOr<mlir::AffineMap>>(m, "FailureOr[AffineMap]");
  nb::class_<llvm::SmallVectorImpl<mlir::OpAsmParser::Argument>>(
      m, "SmallVectorImpl[OpAsmParser.Argument]");
  nb::class_<llvm::MutableArrayRef<mlir::Region>>(m, "MutableArrayRef[Region]");

  nb::class_<llvm::FailureOr<mlir::detail::ElementsAttrIndexer>>(
      m, "FailureOr[ElementsAttrIndexer]");
  nb::class_<llvm::SmallPtrSetImpl<mlir::Operation *>>(
      m, "SmallPtrSetImpl[Operation]");
  nb::class_<mlir::ValueUseIterator<mlir::OpOperand>>(
      m, "ValueUseIterator[OpOperand]");
  nb::class_<mlir::ValueUseIterator<mlir::BlockOperand>>(
      m, "ValueUseIterator[BlockOperand]");
  nb::class_<std::initializer_list<mlir::Value>>(m, "initializer_list[Value]");
  nb::class_<std::initializer_list<mlir::Block *>>(m,
                                                   "initializer_list[Block]");
  nb::class_<mlir::StorageUniquer>(m, "StorageUniquer");
  nb::class_<llvm::ThreadPoolInterface>(m, "ThreadPoolInterface");
  nb::class_<llvm::SmallVectorImpl<mlir::Operation *>>(
      m, "SmallVectorImpl[Operation]");
  nb::class_<mlir::DialectBytecodeReader>(m, "DialectBytecodeReader");
  nb::class_<mlir::DataLayoutSpecInterface>(m, "DataLayoutSpecInterface");
  nb::class_<mlir::TargetSystemSpecInterface>(m, "TargetSystemSpecInterface");
  nb::class_<llvm::FailureOr<mlir::AsmDialectResourceHandle>>(
      m, "FailureOr[AsmDialectResourceHandle]");
  nb::class_<llvm::FailureOr<mlir::OperationName>>(m,
                                                   "FailureOr[OperationName]");
  nb::class_<llvm::SmallVectorImpl<mlir::Value>>(m, "SmallVectorImpl[Value]");
  nb::class_<llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>>(
      m, "SmallVectorImpl[OpAsmParser.UnresolvedOperand]");
  nb::class_<mlir::DialectBytecodeWriter>(m, "DialectBytecodeWriter");
  nb::class_<llvm::iterator_range<mlir::Operation::dialect_attr_iterator>>(
      m, "iterator_range[Operation.dialect_attr_iterator]");
  nb::class_<llvm::iterator_range<mlir::ResultRange::UseIterator>>(
      m, "iterator_range[ResultRange.UseIterator]");
  nb::class_<llvm::SourceMgr>(m, "SourceMgr");
  nb::class_<llvm::SmallVectorImpl<mlir::Type>>(m, "SmallVectorImpl[Type]");
  nb::class_<std::initializer_list<mlir::Type>>(m, "initializer_list[Type]");
  nb::class_<mlir::IntegerValueRange>(m, "IntegerValueRange");
  // nb::class_<mlir::SymbolTableCollection>(m, "SymbolTableCollection");

  nb::bind_vec_like<llvm::ArrayRef<mlir::Attribute>>(m, "ArrayRef[Attribute]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::AffineExpr>>(m,
                                                      "ArrayRef[AffineExpr]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::AffineMap>>(m, "ArrayRef[AffineMap]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::IRUnit>>(m, "ArrayRef[IRUnit]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::RegisteredOperationName>>(
      m, "ArrayRef[RegisteredOperationName]");
  nb::bind_vec_like<llvm::ArrayRef<unsigned>>(m, "ArrayRef[unsigned]");
  nb::bind_vec_like<llvm::ArrayRef<unsigned int>>(m, "ArrayRef[unsigned_int]");
  nb::bind_vec_like<llvm::ArrayRef<unsigned long>>(m,
                                                   "ArrayRef[unsigned_long]");
  nb::bind_vec_like<llvm::ArrayRef<bool>>(m, "ArrayRef[bool]");
  nb::bind_vec_like<llvm::ArrayRef<int16_t>>(m, "ArrayRef[int16_t]");
  nb::bind_vec_like<llvm::ArrayRef<int32_t>>(m, "ArrayRef[int32_t]");
  nb::bind_vec_like<llvm::ArrayRef<int64_t>>(m, "ArrayRef[int64_t]");
  nb::bind_vec_like<llvm::ArrayRef<int>>(m, "ArrayRef[int]");
  nb::bind_vec_like<llvm::ArrayRef<long>>(m, "ArrayRef[long]");
  nb::bind_vec_like<llvm::ArrayRef<float>>(m, "ArrayRef[float]");
  nb::bind_vec_like<llvm::ArrayRef<double>>(m, "ArrayRef[double]");
  nb::bind_vec_like<llvm::ArrayRef<char>>(m, "ArrayRef[char]");
  nb::bind_vec_like<llvm::ArrayRef<signed char>>(m, "ArrayRef[signed_char]");
  nb::bind_vec_like<llvm::ArrayRef<llvm::APInt>>(m, "ArrayRef[APInt]");
  nb::bind_vec_like<llvm::ArrayRef<llvm::APFloat>>(m, "ArrayRef[APFloat]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::Value>>(m, "ArrayRef[Value]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::StringAttr>>(m, "ArrayRef[str]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::OperationName>>(
      m, "ArrayRef[OperationName]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::Region *>>(m, "ArrayRef[Region]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::SymbolTable *>>(
      m, "ArrayRef[SymbolTable]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::Operation *>>(m,
                                                       "ArrayRef[Operation]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::NamedAttribute>>(
      m, "ArrayRef[NamedAttribute]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::FlatSymbolRefAttr>>(
      m, "ArrayRef[FlatSymbolRefAttr]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::BlockArgument>>(
      m, "ArrayRef[BlockArgument]");

  nb::bind_vector<std::vector<mlir::Type>>(m, "VectorOfType");
  nb::bind_vec_like<llvm::ArrayRef<mlir::Type>>(m, "ArrayRefOfType")
      .def(nb::init<const std::vector<mlir::Type> &>());
  nb::implicitly_convertible<std::vector<mlir::Type>,
                             llvm::ArrayRef<mlir::Type>>();

  nb::bind_vec_like<llvm::ArrayRef<mlir::Location>>(m, "ArrayRef[Location]");
  nb::bind_vector<std::vector<llvm::StringRef>>(m, "VectorOfStringRef");
  nb::bind_vec_like<llvm::ArrayRef<llvm::StringRef>>(m, "ArrayRef[str]");
  // nb::bind_vec_like<llvm::ArrayRef<mlir::PDLValue>>(m, "ArrayRef[PDLValue]");
  nb::bind_vec_like<llvm::MutableArrayRef<mlir::BlockArgument>>(
      m, "MutableArrayRef[BlockArgument]");
  nb::bind_vec_like<llvm::MutableArrayRef<char>>(m, "MutableArrayRef[char]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::OpAsmParser::Argument>>(
      m, "ArrayRef[OpAsmParser.Argument]");
  nb::bind_vec_like<llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand>>(
      m, "ArrayRef[OpAsmParser.UnresolvedOperand]");
  nb::bind_vec_like<llvm::MutableArrayRef<mlir::OpOperand>,
                    nb::rv_policy::reference_internal>(
      m, "MutableArrayRef[OpOperand]");
  nb::bind_vec_like<llvm::MutableArrayRef<mlir::BlockOperand>,
                    nb::rv_policy::reference_internal>(
      m, "MutableArrayRef[BlockOperand]");

  nb::bind_iter_range<mlir::ValueTypeRange<mlir::ValueRange>, mlir::Type>(
      m, "ValueTypeRange[ValueRange]");
  nb::bind_iter_range<mlir::ValueTypeRange<mlir::OperandRange>, mlir::Type>(
      m, "ValueTypeRange[OperandRange]");
  nb::bind_iter_range<mlir::ValueTypeRange<mlir::ResultRange>, mlir::Type>(
      m, "ValueTypeRange[ResultRange]");

  nb::bind_iter_like<llvm::iplist<mlir::Block>,
                     nb::rv_policy::reference_internal>(m, "iplist[Block]");
  nb::bind_iter_like<llvm::iplist<mlir::Operation>,
                     nb::rv_policy::reference_internal>(m, "iplist[Operation]");

  auto irModule = m.def_submodule("ir");

  nb::bind_vector<std::vector<mlir::Dialect *>>(m, "VectorOfDialect");
  auto mlir_DialectRegistry =
      non_copying_non_moving_class_<mlir::DialectRegistry>(irModule,
                                                           "DialectRegistry")
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
      non_copying_non_moving_class_<mlir::OperationState>(irModule,
                                                          "OperationState")
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

  populateIRModule(irModule);
  auto dialectsModule = m.def_submodule("dialects");
  auto arithModule = dialectsModule.def_submodule("arith");
  populateArithModule(arithModule);
}
