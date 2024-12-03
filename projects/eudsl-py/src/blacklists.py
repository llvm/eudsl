class_blacklist = {
    "mlir::AsmPrinter::Impl",
    "mlir::OperationName::Impl",
    "mlir::DialectRegistry",
    # allocating an object of abstract class type
    "mlir::AsmResourceParser",
    "mlir::AsmResourcePrinter",
    # object of type 'std::pair<std::basic_string<char>, std::unique_ptr<mlir::FallbackAsmResourceMap::ResourceCollection>>' cannot be assigned because its copy assignment operator is implicitly deleted
    "mlir::FallbackAsmResourceMap",
    # error: call to deleted constructor of 'std::unique_ptr<mlir::AsmResourceParser>'
    "mlir::ParserConfig",
    "mlir::SymbolTableCollection",
    "mlir::PDLResultList",
    # pure virtual
    "mlir::AsmParser",
    "mlir::AsmParser::CyclicParseReset",
    # error: overload resolution selected deleted operator '='
    "mlir::PDLPatternConfigSet",
    # wack
    # call to deleted constructor of 'std::unique_ptr<mlir::Region>'
    "mlir::OperationState",
    "mlir::FallbackAsmResourceMap::OpaqueAsmResource",
    # wrong base class
    # "collision on `value` method with ConstantOp
    "mlir::arith::ConstantIntOp",
    "mlir::arith::ConstantFloatOp",
    "mlir::arith::ConstantIndexOp",
}

fn_blacklist = {
    "getImpl()",
    "getAsOpaquePointer()",
    "getFromOpaquePointer(const void *)",
    "WalkResult(ResultEnum)",
    "initChainWithUse(IROperandBase **)",
    "AsmPrinter(Impl &)",
    "insert(std::unique_ptr<OperationName::Impl>, ArrayRef<StringRef>)",
    # these are all collisions with templated overloads
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ::mlir::MLIRContext *, Type, int64_t, ::llvm::ArrayRef<char>)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type, unsigned int, ArrayRef<char>)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type, const APFloat &)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type, double)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type, const APInt &)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ::mlir::MLIRContext *, const APSInt &)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type, int64_t)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, StringAttr, StringRef, Type)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ShapedType, DenseElementsAttr, DenseElementsAttr)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ::mlir::MLIRContext *, int64_t, ::llvm::ArrayRef<int64_t>)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ::mlir::MLIRContext *, unsigned int, SignednessSemantics)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ArrayRef<int64_t>, Type, MemRefLayoutAttrInterface, Attribute)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ArrayRef<int64_t>, Type, AffineMap, Attribute)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ArrayRef<int64_t>, Type, AffineMap, unsigned int)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, StringAttr, StringRef)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ArrayRef<int64_t>, Type, Attribute)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type, Attribute)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type, unsigned int)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, Type)",
    "getChecked(::llvm::function_ref< ::mlir::InFlightDiagnostic ()>, ArrayRef<int64_t>, Type, ArrayRef<bool>)",
    "getChecked(function_ref<InFlightDiagnostic ()>, DynamicAttrDefinition *, ArrayRef<Attribute>)",
    "getChecked(function_ref<InFlightDiagnostic ()>, DynamicTypeDefinition *, ArrayRef<Attribute>)",
    "get(ShapedType, StringRef, AsmResourceBlob)",
    "processAsArg(StringAttr)",
    # mlir::SparseElementsAttr::getValues
    "getValues()",
    "registerHandler(HandlerTy)",
    "emitDiagnostic(Location, Twine, DiagnosticSeverity, bool)",
    "operator++()",
    "clone(::mlir::Type)",
    "printFunctionalType(Operation *)",
    "print(::mlir::OpAsmPrinter &)",
    "insert(StringRef, std::optional<AsmResourceBlob>)",
    # incomplete types
    "AffineBinaryOpExpr(AffineExpr::ImplType *)",
    "AffineDimExpr(AffineExpr::ImplType *)",
    "AffineSymbolExpr(AffineExpr::ImplType *)",
    "AffineConstantExpr(AffineExpr::ImplType *)",
    "AffineExpr(const ImplType *)",
    "AffineMap(ImplType *)",
    "IntegerSet(ImplType *)",
    "Type(const ImplType *)",
    "Attribute(const ImplType *)",
    "Location(const LocationAttr::ImplType *)",
    "parseFloat(const llvm::fltSemantics &, APFloat &)",
    "getFloatSemantics()",
    # const char*
    "convertEndianOfCharForBEmachine(const char *, char *, size_t, size_t)",
    "parseKeywordType(const char *, Type &)",
    # no matching function for call to object of type 'const std::remove_reference_t
    "parseOptionalRegion(std::unique_ptr<Region> &, ArrayRef<Argument>, bool)",
    "parseSuccessor(Block *&)",
    "parseOptionalSuccessor(Block *&)",
    "parseSuccessorAndUseList(Block *&, SmallVectorImpl<Value> &)",
    # call to implicitly-deleted default constructor of
    #  call to deleted constructor
    "NamedAttrList(std::nullopt_t)",
    "OptionalParseResult(std::nullopt_t)",
    "OpPrintingFlags(std::nullopt_t)",
    "AsmResourceBlob(ArrayRef<char>, size_t, DeleterFn, bool)",
    "allocateWithAlign(ArrayRef<char>, size_t, AsmResourceBlob::DeleterFn, bool)",
    "AsmResourceBlob(const AsmResourceBlob &)",
    "InsertionGuard(const InsertionGuard &)",
    "CyclicPrintReset(const CyclicPrintReset &)",
    "CyclicParseReset(const CyclicParseReset &)",
    "PDLPatternModule(OwningOpRef<ModuleOp>)",
    # something weird - linker error - missing symbol in libMLIR.a
    # probably flags?
    "parseAssembly(OpAsmParser &, OperationState &)",
    "getPredicateByName(StringRef)",
}
