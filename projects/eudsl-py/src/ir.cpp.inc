// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2024.

  auto mlir_MLIRContext = nb::class_<mlir::MLIRContext>(m, "MLIRContext")
.def(nb::init<mlir::MLIRContext::Threading>(), "multithreading"_a)
.def(nb::init<const mlir::DialectRegistry &, mlir::MLIRContext::Threading>(), "registry"_a, "multithreading"_a)
.def_prop_ro("loaded_dialects", &mlir::MLIRContext::getLoadedDialects)
.def_prop_ro("dialect_registry", &mlir::MLIRContext::getDialectRegistry)
.def("append_dialect_registry", &mlir::MLIRContext::appendDialectRegistry, "registry"_a)
.def_prop_ro("available_dialects", &mlir::MLIRContext::getAvailableDialects)
.def("get_loaded_dialect", [](mlir::MLIRContext& self, llvm::StringRef name){ return self.getLoadedDialect(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("get_or_load_dynamic_dialect", &mlir::MLIRContext::getOrLoadDynamicDialect, "dialect_namespace"_a, "ctor"_a, nb::rv_policy::reference_internal)
.def("load_all_available_dialects", &mlir::MLIRContext::loadAllAvailableDialects)
.def("get_or_load_dialect", [](mlir::MLIRContext& self, llvm::StringRef name){ return self.getOrLoadDialect(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("allows_unregistered_dialects", &mlir::MLIRContext::allowsUnregisteredDialects)
.def("allow_unregistered_dialects", &mlir::MLIRContext::allowUnregisteredDialects, "allow"_a)
.def("is_multithreading_enabled", &mlir::MLIRContext::isMultithreadingEnabled)
.def("disable_multithreading", &mlir::MLIRContext::disableMultithreading, "disable"_a)
.def("enable_multithreading", &mlir::MLIRContext::enableMultithreading, "enable"_a)
.def("set_thread_pool", &mlir::MLIRContext::setThreadPool, "pool"_a)
.def_prop_ro("num_threads", &mlir::MLIRContext::getNumThreads)
.def_prop_ro("thread_pool", &mlir::MLIRContext::getThreadPool)
.def("should_print_op_on_diagnostic", &mlir::MLIRContext::shouldPrintOpOnDiagnostic)
.def("print_op_on_diagnostic", &mlir::MLIRContext::printOpOnDiagnostic, "enable"_a)
.def("should_print_stack_trace_on_diagnostic", &mlir::MLIRContext::shouldPrintStackTraceOnDiagnostic)
.def("print_stack_trace_on_diagnostic", &mlir::MLIRContext::printStackTraceOnDiagnostic, "enable"_a)
.def_prop_ro("registered_operations", &mlir::MLIRContext::getRegisteredOperations)
.def("get_registered_operations_by_dialect", &mlir::MLIRContext::getRegisteredOperationsByDialect, "dialect_name"_a)
.def("is_operation_registered", &mlir::MLIRContext::isOperationRegistered, "name"_a)
.def_prop_ro("diag_engine", &mlir::MLIRContext::getDiagEngine)
.def_prop_ro("affine_uniquer", &mlir::MLIRContext::getAffineUniquer)
.def_prop_ro("type_uniquer", &mlir::MLIRContext::getTypeUniquer)
.def_prop_ro("attribute_uniquer", &mlir::MLIRContext::getAttributeUniquer)
.def("enter_multi_threaded_execution", &mlir::MLIRContext::enterMultiThreadedExecution)
.def("exit_multi_threaded_execution", &mlir::MLIRContext::exitMultiThreadedExecution)
.def("get_or_load_dialect", [](mlir::MLIRContext& self, llvm::StringRef dialectNamespace, mlir::TypeID dialectID, llvm::function_ref<std::unique_ptr<mlir::Dialect> ()> ctor){ return self.getOrLoadDialect(dialectNamespace, dialectID, std::move(ctor)); }, "dialect_namespace"_a, "dialect_id"_a, "ctor"_a, nb::rv_policy::reference_internal)
.def_prop_ro("registry_hash", &mlir::MLIRContext::getRegistryHash)
.def("register_action_handler", &mlir::MLIRContext::registerActionHandler, "handler"_a)
.def("has_action_handler", &mlir::MLIRContext::hasActionHandler)
.def("execute_action", [](mlir::MLIRContext& self, llvm::function_ref<void ()> actionFn, const mlir::tracing::Action & action){ return self.executeAction(actionFn, action); }, "action_fn"_a, "action"_a)
;

nb::enum_<mlir::MLIRContext::Threading>(m, "Threading")
.value("DISABLED", mlir::MLIRContext::Threading::DISABLED)
.value("ENABLED", mlir::MLIRContext::Threading::ENABLED)
;

auto mlir_WalkResult = nb::class_<mlir::WalkResult>(m, "WalkResult")
.def(nb::init<llvm::LogicalResult>(), "result"_a)
.def(nb::init<mlir::Diagnostic &&>(), "_"_a)
.def(nb::init<mlir::InFlightDiagnostic &&>(), "_"_a)
.def("__eq__", &mlir::WalkResult::operator==, "rhs"_a)
.def("__ne__", &mlir::WalkResult::operator!=, "rhs"_a)
.def_static("interrupt", &mlir::WalkResult::interrupt)
.def_static("advance", &mlir::WalkResult::advance)
.def_static("skip", &mlir::WalkResult::skip)
.def("was_interrupted", &mlir::WalkResult::wasInterrupted)
.def("was_skipped", &mlir::WalkResult::wasSkipped)
;

nb::enum_<mlir::WalkOrder>(m, "WalkOrder")
.value("PreOrder", mlir::WalkOrder::PreOrder)
.value("PostOrder", mlir::WalkOrder::PostOrder)
;

auto mlir_ForwardIterator = nb::class_<mlir::ForwardIterator>(m, "ForwardIterator")
.def_static("make_iterable", [](mlir::Operation & range){ return mlir::ForwardIterator::makeIterable(range); }, "range"_a)
;

auto mlir_WalkStage = nb::class_<mlir::WalkStage>(m, "WalkStage")
.def(nb::init<mlir::Operation *>(), "op"_a)
.def("is_before_all_regions", &mlir::WalkStage::isBeforeAllRegions)
.def("is_before_region", &mlir::WalkStage::isBeforeRegion, "region"_a)
.def("is_after_region", &mlir::WalkStage::isAfterRegion, "region"_a)
.def("is_after_all_regions", &mlir::WalkStage::isAfterAllRegions)
.def("advance", &mlir::WalkStage::advance)
.def_prop_ro("next_region", &mlir::WalkStage::getNextRegion)
;

auto mlir_AttrTypeWalker = nb::class_<mlir::AttrTypeWalker>(m, "AttrTypeWalker")
.def("add_walk", [](mlir::AttrTypeWalker& self, std::function<mlir::WalkResult (mlir::Attribute)> && fn){ return self.addWalk(std::move(fn)); }, "fn"_a)
.def("add_walk", [](mlir::AttrTypeWalker& self, std::function<mlir::WalkResult (mlir::Type)> && fn){ return self.addWalk(std::move(fn)); }, "fn"_a)
;

auto mlir_AttrTypeReplacer = nb::class_<mlir::AttrTypeReplacer>(m, "AttrTypeReplacer")
.def("replace", [](mlir::AttrTypeReplacer& self, mlir::Attribute attr){ return self.replace(attr); }, "attr"_a)
.def("replace", [](mlir::AttrTypeReplacer& self, mlir::Type type){ return self.replace(type); }, "type"_a)
;

auto mlir_CyclicAttrTypeReplacer = nb::class_<mlir::CyclicAttrTypeReplacer>(m, "CyclicAttrTypeReplacer")
.def(nb::init<>())
.def("replace", [](mlir::CyclicAttrTypeReplacer& self, mlir::Attribute attr){ return self.replace(attr); }, "attr"_a)
.def("replace", [](mlir::CyclicAttrTypeReplacer& self, mlir::Type type){ return self.replace(type); }, "type"_a)
.def("add_cycle_breaker", [](mlir::CyclicAttrTypeReplacer& self, std::function<std::optional<mlir::Attribute> (mlir::Attribute)> fn){ return self.addCycleBreaker(fn); }, "fn"_a)
.def("add_cycle_breaker", [](mlir::CyclicAttrTypeReplacer& self, std::function<std::optional<mlir::Type> (mlir::Type)> fn){ return self.addCycleBreaker(fn); }, "fn"_a)
;

auto mlir_AttrTypeImmediateSubElementWalker = nb::class_<mlir::AttrTypeImmediateSubElementWalker>(m, "AttrTypeImmediateSubElementWalker")
.def(nb::init<llvm::function_ref<void (mlir::Attribute)>, llvm::function_ref<void (mlir::Type)>>(), "walk_attrs_fn"_a, "walk_types_fn"_a)
.def("walk", [](mlir::AttrTypeImmediateSubElementWalker& self, mlir::Attribute element){ return self.walk(element); }, "element"_a)
.def("walk", [](mlir::AttrTypeImmediateSubElementWalker& self, mlir::Type element){ return self.walk(element); }, "element"_a)
;

auto mlir_DialectExtensionBase = nb::class_<mlir::DialectExtensionBase>(m, "DialectExtensionBase")
.def_prop_ro("required_dialects", &mlir::DialectExtensionBase::getRequiredDialects)
.def("apply", &mlir::DialectExtensionBase::apply, "context"_a, "dialects"_a)
.def("clone", &mlir::DialectExtensionBase::clone)
;

auto mlir_AbstractType = nb::class_<mlir::AbstractType>(m, "AbstractType")
.def_static("lookup", [](mlir::TypeID typeID, mlir::MLIRContext * context){ return &mlir::AbstractType::lookup(typeID, context); }, "type_id"_a, "context"_a, nb::rv_policy::reference_internal)
.def_static("lookup", [](llvm::StringRef name, mlir::MLIRContext * context){ return mlir::AbstractType::lookup(name, context); }, "name"_a, "context"_a)
.def_static("get", [](mlir::Dialect & dialect, mlir::detail::InterfaceMap && interfaceMap, llvm::unique_function<bool (mlir::TypeID) const> && hasTrait, mlir::AbstractType::WalkImmediateSubElementsFn walkImmediateSubElementsFn, mlir::AbstractType::ReplaceImmediateSubElementsFn replaceImmediateSubElementsFn, mlir::TypeID typeID, llvm::StringRef name){ return mlir::AbstractType::get(dialect, std::move(interfaceMap), std::move(hasTrait), walkImmediateSubElementsFn, replaceImmediateSubElementsFn, typeID, name); }, "dialect"_a, "interface_map"_a, "has_trait"_a, "walk_immediate_sub_elements_fn"_a, "replace_immediate_sub_elements_fn"_a, "type_id"_a, "name"_a)
.def_prop_ro("dialect", &mlir::AbstractType::getDialect)
.def("has_interface", &mlir::AbstractType::hasInterface, "interface_id"_a)
.def("has_trait", [](mlir::AbstractType& self, mlir::TypeID traitID){ return self.hasTrait(traitID); }, "trait_id"_a)
.def("walk_immediate_sub_elements", &mlir::AbstractType::walkImmediateSubElements, "type"_a, "walk_attrs_fn"_a, "walk_types_fn"_a)
.def("replace_immediate_sub_elements", &mlir::AbstractType::replaceImmediateSubElements, "type"_a, "repl_attrs"_a, "repl_types"_a)
.def_prop_ro("type_id", &mlir::AbstractType::getTypeID)
.def_prop_ro("name", &mlir::AbstractType::getName)
;

auto mlir_TypeStorage = nb::class_<mlir::TypeStorage>(m, "TypeStorage")
.def_prop_ro("abstract_type", &mlir::TypeStorage::getAbstractType)
;

auto mlir_detail_TypeUniquer = nb::class_<mlir::detail::TypeUniquer>(m, "TypeUniquer")
;

auto mlir_Type = nb::class_<mlir::Type>(m, "Type")
.def(nb::init<>())
.def(nb::init<const mlir::Type &>(), "other"_a)
.def("__eq__", &mlir::Type::operator==, "other"_a)
.def("__ne__", &mlir::Type::operator!=, "other"_a)
.def_prop_ro("type_id", &mlir::Type::getTypeID)
.def_prop_ro("context", &mlir::Type::getContext)
.def_prop_ro("dialect", &mlir::Type::getDialect)
.def("is_index", &mlir::Type::isIndex)
.def("is_float4e2m1fn", [](Type& self) { return llvm::isa<mlir::Float4E2M1FNType>(self); })
.def("is_float6e2m3fn", [](Type& self) { return llvm::isa<mlir::Float6E2M3FNType>(self); })
.def("is_float6e3m2fn", [](Type& self) { return llvm::isa<mlir::Float6E3M2FNType>(self); })
.def("is_float8e5m2", [](Type& self) { return llvm::isa<mlir::Float8E5M2Type>(self); })
.def("is_float8e4m3", [](Type& self) { return llvm::isa<mlir::Float8E4M3Type>(self); })
.def("is_float8e4m3fn", [](Type& self) { return llvm::isa<mlir::Float8E4M3FNType>(self); })
.def("is_float8e5m2fnuz", [](Type& self) { return llvm::isa<mlir::Float8E5M2FNUZType>(self); })
.def("is_float8e4m3fnuz", [](Type& self) { return llvm::isa<mlir::Float8E4M3FNUZType>(self); })
.def("is_float8e4m3b11fnuz", [](Type& self) { return llvm::isa<mlir::Float8E4M3B11FNUZType>(self); })
.def("is_float8e3m4", [](Type& self) { return llvm::isa<mlir::Float8E3M4Type>(self); })
.def("is_float8e8m0fnu", [](Type& self) { return llvm::isa<mlir::Float8E8M0FNUType>(self); })
.def("is_bf16", &mlir::Type::isBF16)
.def("is_f16", &mlir::Type::isF16)
.def("is_tf32", &mlir::Type::isTF32)
.def("is_f32", &mlir::Type::isF32)
.def("is_f64", &mlir::Type::isF64)
.def("is_f80", &mlir::Type::isF80)
.def("is_f128", &mlir::Type::isF128)
.def("is_integer", [](mlir::Type& self){ return self.isInteger(); })
.def("is_integer", [](mlir::Type& self, unsigned int width){ return self.isInteger(width); }, "width"_a)
.def("is_signless_integer", [](mlir::Type& self){ return self.isSignlessInteger(); })
.def("is_signless_integer", [](mlir::Type& self, unsigned int width){ return self.isSignlessInteger(width); }, "width"_a)
.def("is_signed_integer", [](mlir::Type& self){ return self.isSignedInteger(); })
.def("is_signed_integer", [](mlir::Type& self, unsigned int width){ return self.isSignedInteger(width); }, "width"_a)
.def("is_unsigned_integer", [](mlir::Type& self){ return self.isUnsignedInteger(); })
.def("is_unsigned_integer", [](mlir::Type& self, unsigned int width){ return self.isUnsignedInteger(width); }, "width"_a)
.def_prop_ro("int_or_float_bit_width", &mlir::Type::getIntOrFloatBitWidth)
.def("is_signless_int_or_index", &mlir::Type::isSignlessIntOrIndex)
.def("is_signless_int_or_index_or_float", &mlir::Type::isSignlessIntOrIndexOrFloat)
.def("is_signless_int_or_float", &mlir::Type::isSignlessIntOrFloat)
.def("is_int_or_index", &mlir::Type::isIntOrIndex)
.def("is_int_or_float", &mlir::Type::isIntOrFloat)
.def("is_int_or_index_or_float", &mlir::Type::isIntOrIndexOrFloat)
.def("print", [](mlir::Type& self, llvm::raw_ostream & os){ return self.print(os); }, "os"_a)
.def("print", [](mlir::Type& self, llvm::raw_ostream & os, mlir::AsmState & state){ return self.print(os, state); }, "os"_a, "state"_a)
.def("dump", &mlir::Type::dump)
.def_prop_ro("abstract_type", &mlir::Type::getAbstractType)
.def("walk_immediate_sub_elements", &mlir::Type::walkImmediateSubElements, "walk_attrs_fn"_a, "walk_types_fn"_a)
.def("replace_immediate_sub_elements", &mlir::Type::replaceImmediateSubElements, "repl_attrs"_a, "repl_types"_a)
.def("__str__", [](mlir::Type& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_AbstractAttribute = nb::class_<mlir::AbstractAttribute>(m, "AbstractAttribute")
.def_static("lookup", [](mlir::TypeID typeID, mlir::MLIRContext * context){ return &mlir::AbstractAttribute::lookup(typeID, context); }, "type_id"_a, "context"_a, nb::rv_policy::reference_internal)
.def_static("lookup", [](llvm::StringRef name, mlir::MLIRContext * context){ return mlir::AbstractAttribute::lookup(name, context); }, "name"_a, "context"_a)
.def_static("get", [](mlir::Dialect & dialect, mlir::detail::InterfaceMap && interfaceMap, llvm::unique_function<bool (mlir::TypeID) const> && hasTrait, mlir::AbstractAttribute::WalkImmediateSubElementsFn walkImmediateSubElementsFn, mlir::AbstractAttribute::ReplaceImmediateSubElementsFn replaceImmediateSubElementsFn, mlir::TypeID typeID, llvm::StringRef name){ return mlir::AbstractAttribute::get(dialect, std::move(interfaceMap), std::move(hasTrait), walkImmediateSubElementsFn, replaceImmediateSubElementsFn, typeID, name); }, "dialect"_a, "interface_map"_a, "has_trait"_a, "walk_immediate_sub_elements_fn"_a, "replace_immediate_sub_elements_fn"_a, "type_id"_a, "name"_a)
.def_prop_ro("dialect", &mlir::AbstractAttribute::getDialect)
.def("has_interface", &mlir::AbstractAttribute::hasInterface, "interface_id"_a)
.def("has_trait", [](mlir::AbstractAttribute& self, mlir::TypeID traitID){ return self.hasTrait(traitID); }, "trait_id"_a)
.def("walk_immediate_sub_elements", &mlir::AbstractAttribute::walkImmediateSubElements, "attr"_a, "walk_attrs_fn"_a, "walk_types_fn"_a)
.def("replace_immediate_sub_elements", &mlir::AbstractAttribute::replaceImmediateSubElements, "attr"_a, "repl_attrs"_a, "repl_types"_a)
.def_prop_ro("type_id", &mlir::AbstractAttribute::getTypeID)
.def_prop_ro("name", &mlir::AbstractAttribute::getName)
;

auto mlir_AttributeStorage = nb::class_<mlir::AttributeStorage>(m, "AttributeStorage")
.def_prop_ro("abstract_attribute", &mlir::AttributeStorage::getAbstractAttribute)
;

auto mlir_detail_AttributeUniquer = nb::class_<mlir::detail::AttributeUniquer>(m, "AttributeUniquer")
;

auto mlir_Attribute = nb::class_<mlir::Attribute>(m, "Attribute")
.def(nb::init<>())
.def(nb::init<const mlir::Attribute &>(), "other"_a)
.def("__eq__", &mlir::Attribute::operator==, "other"_a)
.def("__ne__", &mlir::Attribute::operator!=, "other"_a)
.def_prop_ro("type_id", &mlir::Attribute::getTypeID)
.def_prop_ro("context", &mlir::Attribute::getContext)
.def_prop_ro("dialect", &mlir::Attribute::getDialect)
.def("print", [](mlir::Attribute& self, llvm::raw_ostream & os, bool elideType){ return self.print(os, elideType); }, "os"_a, "elide_type"_a)
.def("print", [](mlir::Attribute& self, llvm::raw_ostream & os, mlir::AsmState & state, bool elideType){ return self.print(os, state, elideType); }, "os"_a, "state"_a, "elide_type"_a)
.def("dump", &mlir::Attribute::dump)
.def("print_stripped", [](mlir::Attribute& self, llvm::raw_ostream & os){ return self.printStripped(os); }, "os"_a)
.def("print_stripped", [](mlir::Attribute& self, llvm::raw_ostream & os, mlir::AsmState & state){ return self.printStripped(os, state); }, "os"_a, "state"_a)
.def_prop_ro("abstract_attribute", &mlir::Attribute::getAbstractAttribute)
.def("walk_immediate_sub_elements", &mlir::Attribute::walkImmediateSubElements, "walk_attrs_fn"_a, "walk_types_fn"_a)
.def("replace_immediate_sub_elements", &mlir::Attribute::replaceImmediateSubElements, "repl_attrs"_a, "repl_types"_a)
.def("__str__", [](mlir::Attribute& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os, /*elideType*/false);
  return s;
});

auto mlir_NamedAttribute = nb::class_<mlir::NamedAttribute>(m, "NamedAttribute")
.def(nb::init<mlir::StringAttr, mlir::Attribute>(), "name"_a, "value"_a)
.def_prop_ro("name", &mlir::NamedAttribute::getName)
.def_prop_ro("name_dialect", &mlir::NamedAttribute::getNameDialect)
.def_prop_ro("value", &mlir::NamedAttribute::getValue)
.def("set_name", &mlir::NamedAttribute::setName, "new_name"_a)
.def("set_value", &mlir::NamedAttribute::setValue, "new_value"_a)
.def("__lt__", [](mlir::NamedAttribute& self, const mlir::NamedAttribute & rhs){ return self.operator<(rhs); }, "rhs"_a)
.def("__lt__", [](mlir::NamedAttribute& self, llvm::StringRef rhs){ return self.operator<(rhs); }, "rhs"_a)
.def("__eq__", &mlir::NamedAttribute::operator==, "rhs"_a)
.def("__ne__", &mlir::NamedAttribute::operator!=, "rhs"_a)
;

auto mlir_AttrTypeSubElementHandler__NamedAttribute__ = nb::class_<mlir::AttrTypeSubElementHandler<NamedAttribute>>(m, "AttrTypeSubElementHandler[NamedAttribute]")
;

auto mlir_LocationAttr = nb::class_<mlir::LocationAttr, mlir::Attribute>(m, "LocationAttr")
.def("walk", &mlir::LocationAttr::walk, "walk_fn"_a)
.def_static("classof", &mlir::LocationAttr::classof, "attr"_a)
;

auto mlir_Location = nb::class_<mlir::Location>(m, "Location")
.def(nb::init<mlir::LocationAttr>(), "loc"_a)
.def_prop_ro("context", &mlir::Location::getContext)
.def("__eq__", &mlir::Location::operator==, "rhs"_a)
.def("__ne__", &mlir::Location::operator!=, "rhs"_a)
.def("print", &mlir::Location::print, "os"_a)
.def("dump", &mlir::Location::dump)
.def_static("classof", &mlir::Location::classof, "attr"_a)
.def("__str__", [](mlir::Location& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_CallSiteLoc = nb::class_<mlir::CallSiteLoc,  mlir::LocationAttr>(m, "CallSiteLoc")
.def_static("get", [](mlir::Location callee, mlir::Location caller){ return mlir::CallSiteLoc::get(callee, caller); }, "callee"_a, "caller"_a)
.def_static("get", [](mlir::Location name, llvm::ArrayRef<mlir::Location> frames){ return mlir::CallSiteLoc::get(name, frames); }, "name"_a, "frames"_a)
.def_prop_ro("callee", &mlir::CallSiteLoc::getCallee)
.def_prop_ro("caller", &mlir::CallSiteLoc::getCaller)
;

auto mlir_FileLineColRange = nb::class_<mlir::FileLineColRange,  mlir::LocationAttr>(m, "FileLineColRange")
.def_prop_ro("filename", &mlir::FileLineColRange::getFilename)
.def_prop_ro("start_line", &mlir::FileLineColRange::getStartLine)
.def_prop_ro("start_column", &mlir::FileLineColRange::getStartColumn)
.def_prop_ro("end_column", &mlir::FileLineColRange::getEndColumn)
.def_prop_ro("end_line", &mlir::FileLineColRange::getEndLine)
.def_static("get", [](mlir::StringAttr filename){ return mlir::FileLineColRange::get(filename); }, "filename"_a)
.def_static("get", [](mlir::StringAttr filename, unsigned int line){ return mlir::FileLineColRange::get(filename, line); }, "filename"_a, "line"_a)
.def_static("get", [](mlir::StringAttr filename, unsigned int line, unsigned int column){ return mlir::FileLineColRange::get(filename, line, column); }, "filename"_a, "line"_a, "column"_a)
.def_static("get", [](mlir::MLIRContext * context, llvm::StringRef filename, unsigned int start_line, unsigned int start_column){ return mlir::FileLineColRange::get(context, filename, start_line, start_column); }, "context"_a, "filename"_a, "start_line"_a, "start_column"_a)
.def_static("get", [](mlir::StringAttr filename, unsigned int line, unsigned int start_column, unsigned int end_column){ return mlir::FileLineColRange::get(filename, line, start_column, end_column); }, "filename"_a, "line"_a, "start_column"_a, "end_column"_a)
.def_static("get", [](mlir::StringAttr filename, unsigned int start_line, unsigned int start_column, unsigned int end_line, unsigned int end_column){ return mlir::FileLineColRange::get(filename, start_line, start_column, end_line, end_column); }, "filename"_a, "start_line"_a, "start_column"_a, "end_line"_a, "end_column"_a)
.def_static("get", [](mlir::MLIRContext * context, llvm::StringRef filename, unsigned int start_line, unsigned int start_column, unsigned int end_line, unsigned int end_column){ return mlir::FileLineColRange::get(context, filename, start_line, start_column, end_line, end_column); }, "context"_a, "filename"_a, "start_line"_a, "start_column"_a, "end_line"_a, "end_column"_a)
;

auto mlir_FusedLoc = nb::class_<mlir::FusedLoc,  mlir::LocationAttr>(m, "FusedLoc")
.def_static("get", [](llvm::ArrayRef<mlir::Location> locs, mlir::Attribute metadata, mlir::MLIRContext * context){ return mlir::FusedLoc::get(locs, metadata, context); }, "locs"_a, "metadata"_a, "context"_a)
.def_static("get", [](mlir::MLIRContext * context, llvm::ArrayRef<mlir::Location> locs){ return mlir::FusedLoc::get(context, locs); }, "context"_a, "locs"_a)
.def_static("get", [](mlir::MLIRContext * context, llvm::ArrayRef<mlir::Location> locations, mlir::Attribute metadata){ return mlir::FusedLoc::get(context, locations, metadata); }, "context"_a, "locations"_a, "metadata"_a)
.def_prop_ro("locations", &mlir::FusedLoc::getLocations)
.def_prop_ro("metadata", &mlir::FusedLoc::getMetadata)
;

auto mlir_NameLoc = nb::class_<mlir::NameLoc,  mlir::LocationAttr>(m, "NameLoc")
.def_static("get", [](mlir::StringAttr name, mlir::Location childLoc){ return mlir::NameLoc::get(name, childLoc); }, "name"_a, "child_loc"_a)
.def_static("get", [](mlir::StringAttr name){ return mlir::NameLoc::get(name); }, "name"_a)
.def_prop_ro("name", &mlir::NameLoc::getName)
.def_prop_ro("child_loc", &mlir::NameLoc::getChildLoc)
;

auto mlir_OpaqueLoc = nb::class_<mlir::OpaqueLoc,  mlir::LocationAttr>(m, "OpaqueLoc")
.def_static("get", [](uintptr_t underlyingLocation, mlir::TypeID underlyingTypeID, mlir::Location fallbackLocation){ return mlir::OpaqueLoc::get(underlyingLocation, underlyingTypeID, fallbackLocation); }, "underlying_location"_a, "underlying_type_id"_a, "fallback_location"_a)
.def_prop_ro("underlying_location", [](mlir::OpaqueLoc& self){ return self.getUnderlyingLocation(); })
.def_prop_ro("underlying_type_id", &mlir::OpaqueLoc::getUnderlyingTypeID)
.def_prop_ro("fallback_location", &mlir::OpaqueLoc::getFallbackLocation)
;

auto mlir_UnknownLoc = nb::class_<mlir::UnknownLoc,  mlir::LocationAttr>(m, "UnknownLoc")
.def_static("get", &mlir::UnknownLoc::get, "context"_a)
;

auto mlir_detail_TypeIDResolver___mlir_CallSiteLoc__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::CallSiteLoc>>(m, "TypeIDResolver[CallSiteLoc]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::CallSiteLoc>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_FileLineColRange__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::FileLineColRange>>(m, "TypeIDResolver[FileLineColRange]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::FileLineColRange>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_FusedLoc__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::FusedLoc>>(m, "TypeIDResolver[FusedLoc]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::FusedLoc>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_NameLoc__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::NameLoc>>(m, "TypeIDResolver[NameLoc]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::NameLoc>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_OpaqueLoc__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::OpaqueLoc>>(m, "TypeIDResolver[OpaqueLoc]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::OpaqueLoc>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_UnknownLoc__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::UnknownLoc>>(m, "TypeIDResolver[UnknownLoc]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::UnknownLoc>::resolveTypeID)
;

auto mlir_FileLineColLoc = nb::class_<mlir::FileLineColLoc, mlir::FileLineColRange>(m, "FileLineColLoc")
.def_static("get", [](mlir::StringAttr filename, unsigned int line, unsigned int column){ return mlir::FileLineColLoc::get(filename, line, column); }, "filename"_a, "line"_a, "column"_a)
.def_static("get", [](mlir::MLIRContext * context, llvm::StringRef fileName, unsigned int line, unsigned int column){ return mlir::FileLineColLoc::get(context, fileName, line, column); }, "context"_a, "file_name"_a, "line"_a, "column"_a)
.def_prop_ro("filename", &mlir::FileLineColLoc::getFilename)
.def_prop_ro("line", &mlir::FileLineColLoc::getLine)
.def_prop_ro("column", &mlir::FileLineColLoc::getColumn)
;

auto mlir_AttrTypeSubElementHandler__Location__ = nb::class_<mlir::AttrTypeSubElementHandler<Location>>(m, "AttrTypeSubElementHandler[Location]")
.def_static("walk", &mlir::AttrTypeSubElementHandler<Location>::walk, "param"_a, "walker"_a)
.def_static("replace", &mlir::AttrTypeSubElementHandler<Location>::replace, "param"_a, "attr_repls"_a, "type_repls"_a)
;

auto mlir_detail_IROperandBase = nb::class_<mlir::detail::IROperandBase>(m, "IROperandBase")
.def_prop_ro("owner", &mlir::detail::IROperandBase::getOwner)
.def_prop_ro("next_operand_using_this_value", &mlir::detail::IROperandBase::getNextOperandUsingThisValue)
.def("link_to", &mlir::detail::IROperandBase::linkTo, "next"_a)
;

auto mlir_detail_ValueImpl = nb::class_<mlir::detail::ValueImpl>(m, "ValueImpl")
.def_prop_ro("type", &mlir::detail::ValueImpl::getType)
.def("set_type", &mlir::detail::ValueImpl::setType, "type"_a)
.def_prop_ro("kind", &mlir::detail::ValueImpl::getKind)
;

nb::enum_<mlir::detail::ValueImpl::Kind>(m, "Kind")
.value("InlineOpResult", mlir::detail::ValueImpl::Kind::InlineOpResult)
.value("OutOfLineOpResult", mlir::detail::ValueImpl::Kind::OutOfLineOpResult)
.value("BlockArgument", mlir::detail::ValueImpl::Kind::BlockArgument)
;

auto mlir_Value = nb::class_<mlir::Value>(m, "Value")
.def(nb::init<mlir::detail::ValueImpl *>(), "impl"_a)
.def("__eq__", &mlir::Value::operator==, "other"_a)
.def("__ne__", &mlir::Value::operator!=, "other"_a)
.def_prop_ro("type", &mlir::Value::getType)
.def_prop_ro("context", &mlir::Value::getContext)
.def("set_type", &mlir::Value::setType, "new_type"_a)
.def_prop_ro("defining_op", [](mlir::Value& self){ return self.getDefiningOp(); })
.def_prop_ro("loc", &mlir::Value::getLoc)
.def("set_loc", &mlir::Value::setLoc, "loc"_a)
.def_prop_ro("parent_region", &mlir::Value::getParentRegion)
.def_prop_ro("parent_block", &mlir::Value::getParentBlock)
.def("drop_all_uses", &mlir::Value::dropAllUses)
.def("replace_all_uses_with", &mlir::Value::replaceAllUsesWith, "new_value"_a)
.def("replace_all_uses_except", [](mlir::Value& self, mlir::Value newValue, const llvm::SmallPtrSetImpl<mlir::Operation *> & exceptions){ return self.replaceAllUsesExcept(newValue, exceptions); }, "new_value"_a, "exceptions"_a)
.def("replace_all_uses_except", [](mlir::Value& self, mlir::Value newValue, mlir::Operation * exceptedUser){ return self.replaceAllUsesExcept(newValue, exceptedUser); }, "new_value"_a, "excepted_user"_a)
.def("replace_uses_with_if", &mlir::Value::replaceUsesWithIf, "new_value"_a, "should_replace"_a)
.def("is_used_outside_of_block", &mlir::Value::isUsedOutsideOfBlock, "block"_a)
.def("shuffle_use_list", &mlir::Value::shuffleUseList, "indices"_a)
.def("use_begin", &mlir::Value::use_begin)
.def("use_end", &mlir::Value::use_end)
.def_prop_ro("uses", &mlir::Value::getUses)
.def("has_one_use", &mlir::Value::hasOneUse)
.def("use_empty", &mlir::Value::use_empty)
.def("user_begin", &mlir::Value::user_begin)
.def("user_end", &mlir::Value::user_end)
.def_prop_ro("users", &mlir::Value::getUsers)
.def("print", [](mlir::Value& self, llvm::raw_ostream & os){ return self.print(os); }, "os"_a)
.def("print", [](mlir::Value& self, llvm::raw_ostream & os, const mlir::OpPrintingFlags & flags){ return self.print(os, flags); }, "os"_a, "flags"_a)
.def("print", [](mlir::Value& self, llvm::raw_ostream & os, mlir::AsmState & state){ return self.print(os, state); }, "os"_a, "state"_a)
.def("dump", &mlir::Value::dump)
.def("print_as_operand", [](mlir::Value& self, llvm::raw_ostream & os, mlir::AsmState & state){ return self.printAsOperand(os, state); }, "os"_a, "state"_a)
.def("print_as_operand", [](mlir::Value& self, llvm::raw_ostream & os, const mlir::OpPrintingFlags & flags){ return self.printAsOperand(os, flags); }, "os"_a, "flags"_a)
.def("__str__", [](mlir::Value& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_OpOperand = nb::class_<mlir::OpOperand>(m, "OpOperand")
.def_static("get_use_list", &mlir::OpOperand::getUseList, "value"_a, nb::rv_policy::reference_internal)
.def_prop_ro("operand_number", &mlir::OpOperand::getOperandNumber)
.def("assign", &mlir::OpOperand::assign, "value"_a)
;

auto mlir_detail_BlockArgumentImpl = nb::class_<mlir::detail::BlockArgumentImpl, mlir::detail::ValueImpl>(m, "BlockArgumentImpl")
.def_static("classof", &mlir::detail::BlockArgumentImpl::classof, "value"_a)
;

auto mlir_BlockArgument = nb::class_<mlir::BlockArgument, mlir::Value>(m, "BlockArgument")
.def_static("classof", &mlir::BlockArgument::classof, "value"_a)
.def_prop_ro("owner", &mlir::BlockArgument::getOwner)
.def_prop_ro("arg_number", &mlir::BlockArgument::getArgNumber)
.def_prop_ro("loc", &mlir::BlockArgument::getLoc)
.def("set_loc", &mlir::BlockArgument::setLoc, "loc"_a)
;

auto mlir_detail_OpResultImpl = nb::class_<mlir::detail::OpResultImpl, mlir::detail::ValueImpl>(m, "OpResultImpl")
.def_static("classof", &mlir::detail::OpResultImpl::classof, "value"_a)
.def_prop_ro("owner", &mlir::detail::OpResultImpl::getOwner)
.def_prop_ro("result_number", &mlir::detail::OpResultImpl::getResultNumber)
.def("get_next_result_at_offset", &mlir::detail::OpResultImpl::getNextResultAtOffset, "offset"_a, nb::rv_policy::reference_internal)
.def_static("max_inline_results", &mlir::detail::OpResultImpl::getMaxInlineResults)
;

auto mlir_detail_InlineOpResult = nb::class_<mlir::detail::InlineOpResult, mlir::detail::OpResultImpl>(m, "InlineOpResult")
.def(nb::init<mlir::Type, unsigned int>(), "type"_a, "result_no"_a)
.def_prop_ro("result_number", &mlir::detail::InlineOpResult::getResultNumber)
.def_static("classof", &mlir::detail::InlineOpResult::classof, "value"_a)
;

auto mlir_detail_OutOfLineOpResult = nb::class_<mlir::detail::OutOfLineOpResult, mlir::detail::OpResultImpl>(m, "OutOfLineOpResult")
.def(nb::init<mlir::Type, uint64_t>(), "type"_a, "out_of_line_index"_a)
.def_static("classof", &mlir::detail::OutOfLineOpResult::classof, "value"_a)
.def_prop_ro("result_number", &mlir::detail::OutOfLineOpResult::getResultNumber)
;

auto mlir_OpResult = nb::class_<mlir::OpResult, mlir::Value>(m, "OpResult")
.def_static("classof", &mlir::OpResult::classof, "value"_a)
.def_prop_ro("owner", &mlir::OpResult::getOwner)
.def_prop_ro("result_number", &mlir::OpResult::getResultNumber)
;

auto mlir_BlockOperand = nb::class_<mlir::BlockOperand>(m, "BlockOperand")
.def_static("get_use_list", &mlir::BlockOperand::getUseList, "value"_a, nb::rv_policy::reference_internal)
.def_prop_ro("operand_number", &mlir::BlockOperand::getOperandNumber)
;

auto mlir_PredecessorIterator = nb::class_<mlir::PredecessorIterator>(m, "PredecessorIterator")
.def(nb::init<mlir::ValueUseIterator<mlir::BlockOperand>>(), "it"_a)
.def(nb::init<mlir::BlockOperand *>(), "operand"_a)
.def_prop_ro("successor_index", &mlir::PredecessorIterator::getSuccessorIndex)
;

auto mlir_SuccessorRange = nb::class_<mlir::SuccessorRange>(m, "SuccessorRange")
.def(nb::init<>())
.def(nb::init<mlir::Block *>(), "block"_a)
.def(nb::init<mlir::Operation *>(), "term"_a)
;

auto mlir_BlockRange = nb::class_<mlir::BlockRange>(m, "BlockRange")
.def(nb::init<llvm::ArrayRef<mlir::Block *>>(), "blocks"_a)
.def(nb::init<mlir::SuccessorRange>(), "successors"_a)
.def(nb::init<std::initializer_list<mlir::Block *>>(), "blocks"_a)
;

nb::enum_<mlir::AffineExprKind>(m, "AffineExprKind")
.value("Add", mlir::AffineExprKind::Add)
.value("Mul", mlir::AffineExprKind::Mul)
.value("Mod", mlir::AffineExprKind::Mod)
.value("FloorDiv", mlir::AffineExprKind::FloorDiv)
.value("CeilDiv", mlir::AffineExprKind::CeilDiv)
.value("LAST_AFFINE_BINARY_OP", mlir::AffineExprKind::LAST_AFFINE_BINARY_OP)
.value("Constant", mlir::AffineExprKind::Constant)
.value("DimId", mlir::AffineExprKind::DimId)
.value("SymbolId", mlir::AffineExprKind::SymbolId)
;

auto mlir_AffineExpr = nb::class_<mlir::AffineExpr>(m, "AffineExpr")
.def(nb::init<>())
.def("__eq__", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.operator==(other); }, "other"_a)
.def("__ne__", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.operator!=(other); }, "other"_a)
.def("__eq__", [](mlir::AffineExpr& self, int64_t v){ return self.operator==(v); }, "v"_a)
.def("__ne__", [](mlir::AffineExpr& self, int64_t v){ return self.operator!=(v); }, "v"_a)
.def_prop_ro("context", &mlir::AffineExpr::getContext)
.def_prop_ro("kind", &mlir::AffineExpr::getKind)
.def("print", &mlir::AffineExpr::print, "os"_a)
.def("dump", &mlir::AffineExpr::dump)
.def("is_symbolic_or_constant", &mlir::AffineExpr::isSymbolicOrConstant)
.def("is_pure_affine", &mlir::AffineExpr::isPureAffine)
.def_prop_ro("largest_known_divisor", &mlir::AffineExpr::getLargestKnownDivisor)
.def("is_multiple_of", &mlir::AffineExpr::isMultipleOf, "factor"_a)
.def("is_function_of_dim", &mlir::AffineExpr::isFunctionOfDim, "position"_a)
.def("is_function_of_symbol", &mlir::AffineExpr::isFunctionOfSymbol, "position"_a)
.def("replace_dims_and_symbols", &mlir::AffineExpr::replaceDimsAndSymbols, "dim_replacements"_a, "sym_replacements"_a)
.def("replace_dims", &mlir::AffineExpr::replaceDims, "dim_replacements"_a)
.def("replace_symbols", &mlir::AffineExpr::replaceSymbols, "sym_replacements"_a)
.def("replace", [](mlir::AffineExpr& self, mlir::AffineExpr expr, mlir::AffineExpr replacement){ return self.replace(expr, replacement); }, "expr"_a, "replacement"_a)
.def("replace", [](mlir::AffineExpr& self, const llvm::DenseMap<mlir::AffineExpr, mlir::AffineExpr> & map){ return self.replace(map); }, "map"_a)
.def("shift_dims", &mlir::AffineExpr::shiftDims, "num_dims"_a, "shift"_a, "offset"_a)
.def("shift_symbols", &mlir::AffineExpr::shiftSymbols, "num_symbols"_a, "shift"_a, "offset"_a)
.def("__add__", [](mlir::AffineExpr& self, int64_t v){ return self.operator+(v); }, "v"_a)
.def("__add__", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.operator+(other); }, "other"_a)
.def("__neg__", [](mlir::AffineExpr& self){ return self.operator-(); })
.def("__neg__", [](mlir::AffineExpr& self, int64_t v){ return self.operator-(v); }, "v"_a)
.def("__neg__", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.operator-(other); }, "other"_a)
.def("__mul__", [](mlir::AffineExpr& self, int64_t v){ return self.operator*(v); }, "v"_a)
.def("__mul__", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.operator*(other); }, "other"_a)
.def("floor_div", [](mlir::AffineExpr& self, uint64_t v){ return self.floorDiv(v); }, "v"_a)
.def("floor_div", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.floorDiv(other); }, "other"_a)
.def("ceil_div", [](mlir::AffineExpr& self, uint64_t v){ return self.ceilDiv(v); }, "v"_a)
.def("ceil_div", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.ceilDiv(other); }, "other"_a)
.def("__mod__", [](mlir::AffineExpr& self, uint64_t v){ return self.operator%(v); }, "v"_a)
.def("__mod__", [](mlir::AffineExpr& self, mlir::AffineExpr other){ return self.operator%(other); }, "other"_a)
.def("compose", &mlir::AffineExpr::compose, "map"_a)
.def("__str__", [](mlir::AffineExpr& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_AffineBinaryOpExpr = nb::class_<mlir::AffineBinaryOpExpr, mlir::AffineExpr>(m, "AffineBinaryOpExpr")
.def_prop_ro("lhs", &mlir::AffineBinaryOpExpr::getLHS)
.def_prop_ro("rhs", &mlir::AffineBinaryOpExpr::getRHS)
;

auto mlir_AffineDimExpr = nb::class_<mlir::AffineDimExpr, mlir::AffineExpr>(m, "AffineDimExpr")
.def_prop_ro("position", &mlir::AffineDimExpr::getPosition)
;

auto mlir_AffineSymbolExpr = nb::class_<mlir::AffineSymbolExpr, mlir::AffineExpr>(m, "AffineSymbolExpr")
.def_prop_ro("position", &mlir::AffineSymbolExpr::getPosition)
;

auto mlir_AffineConstantExpr = nb::class_<mlir::AffineConstantExpr, mlir::AffineExpr>(m, "AffineConstantExpr")
.def_prop_ro("value", &mlir::AffineConstantExpr::getValue)
;

auto mlir_AffineMap = nb::class_<mlir::AffineMap>(m, "AffineMap")
.def(nb::init<>())
.def_static("get", [](mlir::MLIRContext * context){ return mlir::AffineMap::get(context); }, "context"_a)
.def_static("get", [](unsigned int dimCount, unsigned int symbolCount, mlir::MLIRContext * context){ return mlir::AffineMap::get(dimCount, symbolCount, context); }, "dim_count"_a, "symbol_count"_a, "context"_a)
.def_static("get", [](unsigned int dimCount, unsigned int symbolCount, mlir::AffineExpr result){ return mlir::AffineMap::get(dimCount, symbolCount, result); }, "dim_count"_a, "symbol_count"_a, "result"_a)
.def_static("get", [](unsigned int dimCount, unsigned int symbolCount, llvm::ArrayRef<mlir::AffineExpr> results, mlir::MLIRContext * context){ return mlir::AffineMap::get(dimCount, symbolCount, results, context); }, "dim_count"_a, "symbol_count"_a, "results"_a, "context"_a)
.def_static("get_constant_map", &mlir::AffineMap::getConstantMap, "val"_a, "context"_a)
.def_static("get_multi_dim_identity_map", &mlir::AffineMap::getMultiDimIdentityMap, "num_dims"_a, "context"_a)
.def_static("get_minor_identity_map", &mlir::AffineMap::getMinorIdentityMap, "dims"_a, "results"_a, "context"_a)
.def_static("get_filtered_identity_map", &mlir::AffineMap::getFilteredIdentityMap, "ctx"_a, "num_dims"_a, "keep_dim_filter"_a)
.def_static("get_permutation_map", [](llvm::ArrayRef<unsigned int> permutation, mlir::MLIRContext * context){ return mlir::AffineMap::getPermutationMap(permutation, context); }, "permutation"_a, "context"_a)
.def_static("get_permutation_map", [](ArrayRef<int64_t> permutation, mlir::MLIRContext * context){ return mlir::AffineMap::getPermutationMap(permutation, context); }, "permutation"_a, "context"_a)
.def_static("get_multi_dim_map_with_targets", &mlir::AffineMap::getMultiDimMapWithTargets, "num_dims"_a, "targets"_a, "context"_a)
.def_static("infer_from_expr_list", [](llvm::ArrayRef<llvm::ArrayRef<mlir::AffineExpr>> exprsList, mlir::MLIRContext * context){ return mlir::AffineMap::inferFromExprList(exprsList, context); }, "exprs_list"_a, "context"_a)
.def_static("infer_from_expr_list", [](llvm::ArrayRef<llvm::SmallVector<mlir::AffineExpr, 4>> exprsList, mlir::MLIRContext * context){ return mlir::AffineMap::inferFromExprList(exprsList, context); }, "exprs_list"_a, "context"_a)
.def_prop_ro("context", &mlir::AffineMap::getContext)
.def("__eq__", &mlir::AffineMap::operator==, "other"_a)
.def("__ne__", &mlir::AffineMap::operator!=, "other"_a)
.def("is_identity", &mlir::AffineMap::isIdentity)
.def("is_symbol_identity", &mlir::AffineMap::isSymbolIdentity)
.def("is_minor_identity", &mlir::AffineMap::isMinorIdentity)
.def_prop_ro("broadcast_dims", &mlir::AffineMap::getBroadcastDims)
.def("is_minor_identity_with_broadcasting", &mlir::AffineMap::isMinorIdentityWithBroadcasting, "broadcasted_dims"_a)
.def("is_permutation_of_minor_identity_with_broadcasting", &mlir::AffineMap::isPermutationOfMinorIdentityWithBroadcasting, "permuted_dims"_a)
.def("is_empty", &mlir::AffineMap::isEmpty)
.def("is_single_constant", &mlir::AffineMap::isSingleConstant)
.def("is_constant", &mlir::AffineMap::isConstant)
.def_prop_ro("single_constant_result", &mlir::AffineMap::getSingleConstantResult)
.def_prop_ro("constant_results", &mlir::AffineMap::getConstantResults)
.def("print", &mlir::AffineMap::print, "os"_a)
.def("dump", &mlir::AffineMap::dump)
.def_prop_ro("num_dims", &mlir::AffineMap::getNumDims)
.def_prop_ro("num_symbols", &mlir::AffineMap::getNumSymbols)
.def_prop_ro("num_results", &mlir::AffineMap::getNumResults)
.def_prop_ro("num_inputs", &mlir::AffineMap::getNumInputs)
.def_prop_ro("results", &mlir::AffineMap::getResults)
.def("get_result", &mlir::AffineMap::getResult, "idx"_a)
.def("get_dim_position", &mlir::AffineMap::getDimPosition, "idx"_a)
.def("get_result_position", &mlir::AffineMap::getResultPosition, "input"_a)
.def("is_function_of_dim", &mlir::AffineMap::isFunctionOfDim, "position"_a)
.def("is_function_of_symbol", &mlir::AffineMap::isFunctionOfSymbol, "position"_a)
.def("walk_exprs", &mlir::AffineMap::walkExprs, "callback"_a)
.def("replace_dims_and_symbols", &mlir::AffineMap::replaceDimsAndSymbols, "dim_replacements"_a, "sym_replacements"_a, "num_result_dims"_a, "num_result_syms"_a)
.def("replace", [](mlir::AffineMap& self, mlir::AffineExpr expr, mlir::AffineExpr replacement, unsigned int numResultDims, unsigned int numResultSyms){ return self.replace(expr, replacement, numResultDims, numResultSyms); }, "expr"_a, "replacement"_a, "num_result_dims"_a, "num_result_syms"_a)
.def("replace", [](mlir::AffineMap& self, const llvm::DenseMap<mlir::AffineExpr, mlir::AffineExpr> & map){ return self.replace(map); }, "map"_a)
.def("replace", [](mlir::AffineMap& self, const llvm::DenseMap<mlir::AffineExpr, mlir::AffineExpr> & map, unsigned int numResultDims, unsigned int numResultSyms){ return self.replace(map, numResultDims, numResultSyms); }, "map"_a, "num_result_dims"_a, "num_result_syms"_a)
.def("shift_dims", &mlir::AffineMap::shiftDims, "shift"_a, "offset"_a)
.def("shift_symbols", &mlir::AffineMap::shiftSymbols, "shift"_a, "offset"_a)
.def("drop_result", &mlir::AffineMap::dropResult, "pos"_a)
.def("drop_results", [](mlir::AffineMap& self, ArrayRef<int64_t> positions){ return self.dropResults(positions); }, "positions"_a)
.def("drop_results", [](mlir::AffineMap& self, const llvm::SmallBitVector & positions){ return self.dropResults(positions); }, "positions"_a)
.def("insert_result", &mlir::AffineMap::insertResult, "expr"_a, "pos"_a)
.def("constant_fold", &mlir::AffineMap::constantFold, "operand_constants"_a, "results"_a, "has_poison"_a)
.def("partial_constant_fold", &mlir::AffineMap::partialConstantFold, "operand_constants"_a, "results"_a, "has_poison"_a)
.def("compose", [](mlir::AffineMap& self, mlir::AffineMap map){ return self.compose(map); }, "map"_a)
.def("compose", [](mlir::AffineMap& self, ArrayRef<int64_t> values){ return self.compose(values); }, "values"_a)
.def_prop_ro("num_of_zero_results", &mlir::AffineMap::getNumOfZeroResults)
.def("drop_zero_results", &mlir::AffineMap::dropZeroResults)
.def("is_projected_permutation", &mlir::AffineMap::isProjectedPermutation, "allow_zero_in_results"_a)
.def("is_permutation", &mlir::AffineMap::isPermutation)
.def("get_sub_map", &mlir::AffineMap::getSubMap, "result_pos"_a)
.def("get_slice_map", &mlir::AffineMap::getSliceMap, "start"_a, "length"_a)
.def("get_major_sub_map", &mlir::AffineMap::getMajorSubMap, "num_results"_a)
.def("get_minor_sub_map", &mlir::AffineMap::getMinorSubMap, "num_results"_a)
.def_prop_ro("largest_known_divisor_of_map_exprs", &mlir::AffineMap::getLargestKnownDivisorOfMapExprs)
.def("__str__", [](mlir::AffineMap& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_MutableAffineMap = nb::class_<mlir::MutableAffineMap>(m, "MutableAffineMap")
.def(nb::init<>())
.def(nb::init<mlir::AffineMap>(), "map"_a)
.def_prop_ro("results", &mlir::MutableAffineMap::getResults)
.def("get_result", &mlir::MutableAffineMap::getResult, "idx"_a)
.def("set_result", &mlir::MutableAffineMap::setResult, "idx"_a, "result"_a)
.def_prop_ro("num_results", &mlir::MutableAffineMap::getNumResults)
.def_prop_ro("num_dims", &mlir::MutableAffineMap::getNumDims)
.def("set_num_dims", &mlir::MutableAffineMap::setNumDims, "d"_a)
.def_prop_ro("num_symbols", &mlir::MutableAffineMap::getNumSymbols)
.def("set_num_symbols", &mlir::MutableAffineMap::setNumSymbols, "d"_a)
.def_prop_ro("context", &mlir::MutableAffineMap::getContext)
.def("is_multiple_of", &mlir::MutableAffineMap::isMultipleOf, "idx"_a, "factor"_a)
.def("reset", &mlir::MutableAffineMap::reset, "map"_a)
.def("simplify", &mlir::MutableAffineMap::simplify)
.def_prop_ro("affine_map", &mlir::MutableAffineMap::getAffineMap)
;

auto mlir_detail_MemRefElementTypeInterfaceInterfaceTraits = nb::class_<mlir::detail::MemRefElementTypeInterfaceInterfaceTraits>(m, "MemRefElementTypeInterfaceInterfaceTraits")
;

auto mlir_detail_MemRefElementTypeInterfaceInterfaceTraits_Concept = nb::class_<mlir::detail::MemRefElementTypeInterfaceInterfaceTraits::Concept>(mlir_detail_MemRefElementTypeInterfaceInterfaceTraits, "Concept")
;

auto mlir_MemRefElementTypeInterface = nb::class_<mlir::MemRefElementTypeInterface>(m, "MemRefElementTypeInterface")
;

auto mlir_detail_ShapedTypeInterfaceTraits = nb::class_<mlir::detail::ShapedTypeInterfaceTraits>(m, "ShapedTypeInterfaceTraits")
;

auto mlir_detail_ShapedTypeInterfaceTraits_Concept = nb::class_<mlir::detail::ShapedTypeInterfaceTraits::Concept>(mlir_detail_ShapedTypeInterfaceTraits, "Concept")
;

auto mlir_ShapedType = nb::class_<mlir::ShapedType>(m, "ShapedType")
.def("clone_with", &mlir::ShapedType::cloneWith, "shape"_a, "element_type"_a)
.def_prop_ro("element_type", &mlir::ShapedType::getElementType)
.def("has_rank", &mlir::ShapedType::hasRank)
.def_prop_ro("shape", &mlir::ShapedType::getShape)
.def_static("is_dynamic", &mlir::ShapedType::isDynamic, "d_value"_a)
.def_static("is_dynamic_shape", &mlir::ShapedType::isDynamicShape, "d_sizes"_a)
.def_static("get_num_elements", [](ArrayRef<int64_t> shape){ return mlir::ShapedType::getNumElements(shape); }, "shape"_a)
.def("clone", [](mlir::ShapedType& self, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType){ return self.clone(shape, elementType); }, "shape"_a, "element_type"_a)
.def("clone", [](mlir::ShapedType& self, ::llvm::ArrayRef<int64_t> shape){ return self.clone(shape); }, "shape"_a)
.def_prop_ro("element_type_bit_width", &mlir::ShapedType::getElementTypeBitWidth)
.def_prop_ro("rank", &mlir::ShapedType::getRank)
.def_prop_ro("num_elements", [](mlir::ShapedType& self){ return self.getNumElements(); })
.def("is_dynamic_dim", &mlir::ShapedType::isDynamicDim, "idx"_a)
.def("has_static_shape", [](mlir::ShapedType& self){ return self.hasStaticShape(); })
.def("has_static_shape", [](mlir::ShapedType& self, ::llvm::ArrayRef<int64_t> shape){ return self.hasStaticShape(shape); }, "shape"_a)
.def_prop_ro("num_dynamic_dims", &mlir::ShapedType::getNumDynamicDims)
.def("get_dim_size", &mlir::ShapedType::getDimSize, "idx"_a)
.def("get_dynamic_dim_index", &mlir::ShapedType::getDynamicDimIndex, "index"_a)
;

auto mlir_detail_ElementsAttrIndexer = nb::class_<mlir::detail::ElementsAttrIndexer>(m, "ElementsAttrIndexer")
.def(nb::init<>())
.def(nb::init<mlir::detail::ElementsAttrIndexer &&>(), "rhs"_a)
.def(nb::init<const mlir::detail::ElementsAttrIndexer &>(), "rhs"_a)
;

auto mlir_detail_TypedAttrInterfaceTraits = nb::class_<mlir::detail::TypedAttrInterfaceTraits>(m, "TypedAttrInterfaceTraits")
;

auto mlir_detail_TypedAttrInterfaceTraits_Concept = nb::class_<mlir::detail::TypedAttrInterfaceTraits::Concept>(mlir_detail_TypedAttrInterfaceTraits, "Concept")
;

auto mlir_TypedAttr = nb::class_<mlir::TypedAttr>(m, "TypedAttr")
.def_prop_ro("type", &mlir::TypedAttr::getType)
;

auto mlir_detail_ElementsAttrInterfaceTraits = nb::class_<mlir::detail::ElementsAttrInterfaceTraits>(m, "ElementsAttrInterfaceTraits")
;

auto mlir_detail_ElementsAttrInterfaceTraits_Concept = nb::class_<mlir::detail::ElementsAttrInterfaceTraits::Concept>(mlir_detail_ElementsAttrInterfaceTraits, "Concept")
.def("initialize_interface_concept", &mlir::detail::ElementsAttrInterfaceTraits::Concept::initializeInterfaceConcept, "interface_map"_a)
;

auto mlir_ElementsAttr = nb::class_<mlir::ElementsAttr>(m, "ElementsAttr")
.def("get_values_impl", &mlir::ElementsAttr::getValuesImpl, "element_id"_a)
.def("is_splat", &mlir::ElementsAttr::isSplat)
.def_prop_ro("shaped_type", &mlir::ElementsAttr::getShapedType)
.def_prop_ro("element_type", [](mlir::ElementsAttr& self){ return self.getElementType(); })
.def_static("get_element_type", [](mlir::ElementsAttr elementsAttr){ return mlir::ElementsAttr::getElementType(elementsAttr); }, "elements_attr"_a)
.def("is_valid_index", [](mlir::ElementsAttr& self, ArrayRef<uint64_t> index){ return self.isValidIndex(index); }, "index"_a)
.def_static("is_valid_index_static", [](mlir::ShapedType type, ArrayRef<uint64_t> index){ return mlir::ElementsAttr::isValidIndex(type, index); }, "type"_a, "index"_a)
.def_static("is_valid_index_static", [](mlir::ElementsAttr elementsAttr, ArrayRef<uint64_t> index){ return mlir::ElementsAttr::isValidIndex(elementsAttr, index); }, "elements_attr"_a, "index"_a)
.def("get_flattened_index", [](mlir::ElementsAttr& self, ArrayRef<uint64_t> index){ return self.getFlattenedIndex(index); }, "index"_a)
.def_static("get_flattened_index_static", [](mlir::Type type, ArrayRef<uint64_t> index){ return mlir::ElementsAttr::getFlattenedIndex(type, index); }, "type"_a, "index"_a)
.def_static("get_flattened_index_static", [](mlir::ElementsAttr elementsAttr, ArrayRef<uint64_t> index){ return mlir::ElementsAttr::getFlattenedIndex(elementsAttr, index); }, "elements_attr"_a, "index"_a)
.def_prop_ro("num_elements", [](mlir::ElementsAttr& self){ return self.getNumElements(); })
.def_static("get_num_elements", [](mlir::ElementsAttr elementsAttr){ return mlir::ElementsAttr::getNumElements(elementsAttr); }, "elements_attr"_a)
.def("size", &mlir::ElementsAttr::size)
.def("empty", &mlir::ElementsAttr::empty)
.def_prop_ro("type", &mlir::ElementsAttr::getType)
;

auto mlir_detail_MemRefLayoutAttrInterfaceInterfaceTraits = nb::class_<mlir::detail::MemRefLayoutAttrInterfaceInterfaceTraits>(m, "MemRefLayoutAttrInterfaceInterfaceTraits")
;

auto mlir_detail_MemRefLayoutAttrInterfaceInterfaceTraits_Concept = nb::class_<mlir::detail::MemRefLayoutAttrInterfaceInterfaceTraits::Concept>(mlir_detail_MemRefLayoutAttrInterfaceInterfaceTraits, "Concept")
;

auto mlir_MemRefLayoutAttrInterface = nb::class_<mlir::MemRefLayoutAttrInterface>(m, "MemRefLayoutAttrInterface")
.def_prop_ro("affine_map", &mlir::MemRefLayoutAttrInterface::getAffineMap)
.def("is_identity", &mlir::MemRefLayoutAttrInterface::isIdentity)
.def("verify_layout", &mlir::MemRefLayoutAttrInterface::verifyLayout, "shape"_a, "emit_error"_a)
;

auto mlir_DenseElementsAttr = nb::class_<mlir::DenseElementsAttr, mlir::Attribute>(m, "DenseElementsAttr")
.def_static("classof", &mlir::DenseElementsAttr::classof, "attr"_a)
.def_static("get", [](mlir::ShapedType type, llvm::ArrayRef<mlir::Attribute> values){ return mlir::DenseElementsAttr::get(type, values); }, "type"_a, "values"_a)
.def_static("get", [](mlir::ShapedType type, llvm::ArrayRef<bool> values){ return mlir::DenseElementsAttr::get(type, values); }, "type"_a, "values"_a)
.def_static("get", [](mlir::ShapedType type, llvm::ArrayRef<llvm::StringRef> values){ return mlir::DenseElementsAttr::get(type, values); }, "type"_a, "values"_a)
.def_static("get", [](mlir::ShapedType type, llvm::ArrayRef<llvm::APInt> values){ return mlir::DenseElementsAttr::get(type, values); }, "type"_a, "values"_a)
.def_static("get", [](mlir::ShapedType type, llvm::ArrayRef<std::complex<llvm::APInt>> values){ return mlir::DenseElementsAttr::get(type, values); }, "type"_a, "values"_a)
.def_static("get", [](mlir::ShapedType type, llvm::ArrayRef<llvm::APFloat> values){ return mlir::DenseElementsAttr::get(type, values); }, "type"_a, "values"_a)
.def_static("get", [](mlir::ShapedType type, llvm::ArrayRef<std::complex<llvm::APFloat>> values){ return mlir::DenseElementsAttr::get(type, values); }, "type"_a, "values"_a)
.def_static("get_from_raw_buffer", &mlir::DenseElementsAttr::getFromRawBuffer, "type"_a, "raw_buffer"_a)
.def_static("is_valid_raw_buffer", &mlir::DenseElementsAttr::isValidRawBuffer, "type"_a, "raw_buffer"_a, "detected_splat"_a)
.def("is_splat", &mlir::DenseElementsAttr::isSplat)
.def_prop_ro("raw_data", &mlir::DenseElementsAttr::getRawData)
.def_prop_ro("raw_string_data", &mlir::DenseElementsAttr::getRawStringData)
.def_prop_ro("type", &mlir::DenseElementsAttr::getType)
.def_prop_ro("element_type", &mlir::DenseElementsAttr::getElementType)
.def_prop_ro("num_elements", &mlir::DenseElementsAttr::getNumElements)
.def("size", &mlir::DenseElementsAttr::size)
.def("empty", &mlir::DenseElementsAttr::empty)
.def("reshape", &mlir::DenseElementsAttr::reshape, "new_type"_a)
.def("resize_splat", &mlir::DenseElementsAttr::resizeSplat, "new_type"_a)
.def("bitcast", &mlir::DenseElementsAttr::bitcast, "new_el_type"_a)
.def("map_values", [](mlir::DenseElementsAttr& self, mlir::Type newElementType, llvm::function_ref<llvm::APInt (const llvm::APInt &)> mapping){ return self.mapValues(newElementType, mapping); }, "new_element_type"_a, "mapping"_a)
.def("map_values", [](mlir::DenseElementsAttr& self, mlir::Type newElementType, llvm::function_ref<llvm::APInt (const llvm::APFloat &)> mapping){ return self.mapValues(newElementType, mapping); }, "new_element_type"_a, "mapping"_a)
;

auto mlir_DenseElementsAttr_AttributeElementIterator = nb::class_<mlir::DenseElementsAttr::AttributeElementIterator>(mlir_DenseElementsAttr, "AttributeElementIterator")
;

auto mlir_DenseElementsAttr_BoolElementIterator = nb::class_<mlir::DenseElementsAttr::BoolElementIterator>(mlir_DenseElementsAttr, "BoolElementIterator")
;

auto mlir_DenseElementsAttr_IntElementIterator = nb::class_<mlir::DenseElementsAttr::IntElementIterator>(mlir_DenseElementsAttr, "IntElementIterator")
;

auto mlir_DenseElementsAttr_ComplexIntElementIterator = nb::class_<mlir::DenseElementsAttr::ComplexIntElementIterator>(mlir_DenseElementsAttr, "ComplexIntElementIterator")
;

auto mlir_DenseElementsAttr_FloatElementIterator = nb::class_<mlir::DenseElementsAttr::FloatElementIterator>(mlir_DenseElementsAttr, "FloatElementIterator")
.def("map_element", &mlir::DenseElementsAttr::FloatElementIterator::mapElement, "value"_a)
;

auto mlir_DenseElementsAttr_ComplexFloatElementIterator = nb::class_<mlir::DenseElementsAttr::ComplexFloatElementIterator>(mlir_DenseElementsAttr, "ComplexFloatElementIterator")
.def("map_element", &mlir::DenseElementsAttr::ComplexFloatElementIterator::mapElement, "value"_a)
;

auto mlir_SplatElementsAttr = nb::class_<mlir::SplatElementsAttr, mlir::DenseElementsAttr>(m, "SplatElementsAttr")
.def_static("classof", &mlir::SplatElementsAttr::classof, "attr"_a)
;

auto mlir_AffineMapAttr = nb::class_<mlir::AffineMapAttr,  mlir::Attribute>(m, "AffineMapAttr")
.def_prop_ro("affine_map", &mlir::AffineMapAttr::getAffineMap)
.def_static("get", &mlir::AffineMapAttr::get, "value"_a)
.def_prop_ro("value", &mlir::AffineMapAttr::getValue)
;

auto mlir_ArrayAttr = nb::class_<mlir::ArrayAttr,  mlir::Attribute>(m, "ArrayAttr")
.def("__getitem__", &mlir::ArrayAttr::operator[], "idx"_a)
.def("begin", &mlir::ArrayAttr::begin)
.def("end", &mlir::ArrayAttr::end)
.def("size", &mlir::ArrayAttr::size)
.def("empty", &mlir::ArrayAttr::empty)
.def_static("get", &mlir::ArrayAttr::get, "context"_a, "value"_a)
.def_prop_ro("value", &mlir::ArrayAttr::getValue)
;

auto mlir_DenseArrayAttr = nb::class_<mlir::DenseArrayAttr,  mlir::Attribute>(m, "DenseArrayAttr")
.def("size", &mlir::DenseArrayAttr::size)
.def("empty", &mlir::DenseArrayAttr::empty)
.def_static("get", [](mlir::MLIRContext * context, mlir::Type elementType, int64_t size, llvm::ArrayRef<char> rawData){ return mlir::DenseArrayAttr::get(context, elementType, size, rawData); }, "context"_a, "element_type"_a, "size"_a, "raw_data"_a)
.def_static("get", [](mlir::Type elementType, unsigned int size, llvm::ArrayRef<char> rawData){ return mlir::DenseArrayAttr::get(elementType, size, rawData); }, "element_type"_a, "size"_a, "raw_data"_a)
.def_static("verify", &mlir::DenseArrayAttr::verify, "emit_error"_a, "element_type"_a, "size"_a, "raw_data"_a)
.def_static("verify_invariants", &mlir::DenseArrayAttr::verifyInvariants, "emit_error"_a, "element_type"_a, "size"_a, "raw_data"_a)
.def_prop_ro("element_type", &mlir::DenseArrayAttr::getElementType)
.def_prop_ro("size", &mlir::DenseArrayAttr::getSize)
.def_prop_ro("raw_data", &mlir::DenseArrayAttr::getRawData)
;

auto mlir_DenseIntOrFPElementsAttr = nb::class_<mlir::DenseIntOrFPElementsAttr,  mlir::DenseElementsAttr>(m, "DenseIntOrFPElementsAttr")
.def_static("convert_endian_of_array_ref_for_b_emachine", &mlir::DenseIntOrFPElementsAttr::convertEndianOfArrayRefForBEmachine, "in_raw_data"_a, "out_raw_data"_a, "type"_a)
;

auto mlir_DenseStringElementsAttr = nb::class_<mlir::DenseStringElementsAttr,  mlir::DenseElementsAttr>(m, "DenseStringElementsAttr")
.def_static("get", &mlir::DenseStringElementsAttr::get, "type"_a, "values"_a)
;

auto mlir_DenseResourceElementsAttr = nb::class_<mlir::DenseResourceElementsAttr,  mlir::Attribute>(m, "DenseResourceElementsAttr")
.def_static("get", [](mlir::ShapedType type, mlir::DialectResourceBlobHandle<mlir::BuiltinDialect> handle){ return mlir::DenseResourceElementsAttr::get(type, handle); }, "type"_a, "handle"_a)
.def_prop_ro("type", &mlir::DenseResourceElementsAttr::getType)
.def_prop_ro("raw_handle", &mlir::DenseResourceElementsAttr::getRawHandle)
;

auto mlir_DictionaryAttr = nb::class_<mlir::DictionaryAttr,  mlir::Attribute>(m, "DictionaryAttr")
.def_static("get_with_sorted", &mlir::DictionaryAttr::getWithSorted, "context"_a, "value"_a)
.def("get", [](mlir::DictionaryAttr& self, llvm::StringRef name){ return self.get(name); }, "name"_a)
.def("get", [](mlir::DictionaryAttr& self, mlir::StringAttr name){ return self.get(name); }, "name"_a)
.def("get_named", [](mlir::DictionaryAttr& self, llvm::StringRef name){ return self.getNamed(name); }, "name"_a)
.def("get_named", [](mlir::DictionaryAttr& self, mlir::StringAttr name){ return self.getNamed(name); }, "name"_a)
.def("contains", [](mlir::DictionaryAttr& self, llvm::StringRef name){ return self.contains(name); }, "name"_a)
.def("contains", [](mlir::DictionaryAttr& self, mlir::StringAttr name){ return self.contains(name); }, "name"_a)
.def("begin", &mlir::DictionaryAttr::begin)
.def("end", &mlir::DictionaryAttr::end)
.def("empty", &mlir::DictionaryAttr::empty)
.def("size", &mlir::DictionaryAttr::size)
.def_static("sort", &mlir::DictionaryAttr::sort, "values"_a, "storage"_a)
.def_static("sort_in_place", &mlir::DictionaryAttr::sortInPlace, "array"_a)
.def_static("find_duplicate", &mlir::DictionaryAttr::findDuplicate, "array"_a, "is_sorted"_a)
.def_static("get_static", [](mlir::MLIRContext * context, llvm::ArrayRef<mlir::NamedAttribute> value){ return mlir::DictionaryAttr::get(context, value); }, "context"_a, "value"_a)
.def_prop_ro("value", &mlir::DictionaryAttr::getValue)
;

auto mlir_FloatAttr = nb::class_<mlir::FloatAttr,  mlir::Attribute>(m, "FloatAttr")
.def_prop_ro("value_as_double", [](mlir::FloatAttr& self){ return self.getValueAsDouble(); })
.def_static("get_value_as_double", [](llvm::APFloat val){ return mlir::FloatAttr::getValueAsDouble(val); }, "val"_a)
.def_static("get", [](mlir::Type type, const llvm::APFloat & value){ return mlir::FloatAttr::get(type, value); }, "type"_a, "value"_a)
.def_static("get", [](mlir::Type type, double value){ return mlir::FloatAttr::get(type, value); }, "type"_a, "value"_a)
.def_static("verify", &mlir::FloatAttr::verify, "emit_error"_a, "type"_a, "value"_a)
.def_static("verify_invariants", &mlir::FloatAttr::verifyInvariants, "emit_error"_a, "type"_a, "value"_a)
.def_prop_ro("type", &mlir::FloatAttr::getType)
.def_prop_ro("value", &mlir::FloatAttr::getValue)
;

auto mlir_IntegerAttr = nb::class_<mlir::IntegerAttr,  mlir::Attribute>(m, "IntegerAttr")
.def_prop_ro("int", &mlir::IntegerAttr::getInt)
.def_prop_ro("s_int", &mlir::IntegerAttr::getSInt)
.def_prop_ro("u_int", &mlir::IntegerAttr::getUInt)
.def_prop_ro("aps_int", &mlir::IntegerAttr::getAPSInt)
.def_static("get", [](mlir::Type type, const llvm::APInt & value){ return mlir::IntegerAttr::get(type, value); }, "type"_a, "value"_a)
.def_static("get", [](mlir::MLIRContext * context, const llvm::APSInt & value){ return mlir::IntegerAttr::get(context, value); }, "context"_a, "value"_a)
.def_static("get", [](mlir::Type type, int64_t value){ return mlir::IntegerAttr::get(type, value); }, "type"_a, "value"_a)
.def_static("verify", &mlir::IntegerAttr::verify, "emit_error"_a, "type"_a, "value"_a)
.def_static("verify_invariants", &mlir::IntegerAttr::verifyInvariants, "emit_error"_a, "type"_a, "value"_a)
.def_prop_ro("type", &mlir::IntegerAttr::getType)
.def_prop_ro("value", &mlir::IntegerAttr::getValue)
;

auto mlir_IntegerSetAttr = nb::class_<mlir::IntegerSetAttr,  mlir::Attribute>(m, "IntegerSetAttr")
.def_static("get", &mlir::IntegerSetAttr::get, "value"_a)
.def_prop_ro("value", &mlir::IntegerSetAttr::getValue)
;

auto mlir_OpaqueAttr = nb::class_<mlir::OpaqueAttr,  mlir::Attribute>(m, "OpaqueAttr")
.def_static("get", &mlir::OpaqueAttr::get, "dialect"_a, "attr_data"_a, "type"_a)
.def_static("verify", &mlir::OpaqueAttr::verify, "emit_error"_a, "dialect_namespace"_a, "attr_data"_a, "type"_a)
.def_static("verify_invariants", &mlir::OpaqueAttr::verifyInvariants, "emit_error"_a, "dialect_namespace"_a, "attr_data"_a, "type"_a)
.def_prop_ro("dialect_namespace", &mlir::OpaqueAttr::getDialectNamespace)
.def_prop_ro("attr_data", &mlir::OpaqueAttr::getAttrData)
.def_prop_ro("type", &mlir::OpaqueAttr::getType)
;

auto mlir_SparseElementsAttr = nb::class_<mlir::SparseElementsAttr,  mlir::Attribute>(m, "SparseElementsAttr")
.def_static("get", &mlir::SparseElementsAttr::get, "type"_a, "indices"_a, "values"_a)
.def_static("verify", &mlir::SparseElementsAttr::verify, "emit_error"_a, "type"_a, "indices"_a, "values"_a)
.def_static("verify_invariants", &mlir::SparseElementsAttr::verifyInvariants, "emit_error"_a, "type"_a, "indices"_a, "values"_a)
.def_prop_ro("type", &mlir::SparseElementsAttr::getType)
.def_prop_ro("indices", &mlir::SparseElementsAttr::getIndices)
;

auto mlir_StridedLayoutAttr = nb::class_<mlir::StridedLayoutAttr,  mlir::Attribute>(m, "StridedLayoutAttr")
.def("print", &mlir::StridedLayoutAttr::print, "os"_a)
.def("has_static_layout", &mlir::StridedLayoutAttr::hasStaticLayout)
.def_static("get", &mlir::StridedLayoutAttr::get, "context"_a, "offset"_a, "strides"_a)
.def_static("verify", &mlir::StridedLayoutAttr::verify, "emit_error"_a, "offset"_a, "strides"_a)
.def_static("verify_invariants", &mlir::StridedLayoutAttr::verifyInvariants, "emit_error"_a, "offset"_a, "strides"_a)
.def_prop_ro("offset", &mlir::StridedLayoutAttr::getOffset)
.def_prop_ro("strides", &mlir::StridedLayoutAttr::getStrides)
.def_prop_ro("affine_map", &mlir::StridedLayoutAttr::getAffineMap)
.def("verify_layout", &mlir::StridedLayoutAttr::verifyLayout, "shape"_a, "emit_error"_a)
.def("__str__", [](mlir::StridedLayoutAttr& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_StringAttr = nb::class_<mlir::StringAttr,  mlir::Attribute>(m, "StringAttr")
.def_prop_ro("referenced_dialect", &mlir::StringAttr::getReferencedDialect)
.def("strref", &mlir::StringAttr::strref)
.def("str", &mlir::StringAttr::str)
.def("data", &mlir::StringAttr::data, nb::rv_policy::reference_internal)
.def("size", &mlir::StringAttr::size)
.def("empty", &mlir::StringAttr::empty)
.def("begin", &mlir::StringAttr::begin)
.def("end", &mlir::StringAttr::end)
.def("compare", &mlir::StringAttr::compare, "rhs"_a)
.def_static("get", [](const llvm::Twine & bytes, mlir::Type type){ return mlir::StringAttr::get(bytes, type); }, "bytes"_a, "type"_a)
.def_static("get", [](mlir::MLIRContext * context, const llvm::Twine & bytes){ return mlir::StringAttr::get(context, bytes); }, "context"_a, "bytes"_a)
.def_static("get", [](mlir::MLIRContext * context){ return mlir::StringAttr::get(context); }, "context"_a)
.def_prop_ro("value", &mlir::StringAttr::getValue)
.def_prop_ro("type", &mlir::StringAttr::getType)
;

auto mlir_SymbolRefAttr = nb::class_<mlir::SymbolRefAttr,  mlir::Attribute>(m, "SymbolRefAttr")
.def_static("get", [](mlir::MLIRContext * ctx, llvm::StringRef value, llvm::ArrayRef<mlir::FlatSymbolRefAttr> nestedRefs){ return mlir::SymbolRefAttr::get(ctx, value, nestedRefs); }, "ctx"_a, "value"_a, "nested_refs"_a)
.def_static("get", [](mlir::StringAttr value){ return mlir::SymbolRefAttr::get(value); }, "value"_a)
.def_static("get", [](mlir::MLIRContext * ctx, llvm::StringRef value){ return mlir::SymbolRefAttr::get(ctx, value); }, "ctx"_a, "value"_a)
.def_static("get", [](mlir::Operation * symbol){ return mlir::SymbolRefAttr::get(symbol); }, "symbol"_a)
.def_prop_ro("leaf_reference", &mlir::SymbolRefAttr::getLeafReference)
.def_static("get", [](mlir::StringAttr rootReference, llvm::ArrayRef<mlir::FlatSymbolRefAttr> nestedReferences){ return mlir::SymbolRefAttr::get(rootReference, nestedReferences); }, "root_reference"_a, "nested_references"_a)
.def_prop_ro("root_reference", &mlir::SymbolRefAttr::getRootReference)
.def_prop_ro("nested_references", &mlir::SymbolRefAttr::getNestedReferences)
;

auto mlir_TypeAttr = nb::class_<mlir::TypeAttr,  mlir::Attribute>(m, "TypeAttr")
.def_static("get", &mlir::TypeAttr::get, "type"_a)
.def_prop_ro("value", &mlir::TypeAttr::getValue)
;

auto mlir_UnitAttr = nb::class_<mlir::UnitAttr,  mlir::Attribute>(m, "UnitAttr")
.def_static("get", &mlir::UnitAttr::get, "context"_a)
;

auto mlir_detail_TypeIDResolver___mlir_AffineMapAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::AffineMapAttr>>(m, "TypeIDResolver[AffineMapAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::AffineMapAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_ArrayAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::ArrayAttr>>(m, "TypeIDResolver[ArrayAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::ArrayAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_DenseArrayAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::DenseArrayAttr>>(m, "TypeIDResolver[DenseArrayAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::DenseArrayAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_DenseIntOrFPElementsAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::DenseIntOrFPElementsAttr>>(m, "TypeIDResolver[DenseIntOrFPElementsAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::DenseIntOrFPElementsAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_DenseStringElementsAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::DenseStringElementsAttr>>(m, "TypeIDResolver[DenseStringElementsAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::DenseStringElementsAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_DenseResourceElementsAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::DenseResourceElementsAttr>>(m, "TypeIDResolver[DenseResourceElementsAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::DenseResourceElementsAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_DictionaryAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::DictionaryAttr>>(m, "TypeIDResolver[DictionaryAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::DictionaryAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_FloatAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::FloatAttr>>(m, "TypeIDResolver[FloatAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::FloatAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_IntegerAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::IntegerAttr>>(m, "TypeIDResolver[IntegerAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::IntegerAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_IntegerSetAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::IntegerSetAttr>>(m, "TypeIDResolver[IntegerSetAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::IntegerSetAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_OpaqueAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::OpaqueAttr>>(m, "TypeIDResolver[OpaqueAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::OpaqueAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_SparseElementsAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::SparseElementsAttr>>(m, "TypeIDResolver[SparseElementsAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::SparseElementsAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_StridedLayoutAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::StridedLayoutAttr>>(m, "TypeIDResolver[StridedLayoutAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::StridedLayoutAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_StringAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::StringAttr>>(m, "TypeIDResolver[StringAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::StringAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_SymbolRefAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::SymbolRefAttr>>(m, "TypeIDResolver[SymbolRefAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::SymbolRefAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_TypeAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::TypeAttr>>(m, "TypeIDResolver[TypeAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::TypeAttr>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_UnitAttr__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::UnitAttr>>(m, "TypeIDResolver[UnitAttr]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::UnitAttr>::resolveTypeID)
;

auto mlir_detail_DenseArrayAttrImpl__bool__ = nb::class_<mlir::detail::DenseArrayAttrImpl<bool>>(m, "DenseArrayAttrImpl[bool]")
;

auto mlir_detail_DenseArrayAttrImpl__int8_t__ = nb::class_<mlir::detail::DenseArrayAttrImpl<int8_t>>(m, "DenseArrayAttrImpl[int8_t]")
;

auto mlir_detail_DenseArrayAttrImpl__int16_t__ = nb::class_<mlir::detail::DenseArrayAttrImpl<int16_t>>(m, "DenseArrayAttrImpl[int16_t]")
;

auto mlir_detail_DenseArrayAttrImpl__int32_t__ = nb::class_<mlir::detail::DenseArrayAttrImpl<int32_t>>(m, "DenseArrayAttrImpl[int32_t]")
;

auto mlir_detail_DenseArrayAttrImpl__int64_t__ = nb::class_<mlir::detail::DenseArrayAttrImpl<int64_t>>(m, "DenseArrayAttrImpl[int64_t]")
;

auto mlir_detail_DenseArrayAttrImpl__float__ = nb::class_<mlir::detail::DenseArrayAttrImpl<float>>(m, "DenseArrayAttrImpl[float]")
;

auto mlir_detail_DenseArrayAttrImpl__double__ = nb::class_<mlir::detail::DenseArrayAttrImpl<double>>(m, "DenseArrayAttrImpl[double]")
;

auto mlir_detail_DenseResourceElementsAttrBase__bool__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<bool>>(m, "DenseResourceElementsAttrBase[bool]")
;

auto mlir_detail_DenseResourceElementsAttrBase__int8_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<int8_t>>(m, "DenseResourceElementsAttrBase[int8_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__int16_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<int16_t>>(m, "DenseResourceElementsAttrBase[int16_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__int32_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<int32_t>>(m, "DenseResourceElementsAttrBase[int32_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__int64_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<int64_t>>(m, "DenseResourceElementsAttrBase[int64_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__uint8_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<uint8_t>>(m, "DenseResourceElementsAttrBase[uint8_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__uint16_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<uint16_t>>(m, "DenseResourceElementsAttrBase[uint16_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__uint32_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<uint32_t>>(m, "DenseResourceElementsAttrBase[uint32_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__uint64_t__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<uint64_t>>(m, "DenseResourceElementsAttrBase[uint64_t]")
;

auto mlir_detail_DenseResourceElementsAttrBase__float__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<float>>(m, "DenseResourceElementsAttrBase[float]")
;

auto mlir_detail_DenseResourceElementsAttrBase__double__ = nb::class_<mlir::detail::DenseResourceElementsAttrBase<double>>(m, "DenseResourceElementsAttrBase[double]")
;

auto mlir_BoolAttr = nb::class_<mlir::BoolAttr, mlir::Attribute>(m, "BoolAttr")
.def_static("get", &mlir::BoolAttr::get, "context"_a, "value"_a)
.def_prop_ro("value", &mlir::BoolAttr::getValue)
.def_static("classof", &mlir::BoolAttr::classof, "attr"_a)
;

auto mlir_FlatSymbolRefAttr = nb::class_<mlir::FlatSymbolRefAttr, mlir::SymbolRefAttr>(m, "FlatSymbolRefAttr")
.def_static("get", [](mlir::StringAttr value){ return mlir::FlatSymbolRefAttr::get(value); }, "value"_a)
.def_static("get", [](mlir::MLIRContext * ctx, llvm::StringRef value){ return mlir::FlatSymbolRefAttr::get(ctx, value); }, "ctx"_a, "value"_a)
.def_static("get", [](mlir::Operation * symbol){ return mlir::FlatSymbolRefAttr::get(symbol); }, "symbol"_a)
.def_prop_ro("attr", &mlir::FlatSymbolRefAttr::getAttr)
.def_prop_ro("value", &mlir::FlatSymbolRefAttr::getValue)
.def_static("classof", &mlir::FlatSymbolRefAttr::classof, "attr"_a)
;

auto mlir_DenseFPElementsAttr = nb::class_<mlir::DenseFPElementsAttr, mlir::DenseIntOrFPElementsAttr>(m, "DenseFPElementsAttr")
.def("map_values", &mlir::DenseFPElementsAttr::mapValues, "new_element_type"_a, "mapping"_a)
.def("begin", &mlir::DenseFPElementsAttr::begin)
.def("end", &mlir::DenseFPElementsAttr::end)
.def_static("classof", &mlir::DenseFPElementsAttr::classof, "attr"_a)
;

auto mlir_DenseIntElementsAttr = nb::class_<mlir::DenseIntElementsAttr, mlir::DenseIntOrFPElementsAttr>(m, "DenseIntElementsAttr")
.def("map_values", &mlir::DenseIntElementsAttr::mapValues, "new_element_type"_a, "mapping"_a)
.def("begin", &mlir::DenseIntElementsAttr::begin)
.def("end", &mlir::DenseIntElementsAttr::end)
.def_static("classof", &mlir::DenseIntElementsAttr::classof, "attr"_a)
;

auto mlir_DistinctAttr = nb::class_<mlir::DistinctAttr,  mlir::Attribute>(m, "DistinctAttr")
.def_prop_ro("referenced_attr", &mlir::DistinctAttr::getReferencedAttr)
.def_static("create", &mlir::DistinctAttr::create, "referenced_attr"_a)
;

nb::enum_<mlir::DiagnosticSeverity>(m, "DiagnosticSeverity")
.value("Note", mlir::DiagnosticSeverity::Note)
.value("Warning", mlir::DiagnosticSeverity::Warning)
.value("Error", mlir::DiagnosticSeverity::Error)
.value("Remark", mlir::DiagnosticSeverity::Remark)
;

auto mlir_DiagnosticArgument = nb::class_<mlir::DiagnosticArgument>(m, "DiagnosticArgument")
.def(nb::init<mlir::Attribute>(), "attr"_a)
.def(nb::init<double>(), "val"_a)
.def(nb::init<float>(), "val"_a)
.def(nb::init<llvm::StringRef>(), "val"_a)
.def(nb::init<mlir::Type>(), "val"_a)
.def("print", &mlir::DiagnosticArgument::print, "os"_a)
.def_prop_ro("kind", &mlir::DiagnosticArgument::getKind)
.def_prop_ro("as_attribute", &mlir::DiagnosticArgument::getAsAttribute)
.def_prop_ro("as_double", &mlir::DiagnosticArgument::getAsDouble)
.def_prop_ro("as_integer", &mlir::DiagnosticArgument::getAsInteger)
.def_prop_ro("as_string", &mlir::DiagnosticArgument::getAsString)
.def_prop_ro("as_type", &mlir::DiagnosticArgument::getAsType)
.def_prop_ro("as_unsigned", &mlir::DiagnosticArgument::getAsUnsigned)
;

nb::enum_<mlir::DiagnosticArgument::DiagnosticArgumentKind>(m, "DiagnosticArgumentKind")
.value("Attribute", mlir::DiagnosticArgument::DiagnosticArgumentKind::Attribute)
.value("Double", mlir::DiagnosticArgument::DiagnosticArgumentKind::Double)
.value("Integer", mlir::DiagnosticArgument::DiagnosticArgumentKind::Integer)
.value("String", mlir::DiagnosticArgument::DiagnosticArgumentKind::String)
.value("Type", mlir::DiagnosticArgument::DiagnosticArgumentKind::Type)
.value("Unsigned", mlir::DiagnosticArgument::DiagnosticArgumentKind::Unsigned)
;

auto mlir_Diagnostic = nb::class_<mlir::Diagnostic>(m, "Diagnostic")
.def(nb::init<mlir::Location, mlir::DiagnosticSeverity>(), "loc"_a, "severity"_a)
.def(nb::init<mlir::Diagnostic &&>(), "_"_a)
.def_prop_ro("severity", &mlir::Diagnostic::getSeverity)
.def_prop_ro("location", &mlir::Diagnostic::getLocation)
.def_prop_ro("arguments", [](mlir::Diagnostic& self){ return self.getArguments(); })
.def_prop_ro("arguments", [](mlir::Diagnostic& self){ return self.getArguments(); })
.def("append_op", &mlir::Diagnostic::appendOp, "op"_a, "flags"_a, nb::rv_policy::reference_internal)
.def("print", &mlir::Diagnostic::print, "os"_a)
.def("str", &mlir::Diagnostic::str)
.def("attach_note", &mlir::Diagnostic::attachNote, "note_loc"_a, nb::rv_policy::reference_internal)
.def_prop_ro("notes", [](mlir::Diagnostic& self){ return self.getNotes(); })
.def_prop_ro("notes", [](mlir::Diagnostic& self){ return self.getNotes(); })
.def_prop_ro("metadata", &mlir::Diagnostic::getMetadata)
;

auto mlir_InFlightDiagnostic = nb::class_<mlir::InFlightDiagnostic>(m, "InFlightDiagnostic")
.def(nb::init<>())
.def(nb::init<mlir::InFlightDiagnostic &&>(), "rhs"_a)
.def("attach_note", &mlir::InFlightDiagnostic::attachNote, "note_loc"_a, nb::rv_policy::reference_internal)
.def_prop_ro("underlying_diagnostic", &mlir::InFlightDiagnostic::getUnderlyingDiagnostic)
.def("report", &mlir::InFlightDiagnostic::report)
.def("abandon", &mlir::InFlightDiagnostic::abandon)
;

auto mlir_DiagnosticEngine = nb::class_<mlir::DiagnosticEngine>(m, "DiagnosticEngine")
.def("erase_handler", &mlir::DiagnosticEngine::eraseHandler, "id"_a)
.def("emit", [](mlir::DiagnosticEngine& self, mlir::Location loc, mlir::DiagnosticSeverity severity){ return self.emit(loc, severity); }, "loc"_a, "severity"_a)
.def("emit", [](mlir::DiagnosticEngine& self, mlir::Diagnostic && diag){ return self.emit(std::move(diag)); }, "diag"_a)
;

auto mlir_ScopedDiagnosticHandler = nb::class_<mlir::ScopedDiagnosticHandler>(m, "ScopedDiagnosticHandler")
.def(nb::init<mlir::MLIRContext *>(), "ctx"_a)
;

auto mlir_SourceMgrDiagnosticHandler = nb::class_<mlir::SourceMgrDiagnosticHandler, mlir::ScopedDiagnosticHandler>(m, "SourceMgrDiagnosticHandler")
.def(nb::init<llvm::SourceMgr &, mlir::MLIRContext *, llvm::raw_ostream &, llvm::unique_function<bool (mlir::Location)> &&>(), "mgr"_a, "ctx"_a, "os"_a, "should_show_loc_fn"_a)
.def(nb::init<llvm::SourceMgr &, mlir::MLIRContext *, llvm::unique_function<bool (mlir::Location)> &&>(), "mgr"_a, "ctx"_a, "should_show_loc_fn"_a)
;

auto mlir_SourceMgrDiagnosticVerifierHandler = nb::class_<mlir::SourceMgrDiagnosticVerifierHandler, mlir::SourceMgrDiagnosticHandler>(m, "SourceMgrDiagnosticVerifierHandler")
.def(nb::init<llvm::SourceMgr &, mlir::MLIRContext *, llvm::raw_ostream &>(), "src_mgr"_a, "ctx"_a, "out"_a)
.def(nb::init<llvm::SourceMgr &, mlir::MLIRContext *>(), "src_mgr"_a, "ctx"_a)
.def("verify", &mlir::SourceMgrDiagnosticVerifierHandler::verify)
;

auto mlir_ParallelDiagnosticHandler = nb::class_<mlir::ParallelDiagnosticHandler>(m, "ParallelDiagnosticHandler")
.def(nb::init<mlir::MLIRContext *>(), "ctx"_a)
.def("set_order_id_for_thread", &mlir::ParallelDiagnosticHandler::setOrderIDForThread, "order_id"_a)
.def("erase_order_id_for_thread", &mlir::ParallelDiagnosticHandler::eraseOrderIDForThread)
;

auto mlir_OperandRange = nb::class_<mlir::OperandRange>(m, "OperandRange")
.def_prop_ro("types", &mlir::OperandRange::getTypes)
.def_prop_ro("type", &mlir::OperandRange::getType)
.def_prop_ro("begin_operand_index", &mlir::OperandRange::getBeginOperandIndex)
.def("split", &mlir::OperandRange::split, "segment_sizes"_a)
;

auto mlir_OperandRangeRange = nb::class_<mlir::OperandRangeRange>(m, "OperandRangeRange")
.def_prop_ro("types", &mlir::OperandRangeRange::getTypes)
.def_prop_ro("type", &mlir::OperandRangeRange::getType)
.def(nb::init<mlir::OperandRange, mlir::Attribute>(), "operands"_a, "operand_segments"_a)
.def("join", &mlir::OperandRangeRange::join)
;

auto mlir_MutableOperandRange = nb::class_<mlir::MutableOperandRange>(m, "MutableOperandRange")
.def(nb::init<mlir::Operation *, unsigned int, unsigned int, llvm::ArrayRef<std::pair<unsigned int, mlir::NamedAttribute>>>(), "owner"_a, "start"_a, "length"_a, "operand_segments"_a)
.def(nb::init<mlir::Operation *>(), "owner"_a)
.def(nb::init<mlir::OpOperand &>(), "op_operand"_a)
.def("slice", &mlir::MutableOperandRange::slice, "sub_start"_a, "sub_len"_a, "segment"_a)
.def("append", &mlir::MutableOperandRange::append, "values"_a)
.def("assign", [](mlir::MutableOperandRange& self, mlir::ValueRange values){ return self.assign(values); }, "values"_a)
.def("assign", [](mlir::MutableOperandRange& self, mlir::Value value){ return self.assign(value); }, "value"_a)
.def("erase", &mlir::MutableOperandRange::erase, "sub_start"_a, "sub_len"_a)
.def("clear", &mlir::MutableOperandRange::clear)
.def("size", &mlir::MutableOperandRange::size)
.def("empty", &mlir::MutableOperandRange::empty)
.def_prop_ro("as_operand_range", &mlir::MutableOperandRange::getAsOperandRange)
.def_prop_ro("owner", &mlir::MutableOperandRange::getOwner)
.def("split", &mlir::MutableOperandRange::split, "segment_sizes"_a)
.def("__getitem__", &mlir::MutableOperandRange::operator[], "index"_a, nb::rv_policy::reference_internal)
.def("begin", &mlir::MutableOperandRange::begin)
.def("end", &mlir::MutableOperandRange::end)
;

auto mlir_MutableOperandRangeRange = nb::class_<mlir::MutableOperandRangeRange>(m, "MutableOperandRangeRange")
.def(nb::init<const mlir::MutableOperandRange &, mlir::NamedAttribute>(), "operands"_a, "operand_segment_attr"_a)
.def("join", &mlir::MutableOperandRangeRange::join)
;

auto mlir_ResultRange = nb::class_<mlir::ResultRange>(m, "ResultRange")
.def(nb::init<mlir::OpResult>(), "result"_a)
.def_prop_ro("types", &mlir::ResultRange::getTypes)
.def_prop_ro("type", &mlir::ResultRange::getType)
.def_prop_ro("uses", &mlir::ResultRange::getUses)
.def("use_begin", &mlir::ResultRange::use_begin)
.def("use_end", &mlir::ResultRange::use_end)
.def("use_empty", &mlir::ResultRange::use_empty)
.def("replace_all_uses_with", [](mlir::ResultRange& self, mlir::Operation * op){ return self.replaceAllUsesWith(op); }, "op"_a)
.def("replace_uses_with_if", [](mlir::ResultRange& self, mlir::Operation * op, llvm::function_ref<bool (mlir::OpOperand &)> shouldReplace){ return self.replaceUsesWithIf(op, shouldReplace); }, "op"_a, "should_replace"_a)
.def_prop_ro("users", &mlir::ResultRange::getUsers)
.def("user_begin", &mlir::ResultRange::user_begin)
.def("user_end", &mlir::ResultRange::user_end)
;

auto mlir_ResultRange_UseIterator = nb::class_<mlir::ResultRange::UseIterator>(mlir_ResultRange, "UseIterator")
.def(nb::init<mlir::ResultRange, bool>(), "results"_a, "end"_a)
.def("__eq__", &mlir::ResultRange::UseIterator::operator==, "rhs"_a)
.def("__ne__", &mlir::ResultRange::UseIterator::operator!=, "rhs"_a)
;

auto mlir_ValueRange = nb::class_<mlir::ValueRange>(m, "ValueRange")
.def(nb::init<const mlir::Value &>(), "value"_a)
.def(nb::init<const std::initializer_list<mlir::Value> &>(), "values"_a)
.def(nb::init<llvm::iterator_range<llvm::detail::indexed_accessor_range_base<mlir::OperandRange, mlir::OpOperand *, mlir::Value, mlir::Value, mlir::Value>::iterator>>(), "values"_a)
.def(nb::init<llvm::iterator_range<llvm::detail::indexed_accessor_range_base<mlir::ResultRange, mlir::detail::OpResultImpl *, mlir::OpResult, mlir::OpResult, mlir::OpResult>::iterator>>(), "values"_a)
.def(nb::init<llvm::ArrayRef<mlir::BlockArgument>>(), "values"_a)
.def(nb::init<llvm::ArrayRef<mlir::Value>>(), "values"_a)
.def(nb::init<mlir::OperandRange>(), "values"_a)
.def(nb::init<mlir::ResultRange>(), "values"_a)
.def_prop_ro("types", &mlir::ValueRange::getTypes)
.def_prop_ro("type", &mlir::ValueRange::getType)
;

auto mlir_TypeRange = nb::class_<mlir::TypeRange>(m, "TypeRange")
.def(nb::init<llvm::ArrayRef<mlir::Type>>(), "types"_a)
.def(nb::init<mlir::OperandRange>(), "values"_a)
.def(nb::init<mlir::ResultRange>(), "values"_a)
.def(nb::init<mlir::ValueRange>(), "values"_a)
.def(nb::init<std::initializer_list<mlir::Type>>(), "types"_a)
;

auto mlir_TypeRangeRange = nb::class_<mlir::TypeRangeRange>(m, "TypeRangeRange")
;

auto mlir_AttrTypeSubElementHandler__TypeRange__ = nb::class_<mlir::AttrTypeSubElementHandler<TypeRange>>(m, "AttrTypeSubElementHandler[TypeRange]")
.def_static("walk", &mlir::AttrTypeSubElementHandler<TypeRange>::walk, "param"_a, "walker"_a)
.def_static("replace", &mlir::AttrTypeSubElementHandler<TypeRange>::replace, "param"_a, "attr_repls"_a, "type_repls"_a)
;

auto mlir_OpaqueProperties = nb::class_<mlir::OpaqueProperties>(m, "OpaqueProperties")
.def(nb::init<void *>(), "prop"_a)
;

auto mlir_OperationName = nb::class_<mlir::OperationName>(m, "OperationName")
.def(nb::init<llvm::StringRef, mlir::MLIRContext *>(), "name"_a, "context"_a)
.def("is_registered", &mlir::OperationName::isRegistered)
.def_prop_ro("type_id", &mlir::OperationName::getTypeID)
.def_prop_ro("registered_info", &mlir::OperationName::getRegisteredInfo)
.def("fold_hook", &mlir::OperationName::foldHook, "op"_a, "operands"_a, "results"_a)
.def("get_canonicalization_patterns", &mlir::OperationName::getCanonicalizationPatterns, "results"_a, "context"_a)
.def("has_trait", [](mlir::OperationName& self, mlir::TypeID traitID){ return self.hasTrait(traitID); }, "trait_id"_a)
.def("might_have_trait", [](mlir::OperationName& self, mlir::TypeID traitID){ return self.mightHaveTrait(traitID); }, "trait_id"_a)
.def_prop_ro("parse_assembly_fn", &mlir::OperationName::getParseAssemblyFn)
.def("populate_default_attrs", &mlir::OperationName::populateDefaultAttrs, "attrs"_a)
.def("print_assembly", &mlir::OperationName::printAssembly, "op"_a, "p"_a, "default_dialect"_a)
.def("verify_invariants", &mlir::OperationName::verifyInvariants, "op"_a)
.def("verify_region_invariants", &mlir::OperationName::verifyRegionInvariants, "op"_a)
.def_prop_ro("attribute_names", &mlir::OperationName::getAttributeNames)
.def("has_interface", [](mlir::OperationName& self, mlir::TypeID interfaceID){ return self.hasInterface(interfaceID); }, "interface_id"_a)
.def("might_have_interface", [](mlir::OperationName& self, mlir::TypeID interfaceID){ return self.mightHaveInterface(interfaceID); }, "interface_id"_a)
.def("get_inherent_attr", &mlir::OperationName::getInherentAttr, "op"_a, "name"_a)
.def("set_inherent_attr", &mlir::OperationName::setInherentAttr, "op"_a, "name"_a, "value"_a)
.def("populate_inherent_attrs", &mlir::OperationName::populateInherentAttrs, "op"_a, "attrs"_a)
.def("verify_inherent_attrs", &mlir::OperationName::verifyInherentAttrs, "attributes"_a, "emit_error"_a)
.def_prop_ro("op_property_byte_size", &mlir::OperationName::getOpPropertyByteSize)
.def("destroy_op_properties", &mlir::OperationName::destroyOpProperties, "properties"_a)
.def("init_op_properties", &mlir::OperationName::initOpProperties, "storage"_a, "init"_a)
.def("populate_default_properties", &mlir::OperationName::populateDefaultProperties, "properties"_a)
.def("get_op_properties_as_attribute", &mlir::OperationName::getOpPropertiesAsAttribute, "op"_a)
.def("set_op_properties_from_attribute", &mlir::OperationName::setOpPropertiesFromAttribute, "op_name"_a, "properties"_a, "attr"_a, "emit_error"_a)
.def("copy_op_properties", &mlir::OperationName::copyOpProperties, "lhs"_a, "rhs"_a)
.def("compare_op_properties", &mlir::OperationName::compareOpProperties, "lhs"_a, "rhs"_a)
.def("hash_op_properties", &mlir::OperationName::hashOpProperties, "properties"_a)
.def_prop_ro("dialect", &mlir::OperationName::getDialect)
.def_prop_ro("dialect_namespace", &mlir::OperationName::getDialectNamespace)
.def("strip_dialect", &mlir::OperationName::stripDialect)
.def_prop_ro("context", &mlir::OperationName::getContext)
.def_prop_ro("string_ref", &mlir::OperationName::getStringRef)
.def_prop_ro("identifier", &mlir::OperationName::getIdentifier)
.def("print", &mlir::OperationName::print, "os"_a)
.def("dump", &mlir::OperationName::dump)
.def("__eq__", &mlir::OperationName::operator==, "rhs"_a)
.def("__ne__", &mlir::OperationName::operator!=, "rhs"_a)
.def("__str__", [](mlir::OperationName& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_OperationName_InterfaceConcept = nb::class_<mlir::OperationName::InterfaceConcept>(mlir_OperationName, "InterfaceConcept")
.def("fold_hook", &mlir::OperationName::InterfaceConcept::foldHook, "_"_a, "__"_a, "___"_a)
.def("get_canonicalization_patterns", &mlir::OperationName::InterfaceConcept::getCanonicalizationPatterns, "_"_a, "__"_a)
.def("has_trait", &mlir::OperationName::InterfaceConcept::hasTrait, "_"_a)
.def_prop_ro("parse_assembly_fn", &mlir::OperationName::InterfaceConcept::getParseAssemblyFn)
.def("populate_default_attrs", &mlir::OperationName::InterfaceConcept::populateDefaultAttrs, "_"_a, "__"_a)
.def("print_assembly", &mlir::OperationName::InterfaceConcept::printAssembly, "_"_a, "__"_a, "___"_a)
.def("verify_invariants", &mlir::OperationName::InterfaceConcept::verifyInvariants, "_"_a)
.def("verify_region_invariants", &mlir::OperationName::InterfaceConcept::verifyRegionInvariants, "_"_a)
.def("get_inherent_attr", &mlir::OperationName::InterfaceConcept::getInherentAttr, "_"_a, "name"_a)
.def("set_inherent_attr", &mlir::OperationName::InterfaceConcept::setInherentAttr, "op"_a, "name"_a, "value"_a)
.def("populate_inherent_attrs", &mlir::OperationName::InterfaceConcept::populateInherentAttrs, "op"_a, "attrs"_a)
.def("verify_inherent_attrs", &mlir::OperationName::InterfaceConcept::verifyInherentAttrs, "op_name"_a, "attributes"_a, "emit_error"_a)
.def_prop_ro("op_property_byte_size", &mlir::OperationName::InterfaceConcept::getOpPropertyByteSize)
.def("init_properties", &mlir::OperationName::InterfaceConcept::initProperties, "op_name"_a, "storage"_a, "init"_a)
.def("delete_properties", &mlir::OperationName::InterfaceConcept::deleteProperties, "_"_a)
.def("populate_default_properties", &mlir::OperationName::InterfaceConcept::populateDefaultProperties, "op_name"_a, "properties"_a)
.def("set_properties_from_attr", &mlir::OperationName::InterfaceConcept::setPropertiesFromAttr, "_"_a, "__"_a, "___"_a, "emit_error"_a)
.def("get_properties_as_attr", &mlir::OperationName::InterfaceConcept::getPropertiesAsAttr, "_"_a)
.def("copy_properties", &mlir::OperationName::InterfaceConcept::copyProperties, "_"_a, "__"_a)
.def("compare_properties", &mlir::OperationName::InterfaceConcept::compareProperties, "_"_a, "__"_a)
.def("hash_properties", &mlir::OperationName::InterfaceConcept::hashProperties, "_"_a)
;

auto mlir_RegisteredOperationName = nb::class_<mlir::RegisteredOperationName, mlir::OperationName>(m, "RegisteredOperationName")
.def_static("lookup", [](llvm::StringRef name, mlir::MLIRContext * ctx){ return mlir::RegisteredOperationName::lookup(name, ctx); }, "name"_a, "ctx"_a)
.def_static("lookup", [](mlir::TypeID typeID, mlir::MLIRContext * ctx){ return mlir::RegisteredOperationName::lookup(typeID, ctx); }, "type_id"_a, "ctx"_a)
.def_prop_ro("dialect", &mlir::RegisteredOperationName::getDialect)
;

auto mlir_NamedAttrList = nb::class_<mlir::NamedAttrList>(m, "NamedAttrList")
.def(nb::init<>())
.def(nb::init<llvm::ArrayRef<mlir::NamedAttribute>>(), "attributes"_a)
.def(nb::init<mlir::DictionaryAttr>(), "attributes"_a)
.def(nb::init<mlir::NamedAttrList::const_iterator, mlir::NamedAttrList::const_iterator>(), "in_start"_a, "in_end"_a)
.def("__ne__", &mlir::NamedAttrList::operator!=, "other"_a)
.def("__eq__", &mlir::NamedAttrList::operator==, "other"_a)
.def("append", [](mlir::NamedAttrList& self, llvm::StringRef name, mlir::Attribute attr){ return self.append(name, attr); }, "name"_a, "attr"_a)
.def("append", [](mlir::NamedAttrList& self, mlir::StringAttr name, mlir::Attribute attr){ return self.append(name, attr); }, "name"_a, "attr"_a)
.def("append", [](mlir::NamedAttrList& self, mlir::NamedAttribute attr){ return self.append(attr); }, "attr"_a)
.def("assign", [](mlir::NamedAttrList& self, mlir::NamedAttrList::const_iterator inStart, mlir::NamedAttrList::const_iterator inEnd){ return self.assign(inStart, inEnd); }, "in_start"_a, "in_end"_a)
.def("assign", [](mlir::NamedAttrList& self, llvm::ArrayRef<mlir::NamedAttribute> range){ return self.assign(range); }, "range"_a)
.def("clear", &mlir::NamedAttrList::clear)
.def("empty", &mlir::NamedAttrList::empty)
.def("reserve", &mlir::NamedAttrList::reserve, "n"_a)
.def("push_back", &mlir::NamedAttrList::push_back, "new_attribute"_a)
.def("pop_back", &mlir::NamedAttrList::pop_back)
.def("find_duplicate", &mlir::NamedAttrList::findDuplicate)
.def("get_dictionary", &mlir::NamedAttrList::getDictionary, "context"_a)
.def_prop_ro("attrs", &mlir::NamedAttrList::getAttrs)
.def("get", [](mlir::NamedAttrList& self, mlir::StringAttr name){ return self.get(name); }, "name"_a)
.def("get", [](mlir::NamedAttrList& self, llvm::StringRef name){ return self.get(name); }, "name"_a)
.def("get_named", [](mlir::NamedAttrList& self, llvm::StringRef name){ return self.getNamed(name); }, "name"_a)
.def("get_named", [](mlir::NamedAttrList& self, mlir::StringAttr name){ return self.getNamed(name); }, "name"_a)
.def("set", [](mlir::NamedAttrList& self, mlir::StringAttr name, mlir::Attribute value){ return self.set(name, value); }, "name"_a, "value"_a)
.def("set", [](mlir::NamedAttrList& self, llvm::StringRef name, mlir::Attribute value){ return self.set(name, value); }, "name"_a, "value"_a)
.def("erase", [](mlir::NamedAttrList& self, mlir::StringAttr name){ return self.erase(name); }, "name"_a)
.def("erase", [](mlir::NamedAttrList& self, llvm::StringRef name){ return self.erase(name); }, "name"_a)
.def("begin", [](mlir::NamedAttrList& self){ return self.begin(); })
.def("end", [](mlir::NamedAttrList& self){ return self.end(); })
.def("begin", [](mlir::NamedAttrList& self){ return self.begin(); })
.def("end", [](mlir::NamedAttrList& self){ return self.end(); })
;

auto mlir_detail_OperandStorage = nb::class_<mlir::detail::OperandStorage>(m, "OperandStorage")
.def(nb::init<mlir::Operation *, mlir::OpOperand *, mlir::ValueRange>(), "owner"_a, "trailing_operands"_a, "values"_a)
.def("set_operands", [](mlir::detail::OperandStorage& self, mlir::Operation * owner, mlir::ValueRange values){ return self.setOperands(owner, values); }, "owner"_a, "values"_a)
.def("set_operands", [](mlir::detail::OperandStorage& self, mlir::Operation * owner, unsigned int start, unsigned int length, mlir::ValueRange operands){ return self.setOperands(owner, start, length, operands); }, "owner"_a, "start"_a, "length"_a, "operands"_a)
.def("erase_operands", [](mlir::detail::OperandStorage& self, unsigned int start, unsigned int length){ return self.eraseOperands(start, length); }, "start"_a, "length"_a)
.def("erase_operands", [](mlir::detail::OperandStorage& self, const llvm::BitVector & eraseIndices){ return self.eraseOperands(eraseIndices); }, "erase_indices"_a)
.def_prop_ro("operands", &mlir::detail::OperandStorage::getOperands)
.def("size", &mlir::detail::OperandStorage::size)
;

auto mlir_OpPrintingFlags = nb::class_<mlir::OpPrintingFlags>(m, "OpPrintingFlags")
.def(nb::init<>())
.def("elide_large_elements_attrs", &mlir::OpPrintingFlags::elideLargeElementsAttrs, "large_element_limit"_a, nb::rv_policy::reference_internal)
.def("print_large_elements_attr_with_hex", &mlir::OpPrintingFlags::printLargeElementsAttrWithHex, "large_element_limit"_a, nb::rv_policy::reference_internal)
.def("elide_large_resource_string", &mlir::OpPrintingFlags::elideLargeResourceString, "large_resource_limit"_a, nb::rv_policy::reference_internal)
.def("enable_debug_info", &mlir::OpPrintingFlags::enableDebugInfo, "enable"_a, "pretty_form"_a, nb::rv_policy::reference_internal)
.def("print_generic_op_form", &mlir::OpPrintingFlags::printGenericOpForm, "enable"_a, nb::rv_policy::reference_internal)
.def("skip_regions", &mlir::OpPrintingFlags::skipRegions, "skip"_a, nb::rv_policy::reference_internal)
.def("assume_verified", &mlir::OpPrintingFlags::assumeVerified, nb::rv_policy::reference_internal)
.def("use_local_scope", &mlir::OpPrintingFlags::useLocalScope, nb::rv_policy::reference_internal)
.def("print_value_users", &mlir::OpPrintingFlags::printValueUsers, nb::rv_policy::reference_internal)
.def("should_elide_elements_attr", &mlir::OpPrintingFlags::shouldElideElementsAttr, "attr"_a)
.def("should_print_elements_attr_with_hex", &mlir::OpPrintingFlags::shouldPrintElementsAttrWithHex, "attr"_a)
.def_prop_ro("large_elements_attr_limit", &mlir::OpPrintingFlags::getLargeElementsAttrLimit)
.def_prop_ro("large_elements_attr_hex_limit", &mlir::OpPrintingFlags::getLargeElementsAttrHexLimit)
.def_prop_ro("large_resource_string_limit", &mlir::OpPrintingFlags::getLargeResourceStringLimit)
.def("should_print_debug_info", &mlir::OpPrintingFlags::shouldPrintDebugInfo)
.def("should_print_debug_info_pretty_form", &mlir::OpPrintingFlags::shouldPrintDebugInfoPrettyForm)
.def("should_print_generic_op_form", &mlir::OpPrintingFlags::shouldPrintGenericOpForm)
.def("should_skip_regions", &mlir::OpPrintingFlags::shouldSkipRegions)
.def("should_assume_verified", &mlir::OpPrintingFlags::shouldAssumeVerified)
.def("should_use_local_scope", &mlir::OpPrintingFlags::shouldUseLocalScope)
.def("should_print_value_users", &mlir::OpPrintingFlags::shouldPrintValueUsers)
.def("should_print_unique_ssai_ds", &mlir::OpPrintingFlags::shouldPrintUniqueSSAIDs)
;

auto mlir_OperationEquivalence = nb::class_<mlir::OperationEquivalence>(m, "OperationEquivalence")
.def_static("compute_hash", &mlir::OperationEquivalence::computeHash, "op"_a, "hash_operands"_a, "hash_results"_a, "flags"_a)
.def_static("ignore_hash_value", &mlir::OperationEquivalence::ignoreHashValue, "_"_a)
.def_static("direct_hash_value", &mlir::OperationEquivalence::directHashValue, "v"_a)
.def_static("is_equivalent_to", [](mlir::Operation * lhs, mlir::Operation * rhs, llvm::function_ref<llvm::LogicalResult (mlir::Value, mlir::Value)> checkEquivalent, llvm::function_ref<void (mlir::Value, mlir::Value)> markEquivalent, mlir::OperationEquivalence::Flags flags, llvm::function_ref<llvm::LogicalResult (mlir::ValueRange, mlir::ValueRange)> checkCommutativeEquivalent){ return mlir::OperationEquivalence::isEquivalentTo(lhs, rhs, checkEquivalent, markEquivalent, flags, checkCommutativeEquivalent); }, "lhs"_a, "rhs"_a, "check_equivalent"_a, "mark_equivalent"_a, "flags"_a, "check_commutative_equivalent"_a)
.def_static("is_equivalent_to", [](mlir::Operation * lhs, mlir::Operation * rhs, mlir::OperationEquivalence::Flags flags){ return mlir::OperationEquivalence::isEquivalentTo(lhs, rhs, flags); }, "lhs"_a, "rhs"_a, "flags"_a)
.def_static("is_region_equivalent_to", [](mlir::Region * lhs, mlir::Region * rhs, llvm::function_ref<llvm::LogicalResult (mlir::Value, mlir::Value)> checkEquivalent, llvm::function_ref<void (mlir::Value, mlir::Value)> markEquivalent, mlir::OperationEquivalence::Flags flags, llvm::function_ref<llvm::LogicalResult (mlir::ValueRange, mlir::ValueRange)> checkCommutativeEquivalent){ return mlir::OperationEquivalence::isRegionEquivalentTo(lhs, rhs, checkEquivalent, markEquivalent, flags, checkCommutativeEquivalent); }, "lhs"_a, "rhs"_a, "check_equivalent"_a, "mark_equivalent"_a, "flags"_a, "check_commutative_equivalent"_a)
.def_static("is_region_equivalent_to", [](mlir::Region * lhs, mlir::Region * rhs, mlir::OperationEquivalence::Flags flags){ return mlir::OperationEquivalence::isRegionEquivalentTo(lhs, rhs, flags); }, "lhs"_a, "rhs"_a, "flags"_a)
.def_static("ignore_value_equivalence", &mlir::OperationEquivalence::ignoreValueEquivalence, "lhs"_a, "rhs"_a)
.def_static("exact_value_match", &mlir::OperationEquivalence::exactValueMatch, "lhs"_a, "rhs"_a)
;

nb::enum_<mlir::OperationEquivalence::Flags>(m, "Flags")
.value("None", mlir::OperationEquivalence::Flags::None)
.value("IgnoreLocations", mlir::OperationEquivalence::Flags::IgnoreLocations)
.value("LLVM_BITMASK_LARGEST_ENUMERATOR", mlir::OperationEquivalence::Flags::LLVM_BITMASK_LARGEST_ENUMERATOR)
;

auto mlir_OperationFingerPrint = nb::class_<mlir::OperationFingerPrint>(m, "OperationFingerPrint")
.def(nb::init<mlir::Operation *, bool>(), "top_op"_a, "include_nested"_a)
.def(nb::init<const mlir::OperationFingerPrint &>(), "_"_a)
.def("__eq__", &mlir::OperationFingerPrint::operator==, "other"_a)
.def("__ne__", &mlir::OperationFingerPrint::operator!=, "other"_a)
;

auto mlir_IRUnit = nb::class_<mlir::IRUnit>(m, "IRUnit")
.def("print", &mlir::IRUnit::print, "os"_a, "flags"_a)
.def("__str__", [](mlir::IRUnit& self){
  mlir::OpPrintingFlags flags;
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os, flags);
  return s;
});

auto mlir_tracing_Action = nb::class_<mlir::tracing::Action>(m, "Action")
.def_prop_ro("action_id", &mlir::tracing::Action::getActionID)
.def_prop_ro("tag", &mlir::tracing::Action::getTag)
.def("print", &mlir::tracing::Action::print, "os"_a)
.def_prop_ro("context_ir_units", &mlir::tracing::Action::getContextIRUnits)
;

auto mlir_SimpleAffineExprFlattener = nb::class_<mlir::SimpleAffineExprFlattener>(m, "SimpleAffineExprFlattener")
.def(nb::init<unsigned int, unsigned int>(), "num_dims"_a, "num_symbols"_a)
.def("visit_mul_expr", &mlir::SimpleAffineExprFlattener::visitMulExpr, "expr"_a)
.def("visit_add_expr", &mlir::SimpleAffineExprFlattener::visitAddExpr, "expr"_a)
.def("visit_dim_expr", &mlir::SimpleAffineExprFlattener::visitDimExpr, "expr"_a)
.def("visit_symbol_expr", &mlir::SimpleAffineExprFlattener::visitSymbolExpr, "expr"_a)
.def("visit_constant_expr", &mlir::SimpleAffineExprFlattener::visitConstantExpr, "expr"_a)
.def("visit_ceil_div_expr", &mlir::SimpleAffineExprFlattener::visitCeilDivExpr, "expr"_a)
.def("visit_floor_div_expr", &mlir::SimpleAffineExprFlattener::visitFloorDivExpr, "expr"_a)
.def("visit_mod_expr", &mlir::SimpleAffineExprFlattener::visitModExpr, "expr"_a)
;

auto mlir_AsmResourceBlob = nb::class_<mlir::AsmResourceBlob>(m, "AsmResourceBlob")
.def(nb::init<>())
.def(nb::init<mlir::AsmResourceBlob &&>(), "_"_a)
.def_prop_ro("data_alignment", &mlir::AsmResourceBlob::getDataAlignment)
.def_prop_ro("data", &mlir::AsmResourceBlob::getData)
.def_prop_ro("mutable_data", &mlir::AsmResourceBlob::getMutableData)
.def("is_mutable", &mlir::AsmResourceBlob::isMutable)
.def_prop_ro("deleter", [](mlir::AsmResourceBlob& self){ return &self.getDeleter(); })
.def_prop_ro("deleter", [](mlir::AsmResourceBlob& self){ return &self.getDeleter(); })
;

auto mlir_HeapAsmResourceBlob = nb::class_<mlir::HeapAsmResourceBlob>(m, "HeapAsmResourceBlob")
.def_static("allocate", &mlir::HeapAsmResourceBlob::allocate, "size"_a, "align"_a, "data_is_mutable"_a)
.def_static("allocate_and_copy_with_align", &mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign, "data"_a, "align"_a, "data_is_mutable"_a)
;

auto mlir_UnmanagedAsmResourceBlob = nb::class_<mlir::UnmanagedAsmResourceBlob>(m, "UnmanagedAsmResourceBlob")
;

auto mlir_AsmResourceBuilder = nb::class_<mlir::AsmResourceBuilder>(m, "AsmResourceBuilder")
.def("build_bool", &mlir::AsmResourceBuilder::buildBool, "key"_a, "data"_a)
.def("build_string", &mlir::AsmResourceBuilder::buildString, "key"_a, "data"_a)
.def("build_blob", [](mlir::AsmResourceBuilder& self, llvm::StringRef key, llvm::ArrayRef<char> data, uint32_t dataAlignment){ return self.buildBlob(key, data, dataAlignment); }, "key"_a, "data"_a, "data_alignment"_a)
.def("build_blob", [](mlir::AsmResourceBuilder& self, llvm::StringRef key, const mlir::AsmResourceBlob & blob){ return self.buildBlob(key, blob); }, "key"_a, "blob"_a)
;

nb::enum_<mlir::AsmResourceEntryKind>(m, "AsmResourceEntryKind")
.value("Blob", mlir::AsmResourceEntryKind::Blob)
.value("Bool", mlir::AsmResourceEntryKind::Bool)
.value("String", mlir::AsmResourceEntryKind::String)
;

auto mlir_AsmParsedResourceEntry = nb::class_<mlir::AsmParsedResourceEntry>(m, "AsmParsedResourceEntry")
.def_prop_ro("key", &mlir::AsmParsedResourceEntry::getKey)
.def("emit_error", &mlir::AsmParsedResourceEntry::emitError)
.def_prop_ro("kind", &mlir::AsmParsedResourceEntry::getKind)
.def("parse_as_bool", &mlir::AsmParsedResourceEntry::parseAsBool)
.def("parse_as_string", &mlir::AsmParsedResourceEntry::parseAsString)
.def("parse_as_blob", [](mlir::AsmParsedResourceEntry& self, mlir::AsmParsedResourceEntry::BlobAllocatorFn allocator){ return self.parseAsBlob(allocator); }, "allocator"_a)
.def("parse_as_blob", [](mlir::AsmParsedResourceEntry& self){ return self.parseAsBlob(); })
;

auto mlir_AsmState = nb::class_<mlir::AsmState>(m, "AsmState")
.def(nb::init<mlir::Operation *, const mlir::OpPrintingFlags &, llvm::DenseMap<mlir::Operation *, std::pair<unsigned int, unsigned int>> *, mlir::FallbackAsmResourceMap *>(), "op"_a, "printer_flags"_a, "location_map"_a, "map"_a)
.def(nb::init<mlir::MLIRContext *, const mlir::OpPrintingFlags &, llvm::DenseMap<mlir::Operation *, std::pair<unsigned int, unsigned int>> *, mlir::FallbackAsmResourceMap *>(), "ctx"_a, "printer_flags"_a, "location_map"_a, "map"_a)
.def_prop_ro("printer_flags", &mlir::AsmState::getPrinterFlags)
.def("attach_resource_printer", [](mlir::AsmState& self, std::unique_ptr<mlir::AsmResourcePrinter> printer){ return self.attachResourcePrinter(std::move(printer)); }, "printer"_a)
.def("attach_fallback_resource_printer", &mlir::AsmState::attachFallbackResourcePrinter, "map"_a)
.def_prop_ro("dialect_resources", &mlir::AsmState::getDialectResources)
;

auto mlir_Block = nb::class_<mlir::Block>(m, "Block")
.def(nb::init<>())
.def("clear", &mlir::Block::clear)
.def_prop_ro("parent", &mlir::Block::getParent)
.def_prop_ro("parent_op", &mlir::Block::getParentOp)
.def("is_entry_block", &mlir::Block::isEntryBlock)
.def("insert_before", &mlir::Block::insertBefore, "block"_a)
.def("insert_after", &mlir::Block::insertAfter, "block"_a)
.def("move_before", [](mlir::Block& self, mlir::Block * block){ return self.moveBefore(block); }, "block"_a)
.def("move_before", [](mlir::Block& self, mlir::Region * region, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Block, true, false, void, false, void>, false, false> iterator){ return self.moveBefore(region, iterator); }, "region"_a, "iterator"_a)
.def("erase", &mlir::Block::erase)
.def_prop_ro("arguments", &mlir::Block::getArguments)
.def_prop_ro("argument_types", &mlir::Block::getArgumentTypes)
.def("args_begin", &mlir::Block::args_begin)
.def("args_end", &mlir::Block::args_end)
.def("args_rbegin", &mlir::Block::args_rbegin)
.def("args_rend", &mlir::Block::args_rend)
.def("args_empty", &mlir::Block::args_empty)
.def("add_argument", &mlir::Block::addArgument, "type"_a, "loc"_a)
.def("insert_argument", [](mlir::Block& self, mlir::Block::args_iterator it, mlir::Type type, mlir::Location loc){ return self.insertArgument(it, type, loc); }, "it"_a, "type"_a, "loc"_a)
.def("add_arguments", &mlir::Block::addArguments, "types"_a, "locs"_a)
.def("insert_argument", [](mlir::Block& self, unsigned int index, mlir::Type type, mlir::Location loc){ return self.insertArgument(index, type, loc); }, "index"_a, "type"_a, "loc"_a)
.def("erase_argument", &mlir::Block::eraseArgument, "index"_a)
.def("erase_arguments", [](mlir::Block& self, unsigned int start, unsigned int num){ return self.eraseArguments(start, num); }, "start"_a, "num"_a)
.def("erase_arguments", [](mlir::Block& self, const llvm::BitVector & eraseIndices){ return self.eraseArguments(eraseIndices); }, "erase_indices"_a)
.def("erase_arguments", [](mlir::Block& self, llvm::function_ref<bool (mlir::BlockArgument)> shouldEraseFn){ return self.eraseArguments(shouldEraseFn); }, "should_erase_fn"_a)
.def_prop_ro("num_arguments", &mlir::Block::getNumArguments)
.def("get_argument", &mlir::Block::getArgument, "i"_a)
.def_prop_ro("operations", &mlir::Block::getOperations)
.def("begin", &mlir::Block::begin)
.def("end", &mlir::Block::end)
.def("rbegin", &mlir::Block::rbegin)
.def("rend", &mlir::Block::rend)
.def("empty", &mlir::Block::empty)
.def("push_back", &mlir::Block::push_back, "op"_a)
.def("push_front", &mlir::Block::push_front, "op"_a)
.def("back", &mlir::Block::back, nb::rv_policy::reference_internal)
.def("front", &mlir::Block::front, nb::rv_policy::reference_internal)
.def("find_ancestor_op_in_block", &mlir::Block::findAncestorOpInBlock, "op"_a, nb::rv_policy::reference_internal)
.def("drop_all_references", &mlir::Block::dropAllReferences)
.def("drop_all_defined_value_uses", &mlir::Block::dropAllDefinedValueUses)
.def("is_op_order_valid", &mlir::Block::isOpOrderValid)
.def("invalidate_op_order", &mlir::Block::invalidateOpOrder)
.def("verify_op_order", &mlir::Block::verifyOpOrder)
.def("recompute_op_order", &mlir::Block::recomputeOpOrder)
.def("without_terminator", &mlir::Block::without_terminator)
.def_prop_ro("terminator", &mlir::Block::getTerminator)
.def("might_have_terminator", &mlir::Block::mightHaveTerminator)
.def("pred_begin", &mlir::Block::pred_begin)
.def("pred_end", &mlir::Block::pred_end)
.def_prop_ro("predecessors", &mlir::Block::getPredecessors)
.def("has_no_predecessors", &mlir::Block::hasNoPredecessors)
.def("has_no_successors", &mlir::Block::hasNoSuccessors)
.def_prop_ro("single_predecessor", &mlir::Block::getSinglePredecessor)
.def_prop_ro("unique_predecessor", &mlir::Block::getUniquePredecessor)
.def_prop_ro("num_successors", &mlir::Block::getNumSuccessors)
.def("get_successor", &mlir::Block::getSuccessor, "i"_a, nb::rv_policy::reference_internal)
.def("succ_begin", &mlir::Block::succ_begin)
.def("succ_end", &mlir::Block::succ_end)
.def_prop_ro("successors", &mlir::Block::getSuccessors)
.def("is_reachable", &mlir::Block::isReachable, "other"_a, "except_"_a)
.def("split_block", [](mlir::Block& self, mlir::Block::iterator splitBefore){ return self.splitBlock(splitBefore); }, "split_before"_a, nb::rv_policy::reference_internal)
.def("split_block", [](mlir::Block& self, mlir::Operation * splitBeforeOp){ return self.splitBlock(splitBeforeOp); }, "split_before_op"_a, nb::rv_policy::reference_internal)
.def_static("get_sublist_access", &mlir::Block::getSublistAccess, "_"_a, nb::rv_policy::reference_internal)
.def("print", [](mlir::Block& self, llvm::raw_ostream & os){ return self.print(os); }, "os"_a)
.def("print", [](mlir::Block& self, llvm::raw_ostream & os, mlir::AsmState & state){ return self.print(os, state); }, "os"_a, "state"_a)
.def("dump", &mlir::Block::dump)
.def("print_as_operand", [](mlir::Block& self, llvm::raw_ostream & os, bool printType){ return self.printAsOperand(os, printType); }, "os"_a, "print_type"_a)
.def("print_as_operand", [](mlir::Block& self, llvm::raw_ostream & os, mlir::AsmState & state){ return self.printAsOperand(os, state); }, "os"_a, "state"_a)
.def("__str__", [](mlir::Block& self){
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os);
  return s;
});

auto mlir_Dialect = nb::class_<mlir::Dialect>(m, "Dialect")
.def_static("is_valid_namespace", &mlir::Dialect::isValidNamespace, "str"_a)
.def_prop_ro("context", &mlir::Dialect::getContext)
.def_prop_ro("namespace", &mlir::Dialect::getNamespace)
.def_prop_ro("type_id", &mlir::Dialect::getTypeID)
.def("allows_unknown_operations", &mlir::Dialect::allowsUnknownOperations)
.def("allows_unknown_types", &mlir::Dialect::allowsUnknownTypes)
.def("get_canonicalization_patterns", &mlir::Dialect::getCanonicalizationPatterns, "results"_a)
.def("materialize_constant", &mlir::Dialect::materializeConstant, "builder"_a, "value"_a, "type"_a, "loc"_a, nb::rv_policy::reference_internal)
.def("parse_attribute", &mlir::Dialect::parseAttribute, "parser"_a, "type"_a)
.def("print_attribute", &mlir::Dialect::printAttribute, "_"_a, "__"_a)
.def("parse_type", &mlir::Dialect::parseType, "parser"_a)
.def("print_type", &mlir::Dialect::printType, "_"_a, "__"_a)
.def("get_parse_operation_hook", &mlir::Dialect::getParseOperationHook, "op_name"_a)
.def("get_operation_printer", &mlir::Dialect::getOperationPrinter, "op"_a)
.def("verify_region_arg_attribute", &mlir::Dialect::verifyRegionArgAttribute, "_"_a, "region_index"_a, "arg_index"_a, "____"_a)
.def("verify_region_result_attribute", &mlir::Dialect::verifyRegionResultAttribute, "_"_a, "region_index"_a, "result_index"_a, "____"_a)
.def("verify_operation_attribute", &mlir::Dialect::verifyOperationAttribute, "_"_a, "__"_a)
.def("get_registered_interface", [](mlir::Dialect& self, mlir::TypeID interfaceID){ return self.getRegisteredInterface(interfaceID); }, "interface_id"_a, nb::rv_policy::reference_internal)
.def("get_registered_interface_for_op", [](mlir::Dialect& self, mlir::TypeID interfaceID, mlir::OperationName opName){ return self.getRegisteredInterfaceForOp(interfaceID, opName); }, "interface_id"_a, "op_name"_a, nb::rv_policy::reference_internal)
.def("add_interface", [](mlir::Dialect& self, std::unique_ptr<mlir::DialectInterface> interface){ return self.addInterface(std::move(interface)); }, "interface"_a)
.def("handle_use_of_undefined_promised_interface", &mlir::Dialect::handleUseOfUndefinedPromisedInterface, "interface_requestor_id"_a, "interface_id"_a, "interface_name"_a)
.def("handle_addition_of_undefined_promised_interface", &mlir::Dialect::handleAdditionOfUndefinedPromisedInterface, "interface_requestor_id"_a, "interface_id"_a)
.def("has_promised_interface", [](mlir::Dialect& self, mlir::TypeID interfaceRequestorID, mlir::TypeID interfaceID){ return self.hasPromisedInterface(interfaceRequestorID, interfaceID); }, "interface_requestor_id"_a, "interface_id"_a)
;

auto mlir_Region = nb::class_<mlir::Region>(m, "Region")
.def(nb::init<>())
.def(nb::init<mlir::Operation *>(), "container"_a)
.def_prop_ro("context", &mlir::Region::getContext)
.def_prop_ro("loc", &mlir::Region::getLoc)
.def_prop_ro("blocks", &mlir::Region::getBlocks)
.def("emplace_block", &mlir::Region::emplaceBlock, nb::rv_policy::reference_internal)
.def("begin", &mlir::Region::begin)
.def("end", &mlir::Region::end)
.def("rbegin", &mlir::Region::rbegin)
.def("rend", &mlir::Region::rend)
.def("empty", &mlir::Region::empty)
.def("push_back", &mlir::Region::push_back, "block"_a)
.def("push_front", &mlir::Region::push_front, "block"_a)
.def("back", &mlir::Region::back, nb::rv_policy::reference_internal)
.def("front", &mlir::Region::front, nb::rv_policy::reference_internal)
.def("has_one_block", &mlir::Region::hasOneBlock)
.def_static("get_sublist_access", &mlir::Region::getSublistAccess, "_"_a, nb::rv_policy::reference_internal)
.def_prop_ro("arguments", &mlir::Region::getArguments)
.def_prop_ro("argument_types", &mlir::Region::getArgumentTypes)
.def("args_begin", &mlir::Region::args_begin)
.def("args_end", &mlir::Region::args_end)
.def("args_rbegin", &mlir::Region::args_rbegin)
.def("args_rend", &mlir::Region::args_rend)
.def("args_empty", &mlir::Region::args_empty)
.def("add_argument", &mlir::Region::addArgument, "type"_a, "loc"_a)
.def("insert_argument", [](mlir::Region& self, mlir::Region::args_iterator it, mlir::Type type, mlir::Location loc){ return self.insertArgument(it, type, loc); }, "it"_a, "type"_a, "loc"_a)
.def("add_arguments", &mlir::Region::addArguments, "types"_a, "locs"_a)
.def("insert_argument", [](mlir::Region& self, unsigned int index, mlir::Type type, mlir::Location loc){ return self.insertArgument(index, type, loc); }, "index"_a, "type"_a, "loc"_a)
.def("erase_argument", &mlir::Region::eraseArgument, "index"_a)
.def_prop_ro("num_arguments", &mlir::Region::getNumArguments)
.def("get_argument", &mlir::Region::getArgument, "i"_a)
.def("op_begin", [](mlir::Region& self){ return self.op_begin(); })
.def("op_end", [](mlir::Region& self){ return self.op_end(); })
.def_prop_ro("ops", [](mlir::Region& self){ return self.getOps(); })
.def_prop_ro("parent_region", &mlir::Region::getParentRegion)
.def_prop_ro("parent_op", &mlir::Region::getParentOp)
.def_prop_ro("region_number", &mlir::Region::getRegionNumber)
.def("is_proper_ancestor", &mlir::Region::isProperAncestor, "other"_a)
.def("is_ancestor", &mlir::Region::isAncestor, "other"_a)
.def("clone_into", [](mlir::Region& self, mlir::Region * dest, mlir::IRMapping & mapper){ return self.cloneInto(dest, mapper); }, "dest"_a, "mapper"_a)
.def("clone_into", [](mlir::Region& self, mlir::Region * dest, mlir::Region::iterator destPos, mlir::IRMapping & mapper){ return self.cloneInto(dest, destPos, mapper); }, "dest"_a, "dest_pos"_a, "mapper"_a)
.def("take_body", &mlir::Region::takeBody, "other"_a)
.def("find_ancestor_block_in_region", &mlir::Region::findAncestorBlockInRegion, "block"_a, nb::rv_policy::reference_internal)
.def("find_ancestor_op_in_region", &mlir::Region::findAncestorOpInRegion, "op"_a, nb::rv_policy::reference_internal)
.def("drop_all_references", &mlir::Region::dropAllReferences)
.def("view_graph", [](mlir::Region& self, const llvm::Twine & regionName){ return self.viewGraph(regionName); }, "region_name"_a)
.def("view_graph", [](mlir::Region& self){ return self.viewGraph(); })
;

auto mlir_Region_OpIterator = nb::class_<mlir::Region::OpIterator>(mlir_Region, "OpIterator")
.def(nb::init<mlir::Region *, bool>(), "region"_a, "end"_a)
.def("__eq__", &mlir::Region::OpIterator::operator==, "rhs"_a)
.def("__ne__", &mlir::Region::OpIterator::operator!=, "rhs"_a)
;

auto mlir_RegionRange = nb::class_<mlir::RegionRange>(m, "RegionRange")
.def(nb::init<llvm::MutableArrayRef<mlir::Region>>(), "regions"_a)
.def(nb::init<llvm::ArrayRef<std::unique_ptr<mlir::Region>>>(), "regions"_a)
.def(nb::init<llvm::ArrayRef<mlir::Region *>>(), "regions"_a)
;

nb::enum_<mlir::detail::OpProperties>(m, "OpProperties")
;

auto mlir_Operation = nb::class_<mlir::Operation>(m, "Operation")
.def_static("create", [](mlir::Location location, mlir::OperationName name, mlir::TypeRange resultTypes, mlir::ValueRange operands, mlir::NamedAttrList && attributes, mlir::OpaqueProperties properties, mlir::BlockRange successors, unsigned int numRegions){ return mlir::Operation::create(location, name, resultTypes, operands, std::move(attributes), properties, successors, numRegions); }, "location"_a, "name"_a, "result_types"_a, "operands"_a, "attributes"_a, "properties"_a, "successors"_a, "num_regions"_a, nb::rv_policy::reference_internal)
.def_static("create", [](mlir::Location location, mlir::OperationName name, mlir::TypeRange resultTypes, mlir::ValueRange operands, mlir::DictionaryAttr attributes, mlir::OpaqueProperties properties, mlir::BlockRange successors, unsigned int numRegions){ return mlir::Operation::create(location, name, resultTypes, operands, attributes, properties, successors, numRegions); }, "location"_a, "name"_a, "result_types"_a, "operands"_a, "attributes"_a, "properties"_a, "successors"_a, "num_regions"_a, nb::rv_policy::reference_internal)
.def_static("create", [](const mlir::OperationState & state){ return mlir::Operation::create(state); }, "state"_a, nb::rv_policy::reference_internal)
.def_static("create", [](mlir::Location location, mlir::OperationName name, mlir::TypeRange resultTypes, mlir::ValueRange operands, mlir::NamedAttrList && attributes, mlir::OpaqueProperties properties, mlir::BlockRange successors, mlir::RegionRange regions){ return mlir::Operation::create(location, name, resultTypes, operands, std::move(attributes), properties, successors, regions); }, "location"_a, "name"_a, "result_types"_a, "operands"_a, "attributes"_a, "properties"_a, "successors"_a, "regions"_a, nb::rv_policy::reference_internal)
.def_prop_ro("name", &mlir::Operation::getName)
.def_prop_ro("registered_info", &mlir::Operation::getRegisteredInfo)
.def("is_registered", &mlir::Operation::isRegistered)
.def("erase", &mlir::Operation::erase)
.def("remove", &mlir::Operation::remove)
.def("clone", [](mlir::Operation& self, mlir::IRMapping & mapper, mlir::Operation::CloneOptions options){ return self.clone(mapper, options); }, "mapper"_a, "options"_a, nb::rv_policy::reference_internal)
.def("clone", [](mlir::Operation& self, mlir::Operation::CloneOptions options){ return self.clone(options); }, "options"_a, nb::rv_policy::reference_internal)
.def("clone_without_regions", [](mlir::Operation& self, mlir::IRMapping & mapper){ return self.cloneWithoutRegions(mapper); }, "mapper"_a, nb::rv_policy::reference_internal)
.def("clone_without_regions", [](mlir::Operation& self){ return self.cloneWithoutRegions(); }, nb::rv_policy::reference_internal)
.def_prop_ro("block", &mlir::Operation::getBlock)
.def_prop_ro("context", &mlir::Operation::getContext)
.def_prop_ro("dialect", &mlir::Operation::getDialect)
.def_prop_ro("loc", &mlir::Operation::getLoc)
.def("set_loc", &mlir::Operation::setLoc, "loc"_a)
.def_prop_ro("parent_region", &mlir::Operation::getParentRegion)
.def_prop_ro("parent_op", &mlir::Operation::getParentOp)
.def("is_proper_ancestor", &mlir::Operation::isProperAncestor, "other"_a)
.def("is_ancestor", &mlir::Operation::isAncestor, "other"_a)
.def("replace_uses_of_with", &mlir::Operation::replaceUsesOfWith, "from_"_a, "to"_a)
.def("destroy", &mlir::Operation::destroy)
.def("drop_all_references", &mlir::Operation::dropAllReferences)
.def("drop_all_defined_value_uses", &mlir::Operation::dropAllDefinedValueUses)
.def("move_before", [](mlir::Operation& self, mlir::Operation * existingOp){ return self.moveBefore(existingOp); }, "existing_op"_a)
.def("move_before", [](mlir::Operation& self, mlir::Block * block, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false> iterator){ return self.moveBefore(block, iterator); }, "block"_a, "iterator"_a)
.def("move_after", [](mlir::Operation& self, mlir::Operation * existingOp){ return self.moveAfter(existingOp); }, "existing_op"_a)
.def("move_after", [](mlir::Operation& self, mlir::Block * block, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false> iterator){ return self.moveAfter(block, iterator); }, "block"_a, "iterator"_a)
.def("is_before_in_block", &mlir::Operation::isBeforeInBlock, "other"_a)
.def("print", [](mlir::Operation& self, llvm::raw_ostream & os, const mlir::OpPrintingFlags & flags){ return self.print(os, flags); }, "os"_a, "flags"_a)
.def("print", [](mlir::Operation& self, llvm::raw_ostream & os, mlir::AsmState & state){ return self.print(os, state); }, "os"_a, "state"_a)
.def("dump", &mlir::Operation::dump)
.def("set_operands", [](mlir::Operation& self, mlir::ValueRange operands){ return self.setOperands(operands); }, "operands"_a)
.def("set_operands", [](mlir::Operation& self, unsigned int start, unsigned int length, mlir::ValueRange operands){ return self.setOperands(start, length, operands); }, "start"_a, "length"_a, "operands"_a)
.def("insert_operands", &mlir::Operation::insertOperands, "index"_a, "operands"_a)
.def_prop_ro("num_operands", &mlir::Operation::getNumOperands)
.def("get_operand", &mlir::Operation::getOperand, "idx"_a)
.def("set_operand", &mlir::Operation::setOperand, "idx"_a, "value"_a)
.def("erase_operand", &mlir::Operation::eraseOperand, "idx"_a)
.def("erase_operands", [](mlir::Operation& self, unsigned int idx, unsigned int length){ return self.eraseOperands(idx, length); }, "idx"_a, "length"_a)
.def("erase_operands", [](mlir::Operation& self, const llvm::BitVector & eraseIndices){ return self.eraseOperands(eraseIndices); }, "erase_indices"_a)
.def("operand_begin", &mlir::Operation::operand_begin)
.def("operand_end", &mlir::Operation::operand_end)
.def_prop_ro("operands", &mlir::Operation::getOperands)
.def_prop_ro("op_operands", &mlir::Operation::getOpOperands)
.def("get_op_operand", &mlir::Operation::getOpOperand, "idx"_a, nb::rv_policy::reference_internal)
.def("operand_type_begin", &mlir::Operation::operand_type_begin)
.def("operand_type_end", &mlir::Operation::operand_type_end)
.def_prop_ro("operand_types", &mlir::Operation::getOperandTypes)
.def_prop_ro("num_results", &mlir::Operation::getNumResults)
.def("get_result", &mlir::Operation::getResult, "idx"_a)
.def("result_begin", &mlir::Operation::result_begin)
.def("result_end", &mlir::Operation::result_end)
.def_prop_ro("results", &mlir::Operation::getResults)
.def_prop_ro("op_results", &mlir::Operation::getOpResults)
.def("get_op_result", &mlir::Operation::getOpResult, "idx"_a)
.def("result_type_begin", &mlir::Operation::result_type_begin)
.def("result_type_end", &mlir::Operation::result_type_end)
.def_prop_ro("result_types", &mlir::Operation::getResultTypes)
.def("get_inherent_attr", &mlir::Operation::getInherentAttr, "name"_a)
.def("set_inherent_attr", &mlir::Operation::setInherentAttr, "name"_a, "value"_a)
.def("get_discardable_attr", [](mlir::Operation& self, llvm::StringRef name){ return self.getDiscardableAttr(name); }, "name"_a)
.def("get_discardable_attr", [](mlir::Operation& self, mlir::StringAttr name){ return self.getDiscardableAttr(name); }, "name"_a)
.def("set_discardable_attr", [](mlir::Operation& self, mlir::StringAttr name, mlir::Attribute value){ return self.setDiscardableAttr(name, value); }, "name"_a, "value"_a)
.def("set_discardable_attr", [](mlir::Operation& self, llvm::StringRef name, mlir::Attribute value){ return self.setDiscardableAttr(name, value); }, "name"_a, "value"_a)
.def("remove_discardable_attr", [](mlir::Operation& self, mlir::StringAttr name){ return self.removeDiscardableAttr(name); }, "name"_a)
.def("remove_discardable_attr", [](mlir::Operation& self, llvm::StringRef name){ return self.removeDiscardableAttr(name); }, "name"_a)
.def_prop_ro("discardable_attrs", &mlir::Operation::getDiscardableAttrs)
.def_prop_ro("discardable_attr_dictionary", &mlir::Operation::getDiscardableAttrDictionary)
.def_prop_ro("raw_dictionary_attrs", &mlir::Operation::getRawDictionaryAttrs)
.def_prop_ro("attrs", &mlir::Operation::getAttrs)
.def_prop_ro("attr_dictionary", &mlir::Operation::getAttrDictionary)
.def("set_attrs", [](mlir::Operation& self, mlir::DictionaryAttr newAttrs){ return self.setAttrs(newAttrs); }, "new_attrs"_a)
.def("set_attrs", [](mlir::Operation& self, llvm::ArrayRef<mlir::NamedAttribute> newAttrs){ return self.setAttrs(newAttrs); }, "new_attrs"_a)
.def("set_discardable_attrs", [](mlir::Operation& self, mlir::DictionaryAttr newAttrs){ return self.setDiscardableAttrs(newAttrs); }, "new_attrs"_a)
.def("set_discardable_attrs", [](mlir::Operation& self, llvm::ArrayRef<mlir::NamedAttribute> newAttrs){ return self.setDiscardableAttrs(newAttrs); }, "new_attrs"_a)
.def("get_attr", [](mlir::Operation& self, mlir::StringAttr name){ return self.getAttr(name); }, "name"_a)
.def("get_attr", [](mlir::Operation& self, llvm::StringRef name){ return self.getAttr(name); }, "name"_a)
.def("has_attr", [](mlir::Operation& self, mlir::StringAttr name){ return self.hasAttr(name); }, "name"_a)
.def("has_attr", [](mlir::Operation& self, llvm::StringRef name){ return self.hasAttr(name); }, "name"_a)
.def("set_attr", [](mlir::Operation& self, mlir::StringAttr name, mlir::Attribute value){ return self.setAttr(name, value); }, "name"_a, "value"_a)
.def("set_attr", [](mlir::Operation& self, llvm::StringRef name, mlir::Attribute value){ return self.setAttr(name, value); }, "name"_a, "value"_a)
.def("remove_attr", [](mlir::Operation& self, mlir::StringAttr name){ return self.removeAttr(name); }, "name"_a)
.def("remove_attr", [](mlir::Operation& self, llvm::StringRef name){ return self.removeAttr(name); }, "name"_a)
.def_prop_ro("dialect_attrs", &mlir::Operation::getDialectAttrs)
.def("dialect_attr_begin", &mlir::Operation::dialect_attr_begin)
.def("dialect_attr_end", &mlir::Operation::dialect_attr_end)
.def("populate_default_attrs", &mlir::Operation::populateDefaultAttrs)
.def_prop_ro("num_regions", &mlir::Operation::getNumRegions)
.def_prop_ro("regions", &mlir::Operation::getRegions)
.def("get_region", &mlir::Operation::getRegion, "index"_a, nb::rv_policy::reference_internal)
.def_prop_ro("block_operands", &mlir::Operation::getBlockOperands)
.def("successor_begin", &mlir::Operation::successor_begin)
.def("successor_end", &mlir::Operation::successor_end)
.def_prop_ro("successors", &mlir::Operation::getSuccessors)
.def("has_successors", &mlir::Operation::hasSuccessors)
.def_prop_ro("num_successors", &mlir::Operation::getNumSuccessors)
.def("get_successor", &mlir::Operation::getSuccessor, "index"_a, nb::rv_policy::reference_internal)
.def("set_successor", &mlir::Operation::setSuccessor, "block"_a, "index"_a)
.def("fold", [](mlir::Operation& self, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::OpFoldResult> & results){ return self.fold(operands, results); }, "operands"_a, "results"_a)
.def("fold", [](mlir::Operation& self, llvm::SmallVectorImpl<mlir::OpFoldResult> & results){ return self.fold(results); }, "results"_a)
.def("drop_all_uses", &mlir::Operation::dropAllUses)
.def("use_begin", &mlir::Operation::use_begin)
.def("use_end", &mlir::Operation::use_end)
.def_prop_ro("uses", &mlir::Operation::getUses)
.def("has_one_use", &mlir::Operation::hasOneUse)
.def("use_empty", &mlir::Operation::use_empty)
.def("is_used_outside_of_block", &mlir::Operation::isUsedOutsideOfBlock, "block"_a)
.def("user_begin", &mlir::Operation::user_begin)
.def("user_end", &mlir::Operation::user_end)
.def_prop_ro("users", &mlir::Operation::getUsers)
.def("emit_op_error", &mlir::Operation::emitOpError, "message"_a)
.def("emit_error", &mlir::Operation::emitError, "message"_a)
.def("emit_warning", &mlir::Operation::emitWarning, "message"_a)
.def("emit_remark", &mlir::Operation::emitRemark, "message"_a)
.def_prop_ro("properties_storage_size", &mlir::Operation::getPropertiesStorageSize)
.def_prop_ro("properties_storage", [](mlir::Operation& self){ return self.getPropertiesStorage(); })
.def_prop_ro("properties_storage", [](mlir::Operation& self){ return self.getPropertiesStorage(); })
.def_prop_ro("properties_storage_unsafe", &mlir::Operation::getPropertiesStorageUnsafe)
.def_prop_ro("properties_as_attribute", &mlir::Operation::getPropertiesAsAttribute)
.def("set_properties_from_attribute", &mlir::Operation::setPropertiesFromAttribute, "attr"_a, "emit_error"_a)
.def("copy_properties", &mlir::Operation::copyProperties, "rhs"_a)
.def("hash_properties", &mlir::Operation::hashProperties)
.def("__str__", [](mlir::Operation& self){
  mlir::OpPrintingFlags flags;
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os, flags);
  return s;
});

auto mlir_Operation_CloneOptions = nb::class_<mlir::Operation::CloneOptions>(mlir_Operation, "CloneOptions")
.def(nb::init<>())
.def(nb::init<bool, bool>(), "clone_regions"_a, "clone_operands"_a)
.def_static("all", &mlir::Operation::CloneOptions::all)
.def("clone_regions", &mlir::Operation::CloneOptions::cloneRegions, "enable"_a, nb::rv_policy::reference_internal)
.def("should_clone_regions", &mlir::Operation::CloneOptions::shouldCloneRegions)
.def("clone_operands", &mlir::Operation::CloneOptions::cloneOperands, "enable"_a, nb::rv_policy::reference_internal)
.def("should_clone_operands", &mlir::Operation::CloneOptions::shouldCloneOperands)
;

auto mlir_Operation_dialect_attr_iterator = nb::class_<mlir::Operation::dialect_attr_iterator>(mlir_Operation, "dialect_attr_iterator")
;

auto mlir_OptionalParseResult = nb::class_<mlir::OptionalParseResult>(m, "OptionalParseResult")
.def(nb::init<>())
.def(nb::init<llvm::LogicalResult>(), "result"_a)
.def(nb::init<llvm::ParseResult>(), "result"_a)
.def(nb::init<const mlir::InFlightDiagnostic &>(), "_"_a)
.def("has_value", &mlir::OptionalParseResult::has_value)
.def("value", &mlir::OptionalParseResult::value)
;

auto mlir_EmptyProperties = nb::class_<mlir::EmptyProperties>(m, "EmptyProperties")
;

auto mlir_OpState = nb::class_<mlir::OpState>(m, "OpState")
.def_prop_ro("operation", &mlir::OpState::getOperation)
.def_prop_ro("context", &mlir::OpState::getContext)
.def("print", [](mlir::OpState& self, llvm::raw_ostream & os, mlir::OpPrintingFlags flags){ return self.print(os, flags); }, "os"_a, "flags"_a)
.def("print", [](mlir::OpState& self, llvm::raw_ostream & os, mlir::AsmState & asmState){ return self.print(os, asmState); }, "os"_a, "asm_state"_a)
.def("dump", &mlir::OpState::dump)
.def_prop_ro("loc", &mlir::OpState::getLoc)
.def("use_empty", &mlir::OpState::use_empty)
.def("erase", &mlir::OpState::erase)
.def("emit_op_error", &mlir::OpState::emitOpError, "message"_a)
.def("emit_error", &mlir::OpState::emitError, "message"_a)
.def("emit_warning", &mlir::OpState::emitWarning, "message"_a)
.def("emit_remark", &mlir::OpState::emitRemark, "message"_a)
.def_static("get_canonicalization_patterns", &mlir::OpState::getCanonicalizationPatterns, "results"_a, "context"_a)
.def_static("populate_default_attrs", &mlir::OpState::populateDefaultAttrs, "_"_a, "__"_a)
.def("__str__", [](mlir::OpState& self){
  mlir::OpPrintingFlags flags;
  std::string s;
  llvm::raw_string_ostream os(s);
  self.print(os, flags);
  return s;
});

auto mlir_OpFoldResult = nb::class_<mlir::OpFoldResult>(m, "OpFoldResult")
.def("dump", &mlir::OpFoldResult::dump)
.def_prop_ro("context", &mlir::OpFoldResult::getContext)
;

auto mlir_Builder = nb::class_<mlir::Builder>(m, "Builder")
.def(nb::init<mlir::MLIRContext *>(), "context"_a)
.def(nb::init<mlir::Operation *>(), "op"_a)
.def_prop_ro("context", &mlir::Builder::getContext)
.def_prop_ro("unknown_loc", &mlir::Builder::getUnknownLoc)
.def("get_fused_loc", &mlir::Builder::getFusedLoc, "locs"_a, "metadata"_a)
.def_prop_ro("float4_e2_m1_fn_type", &mlir::Builder::getType<Float4E2M1FNType>)
.def_prop_ro("float6e2m3fn_type", &mlir::Builder::getType<Float6E2M3FNType>)
.def_prop_ro("float6e3m2fn_type", &mlir::Builder::getType<Float6E3M2FNType>)
.def_prop_ro("float8e5m2_type", &mlir::Builder::getType<Float8E5M2Type>)
.def_prop_ro("float8e4m3_type", &mlir::Builder::getType<Float8E4M3Type>)
.def_prop_ro("float8e4m3fn_type", &mlir::Builder::getType<Float8E4M3FNType>)
.def_prop_ro("float8e5m2fnuz_type", &mlir::Builder::getType<Float8E5M2FNUZType>)
.def_prop_ro("float8e4m3fnuz_type", &mlir::Builder::getType<Float8E4M3FNUZType>)
.def_prop_ro("float8e4m3b11fnuz_type", &mlir::Builder::getType<Float8E4M3B11FNUZType>)
.def_prop_ro("float8e3m4_type", &mlir::Builder::getType<Float8E3M4Type>)
.def_prop_ro("float8e8m0fnu_type", &mlir::Builder::getType<Float8E8M0FNUType>)
.def_prop_ro("bf16_type", &mlir::Builder::getBF16Type)
.def_prop_ro("f16_type", &mlir::Builder::getF16Type)
.def_prop_ro("tf32_type", &mlir::Builder::getTF32Type)
.def_prop_ro("f32_type", &mlir::Builder::getF32Type)
.def_prop_ro("f64_type", &mlir::Builder::getF64Type)
.def_prop_ro("f80_type", &mlir::Builder::getF80Type)
.def_prop_ro("f128_type", &mlir::Builder::getF128Type)
.def_prop_ro("index_type", &mlir::Builder::getIndexType)
.def_prop_ro("i1_type", &mlir::Builder::getI1Type)
.def_prop_ro("i2_type", &mlir::Builder::getI2Type)
.def_prop_ro("i4_type", &mlir::Builder::getI4Type)
.def_prop_ro("i8_type", &mlir::Builder::getI8Type)
.def_prop_ro("i16_type", &mlir::Builder::getI16Type)
.def_prop_ro("i32_type", &mlir::Builder::getI32Type)
.def_prop_ro("i64_type", &mlir::Builder::getI64Type)
.def("get_integer_type", [](mlir::Builder& self, unsigned int width){ return self.getIntegerType(width); }, "width"_a)
.def("get_integer_type", [](mlir::Builder& self, unsigned int width, bool isSigned){ return self.getIntegerType(width, isSigned); }, "width"_a, "is_signed"_a)
.def("get_function_type", &mlir::Builder::getFunctionType, "inputs"_a, "results"_a)
.def("get_tuple_type", &mlir::Builder::getTupleType, "element_types"_a)
.def_prop_ro("none_type", &mlir::Builder::getNoneType)
.def("get_named_attr", &mlir::Builder::getNamedAttr, "name"_a, "val"_a)
.def_prop_ro("unit_attr", &mlir::Builder::getUnitAttr)
.def("get_bool_attr", &mlir::Builder::getBoolAttr, "value"_a)
.def("get_dictionary_attr", &mlir::Builder::getDictionaryAttr, "value"_a)
.def("get_integer_attr", [](mlir::Builder& self, mlir::Type type, int64_t value){ return self.getIntegerAttr(type, value); }, "type"_a, "value"_a)
.def("get_integer_attr", [](mlir::Builder& self, mlir::Type type, const llvm::APInt & value){ return self.getIntegerAttr(type, value); }, "type"_a, "value"_a)
.def("get_float_attr", [](mlir::Builder& self, mlir::Type type, double value){ return self.getFloatAttr(type, value); }, "type"_a, "value"_a)
.def("get_float_attr", [](mlir::Builder& self, mlir::Type type, const llvm::APFloat & value){ return self.getFloatAttr(type, value); }, "type"_a, "value"_a)
.def("get_string_attr", &mlir::Builder::getStringAttr, "bytes"_a)
.def("get_array_attr", &mlir::Builder::getArrayAttr, "value"_a)
.def("get_zero_attr", &mlir::Builder::getZeroAttr, "type"_a)
.def("get_one_attr", &mlir::Builder::getOneAttr, "type"_a)
.def("get_f16_float_attr", &mlir::Builder::getF16FloatAttr, "value"_a)
.def("get_f32_float_attr", &mlir::Builder::getF32FloatAttr, "value"_a)
.def("get_f64_float_attr", &mlir::Builder::getF64FloatAttr, "value"_a)
.def("get_i8_integer_attr", &mlir::Builder::getI8IntegerAttr, "value"_a)
.def("get_i16_integer_attr", &mlir::Builder::getI16IntegerAttr, "value"_a)
.def("get_i32_integer_attr", &mlir::Builder::getI32IntegerAttr, "value"_a)
.def("get_i64_integer_attr", &mlir::Builder::getI64IntegerAttr, "value"_a)
.def("get_index_attr", &mlir::Builder::getIndexAttr, "value"_a)
.def("get_si32_integer_attr", &mlir::Builder::getSI32IntegerAttr, "value"_a)
.def("get_ui32_integer_attr", &mlir::Builder::getUI32IntegerAttr, "value"_a)
.def("get_bool_vector_attr", &mlir::Builder::getBoolVectorAttr, "values"_a)
.def("get_i32_vector_attr", &mlir::Builder::getI32VectorAttr, "values"_a)
.def("get_i64_vector_attr", &mlir::Builder::getI64VectorAttr, "values"_a)
.def("get_index_vector_attr", &mlir::Builder::getIndexVectorAttr, "values"_a)
.def("get_f32_vector_attr", &mlir::Builder::getF32VectorAttr, "values"_a)
.def("get_f64_vector_attr", &mlir::Builder::getF64VectorAttr, "values"_a)
.def("get_i32_tensor_attr", &mlir::Builder::getI32TensorAttr, "values"_a)
.def("get_i64_tensor_attr", &mlir::Builder::getI64TensorAttr, "values"_a)
.def("get_index_tensor_attr", &mlir::Builder::getIndexTensorAttr, "values"_a)
.def("get_dense_bool_array_attr", &mlir::Builder::getDenseBoolArrayAttr, "values"_a)
.def("get_dense_i8_array_attr", &mlir::Builder::getDenseI8ArrayAttr, "values"_a)
.def("get_dense_i16_array_attr", &mlir::Builder::getDenseI16ArrayAttr, "values"_a)
.def("get_dense_i32_array_attr", &mlir::Builder::getDenseI32ArrayAttr, "values"_a)
.def("get_dense_i64_array_attr", &mlir::Builder::getDenseI64ArrayAttr, "values"_a)
.def("get_dense_f32_array_attr", &mlir::Builder::getDenseF32ArrayAttr, "values"_a)
.def("get_dense_f64_array_attr", &mlir::Builder::getDenseF64ArrayAttr, "values"_a)
.def("get_affine_map_array_attr", &mlir::Builder::getAffineMapArrayAttr, "values"_a)
.def("get_bool_array_attr", &mlir::Builder::getBoolArrayAttr, "values"_a)
.def("get_i32_array_attr", &mlir::Builder::getI32ArrayAttr, "values"_a)
.def("get_i64_array_attr", &mlir::Builder::getI64ArrayAttr, "values"_a)
.def("get_index_array_attr", &mlir::Builder::getIndexArrayAttr, "values"_a)
.def("get_f32_array_attr", &mlir::Builder::getF32ArrayAttr, "values"_a)
.def("get_f64_array_attr", &mlir::Builder::getF64ArrayAttr, "values"_a)
.def("get_str_array_attr", &mlir::Builder::getStrArrayAttr, "values"_a)
.def("get_type_array_attr", &mlir::Builder::getTypeArrayAttr, "values"_a)
.def("get_affine_dim_expr", &mlir::Builder::getAffineDimExpr, "position"_a)
.def("get_affine_symbol_expr", &mlir::Builder::getAffineSymbolExpr, "position"_a)
.def("get_affine_constant_expr", &mlir::Builder::getAffineConstantExpr, "constant"_a)
.def_prop_ro("empty_affine_map", &mlir::Builder::getEmptyAffineMap)
.def("get_constant_affine_map", &mlir::Builder::getConstantAffineMap, "val"_a)
.def_prop_ro("dim_identity_map", &mlir::Builder::getDimIdentityMap)
.def("get_multi_dim_identity_map", &mlir::Builder::getMultiDimIdentityMap, "rank"_a)
.def_prop_ro("symbol_identity_map", &mlir::Builder::getSymbolIdentityMap)
.def("get_single_dim_shift_affine_map", &mlir::Builder::getSingleDimShiftAffineMap, "shift"_a)
.def("get_shifted_affine_map", &mlir::Builder::getShiftedAffineMap, "map"_a, "shift"_a)
;

auto mlir_OpBuilder = nb::class_<mlir::OpBuilder, mlir::Builder>(m, "OpBuilder")
.def(nb::init<mlir::MLIRContext *, mlir::OpBuilder::Listener *>(), "ctx"_a, "listener"_a)
.def(nb::init<mlir::Region *, mlir::OpBuilder::Listener *>(), "region"_a, "listener"_a)
.def(nb::init<mlir::Region &, mlir::OpBuilder::Listener *>(), "region"_a, "listener"_a)
.def(nb::init<mlir::Operation *, mlir::OpBuilder::Listener *>(), "op"_a, "listener"_a)
.def(nb::init<mlir::Block *, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false>, mlir::OpBuilder::Listener *>(), "block"_a, "insert_point"_a, "listener"_a)
.def_static("at_block_begin", &mlir::OpBuilder::atBlockBegin, "block"_a, "listener"_a)
.def_static("at_block_end", &mlir::OpBuilder::atBlockEnd, "block"_a, "listener"_a)
.def_static("at_block_terminator", &mlir::OpBuilder::atBlockTerminator, "block"_a, "listener"_a)
.def("set_listener", &mlir::OpBuilder::setListener, "new_listener"_a)
.def_prop_ro("listener", &mlir::OpBuilder::getListener)
.def("clear_insertion_point", &mlir::OpBuilder::clearInsertionPoint)
.def("save_insertion_point", &mlir::OpBuilder::saveInsertionPoint)
.def("restore_insertion_point", &mlir::OpBuilder::restoreInsertionPoint, "ip"_a)
.def("set_insertion_point", [](mlir::OpBuilder& self, mlir::Block * block, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false> insertPoint){ return self.setInsertionPoint(block, insertPoint); }, "block"_a, "insert_point"_a)
.def("set_insertion_point", [](mlir::OpBuilder& self, mlir::Operation * op){ return self.setInsertionPoint(op); }, "op"_a)
.def("set_insertion_point_after", &mlir::OpBuilder::setInsertionPointAfter, "op"_a)
.def("set_insertion_point_after_value", &mlir::OpBuilder::setInsertionPointAfterValue, "val"_a)
.def("set_insertion_point_to_start", &mlir::OpBuilder::setInsertionPointToStart, "block"_a)
.def("set_insertion_point_to_end", &mlir::OpBuilder::setInsertionPointToEnd, "block"_a)
.def_prop_ro("insertion_block", &mlir::OpBuilder::getInsertionBlock)
.def_prop_ro("insertion_point", &mlir::OpBuilder::getInsertionPoint)
.def_prop_ro("block", &mlir::OpBuilder::getBlock)
.def("create_block", [](mlir::OpBuilder& self, mlir::Region * parent, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Block, true, false, void, false, void>, false, false> insertPt, mlir::TypeRange argTypes, llvm::ArrayRef<mlir::Location> locs){ return self.createBlock(parent, insertPt, argTypes, locs); }, "parent"_a, "insert_pt"_a, "arg_types"_a, "locs"_a, nb::rv_policy::reference_internal)
.def("create_block", [](mlir::OpBuilder& self, mlir::Block * insertBefore, mlir::TypeRange argTypes, llvm::ArrayRef<mlir::Location> locs){ return self.createBlock(insertBefore, argTypes, locs); }, "insert_before"_a, "arg_types"_a, "locs"_a, nb::rv_policy::reference_internal)
.def("insert", &mlir::OpBuilder::insert, "op"_a, nb::rv_policy::reference_internal)
.def("create", [](mlir::OpBuilder& self, const mlir::OperationState & state){ return self.create(state); }, "state"_a, nb::rv_policy::reference_internal)
.def("create", [](mlir::OpBuilder& self, mlir::Location loc, mlir::StringAttr opName, mlir::ValueRange operands, mlir::TypeRange types, llvm::ArrayRef<mlir::NamedAttribute> attributes, mlir::BlockRange successors, llvm::MutableArrayRef<std::unique_ptr<mlir::Region>> regions){ return self.create(loc, opName, operands, types, attributes, successors, std::move(regions)); }, "loc"_a, "op_name"_a, "operands"_a, "types"_a, "attributes"_a, "successors"_a, "regions"_a, nb::rv_policy::reference_internal)
.def("try_fold", &mlir::OpBuilder::tryFold, "op"_a, "results"_a)
.def("clone", [](mlir::OpBuilder& self, mlir::Operation & op, mlir::IRMapping & mapper){ return self.clone(op, mapper); }, "op"_a, "mapper"_a, nb::rv_policy::reference_internal)
.def("clone", [](mlir::OpBuilder& self, mlir::Operation & op){ return self.clone(op); }, "op"_a, nb::rv_policy::reference_internal)
.def("clone_without_regions", [](mlir::OpBuilder& self, mlir::Operation & op, mlir::IRMapping & mapper){ return self.cloneWithoutRegions(op, mapper); }, "op"_a, "mapper"_a, nb::rv_policy::reference_internal)
.def("clone_without_regions", [](mlir::OpBuilder& self, mlir::Operation & op){ return self.cloneWithoutRegions(op); }, "op"_a, nb::rv_policy::reference_internal)
.def("clone_region_before", [](mlir::OpBuilder& self, mlir::Region & region, mlir::Region & parent, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Block, true, false, void, false, void>, false, false> before, mlir::IRMapping & mapping){ return self.cloneRegionBefore(region, parent, before, mapping); }, "region"_a, "parent"_a, "before"_a, "mapping"_a)
.def("clone_region_before", [](mlir::OpBuilder& self, mlir::Region & region, mlir::Region & parent, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Block, true, false, void, false, void>, false, false> before){ return self.cloneRegionBefore(region, parent, before); }, "region"_a, "parent"_a, "before"_a)
.def("clone_region_before", [](mlir::OpBuilder& self, mlir::Region & region, mlir::Block * before){ return self.cloneRegionBefore(region, before); }, "region"_a, "before"_a)
;

auto mlir_OpBuilder_ListenerBase = nb::class_<mlir::OpBuilder::ListenerBase>(mlir_OpBuilder, "ListenerBase")
.def_prop_ro("kind", &mlir::OpBuilder::ListenerBase::getKind)
;

nb::enum_<mlir::OpBuilder::ListenerBase::Kind>(m, "Kind")
.value("OpBuilderListener", mlir::OpBuilder::ListenerBase::Kind::OpBuilderListener)
.value("RewriterBaseListener", mlir::OpBuilder::ListenerBase::Kind::RewriterBaseListener)
;

auto mlir_OpBuilder_Listener = nb::class_<mlir::OpBuilder::Listener, mlir::OpBuilder::ListenerBase>(mlir_OpBuilder, "Listener")
.def(nb::init<>())
.def("notify_operation_inserted", &mlir::OpBuilder::Listener::notifyOperationInserted, "op"_a, "previous"_a)
.def("notify_block_inserted", &mlir::OpBuilder::Listener::notifyBlockInserted, "block"_a, "previous"_a, "previous_it"_a)
;

auto mlir_OpBuilder_InsertPoint = nb::class_<mlir::OpBuilder::InsertPoint>(mlir_OpBuilder, "InsertPoint")
.def(nb::init<>())
.def(nb::init<mlir::Block *, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false>>(), "insert_block"_a, "insert_pt"_a)
.def("is_set", &mlir::OpBuilder::InsertPoint::isSet)
.def_prop_ro("block", &mlir::OpBuilder::InsertPoint::getBlock)
.def_prop_ro("point", &mlir::OpBuilder::InsertPoint::getPoint)
;

auto mlir_OpBuilder_InsertionGuard = nb::class_<mlir::OpBuilder::InsertionGuard>(mlir_OpBuilder, "InsertionGuard")
.def(nb::init<mlir::OpBuilder &>(), "builder"_a)
.def(nb::init<mlir::OpBuilder::InsertionGuard &&>(), "other"_a)
;

auto mlir_BuiltinDialect = nb::class_<mlir::BuiltinDialect, mlir::Dialect>(m, "BuiltinDialect")
.def_static("dialect_namespace", &mlir::BuiltinDialect::getDialectNamespace)
.def_static("insert_into_registry", [](mlir::DialectRegistry &registry) { registry.insert<mlir::BuiltinDialect>(); })
.def_static("load_into_context", [](mlir::MLIRContext &context) { return context.getOrLoadDialect<mlir::BuiltinDialect>(); })
;

auto mlir_detail_TypeIDResolver___mlir_BuiltinDialect__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::BuiltinDialect>>(m, "TypeIDResolver[BuiltinDialect]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::BuiltinDialect>::resolveTypeID)
;

auto mlir_DialectInterface = nb::class_<mlir::DialectInterface>(m, "DialectInterface")
.def_prop_ro("dialect", &mlir::DialectInterface::getDialect)
.def_prop_ro("context", &mlir::DialectInterface::getContext)
.def_prop_ro("id", &mlir::DialectInterface::getID)
;

auto mlir_detail_DialectInterfaceCollectionBase = nb::class_<mlir::detail::DialectInterfaceCollectionBase>(m, "DialectInterfaceCollectionBase")
.def(nb::init<mlir::MLIRContext *, mlir::TypeID, llvm::StringRef>(), "ctx"_a, "interface_kind"_a, "interface_name"_a)
;

auto mlir_FloatType = nb::class_<mlir::FloatType, mlir::Type>(m, "FloatType")
.def_static("classof", &mlir::FloatType::classof, "type"_a)
.def_prop_ro("width", &mlir::FloatType::getWidth)
.def_prop_ro("fp_mantissa_width", &mlir::FloatType::getFPMantissaWidth)
.def("scale_element_bitwidth", &mlir::FloatType::scaleElementBitwidth, "scale"_a)
;

auto mlir_TensorType = nb::class_<mlir::TensorType>(m, "TensorType")
.def_prop_ro("element_type", &mlir::TensorType::getElementType)
.def("has_rank", &mlir::TensorType::hasRank)
.def_prop_ro("shape", &mlir::TensorType::getShape)
.def("clone_with", &mlir::TensorType::cloneWith, "shape"_a, "element_type"_a)
.def("clone", [](mlir::TensorType& self, ArrayRef<int64_t> shape, mlir::Type elementType){ return self.clone(shape, elementType); }, "shape"_a, "element_type"_a)
.def("clone", [](mlir::TensorType& self, ArrayRef<int64_t> shape){ return self.clone(shape); }, "shape"_a)
.def_static("is_valid_element_type", &mlir::TensorType::isValidElementType, "type"_a)
.def_static("classof", &mlir::TensorType::classof, "type"_a)
;

auto mlir_BaseMemRefType = nb::class_<mlir::BaseMemRefType, mlir::Type>(m, "BaseMemRefType")
.def_prop_ro("element_type", &mlir::BaseMemRefType::getElementType)
.def("has_rank", &mlir::BaseMemRefType::hasRank)
.def_prop_ro("shape", &mlir::BaseMemRefType::getShape)
.def("clone_with", &mlir::BaseMemRefType::cloneWith, "shape"_a, "element_type"_a)
.def("clone", [](mlir::BaseMemRefType& self, ArrayRef<int64_t> shape, mlir::Type elementType){ return self.clone(shape, elementType); }, "shape"_a, "element_type"_a)
.def("clone", [](mlir::BaseMemRefType& self, ArrayRef<int64_t> shape){ return self.clone(shape); }, "shape"_a)
.def_static("is_valid_element_type", &mlir::BaseMemRefType::isValidElementType, "type"_a)
.def_static("classof", &mlir::BaseMemRefType::classof, "type"_a)
.def_prop_ro("memory_space", &mlir::BaseMemRefType::getMemorySpace)
.def_prop_ro("memory_space_as_int", &mlir::BaseMemRefType::getMemorySpaceAsInt)
;

auto mlir_ComplexType = nb::class_<mlir::ComplexType,  mlir::Type>(m, "ComplexType")
.def_static("get", &mlir::ComplexType::get, "element_type"_a)
.def_static("verify", &mlir::ComplexType::verify, "emit_error"_a, "element_type"_a)
.def_static("verify_invariants", &mlir::ComplexType::verifyInvariants, "emit_error"_a, "element_type"_a)
.def_prop_ro("element_type", &mlir::ComplexType::getElementType)
;

auto mlir_Float8E5M2Type = nb::class_<mlir::Float8E5M2Type,  mlir::Type>(m, "Float8E5M2Type")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E5M2Type::get(ctx);}, "context"_a)
;

auto mlir_Float8E4M3Type = nb::class_<mlir::Float8E4M3Type,  mlir::Type>(m, "Float8E4M3Type")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E4M3Type::get(ctx);}, "context"_a)
;

auto mlir_Float8E4M3FNType = nb::class_<mlir::Float8E4M3FNType,  mlir::Type>(m, "Float8E4M3FNType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E4M3FNType::get(ctx);}, "context"_a)
;

auto mlir_Float8E5M2FNUZType = nb::class_<mlir::Float8E5M2FNUZType,  mlir::Type>(m, "Float8E5M2FNUZType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E5M2FNUZType::get(ctx);}, "context"_a)
;

auto mlir_Float8E4M3FNUZType = nb::class_<mlir::Float8E4M3FNUZType,  mlir::Type>(m, "Float8E4M3FNUZType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E4M3FNUZType::get(ctx);}, "context"_a)
;

auto mlir_Float8E4M3B11FNUZType = nb::class_<mlir::Float8E4M3B11FNUZType,  mlir::Type>(m, "Float8E4M3B11FNUZType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E4M3B11FNUZType::get(ctx);}, "context"_a)
;

auto mlir_Float8E3M4Type = nb::class_<mlir::Float8E3M4Type,  mlir::Type>(m, "Float8E3M4Type")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E3M4Type::get(ctx);}, "context"_a)
;

auto mlir_Float4E2M1FNType = nb::class_<mlir::Float4E2M1FNType,  mlir::Type>(m, "Float4E2M1FNType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float4E2M1FNType::get(ctx);}, "context"_a)
;

auto mlir_Float6E2M3FNType = nb::class_<mlir::Float6E2M3FNType,  mlir::Type>(m, "Float6E2M3FNType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float6E2M3FNType::get(ctx);}, "context"_a)
;

auto mlir_Float6E3M2FNType = nb::class_<mlir::Float6E3M2FNType,  mlir::Type>(m, "Float6E3M2FNType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float6E3M2FNType::get(ctx);}, "context"_a)
;

auto mlir_Float8E8M0FNUType = nb::class_<mlir::Float8E8M0FNUType,  mlir::Type>(m, "Float8E8M0FNUType")
.def_static("get", [](MLIRContext *ctx){ return mlir::Float8E8M0FNUType::get(ctx);}, "context"_a)
;

auto mlir_BFloat16Type = nb::class_<mlir::BFloat16Type,  mlir::Type>(m, "BFloat16Type")
.def_static("get", &mlir::BFloat16Type::get, "context"_a)
;

auto mlir_Float16Type = nb::class_<mlir::Float16Type,  mlir::Type>(m, "Float16Type")
.def_static("get", &mlir::Float16Type::get, "context"_a)
;

auto mlir_FloatTF32Type = nb::class_<mlir::FloatTF32Type,  mlir::Type>(m, "FloatTF32Type")
.def_static("get", &mlir::FloatTF32Type::get, "context"_a)
;

auto mlir_Float32Type = nb::class_<mlir::Float32Type,  mlir::Type>(m, "Float32Type")
.def_static("get", &mlir::Float32Type::get, "context"_a)
;

auto mlir_Float64Type = nb::class_<mlir::Float64Type,  mlir::Type>(m, "Float64Type")
.def_static("get", &mlir::Float64Type::get, "context"_a)
;

auto mlir_Float80Type = nb::class_<mlir::Float80Type,  mlir::Type>(m, "Float80Type")
.def_static("get", &mlir::Float80Type::get, "context"_a)
;

auto mlir_Float128Type = nb::class_<mlir::Float128Type,  mlir::Type>(m, "Float128Type")
.def_static("get", &mlir::Float128Type::get, "context"_a)
;

auto mlir_FunctionType = nb::class_<mlir::FunctionType,  mlir::Type>(m, "FunctionType")
.def_prop_ro("num_inputs", &mlir::FunctionType::getNumInputs)
.def("get_input", &mlir::FunctionType::getInput, "i"_a)
.def_prop_ro("num_results", &mlir::FunctionType::getNumResults)
.def("get_result", &mlir::FunctionType::getResult, "i"_a)
.def("clone", &mlir::FunctionType::clone, "inputs"_a, "results"_a)
.def("get_with_args_and_results", &mlir::FunctionType::getWithArgsAndResults, "arg_indices"_a, "arg_types"_a, "result_indices"_a, "result_types"_a)
.def("get_without_args_and_results", &mlir::FunctionType::getWithoutArgsAndResults, "arg_indices"_a, "result_indices"_a)
.def_static("get", &mlir::FunctionType::get, "context"_a, "inputs"_a, "results"_a)
.def_prop_ro("inputs", &mlir::FunctionType::getInputs)
.def_prop_ro("results", &mlir::FunctionType::getResults)
;

auto mlir_IndexType = nb::class_<mlir::IndexType,  mlir::Type>(m, "IndexType")
.def_static("get", &mlir::IndexType::get, "context"_a)
;

auto mlir_IntegerType = nb::class_<mlir::IntegerType,  mlir::Type>(m, "IntegerType")
.def("is_signless", &mlir::IntegerType::isSignless)
.def("is_signed", &mlir::IntegerType::isSigned)
.def("is_unsigned", &mlir::IntegerType::isUnsigned)
.def("scale_element_bitwidth", &mlir::IntegerType::scaleElementBitwidth, "scale"_a)
.def_static("get", &mlir::IntegerType::get, "context"_a, "width"_a, "signedness"_a)
.def_static("verify", &mlir::IntegerType::verify, "emit_error"_a, "width"_a, "signedness"_a)
.def_static("verify_invariants", &mlir::IntegerType::verifyInvariants, "emit_error"_a, "width"_a, "signedness"_a)
.def_prop_ro("width", &mlir::IntegerType::getWidth)
.def_prop_ro("signedness", &mlir::IntegerType::getSignedness)
;

nb::enum_<mlir::IntegerType::SignednessSemantics>(m, "SignednessSemantics")
.value("Signless", mlir::IntegerType::SignednessSemantics::Signless)
.value("Signed", mlir::IntegerType::SignednessSemantics::Signed)
.value("Unsigned", mlir::IntegerType::SignednessSemantics::Unsigned)
;

auto mlir_MemRefType = nb::class_<mlir::MemRefType,  mlir::BaseMemRefType>(m, "MemRefType")
.def_prop_ro("memory_space_as_int", &mlir::MemRefType::getMemorySpaceAsInt)
.def_static("get", [](ArrayRef<int64_t> shape, mlir::Type elementType, mlir::MemRefLayoutAttrInterface layout, mlir::Attribute memorySpace){ return mlir::MemRefType::get(shape, elementType, layout, memorySpace); }, "shape"_a, "element_type"_a, "layout"_a, "memory_space"_a)
.def_static("get", [](ArrayRef<int64_t> shape, mlir::Type elementType, mlir::AffineMap map, mlir::Attribute memorySpace){ return mlir::MemRefType::get(shape, elementType, map, memorySpace); }, "shape"_a, "element_type"_a, "map"_a, "memory_space"_a)
.def_static("get", [](ArrayRef<int64_t> shape, mlir::Type elementType, mlir::AffineMap map, unsigned int memorySpaceInd){ return mlir::MemRefType::get(shape, elementType, map, memorySpaceInd); }, "shape"_a, "element_type"_a, "map"_a, "memory_space_ind"_a)
.def_static("verify", &mlir::MemRefType::verify, "emit_error"_a, "shape"_a, "element_type"_a, "layout"_a, "memory_space"_a)
.def_static("verify_invariants", &mlir::MemRefType::verifyInvariants, "emit_error"_a, "shape"_a, "element_type"_a, "layout"_a, "memory_space"_a)
.def_prop_ro("shape", &mlir::MemRefType::getShape)
.def_prop_ro("element_type", &mlir::MemRefType::getElementType)
.def_prop_ro("layout", &mlir::MemRefType::getLayout)
.def_prop_ro("memory_space", &mlir::MemRefType::getMemorySpace)
;

auto mlir_NoneType = nb::class_<mlir::NoneType,  mlir::Type>(m, "NoneType")
.def_static("get", &mlir::NoneType::get, "context"_a)
;

auto mlir_OpaqueType = nb::class_<mlir::OpaqueType,  mlir::Type>(m, "OpaqueType")
.def_static("get", &mlir::OpaqueType::get, "dialect_namespace"_a, "type_data"_a)
.def_static("verify", &mlir::OpaqueType::verify, "emit_error"_a, "dialect_namespace"_a, "type_data"_a)
.def_static("verify_invariants", &mlir::OpaqueType::verifyInvariants, "emit_error"_a, "dialect_namespace"_a, "type_data"_a)
.def_prop_ro("dialect_namespace", &mlir::OpaqueType::getDialectNamespace)
.def_prop_ro("type_data", &mlir::OpaqueType::getTypeData)
;

auto mlir_RankedTensorType = nb::class_<mlir::RankedTensorType,  mlir::TensorType>(m, "RankedTensorType")
.def_static("get", &mlir::RankedTensorType::get, "shape"_a, "element_type"_a, "encoding"_a)
.def_static("verify", &mlir::RankedTensorType::verify, "emit_error"_a, "shape"_a, "element_type"_a, "encoding"_a)
.def_static("verify_invariants", &mlir::RankedTensorType::verifyInvariants, "emit_error"_a, "shape"_a, "element_type"_a, "encoding"_a)
.def_prop_ro("shape", &mlir::RankedTensorType::getShape)
.def_prop_ro("element_type", &mlir::RankedTensorType::getElementType)
.def_prop_ro("encoding", &mlir::RankedTensorType::getEncoding)
;

auto mlir_TupleType = nb::class_<mlir::TupleType,  mlir::Type>(m, "TupleType")
.def("get_flattened_types", &mlir::TupleType::getFlattenedTypes, "types"_a)
.def("size", &mlir::TupleType::size)
.def("begin", &mlir::TupleType::begin)
.def("end", &mlir::TupleType::end)
.def("get_type", &mlir::TupleType::getType, "index"_a)
.def_static("get", [](mlir::MLIRContext * context, mlir::TypeRange elementTypes){ return mlir::TupleType::get(context, elementTypes); }, "context"_a, "element_types"_a)
.def_static("get", [](mlir::MLIRContext * context){ return mlir::TupleType::get(context); }, "context"_a)
.def_prop_ro("types", &mlir::TupleType::getTypes)
;

auto mlir_UnrankedMemRefType = nb::class_<mlir::UnrankedMemRefType,  mlir::BaseMemRefType>(m, "UnrankedMemRefType")
.def_prop_ro("shape", &mlir::UnrankedMemRefType::getShape)
.def_prop_ro("memory_space_as_int", &mlir::UnrankedMemRefType::getMemorySpaceAsInt)
.def_static("get", [](mlir::Type elementType, mlir::Attribute memorySpace){ return mlir::UnrankedMemRefType::get(elementType, memorySpace); }, "element_type"_a, "memory_space"_a)
.def_static("get", [](mlir::Type elementType, unsigned int memorySpace){ return mlir::UnrankedMemRefType::get(elementType, memorySpace); }, "element_type"_a, "memory_space"_a)
.def_static("verify", &mlir::UnrankedMemRefType::verify, "emit_error"_a, "element_type"_a, "memory_space"_a)
.def_static("verify_invariants", &mlir::UnrankedMemRefType::verifyInvariants, "emit_error"_a, "element_type"_a, "memory_space"_a)
.def_prop_ro("element_type", &mlir::UnrankedMemRefType::getElementType)
.def_prop_ro("memory_space", &mlir::UnrankedMemRefType::getMemorySpace)
;

auto mlir_UnrankedTensorType = nb::class_<mlir::UnrankedTensorType,  mlir::TensorType>(m, "UnrankedTensorType")
.def_prop_ro("shape", &mlir::UnrankedTensorType::getShape)
.def_static("get", &mlir::UnrankedTensorType::get, "element_type"_a)
.def_static("verify", &mlir::UnrankedTensorType::verify, "emit_error"_a, "element_type"_a)
.def_static("verify_invariants", &mlir::UnrankedTensorType::verifyInvariants, "emit_error"_a, "element_type"_a)
.def_prop_ro("element_type", &mlir::UnrankedTensorType::getElementType)
;

auto mlir_VectorType = nb::class_<mlir::VectorType,  mlir::Type>(m, "VectorType")
.def_static("is_valid_element_type", &mlir::VectorType::isValidElementType, "t"_a)
.def("is_scalable", &mlir::VectorType::isScalable)
.def("all_dims_scalable", &mlir::VectorType::allDimsScalable)
.def_prop_ro("num_scalable_dims", &mlir::VectorType::getNumScalableDims)
.def("scale_element_bitwidth", &mlir::VectorType::scaleElementBitwidth, "scale"_a)
.def("has_rank", &mlir::VectorType::hasRank)
.def("clone_with", &mlir::VectorType::cloneWith, "shape"_a, "element_type"_a)
.def_static("get", &mlir::VectorType::get, "shape"_a, "element_type"_a, "scalable_dims"_a)
.def_static("verify_invariants_impl", &mlir::VectorType::verifyInvariantsImpl, "emit_error"_a, "shape"_a, "element_type"_a, "scalable_dims"_a)
.def_static("verify", &mlir::VectorType::verify, "emit_error"_a, "shape"_a, "element_type"_a, "scalable_dims"_a)
.def_static("verify_invariants", &mlir::VectorType::verifyInvariants, "emit_error"_a, "shape"_a, "element_type"_a, "scalable_dims"_a)
.def_prop_ro("shape", &mlir::VectorType::getShape)
.def_prop_ro("element_type", &mlir::VectorType::getElementType)
.def_prop_ro("scalable_dims", &mlir::VectorType::getScalableDims)
;

auto mlir_detail_TypeIDResolver___mlir_ComplexType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::ComplexType>>(m, "TypeIDResolver[ComplexType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::ComplexType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E5M2Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E5M2Type>>(m, "TypeIDResolver[Float8E5M2Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E5M2Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E4M3Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E4M3Type>>(m, "TypeIDResolver[Float8E4M3Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E4M3Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E4M3FNType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E4M3FNType>>(m, "TypeIDResolver[Float8E4M3FNType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E4M3FNType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E5M2FNUZType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E5M2FNUZType>>(m, "TypeIDResolver[Float8E5M2FNUZType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E5M2FNUZType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E4M3FNUZType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E4M3FNUZType>>(m, "TypeIDResolver[Float8E4M3FNUZType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E4M3FNUZType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E4M3B11FNUZType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E4M3B11FNUZType>>(m, "TypeIDResolver[Float8E4M3B11FNUZType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E4M3B11FNUZType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E3M4Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E3M4Type>>(m, "TypeIDResolver[Float8E3M4Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E3M4Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float4E2M1FNType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float4E2M1FNType>>(m, "TypeIDResolver[Float4E2M1FNType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float4E2M1FNType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float6E2M3FNType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float6E2M3FNType>>(m, "TypeIDResolver[Float6E2M3FNType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float6E2M3FNType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float6E3M2FNType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float6E3M2FNType>>(m, "TypeIDResolver[Float6E3M2FNType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float6E3M2FNType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float8E8M0FNUType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float8E8M0FNUType>>(m, "TypeIDResolver[Float8E8M0FNUType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float8E8M0FNUType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_BFloat16Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::BFloat16Type>>(m, "TypeIDResolver[BFloat16Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::BFloat16Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float16Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float16Type>>(m, "TypeIDResolver[Float16Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float16Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_FloatTF32Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::FloatTF32Type>>(m, "TypeIDResolver[FloatTF32Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::FloatTF32Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float32Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float32Type>>(m, "TypeIDResolver[Float32Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float32Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float64Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float64Type>>(m, "TypeIDResolver[Float64Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float64Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float80Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float80Type>>(m, "TypeIDResolver[Float80Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float80Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_Float128Type__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::Float128Type>>(m, "TypeIDResolver[Float128Type]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::Float128Type>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_FunctionType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::FunctionType>>(m, "TypeIDResolver[FunctionType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::FunctionType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_IndexType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::IndexType>>(m, "TypeIDResolver[IndexType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::IndexType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_IntegerType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::IntegerType>>(m, "TypeIDResolver[IntegerType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::IntegerType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_MemRefType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::MemRefType>>(m, "TypeIDResolver[MemRefType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::MemRefType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_NoneType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::NoneType>>(m, "TypeIDResolver[NoneType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::NoneType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_OpaqueType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::OpaqueType>>(m, "TypeIDResolver[OpaqueType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::OpaqueType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_RankedTensorType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::RankedTensorType>>(m, "TypeIDResolver[RankedTensorType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::RankedTensorType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_TupleType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::TupleType>>(m, "TypeIDResolver[TupleType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::TupleType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_UnrankedMemRefType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::UnrankedMemRefType>>(m, "TypeIDResolver[UnrankedMemRefType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::UnrankedMemRefType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_UnrankedTensorType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::UnrankedTensorType>>(m, "TypeIDResolver[UnrankedTensorType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::UnrankedTensorType>::resolveTypeID)
;

auto mlir_detail_TypeIDResolver___mlir_VectorType__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::VectorType>>(m, "TypeIDResolver[VectorType]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::VectorType>::resolveTypeID)
;

auto mlir_MemRefType_Builder = nb::class_<mlir::MemRefType::Builder>(mlir_MemRefType, "Builder")
.def(nb::init<mlir::MemRefType>(), "other"_a)
.def(nb::init<ArrayRef<int64_t>, mlir::Type>(), "shape"_a, "element_type"_a)
.def("set_shape", &mlir::MemRefType::Builder::setShape, "new_shape"_a, nb::rv_policy::reference_internal)
.def("set_element_type", &mlir::MemRefType::Builder::setElementType, "new_element_type"_a, nb::rv_policy::reference_internal)
.def("set_layout", &mlir::MemRefType::Builder::setLayout, "new_layout"_a, nb::rv_policy::reference_internal)
.def("set_memory_space", &mlir::MemRefType::Builder::setMemorySpace, "new_memory_space"_a, nb::rv_policy::reference_internal)
.def("memref_type", [](mlir::MemRefType::Builder& self) -> mlir::MemRefType { return self; })
;

auto mlir_RankedTensorType_Builder = nb::class_<mlir::RankedTensorType::Builder>(mlir_RankedTensorType, "Builder")
.def(nb::init<mlir::RankedTensorType>(), "other"_a)
.def(nb::init<ArrayRef<int64_t>, mlir::Type, mlir::Attribute>(), "shape"_a, "element_type"_a, "encoding"_a)
.def("set_shape", &mlir::RankedTensorType::Builder::setShape, "new_shape"_a, nb::rv_policy::reference_internal)
.def("set_element_type", &mlir::RankedTensorType::Builder::setElementType, "new_element_type"_a, nb::rv_policy::reference_internal)
.def("set_encoding", &mlir::RankedTensorType::Builder::setEncoding, "new_encoding"_a, nb::rv_policy::reference_internal)
.def("drop_dim", &mlir::RankedTensorType::Builder::dropDim, "pos"_a, nb::rv_policy::reference_internal)
.def("insert_dim", &mlir::RankedTensorType::Builder::insertDim, "val"_a, "pos"_a, nb::rv_policy::reference_internal)
;

auto mlir_VectorType_Builder = nb::class_<mlir::VectorType::Builder>(mlir_VectorType, "Builder")
.def(nb::init<mlir::VectorType>(), "other"_a)
.def(nb::init<ArrayRef<int64_t>, mlir::Type, llvm::ArrayRef<bool>>(), "shape"_a, "element_type"_a, "scalable_dims"_a)
.def("set_shape", &mlir::VectorType::Builder::setShape, "new_shape"_a, "new_is_scalable_dim"_a, nb::rv_policy::reference_internal)
.def("set_element_type", &mlir::VectorType::Builder::setElementType, "new_element_type"_a, nb::rv_policy::reference_internal)
.def("drop_dim", &mlir::VectorType::Builder::dropDim, "pos"_a, nb::rv_policy::reference_internal)
.def("set_dim", &mlir::VectorType::Builder::setDim, "pos"_a, "val"_a, nb::rv_policy::reference_internal)
;

nb::enum_<mlir::SliceVerificationResult>(m, "SliceVerificationResult")
.value("Success", mlir::SliceVerificationResult::Success)
.value("RankTooLarge", mlir::SliceVerificationResult::RankTooLarge)
.value("SizeMismatch", mlir::SliceVerificationResult::SizeMismatch)
.value("ElemTypeMismatch", mlir::SliceVerificationResult::ElemTypeMismatch)
.value("MemSpaceMismatch", mlir::SliceVerificationResult::MemSpaceMismatch)
.value("LayoutMismatch", mlir::SliceVerificationResult::LayoutMismatch)
;

auto mlir_AsmDialectResourceHandle = nb::class_<mlir::AsmDialectResourceHandle>(m, "AsmDialectResourceHandle")
.def(nb::init<>())
.def(nb::init<void *, mlir::TypeID, mlir::Dialect *>(), "resource"_a, "resource_id"_a, "dialect"_a)
.def("__eq__", &mlir::AsmDialectResourceHandle::operator==, "other"_a)
.def_prop_ro("resource", &mlir::AsmDialectResourceHandle::getResource)
.def_prop_ro("type_id", &mlir::AsmDialectResourceHandle::getTypeID)
.def_prop_ro("dialect", &mlir::AsmDialectResourceHandle::getDialect)
;

auto mlir_AsmPrinter = nb::class_<mlir::AsmPrinter>(m, "AsmPrinter")
.def_prop_ro("stream", &mlir::AsmPrinter::getStream)
.def("print_float", &mlir::AsmPrinter::printFloat, "value"_a)
.def("print_type", &mlir::AsmPrinter::printType, "type"_a)
.def("print_attribute", &mlir::AsmPrinter::printAttribute, "attr"_a)
.def("print_attribute_without_type", &mlir::AsmPrinter::printAttributeWithoutType, "attr"_a)
.def("print_alias", [](mlir::AsmPrinter& self, mlir::Attribute attr){ return self.printAlias(attr); }, "attr"_a)
.def("print_alias", [](mlir::AsmPrinter& self, mlir::Type type){ return self.printAlias(type); }, "type"_a)
.def("print_keyword_or_string", &mlir::AsmPrinter::printKeywordOrString, "keyword"_a)
.def("print_string", &mlir::AsmPrinter::printString, "string"_a)
.def("print_symbol_name", &mlir::AsmPrinter::printSymbolName, "symbol_ref"_a)
.def("print_resource_handle", &mlir::AsmPrinter::printResourceHandle, "resource"_a)
.def("print_dimension_list", &mlir::AsmPrinter::printDimensionList, "shape"_a)
;

auto mlir_AsmPrinter_CyclicPrintReset = nb::class_<mlir::AsmPrinter::CyclicPrintReset>(mlir_AsmPrinter, "CyclicPrintReset")
.def(nb::init<mlir::AsmPrinter *>(), "printer"_a)
.def(nb::init<mlir::AsmPrinter::CyclicPrintReset &&>(), "rhs"_a)
;

auto mlir_OpAsmPrinter = nb::class_<mlir::OpAsmPrinter, mlir::AsmPrinter>(m, "OpAsmPrinter")
.def("print_optional_location_specifier", &mlir::OpAsmPrinter::printOptionalLocationSpecifier, "loc"_a)
.def("print_newline", &mlir::OpAsmPrinter::printNewline)
.def("increase_indent", &mlir::OpAsmPrinter::increaseIndent)
.def("decrease_indent", &mlir::OpAsmPrinter::decreaseIndent)
.def("print_region_argument", &mlir::OpAsmPrinter::printRegionArgument, "arg"_a, "arg_attrs"_a, "omit_type"_a)
.def("print_operand", [](mlir::OpAsmPrinter& self, mlir::Value value){ return self.printOperand(value); }, "value"_a)
.def("print_operand", [](mlir::OpAsmPrinter& self, mlir::Value value, llvm::raw_ostream & os){ return self.printOperand(value, os); }, "value"_a, "os"_a)
.def("print_successor", &mlir::OpAsmPrinter::printSuccessor, "successor"_a)
.def("print_successor_and_use_list", &mlir::OpAsmPrinter::printSuccessorAndUseList, "successor"_a, "succ_operands"_a)
.def("print_optional_attr_dict", &mlir::OpAsmPrinter::printOptionalAttrDict, "attrs"_a, "elided_attrs"_a)
.def("print_optional_attr_dict_with_keyword", &mlir::OpAsmPrinter::printOptionalAttrDictWithKeyword, "attrs"_a, "elided_attrs"_a)
.def("print_custom_or_generic_op", &mlir::OpAsmPrinter::printCustomOrGenericOp, "op"_a)
.def("print_generic_op", &mlir::OpAsmPrinter::printGenericOp, "op"_a, "print_op_name"_a)
.def("print_region", &mlir::OpAsmPrinter::printRegion, "blocks"_a, "print_entry_block_args"_a, "print_block_terminators"_a, "print_empty_block"_a)
.def("shadow_region_args", &mlir::OpAsmPrinter::shadowRegionArgs, "region"_a, "names_to_use"_a)
.def("print_affine_map_of_ssa_ids", &mlir::OpAsmPrinter::printAffineMapOfSSAIds, "map_attr"_a, "operands"_a)
.def("print_affine_expr_of_ssa_ids", &mlir::OpAsmPrinter::printAffineExprOfSSAIds, "expr"_a, "dim_operands"_a, "sym_operands"_a)
;

nb::enum_<mlir::AsmParser::Delimiter>(m, "Delimiter")
.value("None", mlir::AsmParser::Delimiter::None)
.value("Paren", mlir::AsmParser::Delimiter::Paren)
.value("Square", mlir::AsmParser::Delimiter::Square)
.value("LessGreater", mlir::AsmParser::Delimiter::LessGreater)
.value("Braces", mlir::AsmParser::Delimiter::Braces)
.value("OptionalParen", mlir::AsmParser::Delimiter::OptionalParen)
.value("OptionalSquare", mlir::AsmParser::Delimiter::OptionalSquare)
.value("OptionalLessGreater", mlir::AsmParser::Delimiter::OptionalLessGreater)
.value("OptionalBraces", mlir::AsmParser::Delimiter::OptionalBraces)
;

auto mlir_OpAsmParser = nb::class_<mlir::OpAsmParser>(m, "OpAsmParser")
.def("parse_optional_location_specifier", &mlir::OpAsmParser::parseOptionalLocationSpecifier, "result"_a)
.def("get_result_name", &mlir::OpAsmParser::getResultName, "result_no"_a)
.def_prop_ro("num_results", &mlir::OpAsmParser::getNumResults)
.def("parse_generic_operation", &mlir::OpAsmParser::parseGenericOperation, "insert_block"_a, "insert_pt"_a, nb::rv_policy::reference_internal)
.def("parse_custom_operation_name", &mlir::OpAsmParser::parseCustomOperationName)
.def("parse_generic_operation_after_op_name", &mlir::OpAsmParser::parseGenericOperationAfterOpName, "result"_a, "parsed_operand_type"_a, "parsed_successors"_a, "parsed_regions"_a, "parsed_attributes"_a, "parsed_properties_attribute"_a, "parsed_fn_type"_a)
.def("parse_operand", &mlir::OpAsmParser::parseOperand, "result"_a, "allow_result_number"_a)
.def("parse_optional_operand", &mlir::OpAsmParser::parseOptionalOperand, "result"_a, "allow_result_number"_a)
.def("parse_operand_list", [](mlir::OpAsmParser& self, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> & result, mlir::AsmParser::Delimiter delimiter, bool allowResultNumber, int requiredOperandCount){ return self.parseOperandList(result, delimiter, allowResultNumber, requiredOperandCount); }, "result"_a, "delimiter"_a, "allow_result_number"_a, "required_operand_count"_a)
.def("parse_operand_list", [](mlir::OpAsmParser& self, llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> & result, int requiredOperandCount, mlir::AsmParser::Delimiter delimiter){ return self.parseOperandList(result, requiredOperandCount, delimiter); }, "result"_a, "required_operand_count"_a, "delimiter"_a)
.def("parse_trailing_operand_list", &mlir::OpAsmParser::parseTrailingOperandList, "result"_a, "delimiter"_a)
.def("resolve_operand", &mlir::OpAsmParser::resolveOperand, "operand"_a, "type"_a, "result"_a)
.def("parse_affine_map_of_ssa_ids", &mlir::OpAsmParser::parseAffineMapOfSSAIds, "operands"_a, "map"_a, "attr_name"_a, "attrs"_a, "delimiter"_a)
.def("parse_affine_expr_of_ssa_ids", &mlir::OpAsmParser::parseAffineExprOfSSAIds, "dim_operands"_a, "symb_operands"_a, "expr"_a)
.def("parse_argument", &mlir::OpAsmParser::parseArgument, "result"_a, "allow_type"_a, "allow_attrs"_a)
.def("parse_optional_argument", &mlir::OpAsmParser::parseOptionalArgument, "result"_a, "allow_type"_a, "allow_attrs"_a)
.def("parse_argument_list", &mlir::OpAsmParser::parseArgumentList, "result"_a, "delimiter"_a, "allow_type"_a, "allow_attrs"_a)
.def("parse_region", &mlir::OpAsmParser::parseRegion, "region"_a, "arguments"_a, "enable_name_shadowing"_a)
.def("parse_optional_region", [](mlir::OpAsmParser& self, mlir::Region & region, llvm::ArrayRef<mlir::OpAsmParser::Argument> arguments, bool enableNameShadowing){ return self.parseOptionalRegion(region, arguments, enableNameShadowing); }, "region"_a, "arguments"_a, "enable_name_shadowing"_a)
.def("parse_assignment_list", &mlir::OpAsmParser::parseAssignmentList, "lhs"_a, "rhs"_a)
.def("parse_optional_assignment_list", &mlir::OpAsmParser::parseOptionalAssignmentList, "lhs"_a, "rhs"_a)
;

auto mlir_OpAsmParser_UnresolvedOperand = nb::class_<mlir::OpAsmParser::UnresolvedOperand>(mlir_OpAsmParser, "UnresolvedOperand")
;

auto mlir_OpAsmParser_Argument = nb::class_<mlir::OpAsmParser::Argument>(mlir_OpAsmParser, "Argument")
;

auto mlir_OpAsmDialectInterface = nb::class_<mlir::OpAsmDialectInterface>(m, "OpAsmDialectInterface")
.def(nb::init<mlir::Dialect *>(), "dialect"_a)
.def("get_alias", [](mlir::OpAsmDialectInterface& self, mlir::Attribute attr, llvm::raw_ostream & os){ return self.getAlias(attr, os); }, "attr"_a, "os"_a)
.def("get_alias", [](mlir::OpAsmDialectInterface& self, mlir::Type type, llvm::raw_ostream & os){ return self.getAlias(type, os); }, "type"_a, "os"_a)
.def("declare_resource", &mlir::OpAsmDialectInterface::declareResource, "key"_a)
.def("get_resource_key", &mlir::OpAsmDialectInterface::getResourceKey, "handle"_a)
.def("parse_resource", &mlir::OpAsmDialectInterface::parseResource, "entry"_a)
.def("build_resources", &mlir::OpAsmDialectInterface::buildResources, "op"_a, "referenced_resources"_a, "builder"_a)
;

nb::enum_<mlir::OpAsmDialectInterface::AliasResult>(m, "AliasResult")
.value("NoAlias", mlir::OpAsmDialectInterface::AliasResult::NoAlias)
.value("OverridableAlias", mlir::OpAsmDialectInterface::AliasResult::OverridableAlias)
.value("FinalAlias", mlir::OpAsmDialectInterface::AliasResult::FinalAlias)
;

auto mlir_detail_OpAsmOpInterfaceInterfaceTraits = nb::class_<mlir::detail::OpAsmOpInterfaceInterfaceTraits>(m, "OpAsmOpInterfaceInterfaceTraits")
;

auto mlir_detail_OpAsmOpInterfaceInterfaceTraits_Concept = nb::class_<mlir::detail::OpAsmOpInterfaceInterfaceTraits::Concept>(mlir_detail_OpAsmOpInterfaceInterfaceTraits, "Concept")
;

auto mlir_OpAsmOpInterface = nb::class_<mlir::OpAsmOpInterface>(m, "OpAsmOpInterface")
.def("get_asm_result_names", &mlir::OpAsmOpInterface::getAsmResultNames, "set_name_fn"_a)
.def("get_asm_block_argument_names", &mlir::OpAsmOpInterface::getAsmBlockArgumentNames, "region"_a, "set_name_fn"_a)
.def("get_asm_block_names", &mlir::OpAsmOpInterface::getAsmBlockNames, "set_name_fn"_a)
.def_prop_ro("default_dialect", &mlir::OpAsmOpInterface::getDefaultDialect)
;

nb::enum_<mlir::RegionKind>(m, "RegionKind")
.value("SSACFG", mlir::RegionKind::SSACFG)
.value("Graph", mlir::RegionKind::Graph)
;

auto mlir_detail_RegionKindInterfaceInterfaceTraits = nb::class_<mlir::detail::RegionKindInterfaceInterfaceTraits>(m, "RegionKindInterfaceInterfaceTraits")
;

auto mlir_detail_RegionKindInterfaceInterfaceTraits_Concept = nb::class_<mlir::detail::RegionKindInterfaceInterfaceTraits::Concept>(mlir_detail_RegionKindInterfaceInterfaceTraits, "Concept")
;

auto mlir_RegionKindInterface = nb::class_<mlir::RegionKindInterface>(m, "RegionKindInterface")
.def("get_region_kind", &mlir::RegionKindInterface::getRegionKind, "index"_a)
.def("has_ssa_dominance", &mlir::RegionKindInterface::hasSSADominance, "index"_a)
;

auto mlir_SymbolTable = nb::class_<mlir::SymbolTable>(m, "SymbolTable")
.def(nb::init<mlir::Operation *>(), "symbol_table_op"_a)
.def("lookup", [](mlir::SymbolTable& self, llvm::StringRef name){ return self.lookup(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("lookup", [](mlir::SymbolTable& self, mlir::StringAttr name){ return self.lookup(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("remove", &mlir::SymbolTable::remove, "op"_a)
.def("erase", &mlir::SymbolTable::erase, "symbol"_a)
.def("insert", &mlir::SymbolTable::insert, "symbol"_a, "insert_pt"_a)
.def("rename", [](mlir::SymbolTable& self, mlir::StringAttr from_, mlir::StringAttr to){ return self.rename(from_, to); }, "from_"_a, "to"_a)
.def("rename", [](mlir::SymbolTable& self, mlir::Operation * op, mlir::StringAttr to){ return self.rename(op, to); }, "op"_a, "to"_a)
.def("rename", [](mlir::SymbolTable& self, mlir::StringAttr from_, llvm::StringRef to){ return self.rename(from_, to); }, "from_"_a, "to"_a)
.def("rename", [](mlir::SymbolTable& self, mlir::Operation * op, llvm::StringRef to){ return self.rename(op, to); }, "op"_a, "to"_a)
.def("rename_to_unique", [](mlir::SymbolTable& self, mlir::StringAttr from_, llvm::ArrayRef<mlir::SymbolTable *> others){ return self.renameToUnique(from_, others); }, "from_"_a, "others"_a)
.def("rename_to_unique", [](mlir::SymbolTable& self, mlir::Operation * op, llvm::ArrayRef<mlir::SymbolTable *> others){ return self.renameToUnique(op, others); }, "op"_a, "others"_a)
.def_static("symbol_attr_name", &mlir::SymbolTable::getSymbolAttrName)
.def_prop_ro("op", &mlir::SymbolTable::getOp)
.def_static("visibility_attr_name", &mlir::SymbolTable::getVisibilityAttrName)
.def_static("get_symbol_name", &mlir::SymbolTable::getSymbolName, "symbol"_a)
.def_static("set_symbol_name", [](mlir::Operation * symbol, mlir::StringAttr name){ return mlir::SymbolTable::setSymbolName(symbol, name); }, "symbol"_a, "name"_a)
.def_static("set_symbol_name", [](mlir::Operation * symbol, llvm::StringRef name){ return mlir::SymbolTable::setSymbolName(symbol, name); }, "symbol"_a, "name"_a)
.def_static("get_symbol_visibility", &mlir::SymbolTable::getSymbolVisibility, "symbol"_a)
.def_static("set_symbol_visibility", &mlir::SymbolTable::setSymbolVisibility, "symbol"_a, "vis"_a)
.def_static("get_nearest_symbol_table", &mlir::SymbolTable::getNearestSymbolTable, "from_"_a, nb::rv_policy::reference_internal)
.def_static("walk_symbol_tables", &mlir::SymbolTable::walkSymbolTables, "op"_a, "all_sym_uses_visible"_a, "callback"_a)
.def_static("lookup_symbol_in", [](mlir::Operation * op, mlir::StringAttr symbol){ return mlir::SymbolTable::lookupSymbolIn(op, symbol); }, "op"_a, "symbol"_a, nb::rv_policy::reference_internal)
.def_static("lookup_symbol_in", [](mlir::Operation * op, llvm::StringRef symbol){ return mlir::SymbolTable::lookupSymbolIn(op, symbol); }, "op"_a, "symbol"_a, nb::rv_policy::reference_internal)
.def_static("lookup_symbol_in", [](mlir::Operation * op, mlir::SymbolRefAttr symbol){ return mlir::SymbolTable::lookupSymbolIn(op, symbol); }, "op"_a, "symbol"_a, nb::rv_policy::reference_internal)
.def_static("lookup_symbol_in", [](mlir::Operation * op, mlir::SymbolRefAttr symbol, llvm::SmallVectorImpl<mlir::Operation *> & symbols){ return mlir::SymbolTable::lookupSymbolIn(op, symbol, symbols); }, "op"_a, "symbol"_a, "symbols"_a)
.def_static("lookup_nearest_symbol_from", [](mlir::Operation * from_, mlir::StringAttr symbol){ return mlir::SymbolTable::lookupNearestSymbolFrom(from_, symbol); }, "from_"_a, "symbol"_a, nb::rv_policy::reference_internal)
.def_static("lookup_nearest_symbol_from", [](mlir::Operation * from_, mlir::SymbolRefAttr symbol){ return mlir::SymbolTable::lookupNearestSymbolFrom(from_, symbol); }, "from_"_a, "symbol"_a, nb::rv_policy::reference_internal)
.def_static("get_symbol_uses", [](mlir::Operation * from_){ return mlir::SymbolTable::getSymbolUses(from_); }, "from_"_a)
.def_static("get_symbol_uses", [](mlir::Region * from_){ return mlir::SymbolTable::getSymbolUses(from_); }, "from_"_a)
.def_static("get_symbol_uses", [](mlir::StringAttr symbol, mlir::Operation * from_){ return mlir::SymbolTable::getSymbolUses(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("get_symbol_uses", [](mlir::Operation * symbol, mlir::Operation * from_){ return mlir::SymbolTable::getSymbolUses(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("get_symbol_uses", [](mlir::StringAttr symbol, mlir::Region * from_){ return mlir::SymbolTable::getSymbolUses(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("get_symbol_uses", [](mlir::Operation * symbol, mlir::Region * from_){ return mlir::SymbolTable::getSymbolUses(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("symbol_known_use_empty", [](mlir::StringAttr symbol, mlir::Operation * from_){ return mlir::SymbolTable::symbolKnownUseEmpty(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("symbol_known_use_empty", [](mlir::Operation * symbol, mlir::Operation * from_){ return mlir::SymbolTable::symbolKnownUseEmpty(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("symbol_known_use_empty", [](mlir::StringAttr symbol, mlir::Region * from_){ return mlir::SymbolTable::symbolKnownUseEmpty(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("symbol_known_use_empty", [](mlir::Operation * symbol, mlir::Region * from_){ return mlir::SymbolTable::symbolKnownUseEmpty(symbol, from_); }, "symbol"_a, "from_"_a)
.def_static("replace_all_symbol_uses", [](mlir::StringAttr oldSymbol, mlir::StringAttr newSymbol, mlir::Operation * from_){ return mlir::SymbolTable::replaceAllSymbolUses(oldSymbol, newSymbol, from_); }, "old_symbol"_a, "new_symbol"_a, "from_"_a)
.def_static("replace_all_symbol_uses", [](mlir::Operation * oldSymbol, mlir::StringAttr newSymbolName, mlir::Operation * from_){ return mlir::SymbolTable::replaceAllSymbolUses(oldSymbol, newSymbolName, from_); }, "old_symbol"_a, "new_symbol_name"_a, "from_"_a)
.def_static("replace_all_symbol_uses", [](mlir::StringAttr oldSymbol, mlir::StringAttr newSymbol, mlir::Region * from_){ return mlir::SymbolTable::replaceAllSymbolUses(oldSymbol, newSymbol, from_); }, "old_symbol"_a, "new_symbol"_a, "from_"_a)
.def_static("replace_all_symbol_uses", [](mlir::Operation * oldSymbol, mlir::StringAttr newSymbolName, mlir::Region * from_){ return mlir::SymbolTable::replaceAllSymbolUses(oldSymbol, newSymbolName, from_); }, "old_symbol"_a, "new_symbol_name"_a, "from_"_a)
;

nb::enum_<mlir::SymbolTable::Visibility>(m, "Visibility")
.value("Public", mlir::SymbolTable::Visibility::Public)
.value("Private", mlir::SymbolTable::Visibility::Private)
.value("Nested", mlir::SymbolTable::Visibility::Nested)
;

auto mlir_SymbolTable_SymbolUse = nb::class_<mlir::SymbolTable::SymbolUse>(mlir_SymbolTable, "SymbolUse")
.def(nb::init<mlir::Operation *, mlir::SymbolRefAttr>(), "op"_a, "symbol_ref"_a)
.def_prop_ro("user", &mlir::SymbolTable::SymbolUse::getUser)
.def_prop_ro("symbol_ref", &mlir::SymbolTable::SymbolUse::getSymbolRef)
;

auto mlir_SymbolTable_UseRange = nb::class_<mlir::SymbolTable::UseRange>(mlir_SymbolTable, "UseRange")
.def(nb::init<std::vector<mlir::SymbolTable::SymbolUse> &&>(), "uses"_a)
.def("begin", &mlir::SymbolTable::UseRange::begin)
.def("end", &mlir::SymbolTable::UseRange::end)
.def("empty", &mlir::SymbolTable::UseRange::empty)
;

auto mlir_LockedSymbolTableCollection = nb::class_<mlir::LockedSymbolTableCollection>(m, "LockedSymbolTableCollection")
.def(nb::init<mlir::SymbolTableCollection &>(), "collection"_a)
.def("lookup_symbol_in", [](mlir::LockedSymbolTableCollection& self, mlir::Operation * symbolTableOp, mlir::StringAttr symbol){ return self.lookupSymbolIn(symbolTableOp, symbol); }, "symbol_table_op"_a, "symbol"_a, nb::rv_policy::reference_internal)
.def("lookup_symbol_in", [](mlir::LockedSymbolTableCollection& self, mlir::Operation * symbolTableOp, mlir::FlatSymbolRefAttr symbol){ return self.lookupSymbolIn(symbolTableOp, symbol); }, "symbol_table_op"_a, "symbol"_a, nb::rv_policy::reference_internal)
.def("lookup_symbol_in", [](mlir::LockedSymbolTableCollection& self, mlir::Operation * symbolTableOp, mlir::SymbolRefAttr name){ return self.lookupSymbolIn(symbolTableOp, name); }, "symbol_table_op"_a, "name"_a, nb::rv_policy::reference_internal)
.def("lookup_symbol_in", [](mlir::LockedSymbolTableCollection& self, mlir::Operation * symbolTableOp, mlir::SymbolRefAttr name, llvm::SmallVectorImpl<mlir::Operation *> & symbols){ return self.lookupSymbolIn(symbolTableOp, name, symbols); }, "symbol_table_op"_a, "name"_a, "symbols"_a)
;

auto mlir_SymbolUserMap = nb::class_<mlir::SymbolUserMap>(m, "SymbolUserMap")
.def(nb::init<mlir::SymbolTableCollection &, mlir::Operation *>(), "symbol_table"_a, "symbol_table_op"_a)
.def("get_users", &mlir::SymbolUserMap::getUsers, "symbol"_a)
.def("use_empty", &mlir::SymbolUserMap::useEmpty, "symbol"_a)
.def("replace_all_uses_with", &mlir::SymbolUserMap::replaceAllUsesWith, "symbol"_a, "new_symbol_name"_a)
;

auto mlir_detail_SymbolOpInterfaceInterfaceTraits = nb::class_<mlir::detail::SymbolOpInterfaceInterfaceTraits>(m, "SymbolOpInterfaceInterfaceTraits")
;

auto mlir_detail_SymbolOpInterfaceInterfaceTraits_Concept = nb::class_<mlir::detail::SymbolOpInterfaceInterfaceTraits::Concept>(mlir_detail_SymbolOpInterfaceInterfaceTraits, "Concept")
;

auto mlir_SymbolOpInterface = nb::class_<mlir::SymbolOpInterface>(m, "SymbolOpInterface")
.def_prop_ro("name_attr", &mlir::SymbolOpInterface::getNameAttr)
.def("set_name", [](mlir::SymbolOpInterface& self, mlir::StringAttr name){ return self.setName(name); }, "name"_a)
.def_prop_ro("visibility", &mlir::SymbolOpInterface::getVisibility)
.def("is_nested", &mlir::SymbolOpInterface::isNested)
.def("is_private", &mlir::SymbolOpInterface::isPrivate)
.def("is_public", &mlir::SymbolOpInterface::isPublic)
.def("set_visibility", &mlir::SymbolOpInterface::setVisibility, "vis"_a)
.def("set_nested", &mlir::SymbolOpInterface::setNested)
.def("set_private", &mlir::SymbolOpInterface::setPrivate)
.def("set_public", &mlir::SymbolOpInterface::setPublic)
.def("get_symbol_uses", &mlir::SymbolOpInterface::getSymbolUses, "from_"_a)
.def("symbol_known_use_empty", &mlir::SymbolOpInterface::symbolKnownUseEmpty, "from_"_a)
.def("replace_all_symbol_uses", &mlir::SymbolOpInterface::replaceAllSymbolUses, "new_symbol"_a, "from_"_a)
.def("is_optional_symbol", &mlir::SymbolOpInterface::isOptionalSymbol)
.def("can_discard_on_use_empty", &mlir::SymbolOpInterface::canDiscardOnUseEmpty)
.def("is_declaration", &mlir::SymbolOpInterface::isDeclaration)
.def_prop_ro("name", &mlir::SymbolOpInterface::getName)
.def("set_name", [](mlir::SymbolOpInterface& self, llvm::StringRef name){ return self.setName(name); }, "name"_a)
.def_static("classof", &mlir::SymbolOpInterface::classof, "base"_a)
;

auto mlir_detail_SymbolUserOpInterfaceInterfaceTraits = nb::class_<mlir::detail::SymbolUserOpInterfaceInterfaceTraits>(m, "SymbolUserOpInterfaceInterfaceTraits")
;

auto mlir_detail_SymbolUserOpInterfaceInterfaceTraits_Concept = nb::class_<mlir::detail::SymbolUserOpInterfaceInterfaceTraits::Concept>(mlir_detail_SymbolUserOpInterfaceInterfaceTraits, "Concept")
;

auto mlir_SymbolUserOpInterface = nb::class_<mlir::SymbolUserOpInterface>(m, "SymbolUserOpInterface")
.def("verify_symbol_uses", &mlir::SymbolUserOpInterface::verifySymbolUses, "symbol_table"_a)
;

auto mlir_detail_ModuleOpGenericAdaptorBase = nb::class_<mlir::detail::ModuleOpGenericAdaptorBase>(m, "ModuleOpGenericAdaptorBase")
.def(nb::init<mlir::DictionaryAttr, const mlir::detail::ModuleOpGenericAdaptorBase::Properties &, mlir::RegionRange>(), "attrs"_a, "properties"_a, "regions"_a)
.def(nb::init<mlir::ModuleOp>(), "op"_a)
.def("get_ods_operand_index_and_length", &mlir::detail::ModuleOpGenericAdaptorBase::getODSOperandIndexAndLength, "index"_a, "ods_operands_size"_a)
.def_prop_ro("properties", &mlir::detail::ModuleOpGenericAdaptorBase::getProperties)
.def_prop_ro("attributes", &mlir::detail::ModuleOpGenericAdaptorBase::getAttributes)
.def_prop_ro("sym_name_attr", &mlir::detail::ModuleOpGenericAdaptorBase::getSymNameAttr)
.def_prop_ro("sym_name", &mlir::detail::ModuleOpGenericAdaptorBase::getSymName)
.def_prop_ro("sym_visibility_attr", &mlir::detail::ModuleOpGenericAdaptorBase::getSymVisibilityAttr)
.def_prop_ro("sym_visibility", &mlir::detail::ModuleOpGenericAdaptorBase::getSymVisibility)
.def_prop_ro("body_region", &mlir::detail::ModuleOpGenericAdaptorBase::getBodyRegion)
.def_prop_ro("regions", &mlir::detail::ModuleOpGenericAdaptorBase::getRegions)
;

auto mlir_detail_ModuleOpGenericAdaptorBase_Properties = nb::class_<mlir::detail::ModuleOpGenericAdaptorBase::Properties>(mlir_detail_ModuleOpGenericAdaptorBase, "Properties")
.def_prop_ro("sym_name", &mlir::detail::ModuleOpGenericAdaptorBase::Properties::getSymName)
.def("set_sym_name", &mlir::detail::ModuleOpGenericAdaptorBase::Properties::setSymName, "prop_value"_a)
.def_prop_ro("sym_visibility", &mlir::detail::ModuleOpGenericAdaptorBase::Properties::getSymVisibility)
.def("set_sym_visibility", &mlir::detail::ModuleOpGenericAdaptorBase::Properties::setSymVisibility, "prop_value"_a)
.def("__eq__", &mlir::detail::ModuleOpGenericAdaptorBase::Properties::operator==, "rhs"_a)
.def("__ne__", &mlir::detail::ModuleOpGenericAdaptorBase::Properties::operator!=, "rhs"_a)
;

auto mlir_ModuleOpAdaptor = nb::class_<mlir::ModuleOpAdaptor>(m, "ModuleOpAdaptor")
.def(nb::init<mlir::ModuleOp>(), "op"_a)
.def("verify", &mlir::ModuleOpAdaptor::verify, "loc"_a)
;

auto mlir_ModuleOp = nb::class_<mlir::ModuleOp,  mlir::OpState>(m, "ModuleOp")
.def_static("attribute_names", &mlir::ModuleOp::getAttributeNames)
.def_prop_ro("sym_name_attr_name", [](mlir::ModuleOp& self){ return self.getSymNameAttrName(); })
.def_static("get_sym_name_attr_name", [](mlir::OperationName name){ return mlir::ModuleOp::getSymNameAttrName(name); }, "name"_a)
.def_prop_ro("sym_visibility_attr_name", [](mlir::ModuleOp& self){ return self.getSymVisibilityAttrName(); })
.def_static("get_sym_visibility_attr_name", [](mlir::OperationName name){ return mlir::ModuleOp::getSymVisibilityAttrName(name); }, "name"_a)
.def_static("operation_name", &mlir::ModuleOp::getOperationName)
.def("get_ods_operand_index_and_length", &mlir::ModuleOp::getODSOperandIndexAndLength, "index"_a)
.def("get_ods_operands", &mlir::ModuleOp::getODSOperands, "index"_a)
.def("get_ods_result_index_and_length", &mlir::ModuleOp::getODSResultIndexAndLength, "index"_a)
.def("get_ods_results", &mlir::ModuleOp::getODSResults, "index"_a)
.def_prop_ro("body_region", &mlir::ModuleOp::getBodyRegion)
.def_static("set_properties_from_attr", &mlir::ModuleOp::setPropertiesFromAttr, "prop"_a, "attr"_a, "emit_error"_a)
.def_static("get_properties_as_attr", &mlir::ModuleOp::getPropertiesAsAttr, "ctx"_a, "prop"_a)
.def_static("compute_properties_hash", &mlir::ModuleOp::computePropertiesHash, "prop"_a)
.def_static("get_inherent_attr", &mlir::ModuleOp::getInherentAttr, "ctx"_a, "prop"_a, "name"_a)
.def_static("set_inherent_attr", &mlir::ModuleOp::setInherentAttr, "prop"_a, "name"_a, "value"_a)
.def_static("populate_inherent_attrs", &mlir::ModuleOp::populateInherentAttrs, "ctx"_a, "prop"_a, "attrs"_a)
.def_static("verify_inherent_attrs", &mlir::ModuleOp::verifyInherentAttrs, "op_name"_a, "attrs"_a, "emit_error"_a)
.def_static("read_properties", &mlir::ModuleOp::readProperties, "reader"_a, "state"_a)
.def("write_properties", &mlir::ModuleOp::writeProperties, "writer"_a)
.def_prop_ro("sym_name_attr", &mlir::ModuleOp::getSymNameAttr)
.def_prop_ro("sym_name", &mlir::ModuleOp::getSymName)
.def_prop_ro("sym_visibility_attr", &mlir::ModuleOp::getSymVisibilityAttr)
.def_prop_ro("sym_visibility", &mlir::ModuleOp::getSymVisibility)
.def("set_sym_name_attr", &mlir::ModuleOp::setSymNameAttr, "attr"_a)
.def("set_sym_name", &mlir::ModuleOp::setSymName, "attr_value"_a)
.def("set_sym_visibility_attr", &mlir::ModuleOp::setSymVisibilityAttr, "attr"_a)
.def("set_sym_visibility", &mlir::ModuleOp::setSymVisibility, "attr_value"_a)
.def("remove_sym_name_attr", &mlir::ModuleOp::removeSymNameAttr)
.def("remove_sym_visibility_attr", &mlir::ModuleOp::removeSymVisibilityAttr)
.def_static("build", &mlir::ModuleOp::build, "ods_builder"_a, "ods_state"_a, "name"_a)
.def("verify_invariants_impl", &mlir::ModuleOp::verifyInvariantsImpl)
.def("verify_invariants", &mlir::ModuleOp::verifyInvariants)
.def("verify", &mlir::ModuleOp::verify)
.def_static("parse", &mlir::ModuleOp::parse, "parser"_a, "result"_a)
.def_static("create", &mlir::ModuleOp::create, "loc"_a, "name"_a)
.def_prop_ro("name", &mlir::ModuleOp::getName)
.def("is_optional_symbol", &mlir::ModuleOp::isOptionalSymbol)
.def_prop_ro("data_layout_spec", &mlir::ModuleOp::getDataLayoutSpec)
.def_prop_ro("tar_system_spec", &mlir::ModuleOp::getTargetSystemSpec)
.def_static("default_dialect", &mlir::ModuleOp::getDefaultDialect)
;

auto mlir_detail_TypeIDResolver___mlir_ModuleOp__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::ModuleOp>>(m, "TypeIDResolver[ModuleOp]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::ModuleOp>::resolveTypeID)
;

auto mlir_detail_UnrealizedConversionCastOpGenericAdaptorBase = nb::class_<mlir::detail::UnrealizedConversionCastOpGenericAdaptorBase>(m, "UnrealizedConversionCastOpGenericAdaptorBase")
.def(nb::init<mlir::DictionaryAttr, const mlir::EmptyProperties &, mlir::RegionRange>(), "attrs"_a, "properties"_a, "regions"_a)
.def(nb::init<mlir::Operation *>(), "op"_a)
.def("get_ods_operand_index_and_length", &mlir::detail::UnrealizedConversionCastOpGenericAdaptorBase::getODSOperandIndexAndLength, "index"_a, "ods_operands_size"_a)
.def_prop_ro("attributes", &mlir::detail::UnrealizedConversionCastOpGenericAdaptorBase::getAttributes)
;

auto mlir_UnrealizedConversionCastOpAdaptor = nb::class_<mlir::UnrealizedConversionCastOpAdaptor>(m, "UnrealizedConversionCastOpAdaptor")
.def(nb::init<mlir::UnrealizedConversionCastOp>(), "op"_a)
.def("verify", &mlir::UnrealizedConversionCastOpAdaptor::verify, "loc"_a)
;

auto mlir_UnrealizedConversionCastOp = nb::class_<mlir::UnrealizedConversionCastOp,  mlir::OpState>(m, "UnrealizedConversionCastOp")
.def_static("attribute_names", &mlir::UnrealizedConversionCastOp::getAttributeNames)
.def_static("operation_name", &mlir::UnrealizedConversionCastOp::getOperationName)
.def("get_ods_operand_index_and_length", &mlir::UnrealizedConversionCastOp::getODSOperandIndexAndLength, "index"_a)
.def("get_ods_operands", &mlir::UnrealizedConversionCastOp::getODSOperands, "index"_a)
.def_prop_ro("inputs", &mlir::UnrealizedConversionCastOp::getInputs)
.def_prop_ro("inputs_mutable", &mlir::UnrealizedConversionCastOp::getInputsMutable)
.def("get_ods_result_index_and_length", &mlir::UnrealizedConversionCastOp::getODSResultIndexAndLength, "index"_a)
.def("get_ods_results", &mlir::UnrealizedConversionCastOp::getODSResults, "index"_a)
.def_prop_ro("outputs", &mlir::UnrealizedConversionCastOp::getOutputs)
.def_static("build", &mlir::UnrealizedConversionCastOp::build, "_"_a, "ods_state"_a, "result_types"_a, "operands"_a, "attributes"_a)
.def("verify_invariants_impl", &mlir::UnrealizedConversionCastOp::verifyInvariantsImpl)
.def("verify_invariants", &mlir::UnrealizedConversionCastOp::verifyInvariants)
.def("verify", &mlir::UnrealizedConversionCastOp::verify)
.def("fold", &mlir::UnrealizedConversionCastOp::fold, "adaptor"_a, "results"_a)
.def_static("parse", &mlir::UnrealizedConversionCastOp::parse, "parser"_a, "result"_a)
.def("get_effects", &mlir::UnrealizedConversionCastOp::getEffects, "effects"_a)
;

auto mlir_detail_TypeIDResolver___mlir_UnrealizedConversionCastOp__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::UnrealizedConversionCastOp>>(m, "TypeIDResolver[UnrealizedConversionCastOp]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::UnrealizedConversionCastOp>::resolveTypeID)
;

auto mlir_DialectAsmPrinter = nb::class_<mlir::DialectAsmPrinter, mlir::AsmPrinter>(m, "DialectAsmPrinter")
;

auto mlir_DialectAsmParser = nb::class_<mlir::DialectAsmParser>(m, "DialectAsmParser")
.def_prop_ro("full_symbol_spec", &mlir::DialectAsmParser::getFullSymbolSpec)
;

auto mlir_FieldParser__std_string__ = nb::class_<mlir::FieldParser<std::string>>(m, "FieldParser[std::string]")
.def_static("parse", &mlir::FieldParser<std::string>::parse, "parser"_a)
;

auto mlir_FieldParser__AffineMap__ = nb::class_<mlir::FieldParser<AffineMap>>(m, "FieldParser[AffineMap]")
.def_static("parse", &mlir::FieldParser<AffineMap>::parse, "parser"_a)
;

auto mlir_DialectResourceBlobManager = nb::class_<mlir::DialectResourceBlobManager>(m, "DialectResourceBlobManager")
.def("lookup", [](mlir::DialectResourceBlobManager& self, llvm::StringRef name){ return self.lookup(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("lookup", [](mlir::DialectResourceBlobManager& self, llvm::StringRef name){ return self.lookup(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("update", &mlir::DialectResourceBlobManager::update, "name"_a, "new_blob"_a)
;

auto mlir_DialectResourceBlobManager_BlobEntry = nb::class_<mlir::DialectResourceBlobManager::BlobEntry>(mlir_DialectResourceBlobManager, "BlobEntry")
.def_prop_ro("key", &mlir::DialectResourceBlobManager::BlobEntry::getKey)
.def_prop_ro("blob", [](mlir::DialectResourceBlobManager::BlobEntry& self){ return self.getBlob(); })
.def_prop_ro("blob", [](mlir::DialectResourceBlobManager::BlobEntry& self){ return self.getBlob(); })
.def("set_blob", &mlir::DialectResourceBlobManager::BlobEntry::setBlob, "new_blob"_a)
;

auto mlir_ResourceBlobManagerDialectInterface = nb::class_<mlir::ResourceBlobManagerDialectInterface>(m, "ResourceBlobManagerDialectInterface")
.def(nb::init<mlir::Dialect *>(), "dialect"_a)
.def_prop_ro("blob_manager", [](mlir::ResourceBlobManagerDialectInterface& self){ return &self.getBlobManager(); })
.def_prop_ro("blob_manager", [](mlir::ResourceBlobManagerDialectInterface& self){ return &self.getBlobManager(); })
.def("set_blob_manager", &mlir::ResourceBlobManagerDialectInterface::setBlobManager, "new_blob_manager"_a)
;

auto mlir_detail_DominanceInfoBase__true__ = nb::class_<mlir::detail::DominanceInfoBase<true>>(m, "DominanceInfoBase[true]")
;

auto mlir_detail_DominanceInfoBase__false__ = nb::class_<mlir::detail::DominanceInfoBase<false>>(m, "DominanceInfoBase[false]")
;

auto mlir_DominanceInfo = nb::class_<mlir::DominanceInfo, mlir::detail::DominanceInfoBase<false>>(m, "DominanceInfo")
.def("properly_dominates", [](mlir::DominanceInfo& self, mlir::Operation * a, mlir::Operation * b, bool enclosingOpOk){ return self.properlyDominates(a, b, enclosingOpOk); }, "a"_a, "b"_a, "enclosing_op_ok"_a)
.def("dominates", [](mlir::DominanceInfo& self, mlir::Operation * a, mlir::Operation * b){ return self.dominates(a, b); }, "a"_a, "b"_a)
.def("properly_dominates", [](mlir::DominanceInfo& self, mlir::Value a, mlir::Operation * b){ return self.properlyDominates(a, b); }, "a"_a, "b"_a)
.def("dominates", [](mlir::DominanceInfo& self, mlir::Value a, mlir::Operation * b){ return self.dominates(a, b); }, "a"_a, "b"_a)
.def("dominates", [](mlir::DominanceInfo& self, mlir::Block * a, mlir::Block * b){ return self.dominates(a, b); }, "a"_a, "b"_a)
.def("properly_dominates", [](mlir::DominanceInfo& self, mlir::Block * a, mlir::Block * b){ return self.properlyDominates(a, b); }, "a"_a, "b"_a)
;

auto mlir_PostDominanceInfo = nb::class_<mlir::PostDominanceInfo, mlir::detail::DominanceInfoBase<true>>(m, "PostDominanceInfo")
.def("properly_post_dominates", [](mlir::PostDominanceInfo& self, mlir::Operation * a, mlir::Operation * b, bool enclosingOpOk){ return self.properlyPostDominates(a, b, enclosingOpOk); }, "a"_a, "b"_a, "enclosing_op_ok"_a)
.def("post_dominates", [](mlir::PostDominanceInfo& self, mlir::Operation * a, mlir::Operation * b){ return self.postDominates(a, b); }, "a"_a, "b"_a)
.def("properly_post_dominates", [](mlir::PostDominanceInfo& self, mlir::Block * a, mlir::Block * b){ return self.properlyPostDominates(a, b); }, "a"_a, "b"_a)
.def("post_dominates", [](mlir::PostDominanceInfo& self, mlir::Block * a, mlir::Block * b){ return self.postDominates(a, b); }, "a"_a, "b"_a)
;

auto mlir_DynamicAttrDefinition = nb::class_<mlir::DynamicAttrDefinition>(m, "DynamicAttrDefinition")
.def_static("get", [](llvm::StringRef name, mlir::ExtensibleDialect * dialect, llvm::unique_function<llvm::LogicalResult (llvm::function_ref<mlir::InFlightDiagnostic ()>, llvm::ArrayRef<mlir::Attribute>) const> && verifier){ return mlir::DynamicAttrDefinition::get(name, dialect, std::move(verifier)); }, "name"_a, "dialect"_a, "verifier"_a)
.def_static("get", [](llvm::StringRef name, mlir::ExtensibleDialect * dialect, llvm::unique_function<llvm::LogicalResult (llvm::function_ref<mlir::InFlightDiagnostic ()>, llvm::ArrayRef<mlir::Attribute>) const> && verifier, llvm::unique_function<llvm::ParseResult (mlir::AsmParser &, llvm::SmallVectorImpl<mlir::Attribute> &) const> && parser, llvm::unique_function<void (mlir::AsmPrinter &, llvm::ArrayRef<mlir::Attribute>) const> && printer){ return mlir::DynamicAttrDefinition::get(name, dialect, std::move(verifier), std::move(parser), std::move(printer)); }, "name"_a, "dialect"_a, "verifier"_a, "parser"_a, "printer"_a)
.def("set_verify_fn", &mlir::DynamicAttrDefinition::setVerifyFn, "verify"_a)
.def("set_parse_fn", &mlir::DynamicAttrDefinition::setParseFn, "parse"_a)
.def("set_print_fn", &mlir::DynamicAttrDefinition::setPrintFn, "print"_a)
.def("verify", &mlir::DynamicAttrDefinition::verify, "emit_error"_a, "params"_a)
.def_prop_ro("context", &mlir::DynamicAttrDefinition::getContext)
.def_prop_ro("name", &mlir::DynamicAttrDefinition::getName)
.def_prop_ro("dialect", &mlir::DynamicAttrDefinition::getDialect)
;

auto mlir_DynamicAttr = nb::class_<mlir::DynamicAttr,  mlir::Attribute>(m, "DynamicAttr")
.def_static("get", &mlir::DynamicAttr::get, "attr_def"_a, "params"_a)
.def_prop_ro("attr_def", &mlir::DynamicAttr::getAttrDef)
.def_prop_ro("params", &mlir::DynamicAttr::getParams)
.def_static("isa", &mlir::DynamicAttr::isa, "attr"_a, "attr_def"_a)
.def_static("classof", &mlir::DynamicAttr::classof, "attr"_a)
.def_static("parse", &mlir::DynamicAttr::parse, "parser"_a, "attr_def"_a, "parsed_attr"_a)
.def("print", &mlir::DynamicAttr::print, "printer"_a)
;

auto mlir_DynamicTypeDefinition = nb::class_<mlir::DynamicTypeDefinition>(m, "DynamicTypeDefinition")
.def_static("get", [](llvm::StringRef name, mlir::ExtensibleDialect * dialect, llvm::unique_function<llvm::LogicalResult (llvm::function_ref<mlir::InFlightDiagnostic ()>, llvm::ArrayRef<mlir::Attribute>) const> && verifier){ return mlir::DynamicTypeDefinition::get(name, dialect, std::move(verifier)); }, "name"_a, "dialect"_a, "verifier"_a)
.def_static("get", [](llvm::StringRef name, mlir::ExtensibleDialect * dialect, llvm::unique_function<llvm::LogicalResult (llvm::function_ref<mlir::InFlightDiagnostic ()>, llvm::ArrayRef<mlir::Attribute>) const> && verifier, llvm::unique_function<llvm::ParseResult (mlir::AsmParser &, llvm::SmallVectorImpl<mlir::Attribute> &) const> && parser, llvm::unique_function<void (mlir::AsmPrinter &, llvm::ArrayRef<mlir::Attribute>) const> && printer){ return mlir::DynamicTypeDefinition::get(name, dialect, std::move(verifier), std::move(parser), std::move(printer)); }, "name"_a, "dialect"_a, "verifier"_a, "parser"_a, "printer"_a)
.def("set_verify_fn", &mlir::DynamicTypeDefinition::setVerifyFn, "verify"_a)
.def("set_parse_fn", &mlir::DynamicTypeDefinition::setParseFn, "parse"_a)
.def("set_print_fn", &mlir::DynamicTypeDefinition::setPrintFn, "print"_a)
.def("verify", &mlir::DynamicTypeDefinition::verify, "emit_error"_a, "params"_a)
.def_prop_ro("context", &mlir::DynamicTypeDefinition::getContext)
.def_prop_ro("name", &mlir::DynamicTypeDefinition::getName)
.def_prop_ro("dialect", &mlir::DynamicTypeDefinition::getDialect)
;

auto mlir_DynamicType = nb::class_<mlir::DynamicType,  mlir::Type>(m, "DynamicType")
.def_static("get", &mlir::DynamicType::get, "type_def"_a, "params"_a)
.def_prop_ro("type_def", &mlir::DynamicType::getTypeDef)
.def_prop_ro("params", &mlir::DynamicType::getParams)
.def_static("isa", &mlir::DynamicType::isa, "type"_a, "type_def"_a)
.def_static("classof", &mlir::DynamicType::classof, "type"_a)
.def_static("parse", &mlir::DynamicType::parse, "parser"_a, "type_def"_a, "parsed_type"_a)
.def("print", &mlir::DynamicType::print, "printer"_a)
;

auto mlir_DynamicOpDefinition = nb::class_<mlir::DynamicOpDefinition>(m, "DynamicOpDefinition")
.def_static("get", [](llvm::StringRef name, mlir::ExtensibleDialect * dialect, llvm::unique_function<llvm::LogicalResult (mlir::Operation *) const> && verifyFn, llvm::unique_function<llvm::LogicalResult (mlir::Operation *) const> && verifyRegionFn){ return mlir::DynamicOpDefinition::get(name, dialect, std::move(verifyFn), std::move(verifyRegionFn)); }, "name"_a, "dialect"_a, "verify_fn"_a, "verify_region_fn"_a)
.def_static("get", [](llvm::StringRef name, mlir::ExtensibleDialect * dialect, llvm::unique_function<llvm::LogicalResult (mlir::Operation *) const> && verifyFn, llvm::unique_function<llvm::LogicalResult (mlir::Operation *) const> && verifyRegionFn, llvm::unique_function<llvm::ParseResult (mlir::OpAsmParser &, mlir::OperationState &)> && parseFn, llvm::unique_function<void (mlir::Operation *, mlir::OpAsmPrinter &, llvm::StringRef) const> && printFn){ return mlir::DynamicOpDefinition::get(name, dialect, std::move(verifyFn), std::move(verifyRegionFn), std::move(parseFn), std::move(printFn)); }, "name"_a, "dialect"_a, "verify_fn"_a, "verify_region_fn"_a, "parse_fn"_a, "print_fn"_a)
.def_static("get", [](llvm::StringRef name, mlir::ExtensibleDialect * dialect, llvm::unique_function<llvm::LogicalResult (mlir::Operation *) const> && verifyFn, llvm::unique_function<llvm::LogicalResult (mlir::Operation *) const> && verifyRegionFn, llvm::unique_function<llvm::ParseResult (mlir::OpAsmParser &, mlir::OperationState &)> && parseFn, llvm::unique_function<void (mlir::Operation *, mlir::OpAsmPrinter &, llvm::StringRef) const> && printFn, llvm::unique_function<llvm::LogicalResult (mlir::Operation *, llvm::ArrayRef<mlir::Attribute>, llvm::SmallVectorImpl<mlir::OpFoldResult> &) const> && foldHookFn, llvm::unique_function<void (mlir::RewritePatternSet &, mlir::MLIRContext *) const> && getCanonicalizationPatternsFn, llvm::unique_function<void (const mlir::OperationName &, mlir::NamedAttrList &) const> && populateDefaultAttrsFn){ return mlir::DynamicOpDefinition::get(name, dialect, std::move(verifyFn), std::move(verifyRegionFn), std::move(parseFn), std::move(printFn), std::move(foldHookFn), std::move(getCanonicalizationPatternsFn), std::move(populateDefaultAttrsFn)); }, "name"_a, "dialect"_a, "verify_fn"_a, "verify_region_fn"_a, "parse_fn"_a, "print_fn"_a, "fold_hook_fn"_a, "get_canonicalization_patterns_fn"_a, "populate_default_attrs_fn"_a)
.def_prop_ro("type_id", &mlir::DynamicOpDefinition::getTypeID)
.def("set_verify_fn", &mlir::DynamicOpDefinition::setVerifyFn, "verify"_a)
.def("set_verify_region_fn", &mlir::DynamicOpDefinition::setVerifyRegionFn, "verify"_a)
.def("set_parse_fn", &mlir::DynamicOpDefinition::setParseFn, "parse"_a)
.def("set_print_fn", &mlir::DynamicOpDefinition::setPrintFn, "print"_a)
.def("set_fold_hook_fn", &mlir::DynamicOpDefinition::setFoldHookFn, "fold_hook"_a)
.def("set_get_canonicalization_patterns_fn", &mlir::DynamicOpDefinition::setGetCanonicalizationPatternsFn, "get_canonicalization_patterns"_a)
.def("set_populate_default_attrs_fn", &mlir::DynamicOpDefinition::setPopulateDefaultAttrsFn, "populate_default_attrs"_a)
.def("fold_hook", &mlir::DynamicOpDefinition::foldHook, "op"_a, "attrs"_a, "results"_a)
.def("get_canonicalization_patterns", &mlir::DynamicOpDefinition::getCanonicalizationPatterns, "set"_a, "context"_a)
.def("has_trait", &mlir::DynamicOpDefinition::hasTrait, "id"_a)
.def_prop_ro("parse_assembly_fn", &mlir::DynamicOpDefinition::getParseAssemblyFn)
.def("populate_default_attrs", &mlir::DynamicOpDefinition::populateDefaultAttrs, "name"_a, "attrs"_a)
.def("print_assembly", &mlir::DynamicOpDefinition::printAssembly, "op"_a, "printer"_a, "name"_a)
.def("verify_invariants", &mlir::DynamicOpDefinition::verifyInvariants, "op"_a)
.def("verify_region_invariants", &mlir::DynamicOpDefinition::verifyRegionInvariants, "op"_a)
.def("get_inherent_attr", &mlir::DynamicOpDefinition::getInherentAttr, "op"_a, "name"_a)
.def("set_inherent_attr", &mlir::DynamicOpDefinition::setInherentAttr, "op"_a, "name"_a, "value"_a)
.def("populate_inherent_attrs", &mlir::DynamicOpDefinition::populateInherentAttrs, "op"_a, "attrs"_a)
.def("verify_inherent_attrs", &mlir::DynamicOpDefinition::verifyInherentAttrs, "op_name"_a, "attributes"_a, "emit_error"_a)
.def_prop_ro("op_property_byte_size", &mlir::DynamicOpDefinition::getOpPropertyByteSize)
.def("init_properties", &mlir::DynamicOpDefinition::initProperties, "op_name"_a, "storage"_a, "init"_a)
.def("delete_properties", &mlir::DynamicOpDefinition::deleteProperties, "prop"_a)
.def("populate_default_properties", &mlir::DynamicOpDefinition::populateDefaultProperties, "op_name"_a, "properties"_a)
.def("set_properties_from_attr", &mlir::DynamicOpDefinition::setPropertiesFromAttr, "op_name"_a, "properties"_a, "attr"_a, "emit_error"_a)
.def("get_properties_as_attr", &mlir::DynamicOpDefinition::getPropertiesAsAttr, "op"_a)
.def("copy_properties", &mlir::DynamicOpDefinition::copyProperties, "lhs"_a, "rhs"_a)
.def("compare_properties", &mlir::DynamicOpDefinition::compareProperties, "_"_a, "__"_a)
.def("hash_properties", &mlir::DynamicOpDefinition::hashProperties, "prop"_a)
;

auto mlir_ExtensibleDialect = nb::class_<mlir::ExtensibleDialect, mlir::Dialect>(m, "ExtensibleDialect")
.def(nb::init<llvm::StringRef, mlir::MLIRContext *, mlir::TypeID>(), "name"_a, "ctx"_a, "type_id"_a)
.def("register_dynamic_type", &mlir::ExtensibleDialect::registerDynamicType, "type"_a)
.def("register_dynamic_attr", &mlir::ExtensibleDialect::registerDynamicAttr, "attr"_a)
.def("register_dynamic_op", &mlir::ExtensibleDialect::registerDynamicOp, "type"_a)
.def_static("classof", &mlir::ExtensibleDialect::classof, "dialect"_a)
.def("lookup_type_definition", [](mlir::ExtensibleDialect& self, llvm::StringRef name){ return self.lookupTypeDefinition(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("lookup_type_definition", [](mlir::ExtensibleDialect& self, mlir::TypeID id){ return self.lookupTypeDefinition(id); }, "id"_a, nb::rv_policy::reference_internal)
.def("lookup_attr_definition", [](mlir::ExtensibleDialect& self, llvm::StringRef name){ return self.lookupAttrDefinition(name); }, "name"_a, nb::rv_policy::reference_internal)
.def("lookup_attr_definition", [](mlir::ExtensibleDialect& self, mlir::TypeID id){ return self.lookupAttrDefinition(id); }, "id"_a, nb::rv_policy::reference_internal)
;

auto mlir_DynamicDialect = nb::class_<mlir::DynamicDialect>(m, "DynamicDialect")
.def(nb::init<llvm::StringRef, mlir::MLIRContext *>(), "name"_a, "ctx"_a)
.def_prop_ro("type_id", &mlir::DynamicDialect::getTypeID)
.def_static("classof", &mlir::DynamicDialect::classof, "dialect"_a)
.def("parse_type", &mlir::DynamicDialect::parseType, "parser"_a)
.def("print_type", &mlir::DynamicDialect::printType, "type"_a, "printer"_a)
.def("parse_attribute", &mlir::DynamicDialect::parseAttribute, "parser"_a, "type"_a)
.def("print_attribute", &mlir::DynamicDialect::printAttribute, "attr"_a, "printer"_a)
;

auto mlir_IRMapping = nb::class_<mlir::IRMapping>(m, "IRMapping")
.def("map", [](mlir::IRMapping& self, mlir::Value from_, mlir::Value to){ return self.map(from_, to); }, "from_"_a, "to"_a)
.def("map", [](mlir::IRMapping& self, mlir::Block * from_, mlir::Block * to){ return self.map(from_, to); }, "from_"_a, "to"_a)
.def("map", [](mlir::IRMapping& self, mlir::Operation * from_, mlir::Operation * to){ return self.map(from_, to); }, "from_"_a, "to"_a)
.def("clear", &mlir::IRMapping::clear)
.def_prop_ro("value_map", &mlir::IRMapping::getValueMap)
.def_prop_ro("block_map", &mlir::IRMapping::getBlockMap)
.def_prop_ro("operation_map", &mlir::IRMapping::getOperationMap)
;

auto mlir_ImplicitLocOpBuilder = nb::class_<mlir::ImplicitLocOpBuilder, mlir::OpBuilder>(m, "ImplicitLocOpBuilder")
.def_static("at_block_begin", &mlir::ImplicitLocOpBuilder::atBlockBegin, "loc"_a, "block"_a, "listener"_a)
.def_static("at_block_end", &mlir::ImplicitLocOpBuilder::atBlockEnd, "loc"_a, "block"_a, "listener"_a)
.def_static("at_block_terminator", &mlir::ImplicitLocOpBuilder::atBlockTerminator, "loc"_a, "block"_a, "listener"_a)
.def_prop_ro("loc", &mlir::ImplicitLocOpBuilder::getLoc)
.def("set_loc", &mlir::ImplicitLocOpBuilder::setLoc, "loc"_a)
.def("emit_error", &mlir::ImplicitLocOpBuilder::emitError, "message"_a)
.def("emit_warning", &mlir::ImplicitLocOpBuilder::emitWarning, "message"_a)
.def("emit_remark", &mlir::ImplicitLocOpBuilder::emitRemark, "message"_a)
;

auto mlir_IntegerSet = nb::class_<mlir::IntegerSet>(m, "IntegerSet")
.def(nb::init<>())
.def_static("get", &mlir::IntegerSet::get, "dim_count"_a, "symbol_count"_a, "constraints"_a, "eq_flags"_a)
.def_static("get_empty_set", &mlir::IntegerSet::getEmptySet, "num_dims"_a, "num_symbols"_a, "context"_a)
.def("is_empty_integer_set", &mlir::IntegerSet::isEmptyIntegerSet)
.def("replace_dims_and_symbols", &mlir::IntegerSet::replaceDimsAndSymbols, "dim_replacements"_a, "sym_replacements"_a, "num_result_dims"_a, "num_result_syms"_a)
.def("__eq__", &mlir::IntegerSet::operator==, "other"_a)
.def("__ne__", &mlir::IntegerSet::operator!=, "other"_a)
.def_prop_ro("num_dims", &mlir::IntegerSet::getNumDims)
.def_prop_ro("num_symbols", &mlir::IntegerSet::getNumSymbols)
.def_prop_ro("num_inputs", &mlir::IntegerSet::getNumInputs)
.def_prop_ro("num_constraints", &mlir::IntegerSet::getNumConstraints)
.def_prop_ro("num_equalities", &mlir::IntegerSet::getNumEqualities)
.def_prop_ro("num_inequalities", &mlir::IntegerSet::getNumInequalities)
.def_prop_ro("constraints", &mlir::IntegerSet::getConstraints)
.def("get_constraint", &mlir::IntegerSet::getConstraint, "idx"_a)
.def_prop_ro("eq_flags", &mlir::IntegerSet::getEqFlags)
.def("is_eq", &mlir::IntegerSet::isEq, "idx"_a)
.def_prop_ro("context", &mlir::IntegerSet::getContext)
.def("walk_exprs", &mlir::IntegerSet::walkExprs, "callback"_a)
.def("print", &mlir::IntegerSet::print, "os"_a)
.def("dump", &mlir::IntegerSet::dump)
;

auto mlir_ReverseIterator = nb::class_<mlir::ReverseIterator>(m, "ReverseIterator")
;

auto mlir_detail_constant_op_matcher = nb::class_<mlir::detail::constant_op_matcher>(m, "constant_op_matcher")
.def("match", &mlir::detail::constant_op_matcher::match, "op"_a)
;

auto mlir_detail_NameOpMatcher = nb::class_<mlir::detail::NameOpMatcher>(m, "NameOpMatcher")
.def(nb::init<llvm::StringRef>(), "name"_a)
.def("match", &mlir::detail::NameOpMatcher::match, "op"_a)
;

auto mlir_detail_AttrOpMatcher = nb::class_<mlir::detail::AttrOpMatcher>(m, "AttrOpMatcher")
.def(nb::init<llvm::StringRef>(), "attr_name"_a)
.def("match", &mlir::detail::AttrOpMatcher::match, "op"_a)
;

auto mlir_detail_infer_int_range_op_binder = nb::class_<mlir::detail::infer_int_range_op_binder>(m, "infer_int_range_op_binder")
.def(nb::init<mlir::IntegerValueRange *>(), "bind_value"_a)
.def("match", &mlir::detail::infer_int_range_op_binder::match, "op"_a)
;

auto mlir_detail_constant_float_value_binder = nb::class_<mlir::detail::constant_float_value_binder>(m, "constant_float_value_binder")
.def(nb::init<llvm::APFloat *>(), "bv"_a)
.def("match", [](mlir::detail::constant_float_value_binder& self, mlir::Attribute attr){ return self.match(attr); }, "attr"_a)
.def("match", [](mlir::detail::constant_float_value_binder& self, mlir::Operation * op){ return self.match(op); }, "op"_a)
;

auto mlir_detail_constant_float_predicate_matcher = nb::class_<mlir::detail::constant_float_predicate_matcher>(m, "constant_float_predicate_matcher")
.def("match", [](mlir::detail::constant_float_predicate_matcher& self, mlir::Attribute attr){ return self.match(attr); }, "attr"_a)
.def("match", [](mlir::detail::constant_float_predicate_matcher& self, mlir::Operation * op){ return self.match(op); }, "op"_a)
;

auto mlir_detail_constant_int_value_binder = nb::class_<mlir::detail::constant_int_value_binder>(m, "constant_int_value_binder")
.def(nb::init<llvm::APInt *>(), "bv"_a)
.def("match", [](mlir::detail::constant_int_value_binder& self, mlir::Attribute attr){ return self.match(attr); }, "attr"_a)
.def("match", [](mlir::detail::constant_int_value_binder& self, mlir::Operation * op){ return self.match(op); }, "op"_a)
;

auto mlir_detail_constant_int_predicate_matcher = nb::class_<mlir::detail::constant_int_predicate_matcher>(m, "constant_int_predicate_matcher")
.def("match", [](mlir::detail::constant_int_predicate_matcher& self, mlir::Attribute attr){ return self.match(attr); }, "attr"_a)
.def("match", [](mlir::detail::constant_int_predicate_matcher& self, mlir::Operation * op){ return self.match(op); }, "op"_a)
;

auto mlir_detail_constant_int_range_predicate_matcher = nb::class_<mlir::detail::constant_int_range_predicate_matcher>(m, "constant_int_range_predicate_matcher")
.def("match", [](mlir::detail::constant_int_range_predicate_matcher& self, mlir::Attribute attr){ return self.match(attr); }, "attr"_a)
.def("match", [](mlir::detail::constant_int_range_predicate_matcher& self, mlir::Operation * op){ return self.match(op); }, "op"_a)
;

auto mlir_detail_AnyValueMatcher = nb::class_<mlir::detail::AnyValueMatcher>(m, "AnyValueMatcher")
.def("match", &mlir::detail::AnyValueMatcher::match, "op"_a)
;

auto mlir_detail_AnyCapturedValueMatcher = nb::class_<mlir::detail::AnyCapturedValueMatcher>(m, "AnyCapturedValueMatcher")
.def(nb::init<mlir::Value *>(), "what"_a)
.def("match", &mlir::detail::AnyCapturedValueMatcher::match, "op"_a)
;

auto mlir_detail_PatternMatcherValue = nb::class_<mlir::detail::PatternMatcherValue>(m, "PatternMatcherValue")
.def(nb::init<mlir::Value>(), "val"_a)
.def("match", &mlir::detail::PatternMatcherValue::match, "val"_a)
;

auto mlir_PatternBenefit = nb::class_<mlir::PatternBenefit>(m, "PatternBenefit")
.def(nb::init<>())
.def(nb::init<unsigned int>(), "benefit"_a)
.def(nb::init<const mlir::PatternBenefit &>(), "_"_a)
.def_static("impossible_to_match", &mlir::PatternBenefit::impossibleToMatch)
.def("is_impossible_to_match", &mlir::PatternBenefit::isImpossibleToMatch)
.def_prop_ro("benefit", &mlir::PatternBenefit::getBenefit)
.def("__eq__", &mlir::PatternBenefit::operator==, "rhs"_a)
.def("__ne__", &mlir::PatternBenefit::operator!=, "rhs"_a)
.def("__lt__", &mlir::PatternBenefit::operator<, "rhs"_a)
.def("__gt__", &mlir::PatternBenefit::operator>, "rhs"_a)
.def("__le__", &mlir::PatternBenefit::operator<=, "rhs"_a)
.def("__ge__", &mlir::PatternBenefit::operator>=, "rhs"_a)
;

auto mlir_Pattern = nb::class_<mlir::Pattern>(m, "Pattern")
.def_prop_ro("generated_ops", &mlir::Pattern::getGeneratedOps)
.def_prop_ro("root_kind", &mlir::Pattern::getRootKind)
.def_prop_ro("root_interface_id", &mlir::Pattern::getRootInterfaceID)
.def_prop_ro("root_trait_id", &mlir::Pattern::getRootTraitID)
.def_prop_ro("benefit", &mlir::Pattern::getBenefit)
.def("has_bounded_rewrite_recursion", &mlir::Pattern::hasBoundedRewriteRecursion)
.def_prop_ro("context", &mlir::Pattern::getContext)
.def_prop_ro("debug_name", &mlir::Pattern::getDebugName)
.def("set_debug_name", &mlir::Pattern::setDebugName, "name"_a)
.def_prop_ro("debug_labels", &mlir::Pattern::getDebugLabels)
.def("add_debug_labels", [](mlir::Pattern& self, llvm::ArrayRef<llvm::StringRef> labels){ return self.addDebugLabels(labels); }, "labels"_a)
.def("add_debug_labels", [](mlir::Pattern& self, llvm::StringRef label){ return self.addDebugLabels(label); }, "label"_a)
;

auto mlir_RewritePattern = nb::class_<mlir::RewritePattern, mlir::Pattern>(m, "RewritePattern")
.def("rewrite", &mlir::RewritePattern::rewrite, "op"_a, "rewriter"_a)
.def("match", &mlir::RewritePattern::match, "op"_a)
.def("match_and_rewrite", &mlir::RewritePattern::matchAndRewrite, "op"_a, "rewriter"_a)
;

auto mlir_RewriterBase = nb::class_<mlir::RewriterBase, mlir::OpBuilder>(m, "RewriterBase")
.def("inline_region_before", [](mlir::RewriterBase& self, mlir::Region & region, mlir::Region & parent, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Block, true, false, void, false, void>, false, false> before){ return self.inlineRegionBefore(region, parent, before); }, "region"_a, "parent"_a, "before"_a)
.def("inline_region_before", [](mlir::RewriterBase& self, mlir::Region & region, mlir::Block * before){ return self.inlineRegionBefore(region, before); }, "region"_a, "before"_a)
.def("replace_op", [](mlir::RewriterBase& self, mlir::Operation * op, mlir::ValueRange newValues){ return self.replaceOp(op, newValues); }, "op"_a, "new_values"_a)
.def("replace_op", [](mlir::RewriterBase& self, mlir::Operation * op, mlir::Operation * newOp){ return self.replaceOp(op, newOp); }, "op"_a, "new_op"_a)
.def("erase_op", &mlir::RewriterBase::eraseOp, "op"_a)
.def("erase_block", &mlir::RewriterBase::eraseBlock, "block"_a)
.def("inline_block_before", [](mlir::RewriterBase& self, mlir::Block * source, mlir::Block * dest, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false> before, mlir::ValueRange argValues){ return self.inlineBlockBefore(source, dest, before, argValues); }, "source"_a, "dest"_a, "before"_a, "arg_values"_a)
.def("inline_block_before", [](mlir::RewriterBase& self, mlir::Block * source, mlir::Operation * op, mlir::ValueRange argValues){ return self.inlineBlockBefore(source, op, argValues); }, "source"_a, "op"_a, "arg_values"_a)
.def("merge_blocks", &mlir::RewriterBase::mergeBlocks, "source"_a, "dest"_a, "arg_values"_a)
.def("split_block", &mlir::RewriterBase::splitBlock, "block"_a, "before"_a, nb::rv_policy::reference_internal)
.def("move_op_before", [](mlir::RewriterBase& self, mlir::Operation * op, mlir::Operation * existingOp){ return self.moveOpBefore(op, existingOp); }, "op"_a, "existing_op"_a)
.def("move_op_before", [](mlir::RewriterBase& self, mlir::Operation * op, mlir::Block * block, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false> iterator){ return self.moveOpBefore(op, block, iterator); }, "op"_a, "block"_a, "iterator"_a)
.def("move_op_after", [](mlir::RewriterBase& self, mlir::Operation * op, mlir::Operation * existingOp){ return self.moveOpAfter(op, existingOp); }, "op"_a, "existing_op"_a)
.def("move_op_after", [](mlir::RewriterBase& self, mlir::Operation * op, mlir::Block * block, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Operation, true, false, void, false, void>, false, false> iterator){ return self.moveOpAfter(op, block, iterator); }, "op"_a, "block"_a, "iterator"_a)
.def("move_block_before", [](mlir::RewriterBase& self, mlir::Block * block, mlir::Block * anotherBlock){ return self.moveBlockBefore(block, anotherBlock); }, "block"_a, "another_block"_a)
.def("move_block_before", [](mlir::RewriterBase& self, mlir::Block * block, mlir::Region * region, llvm::ilist_iterator<llvm::ilist_detail::node_options<mlir::Block, true, false, void, false, void>, false, false> iterator){ return self.moveBlockBefore(block, region, iterator); }, "block"_a, "region"_a, "iterator"_a)
.def("start_op_modification", &mlir::RewriterBase::startOpModification, "op"_a)
.def("finalize_op_modification", &mlir::RewriterBase::finalizeOpModification, "op"_a)
.def("cancel_op_modification", &mlir::RewriterBase::cancelOpModification, "op"_a)
.def("replace_all_uses_with", [](mlir::RewriterBase& self, mlir::Value from_, mlir::Value to){ return self.replaceAllUsesWith(from_, to); }, "from_"_a, "to"_a)
.def("replace_all_uses_with", [](mlir::RewriterBase& self, mlir::Block * from_, mlir::Block * to){ return self.replaceAllUsesWith(from_, to); }, "from_"_a, "to"_a)
.def("replace_all_uses_with", [](mlir::RewriterBase& self, mlir::ValueRange from_, mlir::ValueRange to){ return self.replaceAllUsesWith(from_, to); }, "from_"_a, "to"_a)
.def("replace_all_op_uses_with", [](mlir::RewriterBase& self, mlir::Operation * from_, mlir::ValueRange to){ return self.replaceAllOpUsesWith(from_, to); }, "from_"_a, "to"_a)
.def("replace_all_op_uses_with", [](mlir::RewriterBase& self, mlir::Operation * from_, mlir::Operation * to){ return self.replaceAllOpUsesWith(from_, to); }, "from_"_a, "to"_a)
.def("replace_uses_with_if", [](mlir::RewriterBase& self, mlir::Value from_, mlir::Value to, llvm::function_ref<bool (mlir::OpOperand &)> functor, bool * allUsesReplaced){ return self.replaceUsesWithIf(from_, to, functor, allUsesReplaced); }, "from_"_a, "to"_a, "functor"_a, "all_uses_replaced"_a)
.def("replace_uses_with_if", [](mlir::RewriterBase& self, mlir::ValueRange from_, mlir::ValueRange to, llvm::function_ref<bool (mlir::OpOperand &)> functor, bool * allUsesReplaced){ return self.replaceUsesWithIf(from_, to, functor, allUsesReplaced); }, "from_"_a, "to"_a, "functor"_a, "all_uses_replaced"_a)
.def("replace_op_uses_with_if", &mlir::RewriterBase::replaceOpUsesWithIf, "from_"_a, "to"_a, "functor"_a, "all_uses_replaced"_a)
.def("replace_op_uses_within_block", &mlir::RewriterBase::replaceOpUsesWithinBlock, "op"_a, "new_values"_a, "block"_a, "all_uses_replaced"_a)
.def("replace_all_uses_except", [](mlir::RewriterBase& self, mlir::Value from_, mlir::Value to, mlir::Operation * exceptedUser){ return self.replaceAllUsesExcept(from_, to, exceptedUser); }, "from_"_a, "to"_a, "excepted_user"_a)
.def("replace_all_uses_except", [](mlir::RewriterBase& self, mlir::Value from_, mlir::Value to, const llvm::SmallPtrSetImpl<mlir::Operation *> & preservedUsers){ return self.replaceAllUsesExcept(from_, to, preservedUsers); }, "from_"_a, "to"_a, "preserved_users"_a)
;

auto mlir_RewriterBase_Listener = nb::class_<mlir::RewriterBase::Listener, mlir::OpBuilder::Listener>(mlir_RewriterBase, "Listener")
.def(nb::init<>())
.def("notify_block_erased", &mlir::RewriterBase::Listener::notifyBlockErased, "block"_a)
.def("notify_operation_modified", &mlir::RewriterBase::Listener::notifyOperationModified, "op"_a)
.def("notify_operation_replaced", [](mlir::RewriterBase::Listener& self, mlir::Operation * op, mlir::Operation * replacement){ return self.notifyOperationReplaced(op, replacement); }, "op"_a, "replacement"_a)
.def("notify_operation_replaced", [](mlir::RewriterBase::Listener& self, mlir::Operation * op, mlir::ValueRange replacement){ return self.notifyOperationReplaced(op, replacement); }, "op"_a, "replacement"_a)
.def("notify_operation_erased", &mlir::RewriterBase::Listener::notifyOperationErased, "op"_a)
.def("notify_pattern_begin", &mlir::RewriterBase::Listener::notifyPatternBegin, "pattern"_a, "op"_a)
.def("notify_pattern_end", &mlir::RewriterBase::Listener::notifyPatternEnd, "pattern"_a, "status"_a)
.def("notify_match_failure", &mlir::RewriterBase::Listener::notifyMatchFailure, "loc"_a, "reason_callback"_a)
.def_static("classof", &mlir::RewriterBase::Listener::classof, "base"_a)
;

auto mlir_RewriterBase_ForwardingListener = nb::class_<mlir::RewriterBase::ForwardingListener, mlir::RewriterBase::Listener>(mlir_RewriterBase, "ForwardingListener")
.def(nb::init<mlir::OpBuilder::Listener *>(), "listener"_a)
.def("notify_operation_inserted", &mlir::RewriterBase::ForwardingListener::notifyOperationInserted, "op"_a, "previous"_a)
.def("notify_block_inserted", &mlir::RewriterBase::ForwardingListener::notifyBlockInserted, "block"_a, "previous"_a, "previous_it"_a)
.def("notify_block_erased", &mlir::RewriterBase::ForwardingListener::notifyBlockErased, "block"_a)
.def("notify_operation_modified", &mlir::RewriterBase::ForwardingListener::notifyOperationModified, "op"_a)
.def("notify_operation_replaced", [](mlir::RewriterBase::ForwardingListener& self, mlir::Operation * op, mlir::Operation * newOp){ return self.notifyOperationReplaced(op, newOp); }, "op"_a, "new_op"_a)
.def("notify_operation_replaced", [](mlir::RewriterBase::ForwardingListener& self, mlir::Operation * op, mlir::ValueRange replacement){ return self.notifyOperationReplaced(op, replacement); }, "op"_a, "replacement"_a)
.def("notify_operation_erased", &mlir::RewriterBase::ForwardingListener::notifyOperationErased, "op"_a)
.def("notify_pattern_begin", &mlir::RewriterBase::ForwardingListener::notifyPatternBegin, "pattern"_a, "op"_a)
.def("notify_pattern_end", &mlir::RewriterBase::ForwardingListener::notifyPatternEnd, "pattern"_a, "status"_a)
.def("notify_match_failure", &mlir::RewriterBase::ForwardingListener::notifyMatchFailure, "loc"_a, "reason_callback"_a)
;

auto mlir_IRRewriter = nb::class_<mlir::IRRewriter, mlir::RewriterBase>(m, "IRRewriter")
.def(nb::init<mlir::MLIRContext *, mlir::OpBuilder::Listener *>(), "ctx"_a, "listener"_a)
.def(nb::init<const mlir::OpBuilder &>(), "builder"_a)
.def(nb::init<mlir::Operation *, mlir::OpBuilder::Listener *>(), "op"_a, "listener"_a)
;

auto mlir_PatternRewriter = nb::class_<mlir::PatternRewriter, mlir::RewriterBase>(m, "PatternRewriter")
.def(nb::init<mlir::MLIRContext *>(), "ctx"_a)
.def("can_recover_from_rewrite_failure", &mlir::PatternRewriter::canRecoverFromRewriteFailure)
;

auto mlir_PDLValue = nb::class_<mlir::PDLValue>(m, "PDLValue")
.def(nb::init<const mlir::PDLValue &>(), "other"_a)
.def(nb::init<std::nullptr_t>(), "_"_a)
.def(nb::init<mlir::Attribute>(), "value"_a)
.def(nb::init<mlir::Operation *>(), "value"_a)
.def(nb::init<mlir::Type>(), "value"_a)
.def(nb::init<mlir::TypeRange *>(), "value"_a)
.def(nb::init<mlir::Value>(), "value"_a)
.def(nb::init<mlir::ValueRange *>(), "value"_a)
.def_prop_ro("kind", &mlir::PDLValue::getKind)
.def("print", [](mlir::PDLValue& self, llvm::raw_ostream & os){ return self.print(os); }, "os"_a)
.def_static("print_static", [](llvm::raw_ostream & os, mlir::PDLValue::Kind kind){ return mlir::PDLValue::print(os, kind); }, "os"_a, "kind"_a)
;

nb::enum_<mlir::PDLValue::Kind>(m, "Kind")
.value("Attribute", mlir::PDLValue::Kind::Attribute)
.value("Operation", mlir::PDLValue::Kind::Operation)
.value("Type", mlir::PDLValue::Kind::Type)
.value("TypeRange", mlir::PDLValue::Kind::TypeRange)
.value("Value", mlir::PDLValue::Kind::Value)
.value("ValueRange", mlir::PDLValue::Kind::ValueRange)
;

auto mlir_PDLPatternConfig = nb::class_<mlir::PDLPatternConfig>(m, "PDLPatternConfig")
.def("notify_rewrite_begin", &mlir::PDLPatternConfig::notifyRewriteBegin, "rewriter"_a)
.def("notify_rewrite_end", &mlir::PDLPatternConfig::notifyRewriteEnd, "rewriter"_a)
.def_prop_ro("type_id", &mlir::PDLPatternConfig::getTypeID)
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__Attribute__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<Attribute>>(m, "ProcessPDLValue[Attribute]")
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__StringRef__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<StringRef>>(m, "ProcessPDLValue[StringRef]")
.def_static("process_as_result", &mlir::detail::pdl_function_builder::ProcessPDLValue<StringRef>::processAsResult, "rewriter"_a, "results"_a, "value"_a)
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__std_string__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<std::string>>(m, "ProcessPDLValue[std::string]")
.def_static("process_as_result", &mlir::detail::pdl_function_builder::ProcessPDLValue<std::string>::processAsResult, "rewriter"_a, "results"_a, "value"_a)
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__Operation_____ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<Operation *>>(m, "ProcessPDLValue[Operation]")
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__Type__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<Type>>(m, "ProcessPDLValue[Type]")
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__TypeRange__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<TypeRange>>(m, "ProcessPDLValue[TypeRange]")
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__ValueTypeRange__OperandRange____ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<ValueTypeRange<OperandRange>>>(m, "ProcessPDLValue[ValueTypeRange[OperandRange]]")
.def_static("process_as_result", &mlir::detail::pdl_function_builder::ProcessPDLValue<ValueTypeRange<OperandRange>>::processAsResult, "_"_a, "results"_a, "types"_a)
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__ValueTypeRange__ResultRange____ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<ValueTypeRange<ResultRange>>>(m, "ProcessPDLValue[ValueTypeRange[ResultRange]]")
.def_static("process_as_result", &mlir::detail::pdl_function_builder::ProcessPDLValue<ValueTypeRange<ResultRange>>::processAsResult, "_"_a, "results"_a, "types"_a)
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__Value__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<Value>>(m, "ProcessPDLValue[Value]")
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__ValueRange__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<ValueRange>>(m, "ProcessPDLValue[ValueRange]")
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__OperandRange__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<OperandRange>>(m, "ProcessPDLValue[OperandRange]")
.def_static("process_as_result", &mlir::detail::pdl_function_builder::ProcessPDLValue<OperandRange>::processAsResult, "_"_a, "results"_a, "values"_a)
;

auto mlir_detail_pdl_function_builder_ProcessPDLValue__ResultRange__ = nb::class_<mlir::detail::pdl_function_builder::ProcessPDLValue<ResultRange>>(m, "ProcessPDLValue[ResultRange]")
.def_static("process_as_result", &mlir::detail::pdl_function_builder::ProcessPDLValue<ResultRange>::processAsResult, "_"_a, "results"_a, "values"_a)
;

auto mlir_PDLPatternModule = nb::class_<mlir::PDLPatternModule>(m, "PDLPatternModule")
.def(nb::init<>())
.def("merge_in", &mlir::PDLPatternModule::mergeIn, "other"_a)
.def_prop_ro("module", &mlir::PDLPatternModule::getModule)
.def_prop_ro("context", &mlir::PDLPatternModule::getContext)
.def("register_constraint_function", [](mlir::PDLPatternModule& self, llvm::StringRef name, std::function<llvm::LogicalResult (mlir::PatternRewriter &, mlir::PDLResultList &, llvm::ArrayRef<mlir::PDLValue>)> constraintFn){ return self.registerConstraintFunction(name, constraintFn); }, "name"_a, "constraint_fn"_a)
.def("register_rewrite_function", [](mlir::PDLPatternModule& self, llvm::StringRef name, std::function<llvm::LogicalResult (mlir::PatternRewriter &, mlir::PDLResultList &, llvm::ArrayRef<mlir::PDLValue>)> rewriteFn){ return self.registerRewriteFunction(name, rewriteFn); }, "name"_a, "rewrite_fn"_a)
.def_prop_ro("constraint_functions", &mlir::PDLPatternModule::getConstraintFunctions)
.def("take_constraint_functions", &mlir::PDLPatternModule::takeConstraintFunctions)
.def_prop_ro("rewrite_functions", &mlir::PDLPatternModule::getRewriteFunctions)
.def("take_rewrite_functions", &mlir::PDLPatternModule::takeRewriteFunctions)
.def("take_configs", &mlir::PDLPatternModule::takeConfigs)
.def("take_config_map", &mlir::PDLPatternModule::takeConfigMap)
.def("clear", &mlir::PDLPatternModule::clear)
;

auto mlir_RewritePatternSet = nb::class_<mlir::RewritePatternSet>(m, "RewritePatternSet")
.def(nb::init<mlir::MLIRContext *>(), "context"_a)
.def(nb::init<mlir::MLIRContext *, std::unique_ptr<mlir::RewritePattern>>(), "context"_a, "pattern"_a)
.def(nb::init<mlir::PDLPatternModule &&>(), "pattern"_a)
.def_prop_ro("context", &mlir::RewritePatternSet::getContext)
.def_prop_ro("native_patterns", &mlir::RewritePatternSet::getNativePatterns)
.def_prop_ro("pdl_patterns", &mlir::RewritePatternSet::getPDLPatterns)
.def("clear", &mlir::RewritePatternSet::clear)
.def("add", [](mlir::RewritePatternSet& self, std::unique_ptr<mlir::RewritePattern> pattern){ return &self.add(std::move(pattern)); }, "pattern"_a, nb::rv_policy::reference_internal)
.def("add", [](mlir::RewritePatternSet& self, mlir::PDLPatternModule && pattern){ return &self.add(std::move(pattern)); }, "pattern"_a, nb::rv_policy::reference_internal)
.def("insert", [](mlir::RewritePatternSet& self, std::unique_ptr<mlir::RewritePattern> pattern){ return &self.insert(std::move(pattern)); }, "pattern"_a, nb::rv_policy::reference_internal)
.def("insert", [](mlir::RewritePatternSet& self, mlir::PDLPatternModule && pattern){ return &self.insert(std::move(pattern)); }, "pattern"_a, nb::rv_policy::reference_internal)
;

auto mlir_detail_VerifiableTensorEncodingInterfaceTraits = nb::class_<mlir::detail::VerifiableTensorEncodingInterfaceTraits>(m, "VerifiableTensorEncodingInterfaceTraits")
;

auto mlir_detail_VerifiableTensorEncodingInterfaceTraits_Concept = nb::class_<mlir::detail::VerifiableTensorEncodingInterfaceTraits::Concept>(mlir_detail_VerifiableTensorEncodingInterfaceTraits, "Concept")
;

auto mlir_VerifiableTensorEncoding = nb::class_<mlir::VerifiableTensorEncoding>(m, "VerifiableTensorEncoding")
.def("verify_encoding", &mlir::VerifiableTensorEncoding::verifyEncoding, "shape"_a, "element_type"_a, "emit_error"_a)
;

auto mlir_OperandElementTypeIterator = nb::class_<mlir::OperandElementTypeIterator>(m, "OperandElementTypeIterator")
.def("map_element", &mlir::OperandElementTypeIterator::mapElement, "value"_a)
;

auto mlir_ResultElementTypeIterator = nb::class_<mlir::ResultElementTypeIterator>(m, "ResultElementTypeIterator")
.def("map_element", &mlir::ResultElementTypeIterator::mapElement, "value"_a)
;
