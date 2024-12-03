
#include "ir.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
namespace nb = nanobind;
using namespace nb::literals;
void populateControlFlowModule(nanobind::module_ & m) {
using namespace mlir;
using namespace mlir::detail;
using namespace mlir::cf;

auto mlir_cf_ControlFlowDialect = nb::class_<mlir::cf::ControlFlowDialect, mlir::Dialect>(m, "ControlFlowDialect")
.def_static("dialect_namespace", &mlir::cf::ControlFlowDialect::getDialectNamespace)
.def_static("insert_into_registry", [](mlir::DialectRegistry &registry) { registry.insert<mlir::cf::ControlFlowDialect>(); })
.def_static("load_into_context", [](mlir::MLIRContext &context) { return context.getOrLoadDialect<mlir::cf::ControlFlowDialect>(); })
;

auto mlir_detail_TypeIDResolver___mlir_cf_ControlFlowDialect__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::cf::ControlFlowDialect>>(m, "TypeIDResolver[cf::ControlFlowDialect]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::cf::ControlFlowDialect>::resolveTypeID)
;

auto mlir_cf_detail_AssertOpGenericAdaptorBase = nb::class_<mlir::cf::detail::AssertOpGenericAdaptorBase>(m, "AssertOpGenericAdaptorBase")
.def(nb::init<mlir::DictionaryAttr, const mlir::cf::detail::AssertOpGenericAdaptorBase::Properties &, mlir::RegionRange>(), "attrs"_a, "properties"_a, "regions"_a)
.def(nb::init<mlir::cf::AssertOp>(), "op"_a)
.def("get_ods_operand_index_and_length", &mlir::cf::detail::AssertOpGenericAdaptorBase::getODSOperandIndexAndLength, "index"_a, "ods_operands_size"_a)
.def_prop_ro("properties", &mlir::cf::detail::AssertOpGenericAdaptorBase::getProperties)
.def_prop_ro("attributes", &mlir::cf::detail::AssertOpGenericAdaptorBase::getAttributes)
.def_prop_ro("msg_attr", &mlir::cf::detail::AssertOpGenericAdaptorBase::getMsgAttr)
.def_prop_ro("msg", &mlir::cf::detail::AssertOpGenericAdaptorBase::getMsg)
;

auto mlir_cf_detail_AssertOpGenericAdaptorBase_Properties = nb::class_<mlir::cf::detail::AssertOpGenericAdaptorBase::Properties>(mlir_cf_detail_AssertOpGenericAdaptorBase, "Properties")
.def_prop_ro("msg", &mlir::cf::detail::AssertOpGenericAdaptorBase::Properties::getMsg)
.def("set_msg", &mlir::cf::detail::AssertOpGenericAdaptorBase::Properties::setMsg, "prop_value"_a)
.def("__eq__", &mlir::cf::detail::AssertOpGenericAdaptorBase::Properties::operator==, "rhs"_a)
.def("__ne__", &mlir::cf::detail::AssertOpGenericAdaptorBase::Properties::operator!=, "rhs"_a)
;

auto mlir_cf_AssertOpAdaptor = nb::class_<mlir::cf::AssertOpAdaptor>(m, "AssertOpAdaptor")
.def(nb::init<mlir::cf::AssertOp>(), "op"_a)
.def("verify", &mlir::cf::AssertOpAdaptor::verify, "loc"_a)
;

auto mlir_cf_AssertOp = nb::class_<mlir::cf::AssertOp,  mlir::OpState>(m, "AssertOp")
.def_static("attribute_names", &mlir::cf::AssertOp::getAttributeNames)
.def_prop_ro("msg_attr_name", [](mlir::cf::AssertOp& self){ return self.getMsgAttrName(); })
.def_static("get_msg_attr_name", [](mlir::OperationName name){ return mlir::cf::AssertOp::getMsgAttrName(name); }, "name"_a)
.def_static("operation_name", &mlir::cf::AssertOp::getOperationName)
.def("get_ods_operand_index_and_length", &mlir::cf::AssertOp::getODSOperandIndexAndLength, "index"_a)
.def("get_ods_operands", &mlir::cf::AssertOp::getODSOperands, "index"_a)
.def_prop_ro("arg", &mlir::cf::AssertOp::getArg)
.def_prop_ro("arg_mutable", &mlir::cf::AssertOp::getArgMutable)
.def("get_ods_result_index_and_length", &mlir::cf::AssertOp::getODSResultIndexAndLength, "index"_a)
.def("get_ods_results", &mlir::cf::AssertOp::getODSResults, "index"_a)
.def_static("set_properties_from_attr", &mlir::cf::AssertOp::setPropertiesFromAttr, "prop"_a, "attr"_a, "emit_error"_a)
.def_static("get_properties_as_attr", &mlir::cf::AssertOp::getPropertiesAsAttr, "ctx"_a, "prop"_a)
.def_static("compute_properties_hash", &mlir::cf::AssertOp::computePropertiesHash, "prop"_a)
.def_static("get_inherent_attr", &mlir::cf::AssertOp::getInherentAttr, "ctx"_a, "prop"_a, "name"_a)
.def_static("set_inherent_attr", &mlir::cf::AssertOp::setInherentAttr, "prop"_a, "name"_a, "value"_a)
.def_static("populate_inherent_attrs", &mlir::cf::AssertOp::populateInherentAttrs, "ctx"_a, "prop"_a, "attrs"_a)
.def_static("verify_inherent_attrs", &mlir::cf::AssertOp::verifyInherentAttrs, "op_name"_a, "attrs"_a, "emit_error"_a)
.def_static("read_properties", &mlir::cf::AssertOp::readProperties, "reader"_a, "state"_a)
.def("write_properties", &mlir::cf::AssertOp::writeProperties, "writer"_a)
.def_prop_ro("msg_attr", &mlir::cf::AssertOp::getMsgAttr)
.def_prop_ro("msg", &mlir::cf::AssertOp::getMsg)
.def("set_msg_attr", &mlir::cf::AssertOp::setMsgAttr, "attr"_a)
.def("set_msg", &mlir::cf::AssertOp::setMsg, "attr_value"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value arg, mlir::StringAttr msg){ return mlir::cf::AssertOp::build(odsBuilder, odsState, arg, msg); }, "ods_builder"_a, "ods_state"_a, "arg"_a, "msg"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::Value arg, mlir::StringAttr msg){ return mlir::cf::AssertOp::build(odsBuilder, odsState, resultTypes, arg, msg); }, "ods_builder"_a, "ods_state"_a, "result_types"_a, "arg"_a, "msg"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value arg, llvm::StringRef msg){ return mlir::cf::AssertOp::build(odsBuilder, odsState, arg, msg); }, "ods_builder"_a, "ods_state"_a, "arg"_a, "msg"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::Value arg, llvm::StringRef msg){ return mlir::cf::AssertOp::build(odsBuilder, odsState, resultTypes, arg, msg); }, "ods_builder"_a, "ods_state"_a, "result_types"_a, "arg"_a, "msg"_a)
.def_static("build", [](mlir::OpBuilder & _, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::ValueRange operands, llvm::ArrayRef<mlir::NamedAttribute> attributes){ return mlir::cf::AssertOp::build(_, odsState, resultTypes, operands, attributes); }, "_"_a, "ods_state"_a, "result_types"_a, "operands"_a, "attributes"_a)
.def("verify_invariants_impl", &mlir::cf::AssertOp::verifyInvariantsImpl)
.def("verify_invariants", &mlir::cf::AssertOp::verifyInvariants)
.def_static("canonicalize", &mlir::cf::AssertOp::canonicalize, "op"_a, "rewriter"_a)
.def_static("get_canonicalization_patterns", &mlir::cf::AssertOp::getCanonicalizationPatterns, "results"_a, "context"_a)
.def("get_effects", &mlir::cf::AssertOp::getEffects, "effects"_a)
.def_static("parse", &mlir::cf::AssertOp::parse, "parser"_a, "result"_a)
;

auto mlir_detail_TypeIDResolver___mlir_cf_AssertOp__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::cf::AssertOp>>(m, "TypeIDResolver[cf::AssertOp]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::cf::AssertOp>::resolveTypeID)
;

auto mlir_cf_detail_BranchOpGenericAdaptorBase = nb::class_<mlir::cf::detail::BranchOpGenericAdaptorBase>(m, "BranchOpGenericAdaptorBase")
.def(nb::init<mlir::DictionaryAttr, const mlir::EmptyProperties &, mlir::RegionRange>(), "attrs"_a, "properties"_a, "regions"_a)
.def(nb::init<mlir::Operation *>(), "op"_a)
.def("get_ods_operand_index_and_length", &mlir::cf::detail::BranchOpGenericAdaptorBase::getODSOperandIndexAndLength, "index"_a, "ods_operands_size"_a)
.def_prop_ro("attributes", &mlir::cf::detail::BranchOpGenericAdaptorBase::getAttributes)
;

auto mlir_cf_BranchOpAdaptor = nb::class_<mlir::cf::BranchOpAdaptor>(m, "BranchOpAdaptor")
.def(nb::init<mlir::cf::BranchOp>(), "op"_a)
.def("verify", &mlir::cf::BranchOpAdaptor::verify, "loc"_a)
;

auto mlir_cf_BranchOp = nb::class_<mlir::cf::BranchOp,  mlir::OpState>(m, "BranchOp")
.def_static("attribute_names", &mlir::cf::BranchOp::getAttributeNames)
.def_static("operation_name", &mlir::cf::BranchOp::getOperationName)
.def("get_ods_operand_index_and_length", &mlir::cf::BranchOp::getODSOperandIndexAndLength, "index"_a)
.def("get_ods_operands", &mlir::cf::BranchOp::getODSOperands, "index"_a)
.def_prop_ro("dest_operands", &mlir::cf::BranchOp::getDestOperands)
.def_prop_ro("dest_operands_mutable", &mlir::cf::BranchOp::getDestOperandsMutable)
.def("get_ods_result_index_and_length", &mlir::cf::BranchOp::getODSResultIndexAndLength, "index"_a)
.def("get_ods_results", &mlir::cf::BranchOp::getODSResults, "index"_a)
.def_prop_ro("dest", &mlir::cf::BranchOp::getDest)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Block * dest, mlir::ValueRange destOperands){ return mlir::cf::BranchOp::build(odsBuilder, odsState, dest, destOperands); }, "ods_builder"_a, "ods_state"_a, "dest"_a, "dest_operands"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::ValueRange destOperands, mlir::Block * dest){ return mlir::cf::BranchOp::build(odsBuilder, odsState, destOperands, dest); }, "ods_builder"_a, "ods_state"_a, "dest_operands"_a, "dest"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::ValueRange destOperands, mlir::Block * dest){ return mlir::cf::BranchOp::build(odsBuilder, odsState, resultTypes, destOperands, dest); }, "ods_builder"_a, "ods_state"_a, "result_types"_a, "dest_operands"_a, "dest"_a)
.def_static("build", [](mlir::OpBuilder & _, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::ValueRange operands, llvm::ArrayRef<mlir::NamedAttribute> attributes){ return mlir::cf::BranchOp::build(_, odsState, resultTypes, operands, attributes); }, "_"_a, "ods_state"_a, "result_types"_a, "operands"_a, "attributes"_a)
.def("verify_invariants_impl", &mlir::cf::BranchOp::verifyInvariantsImpl)
.def("verify_invariants", &mlir::cf::BranchOp::verifyInvariants)
.def_static("canonicalize", &mlir::cf::BranchOp::canonicalize, "op"_a, "rewriter"_a)
.def_static("get_canonicalization_patterns", &mlir::cf::BranchOp::getCanonicalizationPatterns, "results"_a, "context"_a)
.def("get_successor_operands", &mlir::cf::BranchOp::getSuccessorOperands, "index"_a)
.def("get_successor_for_operands", &mlir::cf::BranchOp::getSuccessorForOperands, "operands"_a, nb::rv_policy::reference_internal)
.def_static("parse", &mlir::cf::BranchOp::parse, "parser"_a, "result"_a)
.def("get_effects", &mlir::cf::BranchOp::getEffects, "effects"_a)
.def("set_dest", &mlir::cf::BranchOp::setDest, "block"_a)
.def("erase_operand", &mlir::cf::BranchOp::eraseOperand, "index"_a)
;

auto mlir_detail_TypeIDResolver___mlir_cf_BranchOp__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::cf::BranchOp>>(m, "TypeIDResolver[cf::BranchOp]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::cf::BranchOp>::resolveTypeID)
;

auto mlir_cf_detail_CondBranchOpGenericAdaptorBase = nb::class_<mlir::cf::detail::CondBranchOpGenericAdaptorBase>(m, "CondBranchOpGenericAdaptorBase")
.def(nb::init<mlir::DictionaryAttr, const mlir::cf::detail::CondBranchOpGenericAdaptorBase::Properties &, mlir::RegionRange>(), "attrs"_a, "properties"_a, "regions"_a)
.def(nb::init<mlir::cf::CondBranchOp>(), "op"_a)
.def("get_ods_operand_index_and_length", &mlir::cf::detail::CondBranchOpGenericAdaptorBase::getODSOperandIndexAndLength, "index"_a, "ods_operands_size"_a)
.def_prop_ro("properties", &mlir::cf::detail::CondBranchOpGenericAdaptorBase::getProperties)
.def_prop_ro("attributes", &mlir::cf::detail::CondBranchOpGenericAdaptorBase::getAttributes)
;

auto mlir_cf_detail_CondBranchOpGenericAdaptorBase_Properties = nb::class_<mlir::cf::detail::CondBranchOpGenericAdaptorBase::Properties>(mlir_cf_detail_CondBranchOpGenericAdaptorBase, "Properties")
.def_prop_ro("operand_segment_sizes", &mlir::cf::detail::CondBranchOpGenericAdaptorBase::Properties::getOperandSegmentSizes)
.def("set_operand_segment_sizes", &mlir::cf::detail::CondBranchOpGenericAdaptorBase::Properties::setOperandSegmentSizes, "prop_value"_a)
.def("__eq__", &mlir::cf::detail::CondBranchOpGenericAdaptorBase::Properties::operator==, "rhs"_a)
.def("__ne__", &mlir::cf::detail::CondBranchOpGenericAdaptorBase::Properties::operator!=, "rhs"_a)
;

auto mlir_cf_CondBranchOpAdaptor = nb::class_<mlir::cf::CondBranchOpAdaptor>(m, "CondBranchOpAdaptor")
.def(nb::init<mlir::cf::CondBranchOp>(), "op"_a)
.def("verify", &mlir::cf::CondBranchOpAdaptor::verify, "loc"_a)
;

auto mlir_cf_CondBranchOp = nb::class_<mlir::cf::CondBranchOp,  mlir::OpState>(m, "CondBranchOp")
.def_static("attribute_names", &mlir::cf::CondBranchOp::getAttributeNames)
.def_prop_ro("operand_segment_sizes_attr_name", [](mlir::cf::CondBranchOp& self){ return self.getOperandSegmentSizesAttrName(); })
.def_static("get_operand_segment_sizes_attr_name", [](mlir::OperationName name){ return mlir::cf::CondBranchOp::getOperandSegmentSizesAttrName(name); }, "name"_a)
.def_static("operation_name", &mlir::cf::CondBranchOp::getOperationName)
.def("get_ods_operand_index_and_length", &mlir::cf::CondBranchOp::getODSOperandIndexAndLength, "index"_a)
.def("get_ods_operands", &mlir::cf::CondBranchOp::getODSOperands, "index"_a)
.def_prop_ro("condition", &mlir::cf::CondBranchOp::getCondition)
.def_prop_ro("true_dest_operands", &mlir::cf::CondBranchOp::getTrueDestOperands)
.def_prop_ro("false_dest_operands", &mlir::cf::CondBranchOp::getFalseDestOperands)
.def_prop_ro("condition_mutable", &mlir::cf::CondBranchOp::getConditionMutable)
.def_prop_ro("true_dest_operands_mutable", &mlir::cf::CondBranchOp::getTrueDestOperandsMutable)
.def_prop_ro("false_dest_operands_mutable", &mlir::cf::CondBranchOp::getFalseDestOperandsMutable)
.def("get_ods_result_index_and_length", &mlir::cf::CondBranchOp::getODSResultIndexAndLength, "index"_a)
.def("get_ods_results", &mlir::cf::CondBranchOp::getODSResults, "index"_a)
.def_prop_ro("true_dest", &mlir::cf::CondBranchOp::getTrueDest)
.def_prop_ro("false_dest", &mlir::cf::CondBranchOp::getFalseDest)
.def_static("set_properties_from_attr", &mlir::cf::CondBranchOp::setPropertiesFromAttr, "prop"_a, "attr"_a, "emit_error"_a)
.def_static("get_properties_as_attr", &mlir::cf::CondBranchOp::getPropertiesAsAttr, "ctx"_a, "prop"_a)
.def_static("compute_properties_hash", &mlir::cf::CondBranchOp::computePropertiesHash, "prop"_a)
.def_static("get_inherent_attr", &mlir::cf::CondBranchOp::getInherentAttr, "ctx"_a, "prop"_a, "name"_a)
.def_static("set_inherent_attr", &mlir::cf::CondBranchOp::setInherentAttr, "prop"_a, "name"_a, "value"_a)
.def_static("populate_inherent_attrs", &mlir::cf::CondBranchOp::populateInherentAttrs, "ctx"_a, "prop"_a, "attrs"_a)
.def_static("verify_inherent_attrs", &mlir::cf::CondBranchOp::verifyInherentAttrs, "op_name"_a, "attrs"_a, "emit_error"_a)
.def_static("read_properties", &mlir::cf::CondBranchOp::readProperties, "reader"_a, "state"_a)
.def("write_properties", &mlir::cf::CondBranchOp::writeProperties, "writer"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value condition, mlir::Block * trueDest, mlir::ValueRange trueOperands, mlir::Block * falseDest, mlir::ValueRange falseOperands){ return mlir::cf::CondBranchOp::build(odsBuilder, odsState, condition, trueDest, trueOperands, falseDest, falseOperands); }, "ods_builder"_a, "ods_state"_a, "condition"_a, "true_dest"_a, "true_operands"_a, "false_dest"_a, "false_operands"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value condition, mlir::Block * trueDest, mlir::Block * falseDest, mlir::ValueRange falseOperands){ return mlir::cf::CondBranchOp::build(odsBuilder, odsState, condition, trueDest, falseDest, falseOperands); }, "ods_builder"_a, "ods_state"_a, "condition"_a, "true_dest"_a, "false_dest"_a, "false_operands"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value condition, mlir::ValueRange trueDestOperands, mlir::ValueRange falseDestOperands, mlir::Block * trueDest, mlir::Block * falseDest){ return mlir::cf::CondBranchOp::build(odsBuilder, odsState, condition, trueDestOperands, falseDestOperands, trueDest, falseDest); }, "ods_builder"_a, "ods_state"_a, "condition"_a, "true_dest_operands"_a, "false_dest_operands"_a, "true_dest"_a, "false_dest"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::Value condition, mlir::ValueRange trueDestOperands, mlir::ValueRange falseDestOperands, mlir::Block * trueDest, mlir::Block * falseDest){ return mlir::cf::CondBranchOp::build(odsBuilder, odsState, resultTypes, condition, trueDestOperands, falseDestOperands, trueDest, falseDest); }, "ods_builder"_a, "ods_state"_a, "result_types"_a, "condition"_a, "true_dest_operands"_a, "false_dest_operands"_a, "true_dest"_a, "false_dest"_a)
.def_static("build", [](mlir::OpBuilder & _, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::ValueRange operands, llvm::ArrayRef<mlir::NamedAttribute> attributes){ return mlir::cf::CondBranchOp::build(_, odsState, resultTypes, operands, attributes); }, "_"_a, "ods_state"_a, "result_types"_a, "operands"_a, "attributes"_a)
.def("verify_invariants_impl", &mlir::cf::CondBranchOp::verifyInvariantsImpl)
.def("verify_invariants", &mlir::cf::CondBranchOp::verifyInvariants)
.def_static("get_canonicalization_patterns", &mlir::cf::CondBranchOp::getCanonicalizationPatterns, "results"_a, "context"_a)
.def("get_successor_operands", &mlir::cf::CondBranchOp::getSuccessorOperands, "index"_a)
.def("get_successor_for_operands", &mlir::cf::CondBranchOp::getSuccessorForOperands, "operands"_a, nb::rv_policy::reference_internal)
.def_static("parse", &mlir::cf::CondBranchOp::parse, "parser"_a, "result"_a)
.def("get_effects", &mlir::cf::CondBranchOp::getEffects, "effects"_a)
.def("get_true_operand", &mlir::cf::CondBranchOp::getTrueOperand, "idx"_a)
.def("set_true_operand", &mlir::cf::CondBranchOp::setTrueOperand, "idx"_a, "value"_a)
.def_prop_ro("num_true_operands", &mlir::cf::CondBranchOp::getNumTrueOperands)
.def("erase_true_operand", &mlir::cf::CondBranchOp::eraseTrueOperand, "index"_a)
.def("get_false_operand", &mlir::cf::CondBranchOp::getFalseOperand, "idx"_a)
.def("set_false_operand", &mlir::cf::CondBranchOp::setFalseOperand, "idx"_a, "value"_a)
.def_prop_ro("true_operands", &mlir::cf::CondBranchOp::getTrueOperands)
.def_prop_ro("false_operands", &mlir::cf::CondBranchOp::getFalseOperands)
.def_prop_ro("num_false_operands", &mlir::cf::CondBranchOp::getNumFalseOperands)
.def("erase_false_operand", &mlir::cf::CondBranchOp::eraseFalseOperand, "index"_a)
;

auto mlir_detail_TypeIDResolver___mlir_cf_CondBranchOp__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::cf::CondBranchOp>>(m, "TypeIDResolver[cf::CondBranchOp]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::cf::CondBranchOp>::resolveTypeID)
;

auto mlir_cf_detail_SwitchOpGenericAdaptorBase = nb::class_<mlir::cf::detail::SwitchOpGenericAdaptorBase>(m, "SwitchOpGenericAdaptorBase")
.def(nb::init<mlir::DictionaryAttr, const mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties &, mlir::RegionRange>(), "attrs"_a, "properties"_a, "regions"_a)
.def(nb::init<mlir::cf::SwitchOp>(), "op"_a)
.def("get_ods_operand_index_and_length", &mlir::cf::detail::SwitchOpGenericAdaptorBase::getODSOperandIndexAndLength, "index"_a, "ods_operands_size"_a)
.def_prop_ro("properties", &mlir::cf::detail::SwitchOpGenericAdaptorBase::getProperties)
.def_prop_ro("attributes", &mlir::cf::detail::SwitchOpGenericAdaptorBase::getAttributes)
.def_prop_ro("case_values_attr", &mlir::cf::detail::SwitchOpGenericAdaptorBase::getCaseValuesAttr)
.def_prop_ro("case_values", &mlir::cf::detail::SwitchOpGenericAdaptorBase::getCaseValues)
.def_prop_ro("case_operand_segments_attr", &mlir::cf::detail::SwitchOpGenericAdaptorBase::getCaseOperandSegmentsAttr)
.def_prop_ro("case_operand_segments", &mlir::cf::detail::SwitchOpGenericAdaptorBase::getCaseOperandSegments)
;

auto mlir_cf_detail_SwitchOpGenericAdaptorBase_Properties = nb::class_<mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties>(mlir_cf_detail_SwitchOpGenericAdaptorBase, "Properties")
.def_prop_ro("case_operand_segments", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::getCaseOperandSegments)
.def("set_case_operand_segments", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::setCaseOperandSegments, "prop_value"_a)
.def_prop_ro("case_values", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::getCaseValues)
.def("set_case_values", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::setCaseValues, "prop_value"_a)
.def_prop_ro("operand_segment_sizes", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::getOperandSegmentSizes)
.def("set_operand_segment_sizes", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::setOperandSegmentSizes, "prop_value"_a)
.def("__eq__", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::operator==, "rhs"_a)
.def("__ne__", &mlir::cf::detail::SwitchOpGenericAdaptorBase::Properties::operator!=, "rhs"_a)
;

auto mlir_cf_SwitchOpAdaptor = nb::class_<mlir::cf::SwitchOpAdaptor>(m, "SwitchOpAdaptor")
.def(nb::init<mlir::cf::SwitchOp>(), "op"_a)
.def("verify", &mlir::cf::SwitchOpAdaptor::verify, "loc"_a)
;

auto mlir_cf_SwitchOp = nb::class_<mlir::cf::SwitchOp,  mlir::OpState>(m, "SwitchOp")
.def_static("attribute_names", &mlir::cf::SwitchOp::getAttributeNames)
.def_prop_ro("case_operand_segments_attr_name", [](mlir::cf::SwitchOp& self){ return self.getCaseOperandSegmentsAttrName(); })
.def_static("get_case_operand_segments_attr_name", [](mlir::OperationName name){ return mlir::cf::SwitchOp::getCaseOperandSegmentsAttrName(name); }, "name"_a)
.def_prop_ro("case_values_attr_name", [](mlir::cf::SwitchOp& self){ return self.getCaseValuesAttrName(); })
.def_static("get_case_values_attr_name", [](mlir::OperationName name){ return mlir::cf::SwitchOp::getCaseValuesAttrName(name); }, "name"_a)
.def_prop_ro("operand_segment_sizes_attr_name", [](mlir::cf::SwitchOp& self){ return self.getOperandSegmentSizesAttrName(); })
.def_static("get_operand_segment_sizes_attr_name", [](mlir::OperationName name){ return mlir::cf::SwitchOp::getOperandSegmentSizesAttrName(name); }, "name"_a)
.def_static("operation_name", &mlir::cf::SwitchOp::getOperationName)
.def("get_ods_operand_index_and_length", &mlir::cf::SwitchOp::getODSOperandIndexAndLength, "index"_a)
.def("get_ods_operands", &mlir::cf::SwitchOp::getODSOperands, "index"_a)
.def_prop_ro("flag", &mlir::cf::SwitchOp::getFlag)
.def_prop_ro("default_operands", &mlir::cf::SwitchOp::getDefaultOperands)
.def_prop_ro("case_operands", [](mlir::cf::SwitchOp& self){ return self.getCaseOperands(); })
.def_prop_ro("flag_mutable", &mlir::cf::SwitchOp::getFlagMutable)
.def_prop_ro("default_operands_mutable", &mlir::cf::SwitchOp::getDefaultOperandsMutable)
.def_prop_ro("case_operands_mutable", [](mlir::cf::SwitchOp& self){ return self.getCaseOperandsMutable(); })
.def("get_ods_result_index_and_length", &mlir::cf::SwitchOp::getODSResultIndexAndLength, "index"_a)
.def("get_ods_results", &mlir::cf::SwitchOp::getODSResults, "index"_a)
.def_prop_ro("default_destination", &mlir::cf::SwitchOp::getDefaultDestination)
.def_prop_ro("case_destinations", &mlir::cf::SwitchOp::getCaseDestinations)
.def_static("set_properties_from_attr", &mlir::cf::SwitchOp::setPropertiesFromAttr, "prop"_a, "attr"_a, "emit_error"_a)
.def_static("get_properties_as_attr", &mlir::cf::SwitchOp::getPropertiesAsAttr, "ctx"_a, "prop"_a)
.def_static("compute_properties_hash", &mlir::cf::SwitchOp::computePropertiesHash, "prop"_a)
.def_static("get_inherent_attr", &mlir::cf::SwitchOp::getInherentAttr, "ctx"_a, "prop"_a, "name"_a)
.def_static("set_inherent_attr", &mlir::cf::SwitchOp::setInherentAttr, "prop"_a, "name"_a, "value"_a)
.def_static("populate_inherent_attrs", &mlir::cf::SwitchOp::populateInherentAttrs, "ctx"_a, "prop"_a, "attrs"_a)
.def_static("verify_inherent_attrs", &mlir::cf::SwitchOp::verifyInherentAttrs, "op_name"_a, "attrs"_a, "emit_error"_a)
.def_static("read_properties", &mlir::cf::SwitchOp::readProperties, "reader"_a, "state"_a)
.def("write_properties", &mlir::cf::SwitchOp::writeProperties, "writer"_a)
.def_prop_ro("case_values_attr", &mlir::cf::SwitchOp::getCaseValuesAttr)
.def_prop_ro("case_values", &mlir::cf::SwitchOp::getCaseValues)
.def_prop_ro("case_operand_segments_attr", &mlir::cf::SwitchOp::getCaseOperandSegmentsAttr)
.def_prop_ro("case_operand_segments", &mlir::cf::SwitchOp::getCaseOperandSegments)
.def("set_case_values_attr", &mlir::cf::SwitchOp::setCaseValuesAttr, "attr"_a)
.def("set_case_operand_segments_attr", &mlir::cf::SwitchOp::setCaseOperandSegmentsAttr, "attr"_a)
.def("set_case_operand_segments", &mlir::cf::SwitchOp::setCaseOperandSegments, "attr_value"_a)
.def("remove_case_values_attr", &mlir::cf::SwitchOp::removeCaseValuesAttr)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value flag, mlir::Block * defaultDestination, mlir::ValueRange defaultOperands, llvm::ArrayRef<llvm::APInt> caseValues, mlir::BlockRange caseDestinations, llvm::ArrayRef<mlir::ValueRange> caseOperands){ return mlir::cf::SwitchOp::build(odsBuilder, odsState, flag, defaultDestination, defaultOperands, caseValues, caseDestinations, caseOperands); }, "ods_builder"_a, "ods_state"_a, "flag"_a, "default_destination"_a, "default_operands"_a, "case_values"_a, "case_destinations"_a, "case_operands"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value flag, mlir::Block * defaultDestination, mlir::ValueRange defaultOperands, ArrayRef<int32_t> caseValues, mlir::BlockRange caseDestinations, llvm::ArrayRef<mlir::ValueRange> caseOperands){ return mlir::cf::SwitchOp::build(odsBuilder, odsState, flag, defaultDestination, defaultOperands, caseValues, caseDestinations, caseOperands); }, "ods_builder"_a, "ods_state"_a, "flag"_a, "default_destination"_a, "default_operands"_a, "case_values"_a, "case_destinations"_a, "case_operands"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value flag, mlir::Block * defaultDestination, mlir::ValueRange defaultOperands, mlir::DenseIntElementsAttr caseValues, mlir::BlockRange caseDestinations, llvm::ArrayRef<mlir::ValueRange> caseOperands){ return mlir::cf::SwitchOp::build(odsBuilder, odsState, flag, defaultDestination, defaultOperands, caseValues, caseDestinations, caseOperands); }, "ods_builder"_a, "ods_state"_a, "flag"_a, "default_destination"_a, "default_operands"_a, "case_values"_a, "case_destinations"_a, "case_operands"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::Value flag, mlir::ValueRange defaultOperands, llvm::ArrayRef<mlir::ValueRange> caseOperands, mlir::DenseIntElementsAttr case_values, mlir::Block * defaultDestination, mlir::BlockRange caseDestinations){ return mlir::cf::SwitchOp::build(odsBuilder, odsState, flag, defaultOperands, caseOperands, case_values, defaultDestination, caseDestinations); }, "ods_builder"_a, "ods_state"_a, "flag"_a, "default_operands"_a, "case_operands"_a, "case_values"_a, "default_destination"_a, "case_destinations"_a)
.def_static("build", [](mlir::OpBuilder & odsBuilder, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::Value flag, mlir::ValueRange defaultOperands, llvm::ArrayRef<mlir::ValueRange> caseOperands, mlir::DenseIntElementsAttr case_values, mlir::Block * defaultDestination, mlir::BlockRange caseDestinations){ return mlir::cf::SwitchOp::build(odsBuilder, odsState, resultTypes, flag, defaultOperands, caseOperands, case_values, defaultDestination, caseDestinations); }, "ods_builder"_a, "ods_state"_a, "result_types"_a, "flag"_a, "default_operands"_a, "case_operands"_a, "case_values"_a, "default_destination"_a, "case_destinations"_a)
.def_static("build", [](mlir::OpBuilder & _, mlir::OperationState & odsState, mlir::TypeRange resultTypes, mlir::ValueRange operands, llvm::ArrayRef<mlir::NamedAttribute> attributes){ return mlir::cf::SwitchOp::build(_, odsState, resultTypes, operands, attributes); }, "_"_a, "ods_state"_a, "result_types"_a, "operands"_a, "attributes"_a)
.def("verify_invariants_impl", &mlir::cf::SwitchOp::verifyInvariantsImpl)
.def("verify_invariants", &mlir::cf::SwitchOp::verifyInvariants)
.def("verify", &mlir::cf::SwitchOp::verify)
.def_static("get_canonicalization_patterns", &mlir::cf::SwitchOp::getCanonicalizationPatterns, "results"_a, "context"_a)
.def("get_successor_operands", &mlir::cf::SwitchOp::getSuccessorOperands, "index"_a)
.def("get_successor_for_operands", &mlir::cf::SwitchOp::getSuccessorForOperands, "operands"_a, nb::rv_policy::reference_internal)
.def_static("parse", &mlir::cf::SwitchOp::parse, "parser"_a, "result"_a)
.def("get_effects", &mlir::cf::SwitchOp::getEffects, "effects"_a)
.def("get_case_operands", [](mlir::cf::SwitchOp& self, unsigned int index){ return self.getCaseOperands(index); }, "index"_a)
.def("get_case_operands_mutable", [](mlir::cf::SwitchOp& self, unsigned int index){ return self.getCaseOperandsMutable(index); }, "index"_a)
;

auto mlir_detail_TypeIDResolver___mlir_cf_SwitchOp__ = nb::class_<mlir::detail::TypeIDResolver< ::mlir::cf::SwitchOp>>(m, "TypeIDResolver[cf::SwitchOp]")
.def_static("resolve_type_id", &mlir::detail::TypeIDResolver< ::mlir::cf::SwitchOp>::resolveTypeID)
;

}
