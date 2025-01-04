// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2024.

#include "TGParser.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Builder.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Constraint.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pass.h"
#include "mlir/TableGen/Pattern.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/SideEffects.h"
#include "mlir/TableGen/Successor.h"
#include "mlir/TableGen/Trait.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Record.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

using namespace llvm;

namespace nb = nanobind;
using namespace nb::literals;

template <typename NewReturn, typename Return, typename... Args>
constexpr auto coerceReturn(Return (*pf)(Args...)) noexcept {
  return [&pf](Args &&...args) -> NewReturn {
    return pf(std::forward<Args>(args)...);
  };
}

template <typename NewReturn, typename Return, typename Class, typename... Args>
constexpr auto coerceReturn(Return (Class::*pmf)(Args...),
                            std::false_type = {}) noexcept {
  return [&pmf](Class *cls, Args &&...args) -> NewReturn {
    return (cls->*pmf)(std::forward<Args>(args)...);
  };
}

/*
 * If you get
 * ```
 * Called object type 'void(MyClass::*)(vector<Item>&,int)' is not a function or
 * function pointer
 * ```
 * it's because you're calling a member function without
 * passing the `this` pointer as the first arg
 */
template <typename NewReturn, typename Return, typename Class, typename... Args>
constexpr auto coerceReturn(Return (Class::*pmf)(Args...) const,
                            std::true_type) noexcept {
  // copy the *pmf, not capture by ref
  return [pmf](const Class &cls, Args &&...args) -> NewReturn {
    return (cls.*pmf)(std::forward<Args>(args)...);
  };
}

template <>
struct nb::detail::type_caster<StringRef> {
  NB_TYPE_CASTER(StringRef, const_name("str"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    Py_ssize_t size;
    const char *str = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
    if (!str) {
      PyErr_Clear();
      return false;
    }
    value = StringRef(str, (size_t)size);
    return true;
  }

  static handle from_cpp(StringRef value, rv_policy, cleanup_list *) noexcept {
    return PyUnicode_FromStringAndSize(value.data(), value.size());
  }
};

// hack to expose protected Init::InitKind
struct HackInit : public Init {
  using InitKind = Init::InitKind;
};

NB_MODULE(eudsl_tblgen_ext, m) {
  auto recty = nb::class_<RecTy>(m, "RecTy");

  nb::enum_<RecTy::RecTyKind>(m, "RecTyKind")
      .value("BitRecTyKind", RecTy::RecTyKind::BitRecTyKind)
      .value("BitsRecTyKind", RecTy::RecTyKind::BitsRecTyKind)
      .value("IntRecTyKind", RecTy::RecTyKind::IntRecTyKind)
      .value("StringRecTyKind", RecTy::RecTyKind::StringRecTyKind)
      .value("ListRecTyKind", RecTy::RecTyKind::ListRecTyKind)
      .value("DagRecTyKind", RecTy::RecTyKind::DagRecTyKind)
      .value("RecordRecTyKind", RecTy::RecTyKind::RecordRecTyKind);

  recty.def_prop_ro("rec_ty_kind", &RecTy::getRecTyKind)
      .def_prop_ro("record_keeper", &RecTy::getRecordKeeper)
      .def_prop_ro("as_string", &RecTy::getAsString)
      .def("__str__", &RecTy::getAsString)
      .def("type_is_a", &RecTy::typeIsA, "rhs"_a)
      .def("type_is_convertible_to", &RecTy::typeIsConvertibleTo, "rhs"_a);

  nb::class_<RecordRecTy, RecTy>(m, "RecordRecTy")
      .def_prop_ro("classes", coerceReturn<std::vector<const Record *>>(
                                  &RecordRecTy::getClasses, nb::const_))
      .def("is_sub_class_of", &RecordRecTy::isSubClassOf, "class_"_a);

  nb::enum_<HackInit::InitKind>(m, "InitKind")
      .value("IK_FirstTypedInit", HackInit::InitKind::IK_FirstTypedInit)
      .value("IK_BitInit", HackInit::InitKind::IK_BitInit)
      .value("IK_BitsInit", HackInit::InitKind::IK_BitsInit)
      .value("IK_DagInit", HackInit::InitKind::IK_DagInit)
      .value("IK_DefInit", HackInit::InitKind::IK_DefInit)
      .value("IK_FieldInit", HackInit::InitKind::IK_FieldInit)
      .value("IK_IntInit", HackInit::InitKind::IK_IntInit)
      .value("IK_ListInit", HackInit::InitKind::IK_ListInit)
      .value("IK_FirstOpInit", HackInit::InitKind::IK_FirstOpInit)
      .value("IK_BinOpInit", HackInit::InitKind::IK_BinOpInit)
      .value("IK_TernOpInit", HackInit::InitKind::IK_TernOpInit)
      .value("IK_UnOpInit", HackInit::InitKind::IK_UnOpInit)
      .value("IK_LastOpInit", HackInit::InitKind::IK_LastOpInit)
      .value("IK_CondOpInit", HackInit::InitKind::IK_CondOpInit)
      .value("IK_FoldOpInit", HackInit::InitKind::IK_FoldOpInit)
      .value("IK_IsAOpInit", HackInit::InitKind::IK_IsAOpInit)
      .value("IK_ExistsOpInit", HackInit::InitKind::IK_ExistsOpInit)
      .value("IK_AnonymousNameInit", HackInit::InitKind::IK_AnonymousNameInit)
      .value("IK_StringInit", HackInit::InitKind::IK_StringInit)
      .value("IK_VarInit", HackInit::InitKind::IK_VarInit)
      .value("IK_VarBitInit", HackInit::InitKind::IK_VarBitInit)
      .value("IK_VarDefInit", HackInit::InitKind::IK_VarDefInit)
      .value("IK_LastTypedInit", HackInit::InitKind::IK_LastTypedInit)
      .value("IK_UnsetInit", HackInit::InitKind::IK_UnsetInit)
      .value("IK_ArgumentInit", HackInit::InitKind::IK_ArgumentInit);

  nb::class_<Init>(m, "Init")
      .def_prop_ro("kind", &Init::getKind)
      .def_prop_ro("as_string", &Init::getAsUnquotedString)
      .def("__str__", &Init::getAsUnquotedString)
      .def("is_complete", &Init::isComplete)
      .def("is_concrete", &Init::isConcrete)
      .def("get_field_type", &Init::getFieldType, "field_name"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &Init::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  nb::class_<TypedInit, Init>(m, "TypedInit")
      .def_prop_ro("record_keeper", &TypedInit::getRecordKeeper)
      .def_prop_ro("type", &TypedInit::getType);

  nb::class_<UnsetInit, Init>(m, "UnsetInit");

  nb::class_<ArgumentInit, Init>(m, "ArgumentInit")
      .def("is_positional", &ArgumentInit::isPositional)
      .def("is_named", &ArgumentInit::isNamed)
      .def_prop_ro("value", &ArgumentInit::getValue)
      .def_prop_ro("index", &ArgumentInit::getIndex)
      .def_prop_ro("name", &ArgumentInit::getName);

  nb::class_<BitInit, TypedInit>(m, "BitInit")
      .def_prop_ro("value", &BitInit::getValue)
      .def("__bool__", &BitInit::getValue);

  nb::class_<BitsInit, TypedInit>(m, "BitsInit")
      .def_prop_ro("num_bits", &BitsInit::getNumBits)
      .def("all_incomplete", &BitsInit::allInComplete);

  nb::class_<IntInit, TypedInit>(m, "IntInit")
      .def_prop_ro("value", &IntInit::getValue);

  nb::class_<AnonymousNameInit, TypedInit>(m, "AnonymousNameInit")
      .def_prop_ro("value", &AnonymousNameInit::getValue)
      .def_prop_ro("name_init", &AnonymousNameInit::getNameInit);

  auto stringInit = nb::class_<StringInit, TypedInit>(m, "StringInit");

  nb::enum_<StringInit::StringFormat>(m, "StringFormat")
      .value("SF_String", StringInit::StringFormat::SF_String)
      .value("SF_Code", StringInit::StringFormat::SF_Code);

  stringInit.def_prop_ro("value", &StringInit::getValue)
      .def_prop_ro("format", &StringInit::getFormat)
      .def("has_code_format", &StringInit::hasCodeFormat);

  nb::class_<ListInit, TypedInit>(m, "ListInit")
      .def("__len__", [](const ListInit &v) { return v.size(); })
      .def("__bool__", [](const ListInit &v) { return !v.empty(); })
      .def(
          "__iter__",
          [](ListInit &v) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<ListInit>(), "Iterator", v.begin(), v.end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](ListInit &v, Py_ssize_t i) {
            return v.getElement(nb::detail::wrap(i, v.size()));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("element_type", &ListInit::getElementType)
      .def("get_element_as_record", &ListInit::getElementAsRecord, "i"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("values", coerceReturn<std::vector<const Init *>>(
                                 &ListInit::getValues, nb::const_));

  nb::class_<OpInit, TypedInit>(m, "OpInit")
      .def("bit", &OpInit::getBit, "bit"_a, nb::rv_policy::reference_internal);

  auto unaryOpInit = nb::class_<UnOpInit, OpInit>(m, "UnOpInit");
  nb::enum_<UnOpInit::UnaryOp>(m, "UnaryOp")
      .value("TOLOWER", UnOpInit::UnaryOp::TOLOWER)
      .value("TOUPPER", UnOpInit::UnaryOp::TOUPPER)
      .value("CAST", UnOpInit::UnaryOp::CAST)
      .value("NOT", UnOpInit::UnaryOp::NOT)
      .value("HEAD", UnOpInit::UnaryOp::HEAD)
      .value("TAIL", UnOpInit::UnaryOp::TAIL)
      .value("SIZE", UnOpInit::UnaryOp::SIZE)
      .value("EMPTY", UnOpInit::UnaryOp::EMPTY)
      .value("GETDAGOP", UnOpInit::UnaryOp::GETDAGOP)
      .value("LOG2", UnOpInit::UnaryOp::LOG2)
      .value("REPR", UnOpInit::UnaryOp::REPR)
      .value("LISTFLATTEN", UnOpInit::UnaryOp::LISTFLATTEN);
  unaryOpInit.def_prop_ro("opcode", &UnOpInit::getOpcode);

  auto binaryOpInit = nb::class_<BinOpInit, OpInit>(m, "BinOpInit");
  nb::enum_<BinOpInit::BinaryOp>(m, "BinaryOp")
      .value("ADD", BinOpInit::BinaryOp::ADD)
      .value("SUB", BinOpInit::BinaryOp::SUB)
      .value("MUL", BinOpInit::BinaryOp::MUL)
      .value("DIV", BinOpInit::BinaryOp::DIV)
      .value("AND", BinOpInit::BinaryOp::AND)
      .value("OR", BinOpInit::BinaryOp::OR)
      .value("XOR", BinOpInit::BinaryOp::XOR)
      .value("SHL", BinOpInit::BinaryOp::SHL)
      .value("SRA", BinOpInit::BinaryOp::SRA)
      .value("SRL", BinOpInit::BinaryOp::SRL)
      .value("LISTCONCAT", BinOpInit::BinaryOp::LISTCONCAT)
      .value("LISTSPLAT", BinOpInit::BinaryOp::LISTSPLAT)
      .value("LISTREMOVE", BinOpInit::BinaryOp::LISTREMOVE)
      .value("LISTELEM", BinOpInit::BinaryOp::LISTELEM)
      .value("LISTSLICE", BinOpInit::BinaryOp::LISTSLICE)
      .value("RANGEC", BinOpInit::BinaryOp::RANGEC)
      .value("STRCONCAT", BinOpInit::BinaryOp::STRCONCAT)
      .value("INTERLEAVE", BinOpInit::BinaryOp::INTERLEAVE)
      .value("CONCAT", BinOpInit::BinaryOp::CONCAT)
      .value("EQ", BinOpInit::BinaryOp::EQ)
      .value("NE", BinOpInit::BinaryOp::NE)
      .value("LE", BinOpInit::BinaryOp::LE)
      .value("LT", BinOpInit::BinaryOp::LT)
      .value("GE", BinOpInit::BinaryOp::GE)
      .value("GT", BinOpInit::BinaryOp::GT)
      .value("GETDAGARG", BinOpInit::BinaryOp::GETDAGARG)
      .value("GETDAGNAME", BinOpInit::BinaryOp::GETDAGNAME)
      .value("SETDAGOP", BinOpInit::BinaryOp::SETDAGOP);
  binaryOpInit.def_prop_ro("opcode", &BinOpInit::getOpcode)
      .def_prop_ro("lhs", &BinOpInit::getLHS)
      .def_prop_ro("rhs", &BinOpInit::getRHS);

  auto ternaryOpInit = nb::class_<TernOpInit, OpInit>(m, "TernOpInit");
  nb::enum_<TernOpInit::TernaryOp>(m, "TernaryOp")
      .value("SUBST", TernOpInit::TernaryOp::SUBST)
      .value("FOREACH", TernOpInit::TernaryOp::FOREACH)
      .value("FILTER", TernOpInit::TernaryOp::FILTER)
      .value("IF", TernOpInit::TernaryOp::IF)
      .value("DAG", TernOpInit::TernaryOp::DAG)
      .value("RANGE", TernOpInit::TernaryOp::RANGE)
      .value("SUBSTR", TernOpInit::TernaryOp::SUBSTR)
      .value("FIND", TernOpInit::TernaryOp::FIND)
      .value("SETDAGARG", TernOpInit::TernaryOp::SETDAGARG)
      .value("SETDAGNAME", TernOpInit::TernaryOp::SETDAGNAME);
  ternaryOpInit.def_prop_ro("opcode", &TernOpInit::getOpcode)
      .def_prop_ro("lhs", &TernOpInit::getLHS)
      .def_prop_ro("mhs", &TernOpInit::getMHS)
      .def_prop_ro("rhs", &TernOpInit::getRHS);

  nb::class_<CondOpInit, TypedInit>(m, "CondOpInit");
  nb::class_<FoldOpInit, TypedInit>(m, "FoldOpInit");
  nb::class_<IsAOpInit, TypedInit>(m, "IsAOpInit");
  nb::class_<ExistsOpInit, TypedInit>(m, "ExistsOpInit");

  nb::class_<VarInit, TypedInit>(m, "VarInit")
      .def_prop_ro("name", &VarInit::getName)
      .def_prop_ro("name_init", &VarInit::getNameInit)
      .def_prop_ro("name_init_as_string", &VarInit::getNameInitAsString);

  nb::class_<VarBitInit, TypedInit>(m, "VarBitInit")
      .def_prop_ro("bit_var", &VarBitInit::getBitVar)
      .def_prop_ro("bit_num", &VarBitInit::getBitNum);

  nb::class_<DefInit, TypedInit>(m, "DefInit")
      .def_prop_ro("def_", &DefInit::getDef);

  nb::class_<VarDefInit, TypedInit>(m, "VarDefInit")
      .def("get_arg", &VarDefInit::getArg, "i"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro("args", coerceReturn<std::vector<const ArgumentInit *>>(
                               &VarDefInit::args, nb::const_))
      .def("__len__", [](const VarDefInit &v) { return v.args_size(); })
      .def("__bool__", [](const VarDefInit &v) { return !v.args_empty(); })
      .def(
          "__iter__",
          [](VarDefInit &v) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<VarDefInit>(), "Iterator", v.args_begin(),
                v.args_end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](VarDefInit &v, Py_ssize_t i) {
            return v.getArg(nb::detail::wrap(i, v.args_size()));
          },
          nb::rv_policy::reference_internal);

  nb::class_<FieldInit, TypedInit>(m, "FieldInit")
      .def_prop_ro("record", &FieldInit::getRecord)
      .def_prop_ro("field_name", &FieldInit::getFieldName);

  nb::class_<DagInit, TypedInit>(m, "DagInit")
      .def_prop_ro("operator", &DagInit::getOperator)
      .def_prop_ro("name_init", &DagInit::getName)
      .def_prop_ro("name_str", &DagInit::getNameStr)
      .def_prop_ro("num_args", &DagInit::getNumArgs)
      .def("get_arg", &DagInit::getArg, "num"_a,
           nb::rv_policy::reference_internal)
      .def("get_arg_no", &DagInit::getArgNo, "name"_a)
      .def("get_arg_name_init", &DagInit::getArgName, "num"_a,
           nb::rv_policy::reference_internal)
      .def("get_arg_name_str", &DagInit::getArgNameStr, "num"_a)
      .def("get_arg_name_inits",
           coerceReturn<std::vector<const StringInit *>>(&DagInit::getArgNames,
                                                         nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_args",
           coerceReturn<std::vector<const Init *>>(&DagInit::getArgs,
                                                   nb::const_),
           nb::rv_policy::reference_internal)
      .def("__len__", [](const DagInit &v) { return v.arg_size(); })
      .def("__bool__", [](const DagInit &v) { return !v.arg_empty(); })
      .def(
          "__iter__",
          [](DagInit &v) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<DagInit>(), "Iterator", v.arg_begin(), v.arg_end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](DagInit &v, Py_ssize_t i) {
            return v.getArg(nb::detail::wrap(i, v.arg_size()));
          },
          nb::rv_policy::reference_internal);

  nb::class_<RecordVal>(m, "RecordVal")
      .def("dump", &RecordVal::dump)
      .def_prop_ro("name", &RecordVal::getName)
      .def_prop_ro("name_init_as_string", &RecordVal::getNameInitAsString)
      .def_prop_ro("print_type", &RecordVal::getPrintType)
      .def_prop_ro("record_keeper", &RecordVal::getRecordKeeper)
      .def_prop_ro("type", &RecordVal::getType)
      .def_prop_ro("is_nonconcrete_ok", &RecordVal::isNonconcreteOK)
      .def_prop_ro("is_template_arg", &RecordVal::isTemplateArg)
      .def_prop_ro("value", &RecordVal::getValue)
      .def("__str__",
           [](const RecordVal &self) {
             return self.getValue()->getAsUnquotedString();
           })
      .def_prop_ro("is_used", &RecordVal::isUsed);

  struct RecordValues {};
  nb::class_<RecordValues>(m, "RecordValues", nb::dynamic_attr())
      .def("__repr__", [](const nb::object &self) {
        nb::str s{"RecordValues("};
        auto dic = nb::cast<nb::dict>(nb::getattr(self, "__dict__"));
        int i = 0;
        for (auto [key, value] : dic) {
          s += key + nb::str("=") +
               nb::str(nb::cast<RecordVal>(value)
                           .getValue()
                           ->getAsUnquotedString()
                           .c_str());
          if (i < dic.size() - 1)
            s += nb::str(", ");
          ++i;
        }
        s += nb::str(")");
        return s;
      });

  nb::class_<Record>(m, "Record")
      .def_prop_ro("direct_super_classes",
                   [](const Record &self) -> std::vector<const Record *> {
                     SmallVector<const Record *> Classes;
                     self.getDirectSuperClasses(Classes);
                     return {Classes.begin(), Classes.end()};
                   })
      .def_prop_ro("id", &Record::getID)
      .def_prop_ro("name", &Record::getName)
      .def_prop_ro("name_init_as_string", &Record::getNameInitAsString)
      .def_prop_ro("records", &Record::getRecords)
      .def_prop_ro("type", &Record::getType)
      .def("get_value", nb::overload_cast<StringRef>(&Record::getValue),
           "name"_a, nb::rv_policy::reference_internal)
      .def("get_value_as_bit", &Record::getValueAsBit, "field_name"_a)
      .def("get_value_as_def", &Record::getValueAsDef, "field_name"_a)
      .def("get_value_as_int", &Record::getValueAsInt, "field_name"_a)
      .def("get_value_as_list_of_defs", &Record::getValueAsListOfDefs,
           "field_name"_a, nb::rv_policy::reference_internal)
      .def("get_value_as_list_of_ints", &Record::getValueAsListOfInts,
           "field_name"_a)
      .def("get_value_as_list_of_strings", &Record::getValueAsListOfStrings,
           "field_name"_a)
      .def("get_value_as_optional_def", &Record::getValueAsOptionalDef,
           "field_name"_a, nb::rv_policy::reference_internal)
      .def("get_value_as_optional_string", &Record::getValueAsOptionalString,
           nb::sig("def get_value_as_optional_string(self, field_name: str, /) "
                   "-> Optional[str]"))
      .def("get_value_as_string", &Record::getValueAsString, "field_name"_a)
      .def("get_value_as_bit_or_unset", &Record::getValueAsBitOrUnset,
           "field_name"_a, "unset"_a)
      .def("get_value_as_bits_init", &Record::getValueAsBitsInit,
           "field_name"_a, nb::rv_policy::reference_internal)
      .def("get_value_as_dag", &Record::getValueAsDag, "field_name"_a,
           nb::rv_policy::reference_internal)
      .def("get_value_as_list_init", &Record::getValueAsListInit,
           "field_name"_a, nb::rv_policy::reference_internal)
      .def("get_value_init", &Record::getValueInit, "field_name"_a,
           nb::rv_policy::reference_internal)
      .def_prop_ro(
          "values",
          [](Record &self) {
            // you can't just call the class_->operator()
            nb::handle recordValsInstTy = nb::type<RecordValues>();
            assert(recordValsInstTy.is_valid() &&
                   nb::type_check(recordValsInstTy));
            nb::object recordValsInst = nb::inst_alloc(recordValsInstTy);
            assert(nb::inst_check(recordValsInst) &&
                   recordValsInst.type().is(recordValsInstTy) &&
                   !nb::inst_ready(recordValsInst));

            std::vector<RecordVal> values = self.getValues();
            for (const RecordVal &recordVal : values) {
              nb::setattr(recordValsInst, recordVal.getName().str().c_str(),
                          nb::borrow(nb::cast(recordVal)));
            }
            return recordValsInst;
          })
      .def("has_direct_super_class", &Record::hasDirectSuperClass,
           "super_class"_a)
      .def_prop_ro("is_anonymous", &Record::isAnonymous)
      .def_prop_ro("is_class", &Record::isClass)
      .def_prop_ro("is_multi_class", &Record::isMultiClass)
      .def("is_sub_class_of",
           nb::overload_cast<const Record *>(&Record::isSubClassOf, nb::const_),
           "r"_a)
      .def("is_sub_class_of",
           nb::overload_cast<StringRef>(&Record::isSubClassOf, nb::const_),
           "name"_a)
      .def("is_value_unset", &Record::isValueUnset, "field_name"_a)
      .def_prop_ro("def_init", &Record::getDefInit)
      .def_prop_ro("name_init", &Record::getNameInit)
      .def_prop_ro("template_args", coerceReturn<std::vector<const Init *>>(
                                        &Record::getTemplateArgs, nb::const_))
      .def("is_template_arg", &Record::isTemplateArg, "name"_a);

  using RecordMap = std::map<std::string, std::unique_ptr<Record>, std::less<>>;
  using GlobalMap = std::map<std::string, const Init *, std::less<>>;
  nb::bind_map<GlobalMap, nb::rv_policy::reference_internal>(m, "GlobalMap");

  nb::class_<RecordMap>(m, "RecordMap")
      .def("__len__", [](const RecordMap &m) { return m.size(); })
      .def("__bool__", [](const RecordMap &m) { return !m.empty(); })
      .def("__contains__",
           [](const RecordMap &m, const std::string &k) {
             return m.find(k) != m.end();
           })
      .def("__contains__", [](const RecordMap &, nb::handle) { return false; })
      .def(
          "__iter__",
          [](RecordMap &m) {
            return nb::make_key_iterator<nb::rv_policy::reference>(
                nb::type<RecordMap>(), "KeyIterator", m.begin(), m.end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "keys",
          [](RecordMap &m) {
            return nb::make_key_iterator<nb::rv_policy::reference>(
                nb::type<RecordMap>(), "KeyIterator", m.begin(), m.end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "__getitem__",
          [](RecordMap &m, const std::string &k) {
            if (!m.count(k))
              throw nb::key_error();
            // hack because by default (i.e, when using bind_map)
            // the unique_ptr is moved out of the map
            return m[k].get();
          },
          nb::rv_policy::reference_internal);

  nb::class_<RecordKeeper>(m, "RecordKeeper")
      .def(nb::init<>())
      .def(
          "parse_td",
          [](RecordKeeper &self, const std::string &inputFilename,
             const std::vector<std::string> &includeDirs,
             const std::vector<std::string> &macroNames,
             bool noWarnOnUnusedTemplateArgs) {
            ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
                MemoryBuffer::getFileOrSTDIN(inputFilename, /*IsText=*/true);
            if (std::error_code EC = fileOrErr.getError())
              throw std::runtime_error("Could not open input file '" +
                                       inputFilename + "': " + EC.message() +
                                       "\n");
            self.saveInputFilename(inputFilename);
            SourceMgr srcMgr;
            srcMgr.setIncludeDirs(includeDirs);
            srcMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
            TGParser tgParser(srcMgr, macroNames, self,
                              noWarnOnUnusedTemplateArgs);
            if (tgParser.ParseFile())
              throw std::runtime_error("Could not parse file '" +
                                       inputFilename);
            return &self;
          },
          "input_filename"_a, "include_dirs"_a = nb::list(),
          "macro_names"_a = nb::list(),
          "no_warn_on_unused_template_args"_a = true)
      .def_prop_ro("input_filename", &RecordKeeper::getInputFilename)
      .def_prop_ro("classes", &RecordKeeper::getClasses)
      .def_prop_ro("defs", &RecordKeeper::getDefs)
      .def_prop_ro("globals", &RecordKeeper::getGlobals)
      .def(
          "get_all_derived_definitions",
          [](RecordKeeper &self,
             const std::string &className) -> std::vector<const Record *> {
            return self.getAllDerivedDefinitions(className);
          },
          "class_name"_a, nb::rv_policy::reference_internal);

  nb::class_<llvm::raw_ostream>(m, "raw_ostream");

  auto mlir_tblgen_Pred =
      nb::class_<mlir::tblgen::Pred>(m, "Pred")
          .def(nb::init<>())
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::Init *>(), "init"_a)
          .def("is_null", &mlir::tblgen::Pred::isNull)
          .def("get_condition", &mlir::tblgen::Pred::getCondition)
          .def("is_combined", &mlir::tblgen::Pred::isCombined)
          .def("get_loc", &mlir::tblgen::Pred::getLoc)
          .def(
              "operator==",
              [](mlir::tblgen::Pred &self, const mlir::tblgen::Pred &other)
                  -> bool { return self.operator==(other); },
              "other"_a)
          .def("operator bool",
               [](mlir::tblgen::Pred &self) -> bool {
                 return self.operator bool();
               })
          .def("get_def", &mlir::tblgen::Pred::getDef,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_CPred =
      nb::class_<mlir::tblgen::CPred, mlir::tblgen::Pred>(m, "CPred")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::Init *>(), "init"_a);

  mlir_tblgen_CPred.def("get_condition_impl",
                        &mlir::tblgen::CPred::getConditionImpl);

  auto mlir_tblgen_CombinedPred =
      nb::class_<mlir::tblgen::CombinedPred, mlir::tblgen::Pred>(m,
                                                                 "CombinedPred")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::Init *>(), "init"_a)
          .def("get_condition_impl",
               &mlir::tblgen::CombinedPred::getConditionImpl)
          .def("get_combiner_def", &mlir::tblgen::CombinedPred::getCombinerDef,
               nb::rv_policy::reference_internal);

  mlir_tblgen_CombinedPred.def("get_children",
                               &mlir::tblgen::CombinedPred::getChildren);

  auto mlir_tblgen_SubstLeavesPred =
      nb::class_<mlir::tblgen::SubstLeavesPred, mlir::tblgen::CombinedPred>(
          m, "SubstLeavesPred")
          .def("get_pattern", &mlir::tblgen::SubstLeavesPred::getPattern);

  mlir_tblgen_SubstLeavesPred.def(
      "get_replacement", &mlir::tblgen::SubstLeavesPred::getReplacement);

  auto mlir_tblgen_ConcatPred =
      nb::class_<mlir::tblgen::ConcatPred, mlir::tblgen::CombinedPred>(
          m, "ConcatPred")
          .def("get_prefix", &mlir::tblgen::ConcatPred::getPrefix);

  mlir_tblgen_ConcatPred.def("get_suffix",
                             &mlir::tblgen::ConcatPred::getSuffix);

  auto mlir_tblgen_Constraint =
      nb::class_<mlir::tblgen::Constraint>(m, "Constraint")
          .def(nb::init<const llvm::Record *, mlir::tblgen::Constraint::Kind>(),
               "record"_a, "kind"_a)
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(
              "operator==",
              [](mlir::tblgen::Constraint &self,
                 const mlir::tblgen::Constraint &that) -> bool {
                return self.operator==(that);
              },
              "that"_a)
          .def(
              "operator!=",
              [](mlir::tblgen::Constraint &self,
                 const mlir::tblgen::Constraint &that) -> bool {
                return self.operator!=(that);
              },
              "that"_a)
          .def("get_predicate", &mlir::tblgen::Constraint::getPredicate)
          .def("get_condition_template",
               &mlir::tblgen::Constraint::getConditionTemplate)
          .def("get_summary", &mlir::tblgen::Constraint::getSummary)
          .def("get_description", &mlir::tblgen::Constraint::getDescription)
          .def("get_def_name", &mlir::tblgen::Constraint::getDefName)
          .def("get_unique_def_name",
               &mlir::tblgen::Constraint::getUniqueDefName)
          .def("get_cpp_function_name",
               &mlir::tblgen::Constraint::getCppFunctionName)
          .def("get_kind", &mlir::tblgen::Constraint::getKind)
          .def("get_def", &mlir::tblgen::Constraint::getDef,
               nb::rv_policy::reference_internal);
  nb::enum_<mlir::tblgen::Constraint::Kind>(mlir_tblgen_Constraint, "Kind")
      .value("CK_Attr", mlir::tblgen::Constraint::CK_Attr)
      .value("CK_Region", mlir::tblgen::Constraint::CK_Region)
      .value("CK_Successor", mlir::tblgen::Constraint::CK_Successor)
      .value("CK_Type", mlir::tblgen::Constraint::CK_Type)
      .value("CK_Uncategorized", mlir::tblgen::Constraint::CK_Uncategorized);

  auto mlir_tblgen_AppliedConstraint =
      nb::class_<mlir::tblgen::AppliedConstraint>(m, "AppliedConstraint")
          .def(nb::init<
                   mlir::tblgen::Constraint &&, llvm::StringRef,
                   std::vector<std::string, std::allocator<std::string>> &&>(),
               "constraint"_a, "self"_a, "entities"_a)
          .def_rw("constraint", &mlir::tblgen::AppliedConstraint::constraint)
          .def_rw("self", &mlir::tblgen::AppliedConstraint::self)
          .def_rw("entities", &mlir::tblgen::AppliedConstraint::entities);

  auto mlir_tblgen_AttrConstraint =
      nb::class_<mlir::tblgen::AttrConstraint, mlir::tblgen::Constraint>(
          m, "AttrConstraint")
          .def_static("classof", &mlir::tblgen::AttrConstraint::classof, "c"_a)
          .def("is_sub_class_of", &mlir::tblgen::AttrConstraint::isSubClassOf,
               "class_name"_a);

  auto mlir_tblgen_Attribute =
      nb::class_<mlir::tblgen::Attribute, mlir::tblgen::AttrConstraint>(
          m, "Attribute")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def("get_storage_type", &mlir::tblgen::Attribute::getStorageType)
          .def("get_return_type", &mlir::tblgen::Attribute::getReturnType)
          .def("get_value_type", &mlir::tblgen::Attribute::getValueType)
          .def("get_convert_from_storage_call",
               &mlir::tblgen::Attribute::getConvertFromStorageCall)
          .def("is_const_buildable", &mlir::tblgen::Attribute::isConstBuildable)
          .def("get_const_builder_template",
               &mlir::tblgen::Attribute::getConstBuilderTemplate)
          .def("get_base_attr", &mlir::tblgen::Attribute::getBaseAttr)
          .def("has_default_value", &mlir::tblgen::Attribute::hasDefaultValue)
          .def("get_default_value", &mlir::tblgen::Attribute::getDefaultValue)
          .def("is_optional", &mlir::tblgen::Attribute::isOptional)
          .def("is_derived_attr", &mlir::tblgen::Attribute::isDerivedAttr)
          .def("is_type_attr", &mlir::tblgen::Attribute::isTypeAttr)
          .def("is_symbol_ref_attr", &mlir::tblgen::Attribute::isSymbolRefAttr)
          .def("is_enum_attr", &mlir::tblgen::Attribute::isEnumAttr)
          .def("get_attr_def_name", &mlir::tblgen::Attribute::getAttrDefName)
          .def("get_derived_code_body",
               &mlir::tblgen::Attribute::getDerivedCodeBody)
          .def("get_dialect", &mlir::tblgen::Attribute::getDialect)
          .def("get_def", &mlir::tblgen::Attribute::getDef,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_ConstantAttr =
      nb::class_<mlir::tblgen::ConstantAttr>(m, "ConstantAttr")
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def("get_attribute", &mlir::tblgen::ConstantAttr::getAttribute)
          .def("get_constant_value",
               &mlir::tblgen::ConstantAttr::getConstantValue);

  auto mlir_tblgen_EnumAttrCase =
      nb::class_<mlir::tblgen::EnumAttrCase, mlir::tblgen::Attribute>(
          m, "EnumAttrCase")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def("get_symbol", &mlir::tblgen::EnumAttrCase::getSymbol)
          .def("get_str", &mlir::tblgen::EnumAttrCase::getStr)
          .def("get_value", &mlir::tblgen::EnumAttrCase::getValue);

  mlir_tblgen_EnumAttrCase.def("get_def", &mlir::tblgen::EnumAttrCase::getDef,
                               nb::rv_policy::reference_internal);

  auto mlir_tblgen_EnumAttr =
      nb::class_<mlir::tblgen::EnumAttr, mlir::tblgen::Attribute>(m, "EnumAttr")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::Record &>(), "record"_a)
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def_static("classof", &mlir::tblgen::EnumAttr::classof, "attr"_a)
          .def("is_bit_enum", &mlir::tblgen::EnumAttr::isBitEnum)
          .def("get_enum_class_name", &mlir::tblgen::EnumAttr::getEnumClassName)
          .def("get_cpp_namespace", &mlir::tblgen::EnumAttr::getCppNamespace)
          .def("get_underlying_type",
               &mlir::tblgen::EnumAttr::getUnderlyingType)
          .def("get_underlying_to_symbol_fn_name",
               &mlir::tblgen::EnumAttr::getUnderlyingToSymbolFnName)
          .def("get_string_to_symbol_fn_name",
               &mlir::tblgen::EnumAttr::getStringToSymbolFnName)
          .def("get_symbol_to_string_fn_name",
               &mlir::tblgen::EnumAttr::getSymbolToStringFnName)
          .def("get_symbol_to_string_fn_ret_type",
               &mlir::tblgen::EnumAttr::getSymbolToStringFnRetType)
          .def("get_max_enum_val_fn_name",
               &mlir::tblgen::EnumAttr::getMaxEnumValFnName)
          .def("get_all_cases", &mlir::tblgen::EnumAttr::getAllCases)
          .def("gen_specialized_attr",
               &mlir::tblgen::EnumAttr::genSpecializedAttr)
          .def("get_base_attr_class", &mlir::tblgen::EnumAttr::getBaseAttrClass,
               nb::rv_policy::reference_internal)
          .def("get_specialized_attr_class_name",
               &mlir::tblgen::EnumAttr::getSpecializedAttrClassName);

  mlir_tblgen_EnumAttr.def("print_bit_enum_primary_groups",
                           &mlir::tblgen::EnumAttr::printBitEnumPrimaryGroups);

  auto mlir_tblgen_Property =
      nb::class_<mlir::tblgen::Property>(m, "Property")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def(nb::init<llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef>(),
               "summary"_a, "description"_a, "storage_type"_a,
               "interface_type"_a, "convert_from_storage_call"_a,
               "assign_to_storage_call"_a, "convert_to_attribute_call"_a,
               "convert_from_attribute_call"_a, "parser_call"_a,
               "optional_parser_call"_a, "printer_call"_a,
               "read_from_mlir_bytecode_call"_a,
               "write_to_mlir_bytecode_call"_a, "hash_property_call"_a,
               "default_value"_a, "storage_type_value_override"_a)
          .def("get_summary", &mlir::tblgen::Property::getSummary)
          .def("get_description", &mlir::tblgen::Property::getDescription)
          .def("get_storage_type", &mlir::tblgen::Property::getStorageType)
          .def("get_interface_type", &mlir::tblgen::Property::getInterfaceType)
          .def("get_convert_from_storage_call",
               &mlir::tblgen::Property::getConvertFromStorageCall)
          .def("get_assign_to_storage_call",
               &mlir::tblgen::Property::getAssignToStorageCall)
          .def("get_convert_to_attribute_call",
               &mlir::tblgen::Property::getConvertToAttributeCall)
          .def("get_convert_from_attribute_call",
               &mlir::tblgen::Property::getConvertFromAttributeCall)
          .def("get_predicate", &mlir::tblgen::Property::getPredicate)
          .def("get_parser_call", &mlir::tblgen::Property::getParserCall)
          .def("has_optional_parser",
               &mlir::tblgen::Property::hasOptionalParser)
          .def("get_optional_parser_call",
               &mlir::tblgen::Property::getOptionalParserCall)
          .def("get_printer_call", &mlir::tblgen::Property::getPrinterCall)
          .def("get_read_from_mlir_bytecode_call",
               &mlir::tblgen::Property::getReadFromMlirBytecodeCall)
          .def("get_write_to_mlir_bytecode_call",
               &mlir::tblgen::Property::getWriteToMlirBytecodeCall)
          .def("get_hash_property_call",
               &mlir::tblgen::Property::getHashPropertyCall)
          .def("has_default_value", &mlir::tblgen::Property::hasDefaultValue)
          .def("get_default_value", &mlir::tblgen::Property::getDefaultValue)
          .def("has_storage_type_value_override",
               &mlir::tblgen::Property::hasStorageTypeValueOverride)
          .def("get_storage_type_value_override",
               &mlir::tblgen::Property::getStorageTypeValueOverride)
          .def("get_property_def_name",
               &mlir::tblgen::Property::getPropertyDefName)
          .def("get_base_property", &mlir::tblgen::Property::getBaseProperty)
          .def("get_def", &mlir::tblgen::Property::getDef,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_NamedProperty =
      nb::class_<mlir::tblgen::NamedProperty>(m, "NamedProperty")
          .def_rw("name", &mlir::tblgen::NamedProperty::name)
          .def_rw("prop", &mlir::tblgen::NamedProperty::prop);

  auto mlir_tblgen_Dialect =
      nb::class_<mlir::tblgen::Dialect>(m, "Dialect")
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def("get_name", &mlir::tblgen::Dialect::getName)
          .def("get_cpp_namespace", &mlir::tblgen::Dialect::getCppNamespace)
          .def("get_cpp_class_name", &mlir::tblgen::Dialect::getCppClassName)
          .def("get_summary", &mlir::tblgen::Dialect::getSummary)
          .def("get_description", &mlir::tblgen::Dialect::getDescription)
          .def("get_dependent_dialects",
               &mlir::tblgen::Dialect::getDependentDialects)
          .def("get_extra_class_declaration",
               &mlir::tblgen::Dialect::getExtraClassDeclaration)
          .def("has_canonicalizer", &mlir::tblgen::Dialect::hasCanonicalizer)
          .def("has_constant_materializer",
               &mlir::tblgen::Dialect::hasConstantMaterializer)
          .def("has_non_default_destructor",
               &mlir::tblgen::Dialect::hasNonDefaultDestructor)
          .def("has_operation_attr_verify",
               &mlir::tblgen::Dialect::hasOperationAttrVerify)
          .def("has_region_arg_attr_verify",
               &mlir::tblgen::Dialect::hasRegionArgAttrVerify)
          .def("has_region_result_attr_verify",
               &mlir::tblgen::Dialect::hasRegionResultAttrVerify)
          .def("has_operation_interface_fallback",
               &mlir::tblgen::Dialect::hasOperationInterfaceFallback)
          .def("use_default_attribute_printer_parser",
               &mlir::tblgen::Dialect::useDefaultAttributePrinterParser)
          .def("use_default_type_printer_parser",
               &mlir::tblgen::Dialect::useDefaultTypePrinterParser)
          .def("is_extensible", &mlir::tblgen::Dialect::isExtensible)
          .def("use_properties_for_attributes",
               &mlir::tblgen::Dialect::usePropertiesForAttributes)
          .def("get_discardable_attributes",
               &mlir::tblgen::Dialect::getDiscardableAttributes,
               nb::rv_policy::reference_internal)
          .def("get_def", &mlir::tblgen::Dialect::getDef,
               nb::rv_policy::reference_internal)
          .def(
              "operator==",
              [](mlir::tblgen::Dialect &self,
                 const mlir::tblgen::Dialect &other) -> bool {
                return self.operator==(other);
              },
              "other"_a)
          .def(
              "operator!=",
              [](mlir::tblgen::Dialect &self,
                 const mlir::tblgen::Dialect &other) -> bool {
                return self.operator!=(other);
              },
              "other"_a)
          .def(
              "operator<",
              [](mlir::tblgen::Dialect &self,
                 const mlir::tblgen::Dialect &other) -> bool {
                return self.operator<(other);
              },
              "other"_a)
          .def("operator bool", [](mlir::tblgen::Dialect &self) -> bool {
            return self.operator bool();
          });

  auto mlir_tblgen_TypeConstraint =
      nb::class_<mlir::tblgen::TypeConstraint, mlir::tblgen::Constraint>(
          m, "TypeConstraint")
          .def(nb::init<const llvm::DefInit *>(), "record"_a)
          .def_static("classof", &mlir::tblgen::TypeConstraint::classof, "c"_a)
          .def("is_optional", &mlir::tblgen::TypeConstraint::isOptional)
          .def("is_variadic", &mlir::tblgen::TypeConstraint::isVariadic)
          .def("is_variadic_of_variadic",
               &mlir::tblgen::TypeConstraint::isVariadicOfVariadic)
          .def("get_variadic_of_variadic_segment_size_attr",
               &mlir::tblgen::TypeConstraint::
                   getVariadicOfVariadicSegmentSizeAttr)
          .def("is_variable_length",
               &mlir::tblgen::TypeConstraint::isVariableLength)
          .def("get_builder_call",
               &mlir::tblgen::TypeConstraint::getBuilderCall)
          .def("get_cpp_type", &mlir::tblgen::TypeConstraint::getCppType);

  auto mlir_tblgen_Type =
      nb::class_<mlir::tblgen::Type, mlir::tblgen::TypeConstraint>(m, "Type")
          .def(nb::init<const llvm::Record *>(), "record"_a);

  mlir_tblgen_Type.def("get_dialect", &mlir::tblgen::Type::getDialect);

  auto mlir_tblgen_NamedAttribute =
      nb::class_<mlir::tblgen::NamedAttribute>(m, "NamedAttribute")
          .def_rw("name", &mlir::tblgen::NamedAttribute::name)
          .def_rw("attr", &mlir::tblgen::NamedAttribute::attr);

  auto mlir_tblgen_NamedTypeConstraint =
      nb::class_<mlir::tblgen::NamedTypeConstraint>(m, "NamedTypeConstraint")
          .def("has_predicate",
               &mlir::tblgen::NamedTypeConstraint::hasPredicate)
          .def("is_optional", &mlir::tblgen::NamedTypeConstraint::isOptional)
          .def("is_variadic", &mlir::tblgen::NamedTypeConstraint::isVariadic)
          .def("is_variadic_of_variadic",
               &mlir::tblgen::NamedTypeConstraint::isVariadicOfVariadic)
          .def("is_variable_length",
               &mlir::tblgen::NamedTypeConstraint::isVariableLength)
          .def_rw("name", &mlir::tblgen::NamedTypeConstraint::name)
          .def_rw("constraint", &mlir::tblgen::NamedTypeConstraint::constraint);

  auto mlir_tblgen_Builder = nb::class_<mlir::tblgen::Builder>(m, "Builder");

  auto mlir_tblgen_Builder_Parameter =
      nb::class_<mlir::tblgen::Builder::Parameter>(mlir_tblgen_Builder,
                                                   "Parameter")
          .def("get_cpp_type", &mlir::tblgen::Builder::Parameter::getCppType)
          .def("get_name", &mlir::tblgen::Builder::Parameter::getName)
          .def("get_default_value",
               &mlir::tblgen::Builder::Parameter::getDefaultValue);

  mlir_tblgen_Builder
      .def(nb::init<const llvm::Record *, llvm::ArrayRef<llvm::SMLoc>>(),
           "record"_a, "loc"_a)
      .def("get_parameters", &mlir::tblgen::Builder::getParameters)
      .def("get_body", &mlir::tblgen::Builder::getBody)
      .def("get_deprecated_message",
           &mlir::tblgen::Builder::getDeprecatedMessage);

  auto mlir_tblgen_Trait =
      nb::class_<mlir::tblgen::Trait>(m, "Trait")
          .def(nb::init<mlir::tblgen::Trait::Kind, const llvm::Record *>(),
               "kind"_a, "def_"_a)
          .def_static("create", &mlir::tblgen::Trait::create, "init"_a)
          .def("get_kind", &mlir::tblgen::Trait::getKind)
          .def("get_def", &mlir::tblgen::Trait::getDef,
               nb::rv_policy::reference_internal);
  nb::enum_<mlir::tblgen::Trait::Kind>(mlir_tblgen_Trait, "Kind")
      .value("Native", mlir::tblgen::Trait::Kind::Native)
      .value("Pred", mlir::tblgen::Trait::Kind::Pred)
      .value("Internal", mlir::tblgen::Trait::Kind::Internal)
      .value("Interface", mlir::tblgen::Trait::Kind::Interface);

  auto mlir_tblgen_NativeTrait =
      nb::class_<mlir::tblgen::NativeTrait, mlir::tblgen::Trait>(m,
                                                                 "NativeTrait")
          .def("get_fully_qualified_trait_name",
               &mlir::tblgen::NativeTrait::getFullyQualifiedTraitName)
          .def("is_structural_op_trait",
               &mlir::tblgen::NativeTrait::isStructuralOpTrait)
          .def("get_extra_concrete_class_declaration",
               &mlir::tblgen::NativeTrait::getExtraConcreteClassDeclaration)
          .def("get_extra_concrete_class_definition",
               &mlir::tblgen::NativeTrait::getExtraConcreteClassDefinition)
          .def_static("classof", &mlir::tblgen::NativeTrait::classof, "t"_a);

  auto mlir_tblgen_PredTrait =
      nb::class_<mlir::tblgen::PredTrait, mlir::tblgen::Trait>(m, "PredTrait")
          .def("get_pred_template", &mlir::tblgen::PredTrait::getPredTemplate)
          .def("get_summary", &mlir::tblgen::PredTrait::getSummary)
          .def_static("classof", &mlir::tblgen::PredTrait::classof, "t"_a);

  auto mlir_tblgen_InternalTrait =
      nb::class_<mlir::tblgen::InternalTrait, mlir::tblgen::Trait>(
          m, "InternalTrait")
          .def("get_fully_qualified_trait_name",
               &mlir::tblgen::InternalTrait::getFullyQualifiedTraitName)
          .def_static("classof", &mlir::tblgen::InternalTrait::classof, "t"_a);

  auto mlir_tblgen_InterfaceTrait =
      nb::class_<mlir::tblgen::InterfaceTrait, mlir::tblgen::Trait>(
          m, "InterfaceTrait")
          .def("get_interface", &mlir::tblgen::InterfaceTrait::getInterface)
          .def("get_fully_qualified_trait_name",
               &mlir::tblgen::InterfaceTrait::getFullyQualifiedTraitName)
          .def_static("classof", &mlir::tblgen::InterfaceTrait::classof, "t"_a)
          .def("should_declare_methods",
               &mlir::tblgen::InterfaceTrait::shouldDeclareMethods)
          .def("get_always_declared_methods",
               &mlir::tblgen::InterfaceTrait::getAlwaysDeclaredMethods);

  auto mlir_tblgen_AttrOrTypeBuilder =
      nb::class_<mlir::tblgen::AttrOrTypeBuilder, mlir::tblgen::Builder>(
          m, "AttrOrTypeBuilder")
          .def("get_return_type",
               &mlir::tblgen::AttrOrTypeBuilder::getReturnType)
          .def("has_inferred_context_parameter",
               &mlir::tblgen::AttrOrTypeBuilder::hasInferredContextParameter);

  auto mlir_tblgen_AttrOrTypeParameter =
      nb::class_<mlir::tblgen::AttrOrTypeParameter>(m, "AttrOrTypeParameter")
          .def(nb::init<const llvm::DagInit *, unsigned int>(), "def_"_a,
               "index"_a)
          .def("is_anonymous", &mlir::tblgen::AttrOrTypeParameter::isAnonymous)
          .def("get_name", &mlir::tblgen::AttrOrTypeParameter::getName)
          .def("get_accessor_name",
               &mlir::tblgen::AttrOrTypeParameter::getAccessorName)
          .def("get_allocator",
               &mlir::tblgen::AttrOrTypeParameter::getAllocator)
          .def("get_comparator",
               &mlir::tblgen::AttrOrTypeParameter::getComparator)
          .def("get_cpp_type", &mlir::tblgen::AttrOrTypeParameter::getCppType)
          .def("get_cpp_accessor_type",
               &mlir::tblgen::AttrOrTypeParameter::getCppAccessorType)
          .def("get_cpp_storage_type",
               &mlir::tblgen::AttrOrTypeParameter::getCppStorageType)
          .def("get_convert_from_storage",
               &mlir::tblgen::AttrOrTypeParameter::getConvertFromStorage)
          .def("get_parser", &mlir::tblgen::AttrOrTypeParameter::getParser)
          .def("get_constraint",
               &mlir::tblgen::AttrOrTypeParameter::getConstraint)
          .def("get_printer", &mlir::tblgen::AttrOrTypeParameter::getPrinter)
          .def("get_summary", &mlir::tblgen::AttrOrTypeParameter::getSummary)
          .def("get_syntax", &mlir::tblgen::AttrOrTypeParameter::getSyntax)
          .def("is_optional", &mlir::tblgen::AttrOrTypeParameter::isOptional)
          .def("get_default_value",
               &mlir::tblgen::AttrOrTypeParameter::getDefaultValue)
          .def("get_def", &mlir::tblgen::AttrOrTypeParameter::getDef,
               nb::rv_policy::reference_internal)
          .def(
              "operator==",
              [](mlir::tblgen::AttrOrTypeParameter &self,
                 const mlir::tblgen::AttrOrTypeParameter &other) -> bool {
                return self.operator==(other);
              },
              "other"_a)
          .def(
              "operator!=",
              [](mlir::tblgen::AttrOrTypeParameter &self,
                 const mlir::tblgen::AttrOrTypeParameter &other) -> bool {
                return self.operator!=(other);
              },
              "other"_a);

  auto mlir_tblgen_AttributeSelfTypeParameter =
      nb::class_<mlir::tblgen::AttributeSelfTypeParameter,
                 mlir::tblgen::AttrOrTypeParameter>(
          m, "AttributeSelfTypeParameter")
          .def_static("classof",
                      &mlir::tblgen::AttributeSelfTypeParameter::classof,
                      "param"_a);

  auto mlir_tblgen_AttrOrTypeDef =
      nb::class_<mlir::tblgen::AttrOrTypeDef>(m, "AttrOrTypeDef")
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def("get_dialect", &mlir::tblgen::AttrOrTypeDef::getDialect)
          .def("get_name", &mlir::tblgen::AttrOrTypeDef::getName)
          .def("has_description", &mlir::tblgen::AttrOrTypeDef::hasDescription)
          .def("get_description", &mlir::tblgen::AttrOrTypeDef::getDescription)
          .def("has_summary", &mlir::tblgen::AttrOrTypeDef::hasSummary)
          .def("get_summary", &mlir::tblgen::AttrOrTypeDef::getSummary)
          .def("get_cpp_class_name",
               &mlir::tblgen::AttrOrTypeDef::getCppClassName)
          .def("get_cpp_base_class_name",
               &mlir::tblgen::AttrOrTypeDef::getCppBaseClassName)
          .def("get_storage_class_name",
               &mlir::tblgen::AttrOrTypeDef::getStorageClassName)
          .def("get_storage_namespace",
               &mlir::tblgen::AttrOrTypeDef::getStorageNamespace)
          .def("gen_storage_class",
               &mlir::tblgen::AttrOrTypeDef::genStorageClass)
          .def("has_storage_custom_constructor",
               &mlir::tblgen::AttrOrTypeDef::hasStorageCustomConstructor)
          .def("get_parameters", &mlir::tblgen::AttrOrTypeDef::getParameters)
          .def("get_num_parameters",
               &mlir::tblgen::AttrOrTypeDef::getNumParameters)
          .def("get_mnemonic", &mlir::tblgen::AttrOrTypeDef::getMnemonic)
          .def("has_custom_assembly_format",
               &mlir::tblgen::AttrOrTypeDef::hasCustomAssemblyFormat)
          .def("get_assembly_format",
               &mlir::tblgen::AttrOrTypeDef::getAssemblyFormat)
          .def("gen_accessors", &mlir::tblgen::AttrOrTypeDef::genAccessors)
          .def("gen_verify_decl", &mlir::tblgen::AttrOrTypeDef::genVerifyDecl)
          .def("gen_verify_invariants_impl",
               &mlir::tblgen::AttrOrTypeDef::genVerifyInvariantsImpl)
          .def("get_extra_decls", &mlir::tblgen::AttrOrTypeDef::getExtraDecls)
          .def("get_extra_defs", &mlir::tblgen::AttrOrTypeDef::getExtraDefs)
          .def("get_loc", &mlir::tblgen::AttrOrTypeDef::getLoc)
          .def("skip_default_builders",
               &mlir::tblgen::AttrOrTypeDef::skipDefaultBuilders)
          .def("get_builders", &mlir::tblgen::AttrOrTypeDef::getBuilders)
          .def("get_traits", &mlir::tblgen::AttrOrTypeDef::getTraits)
          .def(
              "operator==",
              [](mlir::tblgen::AttrOrTypeDef &self,
                 const mlir::tblgen::AttrOrTypeDef &other) -> bool {
                return self.operator==(other);
              },
              "other"_a)
          .def(
              "operator<",
              [](mlir::tblgen::AttrOrTypeDef &self,
                 const mlir::tblgen::AttrOrTypeDef &other) -> bool {
                return self.operator<(other);
              },
              "other"_a)
          .def("operator bool",
               [](mlir::tblgen::AttrOrTypeDef &self) -> bool {
                 return self.operator bool();
               })
          .def("get_def", &mlir::tblgen::AttrOrTypeDef::getDef,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_AttrDef =
      nb::class_<mlir::tblgen::AttrDef, mlir::tblgen::AttrOrTypeDef>(m,
                                                                     "AttrDef")
          .def("get_type_builder", &mlir::tblgen::AttrDef::getTypeBuilder)
          .def_static("classof", &mlir::tblgen::AttrDef::classof, "def_"_a)
          .def("get_attr_name", &mlir::tblgen::AttrDef::getAttrName);

  auto mlir_tblgen_TypeDef =
      nb::class_<mlir::tblgen::TypeDef, mlir::tblgen::AttrOrTypeDef>(m,
                                                                     "TypeDef")
          .def_static("classof", &mlir::tblgen::TypeDef::classof, "def_"_a)
          .def("get_type_name", &mlir::tblgen::TypeDef::getTypeName);

  auto mlir_raw_indented_ostream =
      nb::class_<mlir::raw_indented_ostream, llvm::raw_ostream>(
          m, "raw_indented_ostream")
          .def(nb::init<llvm::raw_ostream &>(), "os"_a)
          .def("get_o_stream", &mlir::raw_indented_ostream::getOStream,
               nb::rv_policy::reference_internal)
          .def("scope", &mlir::raw_indented_ostream::scope, "open"_a, "close"_a,
               "indent"_a)
          .def("print_reindented", &mlir::raw_indented_ostream::printReindented,
               "str"_a, "extra_prefix"_a, nb::rv_policy::reference_internal)
          .def(
              "indent",
              [](mlir::raw_indented_ostream &self)
                  -> mlir::raw_indented_ostream & { return self.indent(); },
              nb::rv_policy::reference_internal)
          .def("unindent", &mlir::raw_indented_ostream::unindent,
               nb::rv_policy::reference_internal)
          .def(
              "indent",
              [](mlir::raw_indented_ostream &self, int with)
                  -> mlir::raw_indented_ostream & { return self.indent(with); },
              "with"_a, nb::rv_policy::reference_internal)
          .def("print_reindented", &mlir::raw_indented_ostream::printReindented,
               "str"_a, "extra_prefix"_a, nb::rv_policy::reference_internal);

  auto mlir_raw_indented_ostream_DelimitedScope =
      nb::class_<mlir::raw_indented_ostream::DelimitedScope>(
          mlir_raw_indented_ostream, "DelimitedScope");
  mlir_raw_indented_ostream_DelimitedScope.def(
      nb::init<mlir::raw_indented_ostream &, llvm::StringRef, llvm::StringRef,
               bool>(),

      "os"_a, "open"_a, "close"_a, "indent"_a);
  auto mlir_tblgen_FmtContext =
      nb::class_<mlir::tblgen::FmtContext>(m, "FmtContext")
          .def(nb::init<>())
          .def(nb::init<llvm::ArrayRef<
                   std::pair<llvm::StringRef, llvm::StringRef>>>(),
               "subs"_a)
          .def("add_subst", &mlir::tblgen::FmtContext::addSubst,
               "placeholder"_a, "subst"_a, nb::rv_policy::reference_internal)
          .def("with_builder", &mlir::tblgen::FmtContext::withBuilder,
               "subst"_a, nb::rv_policy::reference_internal)
          .def("with_self", &mlir::tblgen::FmtContext::withSelf, "subst"_a,
               nb::rv_policy::reference_internal)
          .def(
              "get_subst_for",
              [](mlir::tblgen::FmtContext &self,
                 mlir::tblgen::FmtContext::PHKind placeholder)
                  -> std::optional<llvm::StringRef> {
                return self.getSubstFor(placeholder);
              },
              "placeholder"_a)
          .def(
              "get_subst_for",
              [](mlir::tblgen::FmtContext &self, llvm::StringRef placeholder)
                  -> std::optional<llvm::StringRef> {
                return self.getSubstFor(placeholder);
              },
              "placeholder"_a)
          .def_static("get_place_holder_kind",
                      &mlir::tblgen::FmtContext::getPlaceHolderKind, "str"_a);

  nb::enum_<mlir::tblgen::FmtContext::PHKind>(mlir_tblgen_FmtContext, "PHKind")
      .value("None", mlir::tblgen::FmtContext::PHKind::None)
      .value("Custom", mlir::tblgen::FmtContext::PHKind::Custom)
      .value("Builder", mlir::tblgen::FmtContext::PHKind::Builder)
      .value("Self", mlir::tblgen::FmtContext::PHKind::Self);

  auto mlir_tblgen_FmtReplacement =
      nb::class_<mlir::tblgen::FmtReplacement>(m, "FmtReplacement")
          .def(nb::init<>())
          .def(nb::init<llvm::StringRef>(), "literal"_a)
          .def(nb::init<llvm::StringRef, size_t>(), "spec"_a, "index"_a)
          .def(nb::init<llvm::StringRef, size_t, size_t>(), "spec"_a, "index"_a,
               "end"_a)
          .def(nb::init<llvm::StringRef, mlir::tblgen::FmtContext::PHKind>(),
               "spec"_a, "placeholder"_a)
          .def_rw("type", &mlir::tblgen::FmtReplacement::type)
          .def_rw("spec", &mlir::tblgen::FmtReplacement::spec)
          .def_rw("index", &mlir::tblgen::FmtReplacement::index)
          .def_rw("end", &mlir::tblgen::FmtReplacement::end)
          .def_rw("placeholder", &mlir::tblgen::FmtReplacement::placeholder);
  nb::enum_<mlir::tblgen::FmtReplacement::Type>(mlir_tblgen_FmtReplacement,
                                                "Type")
      .value("Empty", mlir::tblgen::FmtReplacement::Type::Empty)
      .value("Literal", mlir::tblgen::FmtReplacement::Type::Literal)
      .value("PositionalPH", mlir::tblgen::FmtReplacement::Type::PositionalPH)
      .value("PositionalRangePH",
             mlir::tblgen::FmtReplacement::Type::PositionalRangePH)
      .value("SpecialPH", mlir::tblgen::FmtReplacement::Type::SpecialPH);

  mlir_tblgen_FmtReplacement.def_ro_static(
      "k_unset", &mlir::tblgen::FmtReplacement::kUnset);

  auto mlir_tblgen_FmtObjectBase =
      nb::class_<mlir::tblgen::FmtObjectBase>(m, "FmtObjectBase")
          .def(nb::init<llvm::StringRef, const mlir::tblgen::FmtContext *,
                        size_t>(),
               "fmt"_a, "ctx"_a, "num_params"_a)
          .def(nb::init<mlir::tblgen::FmtObjectBase &&>(), "that"_a)
          .def("format", &mlir::tblgen::FmtObjectBase::format, "s"_a)
          .def("str", &mlir::tblgen::FmtObjectBase::str)
          .def("__str__", [](mlir::tblgen::FmtObjectBase &self) -> std::string {
            return self.str();
          });

  auto mlir_tblgen_FmtStrVecObject =
      nb::class_<mlir::tblgen::FmtStrVecObject, mlir::tblgen::FmtObjectBase>(
          m, "FmtStrVecObject")
          .def(nb::init<llvm::StringRef, const mlir::tblgen::FmtContext *,
                        llvm::ArrayRef<std::string>>(),
               "fmt"_a, "ctx"_a, "params"_a)
          .def(nb::init<mlir::tblgen::FmtStrVecObject &&>(), "that"_a);

  m.def(
      "tgfmt",
      [](llvm::StringRef fmt, const mlir::tblgen::FmtContext *ctx,
         llvm::ArrayRef<std::string> params) -> mlir::tblgen::FmtStrVecObject {
        return mlir::tblgen::tgfmt(fmt, ctx, params);
      },
      "fmt"_a, "ctx"_a, "params"_a);

  auto mlir_tblgen_IfDefScope =
      nb::class_<mlir::tblgen::IfDefScope>(m, "IfDefScope")
          .def(nb::init<llvm::StringRef, llvm::raw_ostream &>(), "name"_a,
               "os"_a);

  auto mlir_tblgen_NamespaceEmitter =
      nb::class_<mlir::tblgen::NamespaceEmitter>(m, "NamespaceEmitter")
          .def(nb::init<llvm::raw_ostream &, const mlir::tblgen::Dialect &>(),
               "os"_a, "dialect"_a)
          .def(nb::init<llvm::raw_ostream &, llvm::StringRef>(), "os"_a,
               "cpp_namespace"_a);

  auto mlir_tblgen_StaticVerifierFunctionEmitter =
      nb::class_<mlir::tblgen::StaticVerifierFunctionEmitter>(
          m, "StaticVerifierFunctionEmitter")
          .def(nb::init<llvm::raw_ostream &, const llvm::RecordKeeper &,
                        llvm::StringRef>(),
               "os"_a, "records"_a, "tag"_a)
          .def("collect_op_constraints",
               &mlir::tblgen::StaticVerifierFunctionEmitter::
                   collectOpConstraints,
               "op_defs"_a)
          .def("emit_op_constraints",
               &mlir::tblgen::StaticVerifierFunctionEmitter::emitOpConstraints,
               "op_defs"_a)
          .def(
              "emit_pattern_constraints",
              [](mlir::tblgen::StaticVerifierFunctionEmitter &self,
                 const llvm::ArrayRef<mlir::tblgen::DagLeaf> constraints)
                  -> void { return self.emitPatternConstraints(constraints); },
              "constraints"_a)
          .def(
              "get_type_constraint_fn",
              &mlir::tblgen::StaticVerifierFunctionEmitter::getTypeConstraintFn,
              "constraint"_a)
          .def(
              "get_attr_constraint_fn",
              &mlir::tblgen::StaticVerifierFunctionEmitter::getAttrConstraintFn,
              "constraint"_a)
          .def("get_successor_constraint_fn",
               &mlir::tblgen::StaticVerifierFunctionEmitter::
                   getSuccessorConstraintFn,
               "constraint"_a)
          .def("get_region_constraint_fn",
               &mlir::tblgen::StaticVerifierFunctionEmitter::
                   getRegionConstraintFn,
               "constraint"_a);

  m.def("escape_string", &mlir::tblgen::escapeString, "value"_a);

  auto mlir_tblgen_MethodParameter =
      nb::class_<mlir::tblgen::MethodParameter>(m, "MethodParameter")
          .def("write_decl_to", &mlir::tblgen::MethodParameter::writeDeclTo,
               "os"_a)
          .def("write_def_to", &mlir::tblgen::MethodParameter::writeDefTo,
               "os"_a)
          .def("get_type", &mlir::tblgen::MethodParameter::getType)
          .def("get_name", &mlir::tblgen::MethodParameter::getName)
          .def("has_default_value",
               &mlir::tblgen::MethodParameter::hasDefaultValue);

  auto mlir_tblgen_MethodParameters =
      nb::class_<mlir::tblgen::MethodParameters>(m, "MethodParameters")
          .def(nb::init<std::initializer_list<mlir::tblgen::MethodParameter>>(),
               "parameters"_a)
          .def(nb::init<llvm::SmallVector<mlir::tblgen::MethodParameter, 1>>(),
               "parameters"_a)
          .def("write_decl_to", &mlir::tblgen::MethodParameters::writeDeclTo,
               "os"_a)
          .def("write_def_to", &mlir::tblgen::MethodParameters::writeDefTo,
               "os"_a)
          .def("subsumes", &mlir::tblgen::MethodParameters::subsumes, "other"_a)
          .def("get_num_parameters",
               &mlir::tblgen::MethodParameters::getNumParameters);

  auto mlir_tblgen_MethodSignature =
      nb::class_<mlir::tblgen::MethodSignature>(m, "MethodSignature")
          .def("makes_redundant",
               &mlir::tblgen::MethodSignature::makesRedundant, "other"_a)
          .def("get_name", &mlir::tblgen::MethodSignature::getName)
          .def("get_return_type", &mlir::tblgen::MethodSignature::getReturnType)
          .def("get_num_parameters",
               &mlir::tblgen::MethodSignature::getNumParameters)
          .def("write_decl_to", &mlir::tblgen::MethodSignature::writeDeclTo,
               "os"_a)
          .def("write_def_to", &mlir::tblgen::MethodSignature::writeDefTo,
               "os"_a, "name_prefix"_a)
          .def("write_template_params_to",
               &mlir::tblgen::MethodSignature::writeTemplateParamsTo, "os"_a);

  auto mlir_tblgen_MethodBody =
      nb::class_<mlir::tblgen::MethodBody>(m, "MethodBody")
          .def(nb::init<bool>(), "decl_only"_a)
          .def(nb::init<mlir::tblgen::MethodBody &&>(), "other"_a)
          .def(
              "operator=",
              [](mlir::tblgen::MethodBody &self,
                 mlir::tblgen::MethodBody &&body)
                  -> mlir::tblgen::MethodBody & {
                return self.operator=(std::move(body));
              },
              "body"_a, nb::rv_policy::reference_internal)
          .def("write_to", &mlir::tblgen::MethodBody::writeTo, "os"_a)
          .def("indent", &mlir::tblgen::MethodBody::indent,
               nb::rv_policy::reference_internal)
          .def("unindent", &mlir::tblgen::MethodBody::unindent,
               nb::rv_policy::reference_internal)
          .def("scope", &mlir::tblgen::MethodBody::scope, "open"_a, "close"_a,
               "indent"_a)
          .def("get_stream", &mlir::tblgen::MethodBody::getStream,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_ClassDeclaration =
      nb::class_<mlir::tblgen::ClassDeclaration>(m, "ClassDeclaration");
  nb::enum_<mlir::tblgen::ClassDeclaration::Kind>(mlir_tblgen_ClassDeclaration,
                                                  "Kind")
      .value("Method", mlir::tblgen::ClassDeclaration::Method)
      .value("UsingDeclaration",
             mlir::tblgen::ClassDeclaration::UsingDeclaration)
      .value("VisibilityDeclaration",
             mlir::tblgen::ClassDeclaration::VisibilityDeclaration)
      .value("Field", mlir::tblgen::ClassDeclaration::Field)
      .value("ExtraClassDeclaration",
             mlir::tblgen::ClassDeclaration::ExtraClassDeclaration);

  auto mlir_tblgen_Method =
      nb::class_<mlir::tblgen::Method>(m, "Method")
          .def(nb::init<llvm::StringRef, llvm::StringRef,
                        mlir::tblgen::Method::Properties,
                        std::initializer_list<mlir::tblgen::MethodParameter>>(),
               "ret_type"_a, "name"_a, "properties"_a, "params"_a)
          .def(nb::init<mlir::tblgen::Method &&>(), "_"_a)
          .def(
              "operator=",
              [](mlir::tblgen::Method &self,
                 mlir::tblgen::Method &&_) -> mlir::tblgen::Method & {
                return self.operator=(std::move(_));
              },
              "_"_a, nb::rv_policy::reference_internal)
          .def("body", &mlir::tblgen::Method::body,
               nb::rv_policy::reference_internal)
          .def("set_deprecated", &mlir::tblgen::Method::setDeprecated,
               "message"_a)
          .def("is_static", &mlir::tblgen::Method::isStatic)
          .def("is_private", &mlir::tblgen::Method::isPrivate)
          .def("is_inline", &mlir::tblgen::Method::isInline)
          .def("is_constructor", &mlir::tblgen::Method::isConstructor)
          .def("is_const", &mlir::tblgen::Method::isConst)
          .def("get_name", &mlir::tblgen::Method::getName)
          .def("get_return_type", &mlir::tblgen::Method::getReturnType)
          .def("makes_redundant", &mlir::tblgen::Method::makesRedundant,
               "other"_a)
          .def("write_decl_to", &mlir::tblgen::Method::writeDeclTo, "os"_a)
          .def("write_def_to", &mlir::tblgen::Method::writeDefTo, "os"_a,
               "name_prefix"_a);

  nb::enum_<mlir::tblgen::Method::Properties>(mlir_tblgen_Method, "Properties")
      .value("None", mlir::tblgen::Method::None)
      .value("Static", mlir::tblgen::Method::Static)
      .value("Constructor", mlir::tblgen::Method::Constructor)
      .value("Private", mlir::tblgen::Method::Private)
      .value("Declaration", mlir::tblgen::Method::Declaration)
      .value("Inline", mlir::tblgen::Method::Inline)
      .value("ConstexprValue", mlir::tblgen::Method::ConstexprValue)
      .value("Const", mlir::tblgen::Method::Const)
      .value("Constexpr", mlir::tblgen::Method::Constexpr)
      .value("StaticDeclaration", mlir::tblgen::Method::StaticDeclaration)
      .value("StaticInline", mlir::tblgen::Method::StaticInline)
      .value("ConstInline", mlir::tblgen::Method::ConstInline)
      .value("ConstDeclaration", mlir::tblgen::Method::ConstDeclaration);

  nb::enum_<mlir::tblgen::Visibility>(m, "Visibility")
      .value("Public", mlir::tblgen::Visibility::Public)
      .value("Protected", mlir::tblgen::Visibility::Protected)
      .value("Private", mlir::tblgen::Visibility::Private);

  m.def(
      "operator<<",
      [](llvm::raw_ostream &os,
         mlir::tblgen::Visibility visibility) -> llvm::raw_ostream & {
        return mlir::tblgen::operator<<(os, visibility);
      },
      "os"_a, "visibility"_a);

  auto mlir_tblgen_Constructor =
      nb::class_<mlir::tblgen::Constructor, mlir::tblgen::Method>(m,
                                                                  "Constructor")
          .def("write_decl_to", &mlir::tblgen::Constructor::writeDeclTo, "os"_a)
          .def("write_def_to", &mlir::tblgen::Constructor::writeDefTo, "os"_a,
               "name_prefix"_a)
          .def_static("classof", &mlir::tblgen::Constructor::classof,
                      "other"_a);

  auto mlir_tblgen_Constructor_MemberInitializer =
      nb::class_<mlir::tblgen::Constructor::MemberInitializer>(
          mlir_tblgen_Constructor, "MemberInitializer")
          .def(nb::init<std::string, std::string>(), "name"_a, "value"_a)
          .def("write_to",
               &mlir::tblgen::Constructor::MemberInitializer::writeTo, "os"_a);

  auto mlir_tblgen_ParentClass =
      nb::class_<mlir::tblgen::ParentClass>(m, "ParentClass")
          .def("write_to", &mlir::tblgen::ParentClass::writeTo, "os"_a);

  auto mlir_tblgen_UsingDeclaration =
      nb::class_<mlir::tblgen::UsingDeclaration>(m, "UsingDeclaration")
          .def("write_decl_to", &mlir::tblgen::UsingDeclaration::writeDeclTo,
               "os"_a);

  auto mlir_tblgen_Field =
      nb::class_<mlir::tblgen::Field>(m, "Field")
          .def("write_decl_to", &mlir::tblgen::Field::writeDeclTo, "os"_a);

  auto mlir_tblgen_VisibilityDeclaration =
      nb::class_<mlir::tblgen::VisibilityDeclaration>(m,
                                                      "VisibilityDeclaration")
          .def(nb::init<mlir::tblgen::Visibility>(), "visibility"_a)
          .def("get_visibility",
               &mlir::tblgen::VisibilityDeclaration::getVisibility)
          .def("write_decl_to",
               &mlir::tblgen::VisibilityDeclaration::writeDeclTo, "os"_a);

  auto mlir_tblgen_ExtraClassDeclaration =
      nb::class_<mlir::tblgen::ExtraClassDeclaration>(m,
                                                      "ExtraClassDeclaration")
          .def(nb::init<llvm::StringRef, std::string>(),
               "extra_class_declaration"_a, "extra_class_definition"_a)
          .def(nb::init<std::string, std::string>(),
               "extra_class_declaration"_a, "extra_class_definition"_a)
          .def("write_decl_to",
               &mlir::tblgen::ExtraClassDeclaration::writeDeclTo, "os"_a)
          .def("write_def_to", &mlir::tblgen::ExtraClassDeclaration::writeDefTo,
               "os"_a, "name_prefix"_a);

  //     "add_parent",
  //     [](mlir::tblgen::Class &self, mlir::tblgen::ParentClass parent)
  //         -> mlir::tblgen::ParentClass & { return self.addParent(parent); },
  //     "parent"_a, nb::rv_policy::reference_internal)
  auto mlir_tblgen_Class =
      nb::class_<mlir::tblgen::Class>(m, "Class")
          .def("get_class_name", &mlir::tblgen::Class::getClassName)
          .def(
              "write_decl_to",
              [](mlir::tblgen::Class &self, llvm::raw_ostream &rawOs) -> void {
                return self.writeDeclTo(rawOs);
              },
              "raw_os"_a)
          .def(
              "write_def_to",
              [](mlir::tblgen::Class &self, llvm::raw_ostream &rawOs) -> void {
                return self.writeDefTo(rawOs);
              },
              "raw_os"_a)
          .def(
              "write_decl_to",
              [](mlir::tblgen::Class &self, mlir::raw_indented_ostream &os)
                  -> void { return self.writeDeclTo(os); },
              "os"_a)
          .def(
              "write_def_to",
              [](mlir::tblgen::Class &self, mlir::raw_indented_ostream &os)
                  -> void { return self.writeDefTo(os); },
              "os"_a)
          .def("finalize", &mlir::tblgen::Class::finalize);

  auto mlir_GenInfo =
      nb::class_<mlir::GenInfo>(m, "GenInfo")
          .def(nb::init<llvm::StringRef, llvm::StringRef,
                        std::function<bool(const llvm::RecordKeeper &,
                                           llvm::raw_ostream &)>>(),
               "arg"_a, "description"_a, "generator"_a)
          .def("invoke", &mlir::GenInfo::invoke, "records"_a, "os"_a)
          .def("get_gen_argument", &mlir::GenInfo::getGenArgument)
          .def("get_gen_description", &mlir::GenInfo::getGenDescription);

  auto mlir_GenRegistration =
      nb::class_<mlir::GenRegistration>(m, "GenRegistration");

  mlir_GenRegistration.def(
      nb::init<llvm::StringRef, llvm::StringRef,
               const std::function<bool(const llvm::RecordKeeper &,
                                        llvm::raw_ostream &)> &>(),
      "arg"_a, "description"_a, "function"_a);

  auto mlir_GenNameParser =
      nb::class_<mlir::GenNameParser>(m, "GenNameParser")
          .def(nb::init<llvm::cl::Option &>(), "opt"_a)
          .def("print_option_info", &mlir::GenNameParser::printOptionInfo,
               "o"_a, "global_width"_a);

  auto mlir_tblgen_InterfaceMethod =
      nb::class_<mlir::tblgen::InterfaceMethod>(m, "InterfaceMethod");

  auto mlir_tblgen_InterfaceMethod_Argument =
      nb::class_<mlir::tblgen::InterfaceMethod::Argument>(
          mlir_tblgen_InterfaceMethod, "Argument")
          .def_rw("type", &mlir::tblgen::InterfaceMethod::Argument::type)
          .def_rw("name", &mlir::tblgen::InterfaceMethod::Argument::name);

  mlir_tblgen_InterfaceMethod.def(nb::init<const llvm::Record *>(), "def_"_a)
      .def("get_return_type", &mlir::tblgen::InterfaceMethod::getReturnType)
      .def("get_name", &mlir::tblgen::InterfaceMethod::getName)
      .def("is_static", &mlir::tblgen::InterfaceMethod::isStatic)
      .def("get_body", &mlir::tblgen::InterfaceMethod::getBody)
      .def("get_default_implementation",
           &mlir::tblgen::InterfaceMethod::getDefaultImplementation)
      .def("get_description", &mlir::tblgen::InterfaceMethod::getDescription)
      .def("get_arguments", &mlir::tblgen::InterfaceMethod::getArguments)
      .def("arg_empty", &mlir::tblgen::InterfaceMethod::arg_empty);

  auto mlir_tblgen_Interface =
      nb::class_<mlir::tblgen::Interface>(m, "Interface")
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def(nb::init<const mlir::tblgen::Interface &>(), "rhs"_a)
          .def("get_name", &mlir::tblgen::Interface::getName)
          .def("get_fully_qualified_name",
               &mlir::tblgen::Interface::getFullyQualifiedName)
          .def("get_cpp_namespace", &mlir::tblgen::Interface::getCppNamespace)
          .def("get_methods", &mlir::tblgen::Interface::getMethods)
          .def("get_description", &mlir::tblgen::Interface::getDescription)
          .def("get_extra_class_declaration",
               &mlir::tblgen::Interface::getExtraClassDeclaration)
          .def("get_extra_trait_class_declaration",
               &mlir::tblgen::Interface::getExtraTraitClassDeclaration)
          .def("get_extra_shared_class_declaration",
               &mlir::tblgen::Interface::getExtraSharedClassDeclaration)
          .def("get_extra_class_of", &mlir::tblgen::Interface::getExtraClassOf)
          .def("get_verify", &mlir::tblgen::Interface::getVerify)
          .def("get_base_interfaces",
               &mlir::tblgen::Interface::getBaseInterfaces)
          .def("verify_with_regions",
               &mlir::tblgen::Interface::verifyWithRegions)
          .def("get_def", &mlir::tblgen::Interface::getDef,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_AttrInterface =
      nb::class_<mlir::tblgen::AttrInterface, mlir::tblgen::Interface>(
          m, "AttrInterface")
          .def_static("classof", &mlir::tblgen::AttrInterface::classof,
                      "interface"_a);

  auto mlir_tblgen_OpInterface =
      nb::class_<mlir::tblgen::OpInterface, mlir::tblgen::Interface>(
          m, "OpInterface")
          .def_static("classof", &mlir::tblgen::OpInterface::classof,
                      "interface"_a);

  auto mlir_tblgen_TypeInterface =
      nb::class_<mlir::tblgen::TypeInterface, mlir::tblgen::Interface>(
          m, "TypeInterface")
          .def_static("classof", &mlir::tblgen::TypeInterface::classof,
                      "interface"_a);

  auto mlir_tblgen_Region =
      nb::class_<mlir::tblgen::Region, mlir::tblgen::Constraint>(m, "Region")
          .def_static("classof", &mlir::tblgen::Region::classof, "c"_a)
          .def("is_variadic", &mlir::tblgen::Region::isVariadic);

  auto mlir_tblgen_NamedRegion =
      nb::class_<mlir::tblgen::NamedRegion>(m, "NamedRegion")
          .def("is_variadic", &mlir::tblgen::NamedRegion::isVariadic)
          .def_rw("name", &mlir::tblgen::NamedRegion::name)
          .def_rw("constraint", &mlir::tblgen::NamedRegion::constraint);

  auto mlir_tblgen_Successor =
      nb::class_<mlir::tblgen::Successor, mlir::tblgen::Constraint>(m,
                                                                    "Successor")
          .def_static("classof", &mlir::tblgen::Successor::classof, "c"_a)
          .def("is_variadic", &mlir::tblgen::Successor::isVariadic);

  auto mlir_tblgen_NamedSuccessor =
      nb::class_<mlir::tblgen::NamedSuccessor>(m, "NamedSuccessor")
          .def("is_variadic", &mlir::tblgen::NamedSuccessor::isVariadic)
          .def_rw("name", &mlir::tblgen::NamedSuccessor::name)
          .def_rw("constraint", &mlir::tblgen::NamedSuccessor::constraint);

  auto mlir_tblgen_InferredResultType =
      nb::class_<mlir::tblgen::InferredResultType>(m, "InferredResultType")
          .def(nb::init<int, std::string>(), "index"_a, "transformer"_a)
          .def("is_arg", &mlir::tblgen::InferredResultType::isArg)
          .def("get_index", &mlir::tblgen::InferredResultType::getIndex)
          .def("get_result_index",
               &mlir::tblgen::InferredResultType::getResultIndex)
          .def_static("map_result_index",
                      &mlir::tblgen::InferredResultType::mapResultIndex, "i"_a)
          .def_static("unmap_result_index",
                      &mlir::tblgen::InferredResultType::unmapResultIndex,
                      "i"_a)
          .def_static("is_result_index",
                      &mlir::tblgen::InferredResultType::isResultIndex, "i"_a)
          .def_static("is_arg_index",
                      &mlir::tblgen::InferredResultType::isArgIndex, "i"_a)
          .def("get_transformer",
               &mlir::tblgen::InferredResultType::getTransformer);

  auto mlir_tblgen_Operator =
      nb::class_<mlir::tblgen::Operator>(m, "Operator")
          .def(nb::init<const llvm::Record &>(), "def_"_a)
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def("get_dialect_name", &mlir::tblgen::Operator::getDialectName)
          .def("get_operation_name", &mlir::tblgen::Operator::getOperationName)
          .def("get_cpp_class_name", &mlir::tblgen::Operator::getCppClassName)
          .def("get_qual_cpp_class_name",
               &mlir::tblgen::Operator::getQualCppClassName)
          .def("get_cpp_namespace", &mlir::tblgen::Operator::getCppNamespace)
          .def("get_adaptor_name", &mlir::tblgen::Operator::getAdaptorName)
          .def("get_generic_adaptor_name",
               &mlir::tblgen::Operator::getGenericAdaptorName)
          .def("assert_invariants", &mlir::tblgen::Operator::assertInvariants)

          .def("is_variadic", &mlir::tblgen::Operator::isVariadic)
          .def("skip_default_builders",
               &mlir::tblgen::Operator::skipDefaultBuilders)
          .def("result_begin", &mlir::tblgen::Operator::result_begin,
               nb::rv_policy::reference_internal)
          .def("result_end", &mlir::tblgen::Operator::result_end,
               nb::rv_policy::reference_internal)
          .def("get_results", &mlir::tblgen::Operator::getResults)
          .def("get_num_results", &mlir::tblgen::Operator::getNumResults)
          .def(
              "get_result",
              [](mlir::tblgen::Operator &self,
                 int index) -> mlir::tblgen::NamedTypeConstraint & {
                return self.getResult(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def(
              "get_result",
              [](mlir::tblgen::Operator &self,
                 int index) -> const mlir::tblgen::NamedTypeConstraint & {
                return self.getResult(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def("get_result_type_constraint",
               &mlir::tblgen::Operator::getResultTypeConstraint, "index"_a)
          .def("get_result_name", &mlir::tblgen::Operator::getResultName,
               "index"_a)
          .def("get_result_decorators",
               &mlir::tblgen::Operator::getResultDecorators, "index"_a)
          .def("get_num_variable_length_results",
               &mlir::tblgen::Operator::getNumVariableLengthResults)
          .def(
              "attribute_begin",
              [](mlir::tblgen::Operator &self)
                  -> const mlir::tblgen::NamedAttribute * {
                return self.attribute_begin();
              },
              nb::rv_policy::reference_internal)
          .def(
              "attribute_end",
              [](mlir::tblgen::Operator &self)
                  -> const mlir::tblgen::NamedAttribute * {
                return self.attribute_end();
              },
              nb::rv_policy::reference_internal)
          .def("get_attributes",
               [](mlir::tblgen::Operator &self)
                   -> llvm::iterator_range<
                       const mlir::tblgen::NamedAttribute *> {
                 return self.getAttributes();
               })
          .def(
              "attribute_begin",
              [](mlir::tblgen::Operator &self)
                  -> mlir::tblgen::NamedAttribute * {
                return self.attribute_begin();
              },
              nb::rv_policy::reference_internal)
          .def(
              "attribute_end",
              [](mlir::tblgen::Operator &self)
                  -> mlir::tblgen::NamedAttribute * {
                return self.attribute_end();
              },
              nb::rv_policy::reference_internal)
          .def("get_attributes",
               [](mlir::tblgen::Operator &self)
                   -> llvm::iterator_range<mlir::tblgen::NamedAttribute *> {
                 return self.getAttributes();
               })
          .def("get_num_attributes", &mlir::tblgen::Operator::getNumAttributes)
          .def("get_num_native_attributes",
               &mlir::tblgen::Operator::getNumNativeAttributes)
          .def(
              "get_attribute",
              [](mlir::tblgen::Operator &self,
                 int index) -> mlir::tblgen::NamedAttribute & {
                return self.getAttribute(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def(
              "get_attribute",
              [](mlir::tblgen::Operator &self,
                 int index) -> const mlir::tblgen::NamedAttribute & {
                return self.getAttribute(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def("operand_begin", &mlir::tblgen::Operator::operand_begin,
               nb::rv_policy::reference_internal)
          .def("operand_end", &mlir::tblgen::Operator::operand_end,
               nb::rv_policy::reference_internal)
          .def("get_operands", &mlir::tblgen::Operator::getOperands)
          .def(
              "properties_begin",
              [](mlir::tblgen::Operator &self)
                  -> const mlir::tblgen::NamedProperty * {
                return self.properties_begin();
              },
              nb::rv_policy::reference_internal)
          .def(
              "properties_end",
              [](mlir::tblgen::Operator &self)
                  -> const mlir::tblgen::NamedProperty * {
                return self.properties_end();
              },
              nb::rv_policy::reference_internal)
          .def(
              "get_properties",
              [](mlir::tblgen::Operator &self)
                  -> llvm::iterator_range<const mlir::tblgen::NamedProperty *> {
                return self.getProperties();
              })
          .def(
              "properties_begin",
              [](mlir::tblgen::Operator &self)
                  -> mlir::tblgen::NamedProperty * {
                return self.properties_begin();
              },
              nb::rv_policy::reference_internal)
          .def(
              "properties_end",
              [](mlir::tblgen::Operator &self)
                  -> mlir::tblgen::NamedProperty * {
                return self.properties_end();
              },
              nb::rv_policy::reference_internal)
          .def("get_properties",
               [](mlir::tblgen::Operator &self)
                   -> llvm::iterator_range<mlir::tblgen::NamedProperty *> {
                 return self.getProperties();
               })
          .def("get_num_core_attributes",
               &mlir::tblgen::Operator::getNumCoreAttributes)
          .def(
              "get_property",
              [](mlir::tblgen::Operator &self,
                 int index) -> mlir::tblgen::NamedProperty & {
                return self.getProperty(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def(
              "get_property",
              [](mlir::tblgen::Operator &self,
                 int index) -> const mlir::tblgen::NamedProperty & {
                return self.getProperty(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def("get_num_operands", &mlir::tblgen::Operator::getNumOperands)
          .def(
              "get_operand",
              [](mlir::tblgen::Operator &self,
                 int index) -> mlir::tblgen::NamedTypeConstraint & {
                return self.getOperand(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def(
              "get_operand",
              [](mlir::tblgen::Operator &self,
                 int index) -> const mlir::tblgen::NamedTypeConstraint & {
                return self.getOperand(index);
              },
              "index"_a, nb::rv_policy::reference_internal)
          .def("get_num_variable_length_operands",
               &mlir::tblgen::Operator::getNumVariableLengthOperands)
          .def("get_num_args", &mlir::tblgen::Operator::getNumArgs)
          .def("has_single_variadic_arg",
               &mlir::tblgen::Operator::hasSingleVariadicArg)
          .def("has_single_variadic_result",
               &mlir::tblgen::Operator::hasSingleVariadicResult)
          .def("has_no_variadic_regions",
               &mlir::tblgen::Operator::hasNoVariadicRegions)
          .def("arg_begin", &mlir::tblgen::Operator::arg_begin,
               nb::rv_policy::reference_internal)
          .def("arg_end", &mlir::tblgen::Operator::arg_end,
               nb::rv_policy::reference_internal)
          .def("get_args", &mlir::tblgen::Operator::getArgs)
          .def("get_arg", &mlir::tblgen::Operator::getArg, "index"_a)
          .def("get_arg_name", &mlir::tblgen::Operator::getArgName, "index"_a)
          .def("get_arg_decorators", &mlir::tblgen::Operator::getArgDecorators,
               "index"_a)
          .def("get_trait", &mlir::tblgen::Operator::getTrait, "trait"_a,
               nb::rv_policy::reference_internal)
          .def("region_begin", &mlir::tblgen::Operator::region_begin,
               nb::rv_policy::reference_internal)
          .def("region_end", &mlir::tblgen::Operator::region_end,
               nb::rv_policy::reference_internal)
          .def("get_regions", &mlir::tblgen::Operator::getRegions)
          .def("get_num_regions", &mlir::tblgen::Operator::getNumRegions)
          .def("get_region", &mlir::tblgen::Operator::getRegion, "index"_a,
               nb::rv_policy::reference_internal)
          .def("get_num_variadic_regions",
               &mlir::tblgen::Operator::getNumVariadicRegions)
          .def("successor_begin", &mlir::tblgen::Operator::successor_begin,
               nb::rv_policy::reference_internal)
          .def("successor_end", &mlir::tblgen::Operator::successor_end,
               nb::rv_policy::reference_internal)
          .def("get_successors", &mlir::tblgen::Operator::getSuccessors)
          .def("get_num_successors", &mlir::tblgen::Operator::getNumSuccessors)
          .def("get_successor", &mlir::tblgen::Operator::getSuccessor,
               "index"_a, nb::rv_policy::reference_internal)
          .def("get_num_variadic_successors",
               &mlir::tblgen::Operator::getNumVariadicSuccessors)
          .def("trait_begin", &mlir::tblgen::Operator::trait_begin,
               nb::rv_policy::reference_internal)
          .def("trait_end", &mlir::tblgen::Operator::trait_end,
               nb::rv_policy::reference_internal)
          .def("get_traits", &mlir::tblgen::Operator::getTraits)
          .def("get_loc", &mlir::tblgen::Operator::getLoc)
          .def("has_description", &mlir::tblgen::Operator::hasDescription)
          .def("get_description", &mlir::tblgen::Operator::getDescription)
          .def("has_summary", &mlir::tblgen::Operator::hasSummary)
          .def("get_summary", &mlir::tblgen::Operator::getSummary)
          .def("has_assembly_format",
               &mlir::tblgen::Operator::hasAssemblyFormat)
          .def("get_assembly_format",
               &mlir::tblgen::Operator::getAssemblyFormat)
          .def("get_extra_class_declaration",
               &mlir::tblgen::Operator::getExtraClassDeclaration)
          .def("get_extra_class_definition",
               &mlir::tblgen::Operator::getExtraClassDefinition)
          .def("get_def", &mlir::tblgen::Operator::getDef,
               nb::rv_policy::reference_internal)
          .def("get_dialect", &mlir::tblgen::Operator::getDialect,
               nb::rv_policy::reference_internal)
          .def("print", &mlir::tblgen::Operator::print, "os"_a)
          .def("all_result_types_known",
               &mlir::tblgen::Operator::allResultTypesKnown)
          .def("get_inferred_result_type",
               &mlir::tblgen::Operator::getInferredResultType, "index"_a,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_Operator_VariableDecorator =
      nb::class_<mlir::tblgen::Operator::VariableDecorator>(
          mlir_tblgen_Operator, "VariableDecorator")
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def("get_def", &mlir::tblgen::Operator::VariableDecorator::getDef,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_Operator_OperandOrAttribute =
      nb::class_<mlir::tblgen::Operator::OperandOrAttribute>(
          mlir_tblgen_Operator, "OperandOrAttribute")
          .def(
              nb::init<mlir::tblgen::Operator::OperandOrAttribute::Kind, int>(),
              "kind"_a, "index"_a)
          .def("operand_or_attribute_index",
               &mlir::tblgen::Operator::OperandOrAttribute::
                   operandOrAttributeIndex)
          .def("kind", &mlir::tblgen::Operator::OperandOrAttribute::kind);

  nb::enum_<mlir::tblgen::Operator::OperandOrAttribute::Kind>(
      mlir_tblgen_Operator_OperandOrAttribute, "Kind")
      .value("Operand",
             mlir::tblgen::Operator::OperandOrAttribute::Kind::Operand)
      .value("Attribute",
             mlir::tblgen::Operator::OperandOrAttribute::Kind::Attribute);

  mlir_tblgen_Operator
      .def("get_arg_to_operand_or_attribute",
           &mlir::tblgen::Operator::getArgToOperandOrAttribute, "index"_a)
      .def("get_builders", &mlir::tblgen::Operator::getBuilders)
      .def("get_getter_name", &mlir::tblgen::Operator::getGetterName, "name"_a)
      .def("get_setter_name", &mlir::tblgen::Operator::getSetterName, "name"_a)
      .def("get_remover_name", &mlir::tblgen::Operator::getRemoverName,
           "name"_a)
      .def("has_folder", &mlir::tblgen::Operator::hasFolder)
      .def("use_custom_properties_encoding",
           &mlir::tblgen::Operator::useCustomPropertiesEncoding);

  auto mlir_tblgen_PassOption =
      nb::class_<mlir::tblgen::PassOption>(m, "PassOption")
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def("get_cpp_variable_name",
               &mlir::tblgen::PassOption::getCppVariableName)
          .def("get_argument", &mlir::tblgen::PassOption::getArgument)
          .def("get_type", &mlir::tblgen::PassOption::getType)
          .def("get_default_value", &mlir::tblgen::PassOption::getDefaultValue)
          .def("get_description", &mlir::tblgen::PassOption::getDescription)
          .def("get_additional_flags",
               &mlir::tblgen::PassOption::getAdditionalFlags)
          .def("is_list_option", &mlir::tblgen::PassOption::isListOption);

  auto mlir_tblgen_PassStatistic =
      nb::class_<mlir::tblgen::PassStatistic>(m, "PassStatistic")
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def("get_cpp_variable_name",
               &mlir::tblgen::PassStatistic::getCppVariableName)
          .def("get_name", &mlir::tblgen::PassStatistic::getName)
          .def("get_description", &mlir::tblgen::PassStatistic::getDescription);

  auto mlir_tblgen_Pass =
      nb::class_<mlir::tblgen::Pass>(m, "Pass")
          .def(nb::init<const llvm::Record *>(), "def_"_a)
          .def("get_argument", &mlir::tblgen::Pass::getArgument)
          .def("get_base_class", &mlir::tblgen::Pass::getBaseClass)
          .def("get_summary", &mlir::tblgen::Pass::getSummary)
          .def("get_description", &mlir::tblgen::Pass::getDescription)
          .def("get_constructor", &mlir::tblgen::Pass::getConstructor)
          .def("get_dependent_dialects",
               &mlir::tblgen::Pass::getDependentDialects)
          .def("get_options", &mlir::tblgen::Pass::getOptions)
          .def("get_statistics", &mlir::tblgen::Pass::getStatistics)
          .def("get_def", &mlir::tblgen::Pass::getDef,
               nb::rv_policy::reference_internal);

  auto mlir_tblgen_DagLeaf =
      nb::class_<mlir::tblgen::DagLeaf>(m, "DagLeaf")
          .def(nb::init<const llvm::Init *>(), "def_"_a)
          .def("is_unspecified", &mlir::tblgen::DagLeaf::isUnspecified)
          .def("is_operand_matcher", &mlir::tblgen::DagLeaf::isOperandMatcher)
          .def("is_attr_matcher", &mlir::tblgen::DagLeaf::isAttrMatcher)
          .def("is_native_code_call", &mlir::tblgen::DagLeaf::isNativeCodeCall)
          .def("is_constant_attr", &mlir::tblgen::DagLeaf::isConstantAttr)
          .def("is_enum_attr_case", &mlir::tblgen::DagLeaf::isEnumAttrCase)
          .def("is_string_attr", &mlir::tblgen::DagLeaf::isStringAttr)
          .def("get_as_constraint", &mlir::tblgen::DagLeaf::getAsConstraint)
          .def("get_as_constant_attr",
               &mlir::tblgen::DagLeaf::getAsConstantAttr)
          .def("get_as_enum_attr_case",
               &mlir::tblgen::DagLeaf::getAsEnumAttrCase)
          .def("get_condition_template",
               &mlir::tblgen::DagLeaf::getConditionTemplate)
          .def("get_native_code_template",
               &mlir::tblgen::DagLeaf::getNativeCodeTemplate)
          .def("get_num_returns_of_native_code",
               &mlir::tblgen::DagLeaf::getNumReturnsOfNativeCode)
          .def("get_string_attr", &mlir::tblgen::DagLeaf::getStringAttr)
          .def("print", &mlir::tblgen::DagLeaf::print, "os"_a);

  auto mlir_tblgen_DagNode =
      nb::class_<mlir::tblgen::DagNode>(m, "DagNode")
          .def(nb::init<const llvm::DagInit *>(), "node"_a)
          .def("operator bool",
               [](mlir::tblgen::DagNode &self) -> bool {
                 return self.operator bool();
               })
          .def("get_symbol", &mlir::tblgen::DagNode::getSymbol)
          .def("get_dialect_op", &mlir::tblgen::DagNode::getDialectOp,
               "mapper"_a, nb::rv_policy::reference_internal)
          .def("get_num_ops", &mlir::tblgen::DagNode::getNumOps)
          .def("get_num_args", &mlir::tblgen::DagNode::getNumArgs)
          .def("is_nested_dag_arg", &mlir::tblgen::DagNode::isNestedDagArg,
               "index"_a)
          .def("get_arg_as_nested_dag",
               &mlir::tblgen::DagNode::getArgAsNestedDag, "index"_a)
          .def("get_arg_as_leaf", &mlir::tblgen::DagNode::getArgAsLeaf,
               "index"_a)
          .def("get_arg_name", &mlir::tblgen::DagNode::getArgName, "index"_a)
          .def("is_replace_with_value",
               &mlir::tblgen::DagNode::isReplaceWithValue)
          .def("is_location_directive",
               &mlir::tblgen::DagNode::isLocationDirective)
          .def("is_return_type_directive",
               &mlir::tblgen::DagNode::isReturnTypeDirective)
          .def("is_native_code_call", &mlir::tblgen::DagNode::isNativeCodeCall)
          .def("is_either", &mlir::tblgen::DagNode::isEither)
          .def("is_variadic", &mlir::tblgen::DagNode::isVariadic)
          .def("is_operation", &mlir::tblgen::DagNode::isOperation)
          .def("get_native_code_template",
               &mlir::tblgen::DagNode::getNativeCodeTemplate)
          .def("get_num_returns_of_native_code",
               &mlir::tblgen::DagNode::getNumReturnsOfNativeCode)
          .def("print", &mlir::tblgen::DagNode::print, "os"_a);

  auto mlir_tblgen_SymbolInfoMap =
      nb::class_<mlir::tblgen::SymbolInfoMap>(m, "SymbolInfoMap")
          .def(nb::init<llvm::ArrayRef<llvm::SMLoc>>(), "loc"_a);

  auto mlir_tblgen_SymbolInfoMap_SymbolInfo =
      nb::class_<mlir::tblgen::SymbolInfoMap::SymbolInfo>(
          mlir_tblgen_SymbolInfoMap, "SymbolInfo")
          .def("get_var_type_str",
               &mlir::tblgen::SymbolInfoMap::SymbolInfo::getVarTypeStr,
               "name"_a)
          .def("get_var_decl",
               &mlir::tblgen::SymbolInfoMap::SymbolInfo::getVarDecl, "name"_a)
          .def("get_arg_decl",
               &mlir::tblgen::SymbolInfoMap::SymbolInfo::getArgDecl, "name"_a)
          .def("get_var_name",
               &mlir::tblgen::SymbolInfoMap::SymbolInfo::getVarName, "name"_a);

  mlir_tblgen_SymbolInfoMap
      .def(
          "begin",
          [](mlir::tblgen::SymbolInfoMap &self)
              -> std::__hash_map_iterator<std::__hash_iterator<std::__hash_node<
                  std::__hash_value_type<
                      std::string, mlir::tblgen::SymbolInfoMap::SymbolInfo>,
                  void *> *>> { return self.begin(); })
      .def(
          "end",
          [](mlir::tblgen::SymbolInfoMap &self)
              -> std::__hash_map_iterator<std::__hash_iterator<std::__hash_node<
                  std::__hash_value_type<
                      std::string, mlir::tblgen::SymbolInfoMap::SymbolInfo>,
                  void *> *>> { return self.end(); })
      .def(
          "begin",
          [](mlir::tblgen::SymbolInfoMap &self)
              -> std::__hash_map_const_iterator<
                  std::__hash_const_iterator<std::__hash_node<
                      std::__hash_value_type<
                          std::string, mlir::tblgen::SymbolInfoMap::SymbolInfo>,
                      void *> *>> { return self.begin(); })
      .def(
          "end",
          [](mlir::tblgen::SymbolInfoMap &self)
              -> std::__hash_map_const_iterator<
                  std::__hash_const_iterator<std::__hash_node<
                      std::__hash_value_type<
                          std::string, mlir::tblgen::SymbolInfoMap::SymbolInfo>,
                      void *> *>> { return self.end(); })
      .def("bind_op_argument", &mlir::tblgen::SymbolInfoMap::bindOpArgument,
           "node"_a, "symbol"_a, "op"_a, "arg_index"_a, "variadic_sub_index"_a)
      .def("bind_op_result", &mlir::tblgen::SymbolInfoMap::bindOpResult,
           "symbol"_a, "op"_a)
      .def("bind_values", &mlir::tblgen::SymbolInfoMap::bindValues, "symbol"_a,
           "num_values"_a)
      .def("bind_value", &mlir::tblgen::SymbolInfoMap::bindValue, "symbol"_a)
      .def("bind_multiple_values",
           &mlir::tblgen::SymbolInfoMap::bindMultipleValues, "symbol"_a,
           "num_values"_a)
      .def("bind_attr", &mlir::tblgen::SymbolInfoMap::bindAttr, "symbol"_a)
      .def("contains", &mlir::tblgen::SymbolInfoMap::contains, "symbol"_a)
      .def("find", &mlir::tblgen::SymbolInfoMap::find, "key"_a)
      .def(
          "find_bound_symbol",
          [](mlir::tblgen::SymbolInfoMap &self, llvm::StringRef key,
             mlir::tblgen::DagNode node, const mlir::tblgen::Operator &op,
             int argIndex, std::optional<int> variadicSubIndex)
              -> std::__hash_map_const_iterator<
                  std::__hash_const_iterator<std::__hash_node<
                      std::__hash_value_type<
                          std::string, mlir::tblgen::SymbolInfoMap::SymbolInfo>,
                      void *> *>> {
            return self.findBoundSymbol(key, node, op, argIndex,
                                        variadicSubIndex);
          },
          "key"_a, "node"_a, "op"_a, "arg_index"_a, "variadic_sub_index"_a)
      .def(
          "find_bound_symbol",
          [](mlir::tblgen::SymbolInfoMap &self, llvm::StringRef key,
             const mlir::tblgen::SymbolInfoMap::SymbolInfo &symbolInfo)
              -> std::__hash_map_const_iterator<
                  std::__hash_const_iterator<std::__hash_node<
                      std::__hash_value_type<
                          std::string, mlir::tblgen::SymbolInfoMap::SymbolInfo>,
                      void *> *>> {
            return self.findBoundSymbol(key, symbolInfo);
          },
          "key"_a, "symbol_info"_a)
      .def("get_range_of_equal_elements",
           &mlir::tblgen::SymbolInfoMap::getRangeOfEqualElements, "key"_a)
      .def("count", &mlir::tblgen::SymbolInfoMap::count, "key"_a)
      .def("get_static_value_count",
           &mlir::tblgen::SymbolInfoMap::getStaticValueCount, "symbol"_a)
      .def("get_value_and_range_use",
           &mlir::tblgen::SymbolInfoMap::getValueAndRangeUse, "symbol"_a,
           "fmt"_a, "separator"_a)
      .def("get_all_range_use", &mlir::tblgen::SymbolInfoMap::getAllRangeUse,
           "symbol"_a, "fmt"_a, "separator"_a)
      .def("assign_unique_alternative_names",
           &mlir::tblgen::SymbolInfoMap::assignUniqueAlternativeNames)
      .def_static("get_value_pack_name",
                  &mlir::tblgen::SymbolInfoMap::getValuePackName, "symbol"_a,
                  "index"_a);

  auto mlir_tblgen_Pattern =
      nb::class_<mlir::tblgen::Pattern>(m, "Pattern")
          .def(nb::init<
                   const llvm::Record *,
                   llvm::DenseMap<
                       const llvm::Record *,
                       std::unique_ptr<
                           mlir::tblgen::Operator,
                           std::default_delete<mlir::tblgen::Operator>>,
                       llvm::DenseMapInfo<const llvm::Record *, void>,
                       llvm::detail::DenseMapPair<
                           const llvm::Record *,
                           std::unique_ptr<mlir::tblgen::Operator,
                                           std::default_delete<
                                               mlir::tblgen::Operator>>>> *>(),
               "def_"_a, "mapper"_a)
          .def("get_source_pattern", &mlir::tblgen::Pattern::getSourcePattern)
          .def("get_num_result_patterns",
               &mlir::tblgen::Pattern::getNumResultPatterns)
          .def("get_result_pattern", &mlir::tblgen::Pattern::getResultPattern,
               "index"_a)
          .def("collect_source_pattern_bound_symbols",
               &mlir::tblgen::Pattern::collectSourcePatternBoundSymbols,
               "info_map"_a)
          .def("collect_result_pattern_bound_symbols",
               &mlir::tblgen::Pattern::collectResultPatternBoundSymbols,
               "info_map"_a)
          .def("get_source_root_op", &mlir::tblgen::Pattern::getSourceRootOp,
               nb::rv_policy::reference_internal)
          .def("get_dialect_op", &mlir::tblgen::Pattern::getDialectOp, "node"_a,
               nb::rv_policy::reference_internal)
          .def("get_constraints", &mlir::tblgen::Pattern::getConstraints)
          .def("get_num_supplemental_patterns",
               &mlir::tblgen::Pattern::getNumSupplementalPatterns)
          .def("get_supplemental_pattern",
               &mlir::tblgen::Pattern::getSupplementalPattern, "index"_a)
          .def("get_benefit", &mlir::tblgen::Pattern::getBenefit)
          .def("get_location", &mlir::tblgen::Pattern::getLocation)
          .def("collect_bound_symbols",
               &mlir::tblgen::Pattern::collectBoundSymbols, "tree"_a,
               "info_map"_a, "is_src_pattern"_a);

  auto mlir_tblgen_SideEffect =
      nb::class_<mlir::tblgen::SideEffect,
                 mlir::tblgen::Operator::VariableDecorator>(m, "SideEffect")
          .def("get_name", &mlir::tblgen::SideEffect::getName)
          .def("get_base_effect_name",
               &mlir::tblgen::SideEffect::getBaseEffectName)
          .def("get_interface_trait",
               &mlir::tblgen::SideEffect::getInterfaceTrait)
          .def("get_resource", &mlir::tblgen::SideEffect::getResource)
          .def("get_stage", &mlir::tblgen::SideEffect::getStage)
          .def("get_effect_onfull_region",
               &mlir::tblgen::SideEffect::getEffectOnfullRegion);

  mlir_tblgen_SideEffect.def_static(
      "classof", &mlir::tblgen::SideEffect::classof, "var"_a);

  auto mlir_tblgen_SideEffectTrait =
      nb::class_<mlir::tblgen::SideEffectTrait, mlir::tblgen::InterfaceTrait>(
          m, "SideEffectTrait")
          .def("get_effects", &mlir::tblgen::SideEffectTrait::getEffects)
          .def("get_base_effect_name",
               &mlir::tblgen::SideEffectTrait::getBaseEffectName);

  mlir_tblgen_SideEffectTrait.def_static(
      "classof", &mlir::tblgen::SideEffectTrait::classof, "t"_a);

  m.def("lookup_intrinsic_id", Intrinsic::lookupIntrinsicID, nb::arg("name"));
  m.def("intrinsic_is_overloaded", Intrinsic::isOverloaded, nb::arg("id"));
}
