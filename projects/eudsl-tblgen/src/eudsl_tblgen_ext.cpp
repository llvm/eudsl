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
// #include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Record.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_map.h>
// ReSharper disable once CppUnusedIncludeDirective
#include <nanobind/stl/shared_ptr.h>
// ReSharper disable once CppUnusedIncludeDirective
#include <nanobind/stl/string.h>
// ReSharper disable once CppUnusedIncludeDirective
#include <nanobind/stl/unique_ptr.h>
// ReSharper disable once CppUnusedIncludeDirective
#include <nanobind/stl/vector.h>

#include "eudsl/util.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "eudsl/bind_vec_like.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "eudsl/type_casters.h"

namespace nb = nanobind;
using namespace nb::literals;

// hack to expose protected Init::InitKind
struct HackInit : public llvm::Init {
  using InitKind = Init::InitKind;
};

namespace eudsl {
nb::class_<_SmallVector> smallVector;
nb::class_<_ArrayRef> arrayRef;
nb::class_<_MutableArrayRef> mutableArrayRef;
} // namespace eudsl

NB_MODULE(eudsl_tblgen_ext, m) {
  eudsl::bind_array_ref_smallvector(m);

  auto recty = nb::class_<llvm::RecTy>(m, "RecTy");

  nb::enum_<llvm::RecTy::RecTyKind>(m, "RecTyKind")
      .value("BitRecTyKind", llvm::RecTy::RecTyKind::BitRecTyKind)
      .value("BitsRecTyKind", llvm::RecTy::RecTyKind::BitsRecTyKind)
      .value("IntRecTyKind", llvm::RecTy::RecTyKind::IntRecTyKind)
      .value("StringRecTyKind", llvm::RecTy::RecTyKind::StringRecTyKind)
      .value("ListRecTyKind", llvm::RecTy::RecTyKind::ListRecTyKind)
      .value("DagRecTyKind", llvm::RecTy::RecTyKind::DagRecTyKind)
      .value("RecordRecTyKind", llvm::RecTy::RecTyKind::RecordRecTyKind);

  recty.def("get_rec_ty_kind", &llvm::RecTy::getRecTyKind)
      .def("get_record_keeper", &llvm::RecTy::getRecordKeeper,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::RecTy::getAsString)
      .def("__str__", &llvm::RecTy::getAsString)
      .def("print", &llvm::RecTy::print, "os"_a)
      .def("dump", &llvm::RecTy::dump)
      .def("type_is_convertible_to", &llvm::RecTy::typeIsConvertibleTo, "rhs"_a)
      .def("type_is_a", &llvm::RecTy::typeIsA, "rhs"_a)
      .def("get_list_ty", &llvm::RecTy::getListTy,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::BitRecTy, llvm::RecTy>(m, "BitRecTy")
      .def_static("classof", &llvm::BitRecTy::classof, "rt"_a)
      .def_static("get", &llvm::BitRecTy::get, "rk"_a,
                  nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::BitRecTy::getAsString)
      .def("__str__", &llvm::BitRecTy::getAsString)
      .def("type_is_convertible_to", &llvm::BitRecTy::typeIsConvertibleTo,
           "rhs"_a);

  nb::class_<llvm::IntRecTy, llvm::RecTy>(m, "IntRecTy")
      .def_static("classof", &llvm::IntRecTy::classof, "rt"_a)
      .def_static("get", &llvm::IntRecTy::get, "rk"_a,
                  nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::IntRecTy::getAsString)
      .def("__str__", &llvm::IntRecTy::getAsString)
      .def("type_is_convertible_to", &llvm::IntRecTy::typeIsConvertibleTo,
           "rhs"_a);

  nb::class_<llvm::StringRecTy, llvm::RecTy>(m, "StringRecTy")
      .def_static("classof", &llvm::StringRecTy::classof, "rt"_a)
      .def_static("get", &llvm::StringRecTy::get, "rk"_a,
                  nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::StringRecTy::getAsString)
      .def("__init__", &llvm::StringRecTy::getAsString)
      .def("type_is_convertible_to", &llvm::StringRecTy::typeIsConvertibleTo,
           "rhs"_a);

  nb::class_<llvm::ListRecTy, llvm::RecTy>(m, "ListRecTy")
      .def_static("classof", &llvm::ListRecTy::classof, "rt"_a)
      .def_static("get", &llvm::ListRecTy::get, "t"_a,
                  nb::rv_policy::reference_internal)
      .def("get_element_type", &llvm::ListRecTy::getElementType,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::ListRecTy::getAsString)
      .def("__init__", &llvm::ListRecTy::getAsString)
      .def("type_is_convertible_to", &llvm::ListRecTy::typeIsConvertibleTo,
           "rhs"_a)
      .def("type_is_a", &llvm::ListRecTy::typeIsA, "rhs"_a);

  nb::class_<llvm::DagRecTy, llvm::RecTy>(m, "DagRecTy")
      .def_static("classof", &llvm::DagRecTy::classof, "rt"_a)
      .def_static("get", &llvm::DagRecTy::get, "rk"_a,
                  nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::DagRecTy::getAsString)
      .def("__init__", &llvm::DagRecTy::getAsString);

  nb::class_<llvm::RecordRecTy, llvm::RecTy>(m, "RecordRecTy")
      .def_static("classof", &llvm::RecordRecTy::classof, "rt"_a)
      .def_static(
          "get",
          [](llvm::RecordKeeper &RK,
             llvm::ArrayRef<const llvm::Record *> Classes)
              -> const llvm::RecordRecTy * {
            return llvm::RecordRecTy::get(RK, Classes);
          },
          "rk"_a, "classes"_a, nb::rv_policy::reference_internal)
      .def_static(
          "get",
          [](const llvm::Record *Class) -> const llvm::RecordRecTy * {
            return llvm::RecordRecTy::get(Class);
          },
          "class"_a, nb::rv_policy::reference_internal)
      .def("profile", &llvm::RecordRecTy::Profile, "id"_a)
      .def("get_classes",
           eudsl::coerceReturn<std::vector<const llvm::Record *>>(
               &llvm::RecordRecTy::getClasses, nb::const_),
           nb::rv_policy::reference_internal)
      .def("classes_begin", &llvm::RecordRecTy::classes_begin,
           nb::rv_policy::reference_internal)
      .def("classes_end", &llvm::RecordRecTy::classes_end,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::RecordRecTy::getAsString)
      .def("__str__", &llvm::RecordRecTy::getAsString)
      .def("is_sub_class_of", &llvm::RecordRecTy::isSubClassOf, "class"_a)
      .def("type_is_convertible_to", &llvm::RecordRecTy::typeIsConvertibleTo,
           "rhs"_a)
      .def("type_is_a", &llvm::RecordRecTy::typeIsA, "rhs"_a);

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

  nb::class_<llvm::Init>(m, "Init")
      .def("get_kind", &llvm::Init::getKind)
      .def("get_record_keeper", &llvm::Init::getRecordKeeper,
           nb::rv_policy::reference_internal)
      .def("is_complete", &llvm::Init::isComplete)
      .def("is_concrete", &llvm::Init::isConcrete)
      .def("print", &llvm::Init::print, "os"_a)
      .def("get_as_string", &llvm::Init::getAsString)
      .def("__str__", &llvm::Init::getAsUnquotedString)
      .def("get_as_unquoted_string", &llvm::Init::getAsUnquotedString)
      .def("dump", &llvm::Init::dump)
      .def("get_cast_to", &llvm::Init::getCastTo, "ty"_a,
           nb::rv_policy::reference_internal)
      .def("convert_initializer_to", &llvm::Init::convertInitializerTo, "ty"_a,
           nb::rv_policy::reference_internal)
      .def("convert_initializer_bit_range",
           &llvm::Init::convertInitializerBitRange, "bits"_a,
           nb::rv_policy::reference_internal)
      .def("get_field_type", &llvm::Init::getFieldType, "field_name"_a,
           nb::rv_policy::reference_internal)
      .def("resolve_references", &llvm::Init::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::Init::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::TypedInit, llvm::Init>(m, "TypedInit")
      .def_static("classof", &llvm::TypedInit::classof, "i"_a)
      .def("get_type", &llvm::TypedInit::getType,
           nb::rv_policy::reference_internal)
      .def("get_record_keeper", &llvm::TypedInit::getRecordKeeper,
           nb::rv_policy::reference_internal)
      .def("get_cast_to", &llvm::TypedInit::getCastTo, "ty"_a,
           nb::rv_policy::reference_internal)
      .def("convert_initializer_to", &llvm::TypedInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal)
      .def("convert_initializer_bit_range",
           &llvm::TypedInit::convertInitializerBitRange, "bits"_a,
           nb::rv_policy::reference_internal)
      .def("get_field_type", &llvm::TypedInit::getFieldType, "field_name"_a,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::UnsetInit, llvm::Init>(m, "UnsetInit")
      .def_static("classof", &llvm::UnsetInit::classof, "i"_a)
      .def_static("get", &llvm::UnsetInit::get, "rk"_a,
                  nb::rv_policy::reference_internal)
      .def("get_record_keeper", &llvm::UnsetInit::getRecordKeeper,
           nb::rv_policy::reference_internal)
      .def("get_cast_to", &llvm::UnsetInit::getCastTo, "ty"_a,
           nb::rv_policy::reference_internal)
      .def("convert_initializer_to", &llvm::UnsetInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::UnsetInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("is_complete", &llvm::UnsetInit::isComplete)
      .def("is_concrete", &llvm::UnsetInit::isConcrete)
      .def("get_as_string", &llvm::UnsetInit::getAsString)
      .def("__str__", &llvm::UnsetInit::getAsString);

  auto llvm_ArgumentInit = nb::class_<llvm::ArgumentInit>(m, "ArgumentInit");

  nb::enum_<llvm::ArgumentInit::Kind>(llvm_ArgumentInit, "Kind")
      .value("Positional", llvm::ArgumentInit::Positional)
      .value("Named", llvm::ArgumentInit::Named);

  llvm_ArgumentInit.def_static("classof", &llvm::ArgumentInit::classof, "i"_a)
      .def("get_record_keeper", &llvm::ArgumentInit::getRecordKeeper,
           nb::rv_policy::reference_internal)
      .def_static("get", &llvm::ArgumentInit::get, "value"_a, "aux"_a,
                  nb::rv_policy::reference_internal)
      .def("is_positional", &llvm::ArgumentInit::isPositional)
      .def("is_named", &llvm::ArgumentInit::isNamed)
      .def("get_value", &llvm::ArgumentInit::getValue,
           nb::rv_policy::reference_internal)
      .def("get_index", &llvm::ArgumentInit::getIndex)
      .def("get_name", &llvm::ArgumentInit::getName,
           nb::rv_policy::reference_internal)
      .def("clone_with_value", &llvm::ArgumentInit::cloneWithValue, "value"_a,
           nb::rv_policy::reference_internal)
      .def("profile", &llvm::ArgumentInit::Profile, "id"_a)
      .def("resolve_references", &llvm::ArgumentInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::ArgumentInit::getAsString)
      .def("__str__", &llvm::ArgumentInit::getAsString)
      .def("is_complete", &llvm::ArgumentInit::isComplete)
      .def("is_concrete", &llvm::ArgumentInit::isConcrete)
      .def("get_bit", &llvm::ArgumentInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("get_cast_to", &llvm::ArgumentInit::getCastTo, "ty"_a,
           nb::rv_policy::reference_internal)
      .def("convert_initializer_to", &llvm::ArgumentInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal);

  nb::class_<llvm::BitInit, llvm::TypedInit>(m, "BitInit")
      .def_static("classof", &llvm::BitInit::classof, "i"_a)
      .def_static("get", &llvm::BitInit::get, "rk"_a, "v"_a,
                  nb::rv_policy::reference_internal)
      .def("get_value", &llvm::BitInit::getValue)
      .def("__bool__", &llvm::BitInit::getValue)
      .def("convert_initializer_to", &llvm::BitInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::BitInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("is_concrete", &llvm::BitInit::isConcrete)
      .def("get_as_string", &llvm::BitInit::getAsString)
      .def("__str__", &llvm::BitInit::getAsString);

  nb::class_<llvm::BitsInit, llvm::TypedInit>(m, "BitsInit")
      .def_static("classof", &llvm::BitsInit::classof, "i"_a)
      .def_static("get", &llvm::BitsInit::get, "rk"_a, "range"_a,
                  nb::rv_policy::reference_internal)
      .def("profile", &llvm::BitsInit::Profile, "id"_a)
      .def("get_num_bits", &llvm::BitsInit::getNumBits)
      .def("convert_initializer_to", &llvm::BitsInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal)
      .def("convert_initializer_bit_range",
           &llvm::BitsInit::convertInitializerBitRange, "bits"_a,
           nb::rv_policy::reference_internal)
      .def("convert_initializer_to_int",
           &llvm::BitsInit::convertInitializerToInt)
      .def("is_complete", &llvm::BitsInit::isComplete)
      .def("all_in_complete", &llvm::BitsInit::allInComplete)
      .def("is_concrete", &llvm::BitsInit::isConcrete)
      .def("get_as_string", &llvm::BitsInit::getAsString)
      .def("__str__", &llvm::BitsInit::getAsString)
      .def("resolve_references", &llvm::BitsInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::BitsInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::IntInit, llvm::TypedInit>(m, "IntInit")
      .def_static("classof", &llvm::IntInit::classof, "i"_a)
      .def_static("get", &llvm::IntInit::get, "rk"_a, "v"_a,
                  nb::rv_policy::reference_internal)
      .def("get_value", &llvm::IntInit::getValue)
      .def("__int__", &llvm::IntInit::getValue)
      .def("convert_initializer_to", &llvm::IntInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal)
      .def("convert_initializer_bit_range",
           &llvm::IntInit::convertInitializerBitRange, "bits"_a,
           nb::rv_policy::reference_internal)
      .def("is_concrete", &llvm::IntInit::isConcrete)
      .def("get_as_string", &llvm::IntInit::getAsString)
      .def("__str__", &llvm::IntInit::getAsString)
      .def("get_bit", &llvm::IntInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::AnonymousNameInit, llvm::TypedInit>(m, "AnonymousNameInit")
      .def_static("classof", &llvm::AnonymousNameInit::classof, "i"_a)
      .def_static("get", &llvm::AnonymousNameInit::get, "rk"_a, "__"_a,
                  nb::rv_policy::reference_internal)
      .def("get_value", &llvm::AnonymousNameInit::getValue)
      .def("get_name_init", &llvm::AnonymousNameInit::getNameInit,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::AnonymousNameInit::getAsString)
      .def("__str__", &llvm::AnonymousNameInit::getAsString)
      .def("resolve_references", &llvm::AnonymousNameInit::resolveReferences,
           "r"_a, nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::AnonymousNameInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  auto llvm_StringInit =
      nb::class_<llvm::StringInit, llvm::TypedInit>(m, "StringInit");
  nb::enum_<llvm::StringInit::StringFormat>(llvm_StringInit, "StringFormat")
      .value("SF_String", llvm::StringInit::SF_String)
      .value("SF_Code", llvm::StringInit::SF_Code);

  llvm_StringInit.def_static("classof", &llvm::StringInit::classof, "i"_a)
      .def_static("get", &llvm::StringInit::get, "rk"_a, "__"_a, "fmt"_a,
                  nb::rv_policy::reference_internal)
      .def_static("determine_format", &llvm::StringInit::determineFormat,
                  "fmt1"_a, "fmt2"_a)
      .def("get_value", &llvm::StringInit::getValue)
      .def("get_format", &llvm::StringInit::getFormat)
      .def("has_code_format", &llvm::StringInit::hasCodeFormat)
      .def("convert_initializer_to", &llvm::StringInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal)
      .def("is_concrete", &llvm::StringInit::isConcrete)
      .def("get_as_string", &llvm::StringInit::getAsString)
      .def("__str__", &llvm::StringInit::getAsUnquotedString)
      .def("get_as_unquoted_string", &llvm::StringInit::getAsUnquotedString)
      .def("get_bit", &llvm::StringInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  auto llvm_ListInit =
      nb::class_<llvm::ListInit, llvm::TypedInit>(m, "ListInit")
          .def_static("classof", &llvm::ListInit::classof, "i"_a)
          .def_static("get", &llvm::ListInit::get, "range"_a, "elt_ty"_a,
                      nb::rv_policy::reference_internal)
          .def("profile", &llvm::ListInit::Profile, "id"_a)
          .def("get_element", &llvm::ListInit::getElement, "i"_a,
               nb::rv_policy::reference_internal)
          .def("get_element_type", &llvm::ListInit::getElementType,
               nb::rv_policy::reference_internal)
          .def("get_element_as_record", &llvm::ListInit::getElementAsRecord,
               "i"_a, nb::rv_policy::reference_internal)
          .def("convert_initializer_to", &llvm::ListInit::convertInitializerTo,
               "ty"_a, nb::rv_policy::reference_internal)
          .def("resolve_references", &llvm::ListInit::resolveReferences, "r"_a,
               nb::rv_policy::reference_internal)
          .def("is_complete", &llvm::ListInit::isComplete)
          .def("is_concrete", &llvm::ListInit::isConcrete)
          .def("get_as_string", &llvm::ListInit::getAsString)
          .def("begin", &llvm::ListInit::begin,
               nb::rv_policy::reference_internal)
          .def("end", &llvm::ListInit::end, nb::rv_policy::reference_internal)
          .def("size", &llvm::ListInit::size)
          .def("empty", &llvm::ListInit::empty)
          .def("get_bit", &llvm::ListInit::getBit, "bit"_a,
               nb::rv_policy::reference_internal)
          .def("__len__", [](const llvm::ListInit &v) { return v.size(); })
          .def("__bool__", [](const llvm::ListInit &v) { return !v.empty(); })
          .def(
              "__iter__",
              [](llvm::ListInit &v) {
                return nb::make_iterator<nb::rv_policy::reference_internal>(
                    nb::type<llvm::ListInit>(), "Iterator", v.begin(), v.end());
              },
              nb::rv_policy::reference_internal)
          .def(
              "__getitem__",
              [](llvm::ListInit &v, Py_ssize_t i) {
                return v.getElement(eudsl::wrap(i, v.size()));
              },
              nb::rv_policy::reference_internal)
          .def("get_elements",
               eudsl::coerceReturn<std::vector<const llvm::Init *>>(
                   &llvm::ListInit::getElements, nb::const_));

  auto llvm_OpInit = nb::class_<llvm::OpInit, llvm::TypedInit>(m, "OpInit")
                         .def_static("classof", &llvm::OpInit::classof, "i"_a)
                         .def("get_bit", &llvm::OpInit::getBit, "bit"_a,
                              nb::rv_policy::reference_internal);

  auto unaryOpInit = nb::class_<llvm::UnOpInit, llvm::OpInit>(m, "UnOpInit");
  nb::enum_<llvm::UnOpInit::UnaryOp>(m, "UnaryOp")
      .value("TOLOWER", llvm::UnOpInit::UnaryOp::TOLOWER)
      .value("TOUPPER", llvm::UnOpInit::UnaryOp::TOUPPER)
      .value("CAST", llvm::UnOpInit::UnaryOp::CAST)
      .value("NOT", llvm::UnOpInit::UnaryOp::NOT)
      .value("HEAD", llvm::UnOpInit::UnaryOp::HEAD)
      .value("TAIL", llvm::UnOpInit::UnaryOp::TAIL)
      .value("SIZE", llvm::UnOpInit::UnaryOp::SIZE)
      .value("EMPTY", llvm::UnOpInit::UnaryOp::EMPTY)
      .value("GETDAGOP", llvm::UnOpInit::UnaryOp::GETDAGOP)
      .value("LOG2", llvm::UnOpInit::UnaryOp::LOG2)
      .value("REPR", llvm::UnOpInit::UnaryOp::REPR)
      .value("LISTFLATTEN", llvm::UnOpInit::UnaryOp::LISTFLATTEN);

  unaryOpInit.def_static("classof", &llvm::UnOpInit::classof, "i"_a)
      .def_static("get", &llvm::UnOpInit::get, "opc"_a, "lhs"_a, "type"_a,
                  nb::rv_policy::reference_internal)
      .def("profile", &llvm::UnOpInit::Profile, "id"_a)
      .def(
          "get_operand",
          [](llvm::UnOpInit &self) -> const llvm::Init * {
            return self.getOperand();
          },
          nb::rv_policy::reference_internal)
      .def("get_opcode", &llvm::UnOpInit::getOpcode)
      .def(
          "get_operand",
          [](llvm::UnOpInit &self) -> const llvm::Init * {
            return self.getOperand();
          },
          nb::rv_policy::reference_internal)
      .def("fold", &llvm::UnOpInit::Fold, "cur_rec"_a, "is_final"_a,
           nb::rv_policy::reference_internal)
      .def("resolve_references", &llvm::UnOpInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::UnOpInit::getAsString)
      .def("__str__", &llvm::UnOpInit::getAsUnquotedString);

  auto binaryOpInit = nb::class_<llvm::BinOpInit, llvm::OpInit>(m, "BinOpInit");
  nb::enum_<llvm::BinOpInit::BinaryOp>(m, "BinaryOp")
      .value("ADD", llvm::BinOpInit::BinaryOp::ADD)
      .value("SUB", llvm::BinOpInit::BinaryOp::SUB)
      .value("MUL", llvm::BinOpInit::BinaryOp::MUL)
      .value("DIV", llvm::BinOpInit::BinaryOp::DIV)
      .value("AND", llvm::BinOpInit::BinaryOp::AND)
      .value("OR", llvm::BinOpInit::BinaryOp::OR)
      .value("XOR", llvm::BinOpInit::BinaryOp::XOR)
      .value("SHL", llvm::BinOpInit::BinaryOp::SHL)
      .value("SRA", llvm::BinOpInit::BinaryOp::SRA)
      .value("SRL", llvm::BinOpInit::BinaryOp::SRL)
      .value("LISTCONCAT", llvm::BinOpInit::BinaryOp::LISTCONCAT)
      .value("LISTSPLAT", llvm::BinOpInit::BinaryOp::LISTSPLAT)
      .value("LISTREMOVE", llvm::BinOpInit::BinaryOp::LISTREMOVE)
      .value("LISTELEM", llvm::BinOpInit::BinaryOp::LISTELEM)
      .value("LISTSLICE", llvm::BinOpInit::BinaryOp::LISTSLICE)
      .value("RANGEC", llvm::BinOpInit::BinaryOp::RANGEC)
      .value("STRCONCAT", llvm::BinOpInit::BinaryOp::STRCONCAT)
      .value("INTERLEAVE", llvm::BinOpInit::BinaryOp::INTERLEAVE)
      .value("CONCAT", llvm::BinOpInit::BinaryOp::CONCAT)
      .value("EQ", llvm::BinOpInit::BinaryOp::EQ)
      .value("NE", llvm::BinOpInit::BinaryOp::NE)
      .value("LE", llvm::BinOpInit::BinaryOp::LE)
      .value("LT", llvm::BinOpInit::BinaryOp::LT)
      .value("GE", llvm::BinOpInit::BinaryOp::GE)
      .value("GT", llvm::BinOpInit::BinaryOp::GT)
      .value("GETDAGARG", llvm::BinOpInit::BinaryOp::GETDAGARG)
      .value("GETDAGNAME", llvm::BinOpInit::BinaryOp::GETDAGNAME)
      .value("SETDAGOP", llvm::BinOpInit::BinaryOp::SETDAGOP);

  binaryOpInit.def_static("classof", &llvm::BinOpInit::classof, "i"_a)
      .def_static("get", &llvm::BinOpInit::get, "opc"_a, "lhs"_a, "rhs"_a,
                  "type"_a, nb::rv_policy::reference_internal)
      .def_static("get_str_concat", &llvm::BinOpInit::getStrConcat, "lhs"_a,
                  "rhs"_a, nb::rv_policy::reference_internal)
      .def_static("get_list_concat", &llvm::BinOpInit::getListConcat, "lhs"_a,
                  "rhs"_a, nb::rv_policy::reference_internal)
      .def("profile", &llvm::BinOpInit::Profile, "id"_a)
      .def("get_opcode", &llvm::BinOpInit::getOpcode)
      .def("get_lhs", &llvm::BinOpInit::getLHS,
           nb::rv_policy::reference_internal)
      .def("get_rhs", &llvm::BinOpInit::getRHS,
           nb::rv_policy::reference_internal)
      .def("compare_init", &llvm::BinOpInit::CompareInit, "opc"_a, "lhs"_a,
           "rhs"_a)
      .def("fold", &llvm::BinOpInit::Fold, "cur_rec"_a,
           nb::rv_policy::reference_internal)
      .def("resolve_references", &llvm::BinOpInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::BinOpInit::getAsString)
      .def("__str__", &llvm::BinOpInit::getAsUnquotedString);

  auto ternaryOpInit =
      nb::class_<llvm::TernOpInit, llvm::OpInit>(m, "TernOpInit");
  nb::enum_<llvm::TernOpInit::TernaryOp>(m, "TernaryOp")
      .value("SUBST", llvm::TernOpInit::TernaryOp::SUBST)
      .value("FOREACH", llvm::TernOpInit::TernaryOp::FOREACH)
      .value("FILTER", llvm::TernOpInit::TernaryOp::FILTER)
      .value("IF", llvm::TernOpInit::TernaryOp::IF)
      .value("DAG", llvm::TernOpInit::TernaryOp::DAG)
      .value("RANGE", llvm::TernOpInit::TernaryOp::RANGE)
      .value("SUBSTR", llvm::TernOpInit::TernaryOp::SUBSTR)
      .value("FIND", llvm::TernOpInit::TernaryOp::FIND)
      .value("SETDAGARG", llvm::TernOpInit::TernaryOp::SETDAGARG)
      .value("SETDAGNAME", llvm::TernOpInit::TernaryOp::SETDAGNAME);

  ternaryOpInit.def_static("classof", &llvm::TernOpInit::classof, "i"_a)
      .def_static("get", &llvm::TernOpInit::get, "opc"_a, "lhs"_a, "mhs"_a,
                  "rhs"_a, "type"_a, nb::rv_policy::reference_internal)
      .def("profile", &llvm::TernOpInit::Profile, "id"_a)
      .def("get_opcode", &llvm::TernOpInit::getOpcode)
      .def("get_lhs", &llvm::TernOpInit::getLHS,
           nb::rv_policy::reference_internal)
      .def("get_mhs", &llvm::TernOpInit::getMHS,
           nb::rv_policy::reference_internal)
      .def("get_rhs", &llvm::TernOpInit::getRHS,
           nb::rv_policy::reference_internal)
      .def("fold", &llvm::TernOpInit::Fold, "cur_rec"_a,
           nb::rv_policy::reference_internal)
      .def("is_complete", &llvm::TernOpInit::isComplete)
      .def("resolve_references", &llvm::TernOpInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::TernOpInit::getAsString)
      .def("__str__", &llvm::TernOpInit::getAsUnquotedString);

  nb::class_<llvm::CondOpInit, llvm::TypedInit>(m, "CondOpInit")
      .def_static("classof", &llvm::CondOpInit::classof, "i"_a)
      .def_static("get", &llvm::CondOpInit::get, "c"_a, "v"_a, "type"_a,
                  nb::rv_policy::reference_internal)
      .def("profile", &llvm::CondOpInit::Profile, "id"_a)
      .def("get_val_type", &llvm::CondOpInit::getValType,
           nb::rv_policy::reference_internal)
      .def("get_num_conds", &llvm::CondOpInit::getNumConds)
      .def("get_cond", &llvm::CondOpInit::getCond, "num"_a,
           nb::rv_policy::reference_internal)
      .def("get_val", &llvm::CondOpInit::getVal, "num"_a,
           nb::rv_policy::reference_internal)
      .def("get_conds", &llvm::CondOpInit::getConds)
      .def("get_vals", &llvm::CondOpInit::getVals)
      .def("fold", &llvm::CondOpInit::Fold, "cur_rec"_a,
           nb::rv_policy::reference_internal)
      .def("resolve_references", &llvm::CondOpInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("is_concrete", &llvm::CondOpInit::isConcrete)
      .def("is_complete", &llvm::CondOpInit::isComplete)
      .def("get_as_string", &llvm::CondOpInit::getAsString)
      .def("__str__", &llvm::CondOpInit::getAsUnquotedString)
      .def("arg_begin", &llvm::CondOpInit::arg_begin,
           nb::rv_policy::reference_internal)
      .def("arg_end", &llvm::CondOpInit::arg_end,
           nb::rv_policy::reference_internal)
      .def("case_size", &llvm::CondOpInit::case_size)
      .def("case_empty", &llvm::CondOpInit::case_empty)
      .def("name_begin", &llvm::CondOpInit::name_begin,
           nb::rv_policy::reference_internal)
      .def("name_end", &llvm::CondOpInit::name_end,
           nb::rv_policy::reference_internal)
      .def("val_size", &llvm::CondOpInit::val_size)
      .def("val_empty", &llvm::CondOpInit::val_empty)
      .def("get_bit", &llvm::CondOpInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::FoldOpInit, llvm::TypedInit>(m, "FoldOpInit")
      .def_static("classof", &llvm::FoldOpInit::classof, "i"_a)
      .def_static("get", &llvm::FoldOpInit::get, "start"_a, "list"_a, "a"_a,
                  "b"_a, "expr"_a, "type"_a, nb::rv_policy::reference_internal)
      .def("profile", &llvm::FoldOpInit::Profile, "id"_a)
      .def("fold", &llvm::FoldOpInit::Fold, "cur_rec"_a,
           nb::rv_policy::reference_internal)
      .def("is_complete", &llvm::FoldOpInit::isComplete)
      .def("resolve_references", &llvm::FoldOpInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::FoldOpInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::FoldOpInit::getAsString)
      .def("__str__", &llvm::FoldOpInit::getAsString);

  nb::class_<llvm::IsAOpInit, llvm::TypedInit>(m, "IsAOpInit")
      .def_static("classof", &llvm::IsAOpInit::classof, "i"_a)
      .def_static("get", &llvm::IsAOpInit::get, "check_type"_a, "expr"_a,
                  nb::rv_policy::reference_internal)
      .def("profile", &llvm::IsAOpInit::Profile, "id"_a)
      .def("fold", &llvm::IsAOpInit::Fold, nb::rv_policy::reference_internal)
      .def("is_complete", &llvm::IsAOpInit::isComplete)
      .def("resolve_references", &llvm::IsAOpInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::IsAOpInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("__str__", &llvm::IsAOpInit::getAsString)
      .def("get_as_string", &llvm::IsAOpInit::getAsString);

  nb::class_<llvm::ExistsOpInit, llvm::TypedInit>(m, "ExistsOpInit")
      .def_static("classof", &llvm::ExistsOpInit::classof, "i"_a)
      .def_static("get", &llvm::ExistsOpInit::get, "check_type"_a, "expr"_a,
                  nb::rv_policy::reference_internal)
      .def("profile", &llvm::ExistsOpInit::Profile, "id"_a)
      .def("fold", &llvm::ExistsOpInit::Fold, "cur_rec"_a, "is_final"_a,
           nb::rv_policy::reference_internal)
      .def("is_complete", &llvm::ExistsOpInit::isComplete)
      .def("resolve_references", &llvm::ExistsOpInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::ExistsOpInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::ExistsOpInit::getAsString)
      .def("__str__", &llvm::ExistsOpInit::getAsUnquotedString);

  nb::class_<llvm::VarInit, llvm::TypedInit>(m, "VarInit")
      .def_static("classof", &llvm::VarInit::classof, "i"_a)
      .def_static(
          "get",
          [](llvm::StringRef VN, const llvm::RecTy *T)
              -> const llvm::VarInit * { return llvm::VarInit::get(VN, T); },
          "vn"_a, "t"_a, nb::rv_policy::reference_internal)
      .def_static(
          "get",
          [](const llvm::Init *VN, const llvm::RecTy *T)
              -> const llvm::VarInit * { return llvm::VarInit::get(VN, T); },
          "vn"_a, "t"_a, nb::rv_policy::reference_internal)
      .def("get_name", &llvm::VarInit::getName)
      .def("get_name_init", &llvm::VarInit::getNameInit,
           nb::rv_policy::reference_internal)
      .def("get_name_init_as_string", &llvm::VarInit::getNameInitAsString)
      .def("resolve_references", &llvm::VarInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::VarInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::VarInit::getAsString)
      .def("__str__", &llvm::VarInit::getAsUnquotedString);

  nb::class_<llvm::VarBitInit, llvm::TypedInit>(m, "VarBitInit")
      .def_static("classof", &llvm::VarBitInit::classof, "i"_a)
      .def_static("get", &llvm::VarBitInit::get, "t"_a, "b"_a,
                  nb::rv_policy::reference_internal)
      .def("get_bit_var", &llvm::VarBitInit::getBitVar,
           nb::rv_policy::reference_internal)
      .def("get_bit_num", &llvm::VarBitInit::getBitNum)
      .def("get_as_string", &llvm::VarBitInit::getAsString)
      .def("__str__", &llvm::VarBitInit::getAsUnquotedString)
      .def("resolve_references", &llvm::VarBitInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::VarBitInit::getBit, "b"_a,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::DefInit, llvm::TypedInit>(m, "DefInit")
      .def_static("classof", &llvm::DefInit::classof, "i"_a)
      .def("convert_initializer_to", &llvm::DefInit::convertInitializerTo,
           "ty"_a, nb::rv_policy::reference_internal)
      .def("get_def", &llvm::DefInit::getDef, nb::rv_policy::reference_internal)
      .def("get_field_type", &llvm::DefInit::getFieldType, "field_name"_a,
           nb::rv_policy::reference_internal)
      .def("is_concrete", &llvm::DefInit::isConcrete)
      .def("get_as_string", &llvm::DefInit::getAsString)
      .def("__str__", &llvm::DefInit::getAsUnquotedString)
      .def("get_bit", &llvm::DefInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal);

  nb::class_<llvm::VarDefInit, llvm::TypedInit>(m, "VarDefInit")
      .def_static("classof", &llvm::VarDefInit::classof, "i"_a)
      .def_static("get", &llvm::VarDefInit::get, "loc"_a, "class"_a, "args"_a,
                  nb::rv_policy::reference_internal)
      .def("profile", &llvm::VarDefInit::Profile, "id"_a)
      .def("resolve_references", &llvm::VarDefInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("fold", &llvm::VarDefInit::Fold, nb::rv_policy::reference_internal)
      .def("get_as_string", &llvm::VarDefInit::getAsString)
      .def("__str__", &llvm::VarDefInit::getAsUnquotedString)
      .def("get_arg", &llvm::VarDefInit::getArg, "i"_a,
           nb::rv_policy::reference_internal)
      .def("args_begin", &llvm::VarDefInit::args_begin,
           nb::rv_policy::reference_internal)
      .def("args_end", &llvm::VarDefInit::args_end,
           nb::rv_policy::reference_internal)
      .def("args_size", &llvm::VarDefInit::args_size)
      .def("args_empty", &llvm::VarDefInit::args_empty)
      .def("get_bit", &llvm::VarDefInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("args",
           eudsl::coerceReturn<std::vector<const llvm::ArgumentInit *>>(
               &llvm::VarDefInit::args, nb::const_),
           nb::rv_policy::reference_internal)
      .def("__len__", [](const llvm::VarDefInit &v) { return v.args_size(); })
      .def("__bool__",
           [](const llvm::VarDefInit &v) { return !v.args_empty(); })
      .def(
          "__iter__",
          [](llvm::VarDefInit &v) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<llvm::VarDefInit>(), "Iterator", v.args_begin(),
                v.args_end());
          },
          nb::rv_policy::reference_internal)
      .def(
          "__getitem__",
          [](llvm::VarDefInit &v, Py_ssize_t i) {
            return v.getArg(eudsl::wrap(i, v.args_size()));
          },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::FieldInit, llvm::TypedInit>(m, "FieldInit")
      .def_static("classof", &llvm::FieldInit::classof, "i"_a)
      .def_static("get", &llvm::FieldInit::get, "r"_a, "fn"_a,
                  nb::rv_policy::reference_internal)
      .def("get_record", &llvm::FieldInit::getRecord,
           nb::rv_policy::reference_internal)
      .def("get_field_name", &llvm::FieldInit::getFieldName,
           nb::rv_policy::reference_internal)
      .def("get_bit", &llvm::FieldInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("resolve_references", &llvm::FieldInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("fold", &llvm::FieldInit::Fold, "cur_rec"_a,
           nb::rv_policy::reference_internal)
      .def("is_concrete", &llvm::FieldInit::isConcrete)
      .def("get_as_string", &llvm::FieldInit::getAsString)
      .def("__str__", &llvm::FieldInit::getAsUnquotedString);

  nb::class_<llvm::DagInit, llvm::TypedInit>(m, "DagInit")
      .def("profile", &llvm::DagInit::Profile, "id"_a)
      .def("get_operator", &llvm::DagInit::getOperator,
           nb::rv_policy::reference_internal)
      .def("get_operator_as_def", &llvm::DagInit::getOperatorAsDef, "loc"_a,
           nb::rv_policy::reference_internal)
      .def("get_name", &llvm::DagInit::getName,
           nb::rv_policy::reference_internal)
      .def("get_name_str", &llvm::DagInit::getNameStr)
      .def("get_num_args", &llvm::DagInit::getNumArgs)
      .def("get_arg", &llvm::DagInit::getArg, "num"_a,
           nb::rv_policy::reference_internal)
      .def("get_arg_no", &llvm::DagInit::getArgNo, "name"_a)
      .def("get_arg_name", &llvm::DagInit::getArgName, "num"_a,
           nb::rv_policy::reference_internal)
      .def("get_arg_name_str", &llvm::DagInit::getArgNameStr, "num"_a)
      .def("resolve_references", &llvm::DagInit::resolveReferences, "r"_a,
           nb::rv_policy::reference_internal)
      .def("is_concrete", &llvm::DagInit::isConcrete)
      .def("get_as_string", &llvm::DagInit::getAsString)
      .def("arg_begin", &llvm::DagInit::arg_begin,
           nb::rv_policy::reference_internal)
      .def("arg_end", &llvm::DagInit::arg_end,
           nb::rv_policy::reference_internal)
      .def("arg_size", &llvm::DagInit::arg_size)
      .def("arg_empty", &llvm::DagInit::arg_empty)
      .def("name_begin", &llvm::DagInit::name_begin,
           nb::rv_policy::reference_internal)
      .def("name_end", &llvm::DagInit::name_end,
           nb::rv_policy::reference_internal)
      .def("name_size",
           [](const llvm::DagInit &init) {
             return std::distance(init.name_begin(), init.name_end());
           })
      .def("name_empty",
           [](const llvm::DagInit &init) {
             return std::distance(init.name_begin(), init.name_end()) == 0;
           })
      .def("get_bit", &llvm::DagInit::getBit, "bit"_a,
           nb::rv_policy::reference_internal)
      .def("get_arg_names",
           eudsl::coerceReturn<std::vector<const llvm::StringInit *>>(
               &llvm::DagInit::getArgNames, nb::const_),
           nb::rv_policy::reference_internal)
      .def("get_args",
           eudsl::coerceReturn<std::vector<const llvm::Init *>>(
               &llvm::DagInit::getArgs, nb::const_),
           nb::rv_policy::reference_internal)
      .def("__len__", [](const llvm::DagInit &v) { return v.arg_size(); })
      .def("__bool__", [](const llvm::DagInit &v) { return !v.arg_empty(); })
      .def(
          "__iter__",
          [](llvm::DagInit &v) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<llvm::DagInit>(), "Iterator", v.arg_begin(),
                v.arg_end());
          },
          nb::rv_policy::reference_internal)
      .def(
          "__getitem__",
          [](llvm::DagInit &v, Py_ssize_t i) {
            return v.getArg(eudsl::wrap(i, v.arg_size()));
          },
          nb::rv_policy::reference_internal);

  auto llvm_RecordVal = nb::class_<llvm::RecordVal>(m, "RecordVal");
  nb::enum_<llvm::RecordVal::FieldKind>(llvm_RecordVal, "FieldKind")
      .value("FK_Normal", llvm::RecordVal::FK_Normal)
      .value("FK_NonconcreteOK", llvm::RecordVal::FK_NonconcreteOK)
      .value("FK_TemplateArg", llvm::RecordVal::FK_TemplateArg);

  llvm_RecordVal
      .def("get_record_keeper", &llvm::RecordVal::getRecordKeeper,
           nb::rv_policy::reference_internal)
      .def("get_name", &llvm::RecordVal::getName)
      .def("get_name_init", &llvm::RecordVal::getNameInit,
           nb::rv_policy::reference_internal)
      .def("get_name_init_as_string", &llvm::RecordVal::getNameInitAsString)
      .def("get_loc", &llvm::RecordVal::getLoc,
           nb::rv_policy::reference_internal)
      .def("is_nonconcrete_ok", &llvm::RecordVal::isNonconcreteOK)
      .def("is_template_arg", &llvm::RecordVal::isTemplateArg)
      .def("get_type", &llvm::RecordVal::getType,
           nb::rv_policy::reference_internal)
      .def("get_print_type", &llvm::RecordVal::getPrintType)
      .def("get_value", &llvm::RecordVal::getValue,
           nb::rv_policy::reference_internal)
      .def(
          "set_value",
          [](llvm::RecordVal &self, const llvm::Init *V) -> bool {
            return self.setValue(V);
          },
          "v"_a)
      .def(
          "set_value",
          [](llvm::RecordVal &self, const llvm::Init *V,
             llvm::SMLoc NewLoc) -> bool { return self.setValue(V, NewLoc); },
          "v"_a, "new_loc"_a)
      .def("add_reference_loc", &llvm::RecordVal::addReferenceLoc, "loc"_a)
      .def("get_reference_locs", &llvm::RecordVal::getReferenceLocs)
      .def("set_used", &llvm::RecordVal::setUsed, "used"_a)
      .def("is_used", &llvm::RecordVal::isUsed)
      .def("dump", &llvm::RecordVal::dump)
      .def("print", &llvm::RecordVal::print, "os"_a, "print_sem"_a)
      .def("__str__",
           [](const llvm::RecordVal &self) {
             return self.getValue() ? self.getValue()->getAsUnquotedString()
                                    : "<<NULL>>";
           })
      .def("is_used", &llvm::RecordVal::isUsed);

  struct RecordValues {};
  nb::class_<RecordValues>(m, "RecordValues", nb::dynamic_attr())
      .def("__repr__",
           [](const nb::object &self) {
             nb::str s{"RecordValues("};
             auto dic = nb::cast<nb::dict>(nb::getattr(self, "__dict__"));
             size_t i = 0;
             for (auto [key, value] : dic) {
               s += key + nb::str("=") +
                    nb::str(nb::cast<llvm::RecordVal>(value)
                                .getValue()
                                ->getAsUnquotedString()
                                .c_str());
               if (i < dic.size() - 1)
                 s += nb::str(", ");
               ++i;
             }
             s += nb::str(")");
             return s;
           })
      .def(
          "__iter__",
          [](const nb::object &self) {
            return nb::iter(getattr(self, "__dict__"));
          },
          nb::rv_policy::reference_internal)
      .def(
          "keys",
          [](const nb::object &self) {
            return getattr(getattr(self, "__dict__"), "keys")();
          },
          nb::rv_policy::reference_internal)
      .def(
          "values",
          [](const nb::object &self) {
            return getattr(getattr(self, "__dict__"), "values")();
          },
          nb::rv_policy::reference_internal)
      .def(
          "items",
          [](const nb::object &self) {
            return getattr(getattr(self, "__dict__"), "items")();
          },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::Record>(m, "Record")
      .def("get_direct_super_classes",
           [](const llvm::Record &self) -> std::vector<const llvm::Record *> {
             llvm::SmallVector<const llvm::Record *> Classes;
             for (auto [rec, _] : self.getDirectSuperClasses())
               Classes.push_back(rec);
             return {Classes.begin(), Classes.end()};
           })
      .def(
          "get_values",
          [](llvm::Record &self) {
            // you can't just call the class_->operator()
            nb::handle recordValsInstTy = nb::type<RecordValues>();
            assert(recordValsInstTy.is_valid() &&
                   nb::type_check(recordValsInstTy));
            nb::object recordValsInst = nb::inst_alloc(recordValsInstTy);
            assert(nb::inst_check(recordValsInst) &&
                   recordValsInst.type().is(recordValsInstTy) &&
                   !nb::inst_ready(recordValsInst));

            for (const llvm::RecordVal &recordVal : self.getValues()) {
              nb::setattr(recordValsInst, recordVal.getName().str().c_str(),
                          nb::borrow(nb::cast(recordVal)));
            }
            return recordValsInst;
          },
          nb::rv_policy::reference_internal)
      .def("get_template_args",
           eudsl::coerceReturn<std::vector<const llvm::Init *>>(
               &llvm::Record::getTemplateArgs, nb::const_),
           nb::rv_policy::reference_internal)
      .def_static("get_new_uid", &llvm::Record::getNewUID, "rk"_a)
      .def("get_id", &llvm::Record::getID)
      .def("get_name", &llvm::Record::getName)
      .def("get_name_init", &llvm::Record::getNameInit,
           nb::rv_policy::reference_internal)
      .def("get_name_init_as_string", &llvm::Record::getNameInitAsString)
      .def("set_name", &llvm::Record::setName, "name"_a)
      .def("get_loc", eudsl::coerceReturn<std::vector<llvm::SMLoc>>(
                          &llvm::Record::getLoc, nb::const_))
      .def("append_loc", &llvm::Record::appendLoc, "loc"_a)
      .def("get_forward_declaration_locs",
           &llvm::Record::getForwardDeclarationLocs)
      .def("append_reference_loc", &llvm::Record::appendReferenceLoc, "loc"_a)
      .def("get_reference_locs", &llvm::Record::getReferenceLocs)
      .def("update_class_loc", &llvm::Record::updateClassLoc, "loc"_a)
      .def("get_type", &llvm::Record::getType,
           nb::rv_policy::reference_internal)
      .def("get_def_init", &llvm::Record::getDefInit,
           nb::rv_policy::reference_internal)
      .def("is_class", &llvm::Record::isClass)
      .def("is_multi_class", &llvm::Record::isMultiClass)
      .def("is_anonymous", &llvm::Record::isAnonymous)
      .def("get_template_args", &llvm::Record::getTemplateArgs)
      .def("get_assertions", &llvm::Record::getAssertions)
      .def("get_dumps", &llvm::Record::getDumps)
      .def(
          "get_super_classes",
          [](llvm::Record &self) { return self.getSuperClasses(); },
          nb::rv_policy::reference_internal)
      .def("has_direct_super_class", &llvm::Record::hasDirectSuperClass,
           "super_class"_a)
      .def("is_template_arg", &llvm::Record::isTemplateArg, "name"_a)
      .def(
          "get_value",
          [](llvm::Record &self, const llvm::Init *Name)
              -> const llvm::RecordVal * { return self.getValue(Name); },
          "name"_a, nb::rv_policy::reference_internal)
      .def(
          "get_value",
          [](llvm::Record &self, llvm::StringRef Name)
              -> const llvm::RecordVal * { return self.getValue(Name); },
          "name"_a, nb::rv_policy::reference_internal)
      .def(
          "get_value",
          [](llvm::Record &self, const llvm::Init *Name) -> llvm::RecordVal * {
            return self.getValue(Name);
          },
          "name"_a, nb::rv_policy::reference_internal)
      .def(
          "get_value",
          [](llvm::Record &self, llvm::StringRef Name) -> llvm::RecordVal * {
            return self.getValue(Name);
          },
          "name"_a, nb::rv_policy::reference_internal)
      .def("add_template_arg", &llvm::Record::addTemplateArg, "name"_a)
      .def("add_value", &llvm::Record::addValue, "rv"_a)
      .def(
          "remove_value",
          [](llvm::Record &self, const llvm::Init *Name) -> void {
            return self.removeValue(Name);
          },
          "name"_a)
      .def(
          "remove_value",
          [](llvm::Record &self, llvm::StringRef Name) -> void {
            return self.removeValue(Name);
          },
          "name"_a)
      .def("add_assertion", &llvm::Record::addAssertion, "loc"_a, "condition"_a,
           "message"_a)
      .def("add_dump", &llvm::Record::addDump, "loc"_a, "message"_a)
      .def("append_assertions", &llvm::Record::appendAssertions, "rec"_a)
      .def("append_dumps", &llvm::Record::appendDumps, "rec"_a)
      .def("check_record_assertions", &llvm::Record::checkRecordAssertions)
      .def("emit_record_dumps", &llvm::Record::emitRecordDumps)
      .def("check_unused_template_args", &llvm::Record::checkUnusedTemplateArgs)
      .def(
          "is_sub_class_of",
          [](llvm::Record &self, const llvm::Record *R) -> bool {
            return self.isSubClassOf(R);
          },
          "r"_a)
      .def(
          "is_sub_class_of",
          [](llvm::Record &self, llvm::StringRef Name) -> bool {
            return self.isSubClassOf(Name);
          },
          "name"_a)
      .def("add_direct_super_class", &llvm::Record::addDirectSuperClass, "r"_a,
           "range"_a)
      .def(
          "resolve_references",
          [](llvm::Record &self, const llvm::Init *NewName) -> void {
            return self.resolveReferences(NewName);
          },
          "new_name"_a)
      .def(
          "resolve_references",
          [](llvm::Record &self, llvm::Resolver &R,
             const llvm::RecordVal *SkipVal) -> void {
            return self.resolveReferences(R, SkipVal);
          },
          "r"_a, "skip_val"_a)
      .def("get_records", &llvm::Record::getRecords,
           nb::rv_policy::reference_internal)
      .def("dump", [](llvm::Record &self) { self.dump(); })
      .def("get_field_loc", &llvm::Record::getFieldLoc, "field_name"_a)
      .def("get_value_init", &llvm::Record::getValueInit, "field_name"_a,
           nb::rv_policy::reference_internal)
      .def("is_value_unset", &llvm::Record::isValueUnset, "field_name"_a)
      .def("get_value_as_string", &llvm::Record::getValueAsString,
           "field_name"_a)
      .def("get_value_as_optional_string",
           &llvm::Record::getValueAsOptionalString, "field_name"_a)
      .def("get_value_as_bits_init", &llvm::Record::getValueAsBitsInit,
           "field_name"_a, nb::rv_policy::reference_internal)
      .def("get_value_as_list_init", &llvm::Record::getValueAsListInit,
           "field_name"_a, nb::rv_policy::reference_internal)
      .def("get_value_as_list_of_defs", &llvm::Record::getValueAsListOfDefs,
           "field_name"_a)
      .def("get_value_as_list_of_ints", &llvm::Record::getValueAsListOfInts,
           "field_name"_a)
      .def("get_value_as_list_of_strings",
           &llvm::Record::getValueAsListOfStrings, "field_name"_a)
      .def("get_value_as_def", &llvm::Record::getValueAsDef, "field_name"_a,
           nb::rv_policy::reference_internal)
      .def("get_value_as_optional_def", &llvm::Record::getValueAsOptionalDef,
           "field_name"_a, nb::rv_policy::reference_internal)
      .def("get_value_as_bit", &llvm::Record::getValueAsBit, "field_name"_a)
      .def("get_value_as_bit_or_unset", &llvm::Record::getValueAsBitOrUnset,
           "field_name"_a, "unset"_a)
      .def("get_value_as_int", &llvm::Record::getValueAsInt, "field_name"_a)
      .def("get_value_as_dag", &llvm::Record::getValueAsDag, "field_name"_a,
           nb::rv_policy::reference_internal);

  using RecordMap =
      std::map<std::string, std::unique_ptr<llvm::Record>, std::less<>>;
  using GlobalMap = std::map<std::string, const llvm::Init *, std::less<>>;
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
          nb::rv_policy::reference_internal)
      .def(
          "keys",
          [](RecordMap &m) {
            return nb::make_key_iterator<nb::rv_policy::reference>(
                nb::type<RecordMap>(), "KeyIterator", m.begin(), m.end());
          },
          nb::rv_policy::reference_internal)
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

  nb::class_<llvm::RecordKeeper>(m, "RecordKeeper")
      .def(nb::init<>())
      .def(
          "parse_td",
          [](llvm::RecordKeeper &self, const std::string &inputFilename,
             const std::vector<std::string> &includeDirs,
             const std::vector<std::string> &macroNames,
             bool noWarnOnUnusedTemplateArgs) {
            llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
                llvm::MemoryBuffer::getFile(inputFilename,
                                            /*IsText=*/true);
            if (std::error_code EC = fileOrErr.getError())
              throw std::runtime_error("Could not open input file '" +
                                       inputFilename + "': " + EC.message() +
                                       "\n");
            self.saveInputFilename(inputFilename);
            llvm::SourceMgr srcMgr;
            srcMgr.setIncludeDirs(includeDirs);
            srcMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
            llvm::TGParser tgParser(srcMgr, macroNames, self,
                                    noWarnOnUnusedTemplateArgs);
            if (tgParser.ParseFile())
              throw std::runtime_error("Could not parse file '" +
                                       inputFilename);
            return &self;
          },
          "input_filename"_a, "include_dirs"_a = nb::list(),
          "macro_names"_a = nb::list(),
          "no_warn_on_unused_template_args"_a = true)
      .def("get_input_filename", &llvm::RecordKeeper::getInputFilename)
      .def("get_classes", &llvm::RecordKeeper::getClasses,
           nb::rv_policy::reference_internal)
      .def("get_defs", &llvm::RecordKeeper::getDefs,
           nb::rv_policy::reference_internal)
      .def("get_globals", &llvm::RecordKeeper::getGlobals,
           nb::rv_policy::reference_internal)
      .def("get_class", &llvm::RecordKeeper::getClass, "name"_a,
           nb::rv_policy::reference_internal)
      .def("get_def", &llvm::RecordKeeper::getDef, "name"_a,
           nb::rv_policy::reference_internal)
      .def("get_global", &llvm::RecordKeeper::getGlobal, "name"_a,
           nb::rv_policy::reference_internal)
      .def("save_input_filename", &llvm::RecordKeeper::saveInputFilename,
           "filename"_a)
      .def("add_class", &llvm::RecordKeeper::addClass, "r"_a)
      .def("add_def", &llvm::RecordKeeper::addDef, "r"_a)
      .def("add_extra_global", &llvm::RecordKeeper::addExtraGlobal, "name"_a,
           "i"_a)
      .def("get_new_anonymous_name", &llvm::RecordKeeper::getNewAnonymousName,
           nb::rv_policy::reference_internal)
      .def(
          "get_all_derived_definitions",
          [](llvm::RecordKeeper &self,
             const llvm::ArrayRef<llvm::StringRef> ClassNames)
              -> std::vector<const llvm::Record *,
                             std::allocator<const llvm::Record *>> {
            return self.getAllDerivedDefinitions(ClassNames);
          },
          "class_names"_a)
      .def("dump", &llvm::RecordKeeper::dump)
      .def(
          "get_all_derived_definitions",
          [](llvm::RecordKeeper &self, const std::string &className)
              -> std::vector<const llvm::Record *> {
            return self.getAllDerivedDefinitions(className);
          },
          "class_name"_a, nb::rv_policy::reference_internal)
      .def(
          "get_all_derived_definitions_if_defined",
          [](llvm::RecordKeeper &self, const std::string &className)
              -> std::vector<const llvm::Record *> {
            return self.getAllDerivedDefinitionsIfDefined(className);
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

  auto mlir_tblgen_EnumCase =
      nb::class_<mlir::tblgen::EnumCase>(m, "EnumCase")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def("get_symbol", &mlir::tblgen::EnumCase::getSymbol)
          .def("get_str", &mlir::tblgen::EnumCase::getStr)
          .def("get_value", &mlir::tblgen::EnumCase::getValue);

  mlir_tblgen_EnumCase.def("get_def", &mlir::tblgen::EnumCase::getDef,
                           nb::rv_policy::reference_internal);

  auto mlir_tblgen_EnumInfo =
      nb::class_<mlir::tblgen::EnumInfo>(m, "EnumInfo")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::Record &>(), "record"_a)
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def("is_bit_enum", &mlir::tblgen::EnumInfo::isBitEnum)
          .def("get_enum_class_name", &mlir::tblgen::EnumInfo::getEnumClassName)
          .def("get_cpp_namespace", &mlir::tblgen::EnumInfo::getCppNamespace)
          .def("get_underlying_type",
               &mlir::tblgen::EnumInfo::getUnderlyingType)
          .def("get_underlying_to_symbol_fn_name",
               &mlir::tblgen::EnumInfo::getUnderlyingToSymbolFnName)
          .def("get_string_to_symbol_fn_name",
               &mlir::tblgen::EnumInfo::getStringToSymbolFnName)
          .def("get_symbol_to_string_fn_name",
               &mlir::tblgen::EnumInfo::getSymbolToStringFnName)
          .def("get_symbol_to_string_fn_ret_type",
               &mlir::tblgen::EnumInfo::getSymbolToStringFnRetType)
          .def("get_max_enum_val_fn_name",
               &mlir::tblgen::EnumInfo::getMaxEnumValFnName)
          .def("get_all_cases", &mlir::tblgen::EnumInfo::getAllCases)
          .def("gen_specialized_attr",
               &mlir::tblgen::EnumInfo::genSpecializedAttr)
          .def("get_base_attr_class", &mlir::tblgen::EnumInfo::getBaseAttrClass,
               nb::rv_policy::reference_internal)
          .def("get_specialized_attr_class_name",
               &mlir::tblgen::EnumInfo::getSpecializedAttrClassName);

  mlir_tblgen_EnumInfo.def("print_bit_enum_primary_groups",
                           &mlir::tblgen::EnumInfo::printBitEnumPrimaryGroups);

  auto mlir_tblgen_Property =
      nb::class_<mlir::tblgen::Property>(m, "Property")
          .def(nb::init<const llvm::Record *>(), "record"_a)
          .def(nb::init<const llvm::DefInit *>(), "init"_a)
          .def(nb::init<const llvm::Record *, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef, llvm::StringRef,
                        llvm::StringRef, llvm::StringRef>(),
               "maybe_def"_a, "summary"_a, "description"_a, "storage_type"_a,
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

  eudsl::bind_array_ref<mlir::tblgen::Builder::Parameter>(m);
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

  eudsl::bind_array_ref<mlir::tblgen::AttrOrTypeParameter>(m);
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

  auto mlir_tblgen_DialectNamespaceEmitter =
      nb::class_<mlir::tblgen::DialectNamespaceEmitter>(
          m, "DialectNamespaceEmitter")
          .def(nb::init<llvm::raw_ostream &, const mlir::tblgen::Dialect &>(),
               "os"_a, "dialect"_a);

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

  mlir_tblgen_InterfaceMethod
      .def(nb::init<const llvm::Record *, std::string>(), "def_"_a,
           "unique_name"_a)
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

  auto mlir_tblgen_Operator_OperandAttrOrProp =
      nb::class_<mlir::tblgen::Operator::OperandAttrOrProp>(
          mlir_tblgen_Operator, "OperandAttrOrProp")
          .def(nb::init<mlir::tblgen::Operator::OperandAttrOrProp::Kind, int>(),
               "kind"_a, "index"_a)
          .def("operand_or_attribute_index",
               &mlir::tblgen::Operator::OperandAttrOrProp::
                   operandOrAttributeIndex)
          .def("kind", &mlir::tblgen::Operator::OperandAttrOrProp::kind);

  nb::enum_<mlir::tblgen::Operator::OperandAttrOrProp::Kind>(
      mlir_tblgen_Operator_OperandAttrOrProp, "Kind")
      .value("Operand",
             mlir::tblgen::Operator::OperandAttrOrProp::Kind::Operand)
      .value("Attribute",
             mlir::tblgen::Operator::OperandAttrOrProp::Kind::Attribute)
      .value("Property",
             mlir::tblgen::Operator::OperandAttrOrProp::Kind::Property);

  mlir_tblgen_Operator
      .def("get_arg_to_operand_attr_or_prop",
           &mlir::tblgen::Operator::getArgToOperandAttrOrProp, "index"_a)
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
          .def("is_enum_case", &mlir::tblgen::DagLeaf::isEnumCase)
          .def("is_string_attr", &mlir::tblgen::DagLeaf::isStringAttr)
          .def("get_as_constraint", &mlir::tblgen::DagLeaf::getAsConstraint)
          .def("get_as_constant_attr",
               &mlir::tblgen::DagLeaf::getAsConstantAttr)
          .def("get_as_enum_case", &mlir::tblgen::DagLeaf::getAsEnumCase)
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

  using SymbolInfoMapBaseT =
      std::unordered_multimap<std::string,
                              mlir::tblgen::SymbolInfoMap::SymbolInfo>;
  mlir_tblgen_SymbolInfoMap
      .def("begin",
           [](mlir::tblgen::SymbolInfoMap &self)
               -> SymbolInfoMapBaseT::iterator { return self.begin(); })
      .def("end",
           [](mlir::tblgen::SymbolInfoMap &self)
               -> SymbolInfoMapBaseT::iterator { return self.end(); })
      .def("cbegin",
           [](mlir::tblgen::SymbolInfoMap &self)
               -> SymbolInfoMapBaseT::const_iterator { return self.begin(); })
      .def("cend",
           [](mlir::tblgen::SymbolInfoMap &self)
               -> SymbolInfoMapBaseT::const_iterator { return self.end(); })
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
              -> SymbolInfoMapBaseT::const_iterator {
            return self.findBoundSymbol(key, node, op, argIndex,
                                        variadicSubIndex);
          },
          "key"_a, "node"_a, "op"_a, "arg_index"_a, "variadic_sub_index"_a)
      .def(
          "find_bound_symbol",
          [](mlir::tblgen::SymbolInfoMap &self, llvm::StringRef key,
             const mlir::tblgen::SymbolInfoMap::SymbolInfo &symbolInfo)
              -> SymbolInfoMapBaseT::const_iterator {
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

  m.def("lookup_intrinsic_id", llvm::Intrinsic::lookupIntrinsicID,
        nb::arg("name"));
  m.def("intrinsic_is_overloaded", llvm::Intrinsic::isOverloaded,
        nb::arg("id"));
}
