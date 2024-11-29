#include "TGParser.h"
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
      .def_prop_ro("num_operands", &OpInit::getNumOperands)
      .def("operand", &OpInit::getOperand, "i"_a,
           nb::rv_policy::reference_internal);

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
      .def("get_all_derived_definitions",
           coerceReturn<std::vector<const Record *>, ArrayRef<const Record *>>(
               &RecordKeeper::getAllDerivedDefinitions, nb::const_),
           "class_name"_a, nb::rv_policy::reference_internal);
}
