#include "TGParser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Record.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

using namespace llvm;

namespace nb = nanobind;
using namespace nb::literals;

void parseTD(RecordKeeper &Records, const std::string &InputFilename,
             const std::vector<std::string> &IncludeDirs,
             const std::vector<std::string> &MacroNames,
             bool NoWarnOnUnusedTemplateArgs) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFilename, /*IsText=*/true);
  if (std::error_code EC = FileOrErr.getError())
    throw std::runtime_error("Could not open input file '" + InputFilename +
                             "': " + EC.message() + "\n");
  Records.saveInputFilename(InputFilename);
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(std::move(*FileOrErr), SMLoc());
  SrcMgr.setIncludeDirs(IncludeDirs);
  TGParser Parser(SrcMgr, MacroNames, Records, NoWarnOnUnusedTemplateArgs);
  if (Parser.ParseFile())
    throw std::runtime_error("Could not parse file '" + InputFilename);
}

template <typename NewReturn, typename Return, typename... Args>
constexpr auto coerceReturn(Return (*pf)(Args...)) noexcept {
  return [&pf](Args &&...args) -> NewReturn {
    return pf(std::forward<Args>(args)...);
  };
}

template <typename NewReturn, typename Return, typename Class, typename... Args>
constexpr auto coerceReturn(Return (Class:: *pmf)(Args...),
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
constexpr auto coerceReturn(Return (Class:: *pmf)(Args...) const,
                            std::true_type) noexcept {
  // copy the *pmf, not capture by ref
  return [pmf](const Class &cls, Args &&...args) -> NewReturn {
    return (cls.*pmf)(std::forward<Args>(args)...);
  };
}

template <>
struct nb::detail::type_caster<StringRef> {
  NB_TYPE_CASTER(StringRef, const_name("str_ref"))

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

NB_MODULE(eudsl_tblgen_ext, m) {

  auto recty = nb::class_<RecTy>(m, "RecTy");

  nb::enum_<RecTy::RecTyKind>(m, "RecTyKind")
      .value("BitRecTyKind", RecTy::RecTyKind::BitRecTyKind)
      .value("BitsRecTyKind", RecTy::RecTyKind::BitsRecTyKind)
      .value("IntRecTyKind", RecTy::RecTyKind::IntRecTyKind)
      .value("StringRecTyKind", RecTy::RecTyKind::StringRecTyKind)
      .value("ListRecTyKind", RecTy::RecTyKind::ListRecTyKind)
      .value("DagRecTyKind", RecTy::RecTyKind::DagRecTyKind)
      .value("RecordRecTyKind", RecTy::RecTyKind::RecordRecTyKind)
      .export_values();

  recty.def_prop_ro("RecTyKind", &RecTy::getRecTyKind)
      .def_prop_ro("RecordKeeper", &RecTy::getRecordKeeper)
      .def_prop_ro("AsString", &RecTy::getAsString)
      .def("__str__", &RecTy::getAsString)
      .def("typeIsA", &RecTy::typeIsA)
      .def("typeIsConvertibleTo", &RecTy::typeIsConvertibleTo);

  nb::class_<RecordRecTy, RecTy>(m, "RecordRecTy")
      .def_prop_ro(
          "Classes",
          coerceReturn<std::vector<const Record *>, ArrayRef<const Record *>>(
              &RecordRecTy::getClasses, nb::const_))
      .def("isSubClassOf", &RecordRecTy::isSubClassOf);

  nb::class_<RecordVal>(m, "RecordVal")
      .def("dump", &RecordVal::dump)
      .def_prop_ro("Name", &RecordVal::getName)
      .def_prop_ro("NameInitAsString", &RecordVal::getNameInitAsString)
      .def_prop_ro("PrintType", &RecordVal::getPrintType)
      .def_prop_ro("RecordKeeper", &RecordVal::getRecordKeeper)
      .def_prop_ro("Type", &RecordVal::getType)
      .def_prop_ro("isNonconcreteOK", &RecordVal::isNonconcreteOK)
      .def_prop_ro("isTemplateArg", &RecordVal::isTemplateArg)
      .def_prop_ro("isUsed", &RecordVal::isUsed);
  // .def_prop_ro("Loc", &RecordVal::getLoc)
  // .def_prop_ro("NameInit", &RecordVal::getNameInit)
  // .def_prop_ro("ReferenceLocs", &RecordVal::getReferenceLocs)
  //  .def_prop_ro("Value", &RecordVal::getValue)

  nb::class_<Record>(m, "Record")
      .def_prop_ro("DirectSuperClasses",
                   [](const Record &self) -> std::vector<const Record *> {
                     SmallVector<const Record *> Classes;
                     self.getDirectSuperClasses(Classes);
                     return {Classes.begin(), Classes.end()};
                   })
      .def_prop_ro("ID", &Record::getID)
      .def_prop_ro("Name", &Record::getName)
      .def_prop_ro("NameInitAsString", &Record::getNameInitAsString)
      .def_prop_ro("Records", &Record::getRecords)
      .def_prop_ro("Type", &Record::getType)
      .def("getValue", nb::overload_cast<StringRef>(&Record::getValue))
      .def("getValueAsBit", &Record::getValueAsBit)
      .def("getValueAsDef", &Record::getValueAsDef)
      .def("getValueAsInt", &Record::getValueAsInt)
      .def("getValueAsListOfDefs", &Record::getValueAsListOfDefs)
      .def("getValueAsListOfInts", &Record::getValueAsListOfInts)
      .def("getValueAsListOfStrings", &Record::getValueAsListOfStrings)
      .def("getValueAsOptionalDef", &Record::getValueAsOptionalDef)
      .def("getValueAsOptionalString", &Record::getValueAsOptionalString)
      .def("getValueAsString", &Record::getValueAsString)
      .def_prop_ro("Values", coerceReturn<std::vector<RecordVal>>(
                                 &Record::getValues, nb::const_))
      .def("hasDirectSuperClass", &Record::hasDirectSuperClass)
      .def_prop_ro("isAnonymous", &Record::isAnonymous)
      .def_prop_ro("isClass", &Record::isClass)
      .def_prop_ro("isMultiClass", &Record::isMultiClass)
      .def("isSubClassOf",
           nb::overload_cast<const Record *>(&Record::isSubClassOf, nb::const_))
      .def("isSubClassOf",
           nb::overload_cast<StringRef>(&Record::isSubClassOf, nb::const_))
      .def("isValueUnset", &Record::isValueUnset);
  // .def_prop_ro("Assertions", &Record::getAssertions)
  // .def_prop_ro("DefInit", &Record::getDefInit)
  // .def_prop_ro("Dumps", &Record::getDumps)
  // .def_prop_ro("FieldLoc", &Record::getFieldLoc)
  // .def_prop_ro("ForwardDeclarationLocs", &Record::getForwardDeclarationLocs)
  // .def_prop_ro("Loc", &Record::getLoc)
  // .def_prop_ro("NameInit", &Record::getNameInit)
  // .def_prop_ro("NewUID", &Record::getNewUID)
  // .def_prop_ro("ReferenceLocs", &Record::getReferenceLocs)
  // .def_prop_ro("SuperClasses", &Record::getSuperClasses)
  // .def_prop_ro("TemplateArgs", &Record::getTemplateArgs)
  // .def_prop_ro("ValueAsBitOrUnset", &Record::getValueAsBitOrUnset)
  // .def_prop_ro("ValueAsBitsInit", &Record::getValueAsBitsInit)
  // .def_prop_ro("ValueAsDag", &Record::getValueAsDag)
  // .def_prop_ro("ValueAsListInit", &Record::getValueAsListInit)
  // .def_prop_ro("ValueInit", &Record::getValueInit)
  // .def_prop_ro("isTemplateArg", &Record::isTemplateArg)

  using RecordMap = std::map<std::string, std::unique_ptr<Record>, std::less<>>;

  nb::class_<RecordMap>(m, "RecordMap")
      .def("__len__", [](const RecordMap &m) { return m.size(); })
      .def("__bool__", [](const RecordMap &m) { return !m.empty(); })
      .def("__contains__",
           [](const RecordMap &m, const std::string &k) {
             return m.find(k) != m.end();
           })
      // fallback for incompatible types
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
            Record *r = m[k].release();
            m[k] = std::make_unique<Record>(*r);
            return r;
          },
          nb::rv_policy::reference_internal);

  nb::class_<RecordKeeper>(m, "RecordKeeper")
      .def(nb::init<>())
      .def("parse_td", &parseTD, "input_filename"_a, "include_dirs"_a,
           "macro_names"_a = nb::list(),
           "no_warn_on_unused_template_args"_a = true)
      .def_prop_ro("InputFilename", &RecordKeeper::getInputFilename)
      .def_prop_ro("Classes", &RecordKeeper::getClasses)
      .def_prop_ro("Defs", &RecordKeeper::getDefs)
      .def_prop_ro("Globals", &RecordKeeper::getGlobals)
      .def("getAllDerivedDefinitions",
           coerceReturn<std::vector<const Record *>, ArrayRef<const Record *>>(
               &RecordKeeper::getAllDerivedDefinitions, nb::const_));
}