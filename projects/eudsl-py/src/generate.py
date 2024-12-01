import os
import platform
import sysconfig
from collections import defaultdict
from logging import warning
from pathlib import Path

import clang.cindex
import inflection
import nanobind
from clang.cindex import (
    CursorKind,
    Cursor,
    Type,
    TranslationUnit,
    AccessSpecifier,
    TypeKind,
)

if os.path.exists(
    "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib"
):
    clang.cindex.Config.set_library_file(
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib"
    )
elif os.path.exists("/usr/lib/x86_64-linux-gnu/libclang-20.so.20"):
    clang.cindex.Config.set_library_file("/usr/lib/x86_64-linux-gnu/libclang-20.so.20")
else:
    raise RuntimeError("unknown location for libclang")

THIS_DIR = Path(__file__).parent
filename = THIS_DIR / "eudsl_ext.cpp"
LLVM_INSTALL_DIR = os.getenv(
    "LLVM_INSTALL_DIR", THIS_DIR.parent.parent.parent / "llvm-install"
)
comp_args = [
    f"-I{LLVM_INSTALL_DIR / 'include'}",
    f"-I{sysconfig.get_paths()['include']}",
    f"-I{nanobind.include_dir()}",
    "-std=c++17",
]
if platform.system() == "Darwin":
    comp_args.extend(
        [
            "-isysroot",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.1.sdk",
        ]
    )

tu = TranslationUnit.from_source(
    filename,
    comp_args,
    options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    | TranslationUnit.PARSE_INCOMPLETE
    | TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
)
for d in tu.diagnostics:
    print(d)


# assert len(list(tu.diagnostics)) == 0


def collect_overloaded_methods(node: clang.cindex.Cursor, parent, methods):
    if (
        node.kind
        in {CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR, CursorKind.FUNCTION_TEMPLATE}
        and node.access_specifier == AccessSpecifier.PUBLIC
        and parent.type.get_canonical().spelling == class_being_visited[0]
    ):
        is_getter = (
            node.displayname.startswith("get") and len(list(node.get_arguments())) == 0
        )
        fn_name = node.displayname.split("(")[0]
        methods[fn_name].append(
            {"is_static": node.is_static_method(), "is_getter": is_getter}
        )

    return 1  # means continue adjacent


class_being_visited = [None]

FILE = open("arith.cpp", "w")

print('#include "ir.h"', file=FILE)
print("namespace nb = nanobind;", file=FILE)
print("using namespace nb::literals;", file=FILE)
print(file=FILE)
print("void populateArithModule(nanobind::module_ & m) {", file=FILE)
# FieldParser<AffineMap> is emitted without the canonical namespace for AffineMap
print("using namespace mlir;", file=FILE)
print("using namespace mlir::detail;", file=FILE)
print("using namespace mlir::arith;", file=FILE)

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

dialects = {
    "mlir::arith::ArithDialect",
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

renames = {"from": "from_", "except": "except_"}


def get_default_val(node: Cursor, parent, defaults):
    defaults.append(node)
    return 2


def parse_literal(cursor):
    value = "".join([str(t.spelling) for t in cursor.get_tokens()])
    if cursor.kind == CursorKind.STRING_LITERAL:
        value = "'" + value[1:-1] + "'"  # prefer single quotes
        # value = 'b' + value  # Ensure byte string for compatibility
    return value


def parse_parameter(cursor):
    name = cursor.spelling
    default_value = None
    annotation = None
    for child in cursor.get_children():
        if child.kind == CursorKind.TYPE_REF:
            # the type of the param
            pass
        elif child.kind == CursorKind.UNEXPOSED_EXPR:
            for gc in child.get_children():
                if gc.kind == CursorKind.CXX_NULL_PTR_LITERAL_EXPR:
                    default_value = None
                elif gc.kind == CursorKind.INTEGER_LITERAL:
                    val = parse_literal(gc)
                    if val.endswith("L"):
                        val = val[:-1]
                    if cursor.type.kind == TypeKind.POINTER and int(val) == 0:
                        default_value = None
                    else:
                        default_value = f"{int(val)}"
                elif len(gc.spelling) > 0:
                    default_value = gc.spelling
                elif gc.kind == CursorKind.CXX_UNARY_EXPR:
                    # default value is some expression
                    # so jam the tokens together, and hope it's interpretable by python
                    # unPrimitiveSize = sizeof( VROverlayIntersectionMaskPrimitive_t )
                    tokens = "".join([t.spelling for t in gc.get_tokens()])
                    default_value = tokens
                else:
                    raise NotImplementedError(f"unparsed {gc.kind}")
        elif child.kind == CursorKind.CXX_BOOL_LITERAL_EXPR:
            bool_val = str(next(child.get_tokens()).spelling)
            default_value = str(bool_val == "true")
        elif child.kind == CursorKind.ANNOTATE_ATTR:
            annotation = child.spelling
        elif child.kind == CursorKind.DECL_REF_EXPR:
            default_value = f"{child.type.get_canonical().spelling}::{child.spelling}"
        elif child.kind == CursorKind.NAMESPACE_REF:
            pass  # probably on the parameter type
        else:
            raise NotImplementedError(f"unparsed {child.kind}")
    return dict(
        name=name,
        type_=cursor.type,
        # docstring=clean_comment(cursor),
        default_value=default_value,
        annotation=annotation,
    )


def class_visitor(node: clang.cindex.Cursor, parent, counts):
    if not (
        node.kind in {CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR}
        and node.access_specifier == AccessSpecifier.PUBLIC
        and parent.type.get_canonical().spelling == class_being_visited[0]
        and node.displayname not in fn_blacklist
    ):
        return 2

    # print(node.displayname)

    arg_names = []
    arg_types = []
    for i, a in enumerate(node.get_arguments()):
        t: Type = a.type
        t_parent = t.get_declaration().semantic_parent
        if t_parent and parent and t_parent == parent:
            t_spelling = t.get_named_type().spelling
        elif "_t" not in t.spelling:
            t_spelling = a.type.get_canonical().spelling
        else:
            t_spelling = t.spelling

        arg_types.append(t_spelling)

        a_name = a.displayname
        if a_name in renames:
            a_name = renames[a_name]
        if not len(a_name):
            a_name = "_" * (i + 1)
        arg_names.append(a_name)

    fq_name = parent.type.get_canonical().spelling
    is_getter = (
        node.displayname.startswith("get") and len(list(node.get_arguments())) == 0
    )

    returns_ref = returns_ptr = False
    if node.kind == CursorKind.CONSTRUCTOR:
        func_ref = f"nb::init<{', '.join(arg_types)}>()"
        nb_fn_name = ""
    else:
        fn_name = node.displayname.split("(")[0]

        returns_ref = node.type.get_result().spelling[-1] == "&"
        returns_ptr = node.type.get_result().spelling[-1] == "*"

        if len(counts[fn_name]) > 1:
            if arg_names:
                typed_args = ", ".join(
                    [f"{t} {a}" for t, a in zip(arg_types, arg_names)]
                )
            else:
                typed_args = ""
            new_arg_names = arg_names[:]
            for i, a in enumerate(new_arg_names):
                t = arg_types[i]
                if ("std::unique_ptr" in t and t[-1] != "&") or "&&" in t:
                    new_arg_names[i] = f"std::move({a})"

            if node.is_static_method():
                func_ref = f"[]({typed_args}){{ return {'&' if returns_ref else ''}{fq_name}::{fn_name}({', '.join(new_arg_names)}); }}"
            else:
                if len(typed_args):
                    typed_args = f"self, {typed_args}"
                else:
                    typed_args = "self"

                func_ref = f"[]({fq_name}& {typed_args}){{ return {'&' if returns_ref else ''}self.{node.spelling}({', '.join(new_arg_names)}); }}"
        else:
            func_ref = f"&{fq_name}::{fn_name}"

        if "operator" not in fn_name:
            nb_fn_name = fn_name
            # static method with non-static overloads that aren't also getters (getters are renamed already to break overlap)
            # see mlir::ElementsAttr
            if (
                len(counts[fn_name]) > 1
                and any(
                    props["is_static"] is False and not props["is_getter"]
                    for props in counts[fn_name]
                )
                and node.is_static_method()
                and not is_getter
            ):
                nb_fn_name += "_static"
            if is_getter and nb_fn_name.startswith("get"):
                nb_fn_name = nb_fn_name.replace("get", "")
            nb_fn_name = f'"{inflection.underscore(nb_fn_name)}", '
        else:
            if fn_name == "operator!=":
                fn_name = "__ne__"
            elif fn_name == "operator==":
                fn_name = "__eq__"
            elif fn_name == "operator-":
                fn_name = "__neg__"
            elif fn_name == "operator[]":
                fn_name = "__getitem__"
            elif fn_name == "operator<":
                fn_name = "__lt__"
            elif fn_name == "operator<=":
                fn_name = "__le__"
            elif fn_name == "operator>":
                fn_name = "__gt__"
            elif fn_name == "operator>=":
                fn_name = "__ge__"
            elif fn_name == "operator%":
                fn_name = "__mod__"
            elif fn_name == "operator*":
                if len(arg_names):
                    fn_name = "__mul__"
                else:
                    warning("operator* not supported")
                    return 2
            elif fn_name == "operator+" and len(arg_names):
                fn_name = "__add__"
            elif fn_name == "operator=":
                warning("operator= not supported")
                return 2
            elif fn_name == "operator->":
                warning("operator-> not supported")
                return 2
            elif fn_name == "operator!":
                warning("operator! not supported")
                return 2
            elif fn_name == "operator<<":
                warning("operator<< not supported")
                return 2
            nb_fn_name = f'"{fn_name}", '

    # parsed_args = [parse_parameter(a) for a in node.get_arguments()]
    if arg_names:
        arg_names = list(map(inflection.underscore, arg_names))
        arg_names = ", ".join([f'"{a}"_a' for a in arg_names])
        arg_names = f", {arg_names}"
    else:
        arg_names = ""

    if is_getter:
        if node.is_static_method():
            def_ = "def_static"
        else:
            def_ = "def_prop_ro"
    else:
        if node.is_static_method():
            def_ = "def_static"
        else:
            def_ = "def"

    if (returns_ptr or returns_ref) and not is_getter:
        ref_internal = ", nb::rv_policy::reference_internal"
    else:
        ref_internal = ""

    print(f".{def_}({nb_fn_name}{func_ref}{arg_names}{ref_internal})", file=FILE)
    return 2  # means continue visiting recursively


def enum_visitor(node: clang.cindex.Cursor, parent, userdata):
    print(
        f'.value("{node.spelling}", {node.type.spelling}::{node.spelling})', file=FILE
    )
    return 1  # means continue visiting adjacent


def walk_postorder(cursor):
    for child in cursor.get_children():
        for descendant in walk_postorder(child):
            yield descendant
    yield cursor


def get_public_supers(node: Cursor, parent, supers):
    if node.kind == CursorKind.CXX_BASE_SPECIFIER:
        supers.append(
            node
        )  # print(node.spelling, node.is_definition(), node.kind, parent.spelling)

    return 1  # don't recurse so you won't hit nested classes


def get_ctor(cursor):
    for node in cursor.walk_preorder():
        if node.kind == CursorKind.CONSTRUCTOR:
            return node
    warning(f"couldn't find ctor for {cursor.displayname}")


def canonicalize_class_name(name):
    return (
        name.replace(" ", "")
        .replace("*", "___")
        .replace("::", "_")
        .replace("<", "__")
        .replace(">", "__")
    )


all_classes_enums_structs = set()
for node in tu.cursor.walk_preorder():
    node: Cursor
    ty: Type = node.type
    if (
        node.kind in {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}
        and node.is_definition()
        and ty.spelling.startswith("mlir")
        and node.access_specifier
        not in {AccessSpecifier.PRIVATE, AccessSpecifier.PROTECTED}
        and node.location.file.name.startswith(f"{LLVM_INSTALL_DIR}/include/mlir/Dialect/Arith/IR")
        # and node.location.file.name.startswith(f"{LLVM_INSTALL_DIR}/include/mlir/IR")
        and node.type.get_canonical().spelling not in class_blacklist
    ):
        all_classes_enums_structs.add(
            node.type.get_canonical().spelling.replace(" ", "")
        )

for node in tu.cursor.walk_preorder():
    node: Cursor
    ty: Type = node.type
    if (
        node.kind in {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}
        and node.is_definition()
        and ty.spelling.startswith("mlir")
        and node.access_specifier
        not in {AccessSpecifier.PRIVATE, AccessSpecifier.PROTECTED}
        and node.location.file.name.startswith(f"{LLVM_INSTALL_DIR}/include/mlir/Dialect/Arith/IR")
        # and node.location.file.name.startswith(f"{LLVM_INSTALL_DIR}/include/mlir/IR")
        and node.type.get_canonical().spelling not in class_blacklist
    ):
        methods = defaultdict(list)
        class_being_visited[0] = node.type.get_canonical().spelling.replace(" ", "")
        clang.cindex.conf.lib.clang_visitChildren(
            node,
            clang.cindex.callbacks["cursor_visit"](collect_overloaded_methods),
            methods,
        )

        py_class_name = (
            node.displayname.replace("<", "[")
            .replace(">", "]")
            .replace("::mlir::", "")
            .replace("mlir::", "")
            .replace(" ", "")
            .replace("*", "")
        )

        cpp_class_name: str = node.type.get_canonical().spelling.replace(" ", "")
        if node.semantic_parent.kind in {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}:
            scope = (
                node.semantic_parent.type.get_canonical()
                .spelling.replace("::", "_")
                .replace("<", "__")
                .replace(">", "__")
                .replace("*", "___")
            )
        else:
            scope = "m"

        auto_var = f"""auto {cpp_class_name.replace('::', '_').replace("<", "__").replace(">", "__").replace("*", "___").replace(",", "____")} = """

        superclasses = []
        clang.cindex.conf.lib.clang_visitChildren(
            node,
            clang.cindex.callbacks["cursor_visit"](get_public_supers),
            superclasses,
        )

        super_name = None
        if len(superclasses) > 1:
            warning(f"multiple base classes not supported {node.spelling}")
            class_ = f"{cpp_class_name}"
        elif len(superclasses) == 1:
            super_name = superclasses[0].referenced.type.get_canonical().spelling
            if super_name in all_classes_enums_structs:
                class_ = f"{cpp_class_name}, {super_name}"
            elif super_name.startswith("mlir::Op<"):
                # print(
                #     f'nb::class_<{super_name}, mlir::OpState>({scope}, "mlir_Op[{py_class_name}]");',
                #     file=FILE,
                # )
                class_ = f"{cpp_class_name},  mlir::OpState"
            elif super_name.startswith("mlir::detail::StorageUserBase<"):
                super_name = super_name.split(", ")[1]
                class_ = f"{cpp_class_name}, {super_name}"
            elif super_name.startswith("mlir::IntegerAttr"):
                class_ = f"{cpp_class_name}, mlir::IntegerAttr"
            elif super_name.startswith("mlir::Dialect"):
                class_ = f"{cpp_class_name}, mlir::Dialect"
            else:
                warning(f"unknown super {super_name} for {cpp_class_name}")
                class_ = f"{cpp_class_name}"
        else:
            class_ = f"{cpp_class_name}"

        print(
            f'{auto_var}nb::class_<{class_}>({scope}, "{py_class_name}")',
            file=FILE,
        )
        clang.cindex.conf.lib.clang_visitChildren(
            node, clang.cindex.callbacks["cursor_visit"](class_visitor), methods
        )

        if super_name and super_name.startswith("mlir::Dialect"):
            print(
                f""".def_static("insert_into_registry", [](mlir::DialectRegistry &registry) {{ registry.insert<{cpp_class_name}>(); }})""",
                file=FILE,
            )
            print(
                f""".def_static("load_into_context", [](mlir::MLIRContext &context) {{ return context.getOrLoadDialect<{cpp_class_name}>(); }})""",
                file=FILE,
            )

        print(";", file=FILE)
        class_being_visited[0] = None
        print(file=FILE)

    if (
        node.kind == CursorKind.ENUM_DECL
        and node.is_definition()
        and ty.spelling.startswith("mlir")
        and node.access_specifier
        not in {AccessSpecifier.PRIVATE, AccessSpecifier.PROTECTED}
        and node.location.file.name.startswith(f"{LLVM_INSTALL_DIR}/include/mlir/Dialect/Arith/IR")
        # and node.location.file.name.startswith(f"{LLVM_INSTALL_DIR}/include/mlir/IR")
        and node.type.get_canonical().spelling not in class_blacklist
    ):
        class_being_visited[0] = node.type.get_canonical().spelling
        print(
            f'nb::enum_<{node.type.get_canonical().spelling}>(m, "{node.displayname}")',
            file=FILE,
        )
        clang.cindex.conf.lib.clang_visitChildren(
            node, clang.cindex.callbacks["cursor_visit"](enum_visitor), {}
        )
        print(";", file=FILE)
        class_being_visited[0] = None
        print(file=FILE)

print("}", file=FILE)

FILE.close()


def get_cursor(source, spelling):
    root_cursor = source if isinstance(source, Cursor) else source.cursor
    for cursor in root_cursor.walk_preorder():
        if cursor.spelling == spelling:
            return cursor

    return None


# TODO: don't return void


# src = """
# class A {};
# class B: public A {};
# void main(int b, bool a = false) {
# }
# """
#
# filename = "test.cpp"
# tu = clang.cindex.TranslationUnit.from_source(
#     filename,
#     comp_args,
#     options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
#     | clang.cindex.TranslationUnit.PARSE_INCOMPLETE,
#     unsaved_files=[(filename, src)],
# )
#
# B = get_cursor(tu, "B")
#
#
# def printchild(node: Cursor, parent, _):
#     if not node.kind == CursorKind.MACRO_DEFINITION:
#         print(node.spelling, node.is_definition(), node.kind, parent.spelling)
#     return 2
#
#
# clang.cindex.conf.lib.clang_visitChildren(
#     tu.cursor,
#     clang.cindex.callbacks["cursor_visit"](printchild),
#     {},
# )
