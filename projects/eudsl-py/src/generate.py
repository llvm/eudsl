import ctypes
import glob
import os
import platform
import sysconfig
import warnings
from collections import defaultdict
from logging import warning
from pathlib import Path
from tempfile import TemporaryFile, NamedTemporaryFile
from textwrap import dedent

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
    Index,
    _CXUnsavedFile,
    conf,
    c_object_p,
    c_interop_string,
)
from ctypes import c_char_p, c_void_p
from blacklists import fn_blacklist, class_blacklist

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
    # "-std=c++17",
    "-std=c++17",
    "-fsyntax-only",
    "-fdirectives-only",
]
if platform.system() == "Darwin":
    comp_args.extend(
        [
            "-isysroot",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.1.sdk",
        ]
    )

clang.cindex.register_function(
    conf.lib,
    (
        "clang_parseTranslationUnit2",
        [
            Index,
            clang.cindex.c_interop_string,
            c_void_p,
            ctypes.c_int,
            c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            c_void_p,
        ],
        ctypes.c_uint,
    ),
    False,
)


def b(x):
    if isinstance(x, bytes):
        return x
    return x.encode("utf-8")


def from_source(filename, args=None, unsaved_files=None, options=0, index=None):
    if args is None:
        args = []

    if unsaved_files is None:
        unsaved_files = []

    if index is None:
        index = Index.create()

    args_array = None
    if len(args) > 0:
        args_array = (c_char_p * len(args))(*[b(x) for x in args])

    unsaved_array = None
    if len(unsaved_files) > 0:
        unsaved_array = (_CXUnsavedFile * len(unsaved_files))()
        for i, (name, contents) in enumerate(unsaved_files):
            if hasattr(contents, "read"):
                contents = contents.read()
            unsaved_array[i].name = c_interop_string(name)
            unsaved_array[i].contents = c_interop_string(contents)
            unsaved_array[i].length = len(unsaved_array[i].contents)

    tu = c_object_p()
    err_code = conf.lib.clang_parseTranslationUnit2(
        index,
        os.fspath(filename) if filename is not None else None,
        args_array,
        len(args),
        unsaved_array,
        len(unsaved_files),
        options,
        ctypes.byref(tu),
    )
    if err_code:
        match err_code:
            case 1:
                raise RuntimeError("CXError_Failure")
            case 2:
                raise RuntimeError("CXError_Crashed")
            case 3:
                raise RuntimeError("CXError_InvalidArguments")
            case 4:
                raise RuntimeError("CXError_ASTReadError")

    tu = TranslationUnit(tu, index)
    for d in tu.diagnostics:
        print(d)

    return tu


def parse_file(filepath):
    tu = from_source(
        filepath,
        comp_args,
        options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        | TranslationUnit.PARSE_INCOMPLETE
        | TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
    )
    return tu


def parse_files(source_files: list[Path]):
    with NamedTemporaryFile(mode="w", suffix=".hpp") as t:
        for s in source_files:
            t.write(f'#include "{s}"\n')
        t.flush()
        tu = parse_file(t.name)
    return tu


def parse_directory(dir_):
    source_files = [
        dir_ / p for p in glob.glob("**/*.h", root_dir=dir_, recursive=True)
    ]
    return parse_files(source_files)


def get_canonical_type(node: Cursor):
    return node.type.get_canonical().spelling


def collect_methods_for_class(class_node: Cursor):
    assert class_node.kind in {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}

    def visitor(node: Cursor, parent: Cursor, class_being_visited_methods):
        class_being_visited, methods = class_being_visited_methods
        if (
            node.kind
            in {
                CursorKind.CXX_METHOD,
                CursorKind.CONSTRUCTOR,
                CursorKind.FUNCTION_TEMPLATE,
            }
            and node.access_specifier == AccessSpecifier.PUBLIC
            and get_canonical_type(parent) == class_being_visited
        ):
            is_getter = (
                node.displayname.startswith("get")
                and len(list(node.get_arguments())) == 0
            )
            fn_name = node.displayname.split("(")[0]
            methods[fn_name].append(
                {"is_static": node.is_static_method(), "is_getter": is_getter}
            )

        return 1  # means continue adjacent

    methods = defaultdict(list)
    conf.lib.clang_visitChildren(
        class_node,
        clang.cindex.callbacks["cursor_visit"](visitor),
        (get_canonical_type(class_node), methods),
    )
    return methods


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


def get_public_supers(node: Cursor, parent, supers):
    if node.kind == CursorKind.CXX_BASE_SPECIFIER:
        supers.append(node)
        # print(node.spelling, node.is_definition(), node.kind, parent.spelling)
    return 1  # don't recurse so you won't hit nested classes


def emit_enum_values(node: clang.cindex.Cursor, _parent, FILE):
    print(
        f'.value("{node.spelling}", {node.type.spelling}::{node.spelling})', file=FILE
    )
    return 1  # means continue visiting adjacent


def get_all_classes_enums_structs(tu: TranslationUnit, file_match_string):
    all_classes_enums_structs = set()
    for node in tu.cursor.walk_preorder():
        node: Cursor
        ty: Type = node.type
        if not (
            node.kind in {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}
            and node.is_definition()
            and ty.spelling.startswith("mlir")
            and node.access_specifier
            not in {AccessSpecifier.PRIVATE, AccessSpecifier.PROTECTED}
            and file_match_string in node.location.file.name
            and get_canonical_type(node) not in class_blacklist
        ):
            continue
        all_classes_enums_structs.add(get_canonical_type(node))

    return all_classes_enums_structs


def emit_class_members(node: Cursor, parent: Cursor, counts_class_being_visited_FILE):
    class_being_visited, counts, FILE = counts_class_being_visited_FILE
    if not (
        node.kind in {CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR}
        and node.access_specifier == AccessSpecifier.PUBLIC
        and get_canonical_type(parent) == class_being_visited
        and node.displayname not in fn_blacklist
    ):
        return 2

    arg_names = []
    arg_types = []
    for i, a in enumerate(node.get_arguments()):
        t: Type = a.type
        t_parent = t.get_declaration().semantic_parent
        if t_parent and parent and t_parent == parent:
            t_spelling = t.get_named_type().spelling
        elif "_t" not in t.spelling:
            t_spelling = get_canonical_type(a)
        else:
            t_spelling = t.spelling

        arg_types.append(t_spelling)

        a_name = a.displayname
        if a_name in renames:
            a_name = renames[a_name]
        if not len(a_name):
            a_name = "_" * (i + 1)
        arg_names.append(a_name)

    fq_name = get_canonical_type(parent)
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


def emit_class(node: Cursor, all_classes_enums_structs, FILE):
    if node.semantic_parent.kind in {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}:
        scope = get_nb_bind_class_name(node.semantic_parent)
    else:
        scope = "m"

    superclasses = []
    conf.lib.clang_visitChildren(
        node,
        clang.cindex.callbacks["cursor_visit"](get_public_supers),
        superclasses,
    )

    cpp_class_name: str = get_canonical_type(node)
    super_name = None
    if len(superclasses) > 1:
        warning(f"multiple base classes not supported {node.spelling}")
        class_ = f"{cpp_class_name}"
    elif len(superclasses) == 1:
        super_name = get_canonical_type(superclasses[0].referenced)
        if super_name in all_classes_enums_structs:
            class_ = f"{cpp_class_name}, {super_name}"
        elif super_name.startswith("mlir::Op<"):
            # print(
            #     f'nb::class_<{super_name}, mlir::OpState>({scope}, "mlir_Op[{py_class_name}]");',
            #     file=FILE,
            # )
            class_ = f"{cpp_class_name},  mlir::OpState"
        elif super_name.startswith("mlir::detail::StorageUserBase<"):
            super_name = super_name.split(",")[1]
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

    auto_var = f"""auto {get_nb_bind_class_name(node)} = """
    print(
        f'{auto_var}nb::class_<{class_}>({scope}, "{get_py_class_name(node)}")',
        file=FILE,
    )

    methods = collect_methods_for_class(node)
    conf.lib.clang_visitChildren(
        node,
        clang.cindex.callbacks["cursor_visit"](emit_class_members),
        (get_canonical_type(node), methods, FILE),
    )

    if (
        super_name
        and super_name.startswith("mlir::Dialect")
        and cpp_class_name != "mlir::ExtensibleDialect"
    ):
        print(
            f""".def_static("insert_into_registry", [](mlir::DialectRegistry &registry) {{ registry.insert<{cpp_class_name}>(); }})""",
            file=FILE,
        )
        print(
            f""".def_static("load_into_context", [](mlir::MLIRContext &context) {{ return context.getOrLoadDialect<{cpp_class_name}>(); }})""",
            file=FILE,
        )

    print(";\n", file=FILE)


def get_nb_bind_class_name(node):
    return (
        get_canonical_type(node)
        .replace(" ", "")
        .replace("::", "_")
        .replace("<", "__")
        .replace(">", "__")
        .replace("*", "___")
        .replace(",", "____")
    )


def get_py_class_name(node):
    return (
        node.displayname.replace("<", "[")
        .replace(">", "]")
        .replace("::mlir::", "")
        .replace("mlir::", "")
        .replace(" ", "")
        .replace("*", "")
    )


def emit_enum(node, FILE):
    print(
        f'nb::enum_<{get_canonical_type(node)}>(m, "{node.displayname}")',
        file=FILE,
    )
    conf.lib.clang_visitChildren(
        node, clang.cindex.callbacks["cursor_visit"](emit_enum_values), FILE
    )
    print(";\n", file=FILE)


def emit_file(tu, filename, prologue, epilogue, file_match_string):
    seen = set()
    all_classes_enums_structs = get_all_classes_enums_structs(tu, file_match_string)
    FILE = open(filename, "w")

    print(prologue, file=FILE)

    for node in tu.cursor.walk_preorder():
        node: Cursor
        if node.hash in seen:
            continue
        seen.add(node.hash)
        if not node.location.file or file_match_string not in node.location.file.name:
            continue
        node_ty: Type = node.type
        cpp_class_name: str = get_canonical_type(node)

        if (
            node.kind in {CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL}
            and node.is_definition()
            and cpp_class_name.startswith("mlir")
            and node.access_specifier
            not in {AccessSpecifier.PRIVATE, AccessSpecifier.PROTECTED}
            and get_canonical_type(node) not in class_blacklist
        ):
            emit_class(node, all_classes_enums_structs, FILE)

        if (
            node.kind == CursorKind.ENUM_DECL
            and node.is_definition()
            and node_ty.spelling.startswith("mlir")
            and node.access_specifier
            not in {AccessSpecifier.PRIVATE, AccessSpecifier.PROTECTED}
            and get_canonical_type(node) not in class_blacklist
            and "unnamed enum" not in get_canonical_type(node)
        ):
            emit_enum(node, FILE)

    print(epilogue, file=FILE)


def emit_ir_module():
    prologue = dedent(
        f"""
    #include "ir.h"
    #include "mlir/IR/Action.h"
    #include "mlir/IR/AffineExpr.h"
    #include "mlir/IR/AffineExprVisitor.h"
    #include "mlir/IR/AffineMap.h"
    #include "mlir/IR/IntegerSet.h"
    #include "mlir/IR/AsmState.h"
    #include "mlir/IR/DialectResourceBlobManager.h"
    #include "mlir/IR/Iterators.h"
    #include "llvm/Support/ThreadPool.h"
    namespace nb = nanobind;
    using namespace nb::literals;
    void populateIRModule(nanobind::module_ & m) {{
    using namespace mlir;
    using namespace mlir::detail;
    """
    )
    epilogue = "}"
    # tu = parse_directory(
    #     Path("/Users/mlevental/dev_projects/eudsl/llvm-install/include/mlir/IR")
    # )
    tu = parse_file(
        "/Users/mlevental/dev_projects/eudsl/projects/eudsl-py/src/eudsl_ext.cpp"
    )

    emit_file(tu, "ir.cpp", prologue, epilogue, "include/mlir/IR")


def emit_arith_module():
    prologue = dedent(
        f"""
    #include "ir.h"
    namespace nb = nanobind;
    using namespace nb::literals;
    void populateArithModule(nanobind::module_ & m) {{
    using namespace mlir;
    using namespace mlir::detail;
    using namespace mlir::arith;
    """
    )
    epilogue = "}"
    # tu = parse_directory(
    #     Path("/Users/mlevental/dev_projects/eudsl/llvm-install/include/mlir/IR")
    # )
    tu = parse_file(
        "/Users/mlevental/dev_projects/eudsl/projects/eudsl-py/src/eudsl_ext.cpp"
    )
    emit_file(tu, "arith.cpp", prologue, epilogue, "include/mlir/Dialect/Arith/IR")


def emit_cf_module():
    prologue = dedent(
        f"""
    #include "ir.h"
    #include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
    #include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
    namespace nb = nanobind;
    using namespace nb::literals;
    void populateControlFlowModule(nanobind::module_ & m) {{
    using namespace mlir;
    using namespace mlir::detail;
    using namespace mlir::cf;
    """
    )
    epilogue = "}"
    # tu = parse_directory(
    #     Path("/Users/mlevental/dev_projects/eudsl/llvm-install/include/mlir/IR")
    # )
    tu = parse_file(
        "/Users/mlevental/dev_projects/eudsl/projects/eudsl-py/src/eudsl_ext.cpp"
    )
    emit_file(tu, "cf.cpp", prologue, epilogue, "include/mlir/Dialect/ControlFlow/IR")


def emit_scf_module():
    prologue = dedent(
        f"""
    #include "ir.h"
    namespace nb = nanobind;
    using namespace nb::literals;
    void populateSCFModule(nanobind::module_ & m) {{
    using namespace mlir;
    using namespace mlir::detail;
    using namespace mlir::scf;
    """
    )
    epilogue = "}"
    # tu = parse_directory(
    #     Path("/Users/mlevental/dev_projects/eudsl/llvm-install/include/mlir/IR")
    # )
    tu = parse_file(
        "/Users/mlevental/dev_projects/eudsl/projects/eudsl-py/src/eudsl_ext.cpp"
    )
    emit_file(tu, "scf.cpp", prologue, epilogue, "include/mlir/Dialect/SCF/IR")


def emit_affine_module():
    prologue = dedent(
        f"""
    #include "ir.h"
    #include "mlir/IR/IntegerSet.h"
    #include "mlir/Dialect/Affine/IR/AffineValueMap.h"
    namespace nb = nanobind;
    using namespace nb::literals;
    void populateAffineModule(nanobind::module_ & m) {{
    using namespace mlir;
    using namespace mlir::detail;
    using namespace mlir::affine;
    """
    )
    epilogue = "}"
    # tu = parse_directory(
    #     Path("/Users/mlevental/dev_projects/eudsl/llvm-install/include/mlir/IR")
    # )
    tu = parse_file(
        "/Users/mlevental/dev_projects/eudsl/projects/eudsl-py/src/eudsl_ext.cpp"
    )
    emit_file(tu, "affine.cpp", prologue, epilogue, "include/mlir/Dialect/Affine/IR")


# emit_ir_module()
# emit_arith_module()
# emit_cf_module()
# emit_scf_module()
emit_affine_module()
