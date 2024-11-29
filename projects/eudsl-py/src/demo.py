import shlex

import clang.cindex
from clang.cindex import (
    CursorKind,
    Cursor,
    Type,
    TypeKind,
    TranslationUnit,
    AccessSpecifier,
)
import inflection

comp_args = shlex.split(
    "-D_DEBUG -D_GLIBCXX_ASSERTIONS -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -Deudsl_ext_EXPORTS -I/Users/mlevental/dev_projects/eudsl/llvm-install/include -I/Users/mlevental/miniforge3/envs/eudsl/include/python3.11 -I/Users/mlevental/miniforge3/envs/eudsl/lib/python3.11/site-packages/nanobind/include  -fPIC -fvisibility-inlines-hidden -Werror=date-time -Werror=unguarded-availability-new -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wmissing-field-initializers -Wimplicit-fallthrough -Wcovered-switch-default -Wno-noexcept-type -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -Wsuggest-override -Wstring-conversion -Wmisleading-indentation -Wctad-maybe-unsupported -fdiagnostics-color -O3 -DNDEBUG -std=gnu++17 -arch arm64 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.1.sdk -mmacosx-version-min=14.6 -fPIC -fvisibility=hidden -fcolor-diagnostics -UNDEBUG -fno-stack-protector -Os -fexceptions -frtti"
)


clang.cindex.Config.set_library_file(
    "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib"
)

filename = "/Users/mlevental/dev_projects/eudsl/projects/eudsl-py/src/eudsl_ext.cpp"
comp_args = [
    "-D_DEBUG",
    "-D_GLIBCXX_ASSERTIONS",
    "-D__STDC_CONSTANT_MACROS",
    "-D__STDC_FORMAT_MACROS",
    "-D__STDC_LIMIT_MACROS",
    "-Deudsl_ext_EXPORTS",
    "-I/Users/mlevental/dev_projects/eudsl/llvm-install/include",
    "-I/Users/mlevental/miniforge3/envs/eudsl/include/python3.11",
    "-I/Users/mlevental/miniforge3/envs/eudsl/lib/python3.11/site-packages/nanobind/include",
    "-fPIC",
    "-fvisibility-inlines-hidden",
    "-Werror=date-time",
    "-Werror=unguarded-availability-new",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-Wwrite-strings",
    "-Wcast-qual",
    "-Wmissing-field-initializers",
    "-Wimplicit-fallthrough",
    "-Wcovered-switch-default",
    "-Wno-noexcept-type",
    "-Wnon-virtual-dtor",
    "-Wdelete-non-virtual-dtor",
    "-Wsuggest-override",
    "-Wstring-conversion",
    "-Wmisleading-indentation",
    "-Wctad-maybe-unsupported",
    "-fdiagnostics-color",
    "-O3",
    "-DNDEBUG",
    "-std=gnu++17",
    "-arch",
    "arm64",
    "-isysroot",
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.1.sdk",
    "-mmacosx-version-min=14.6",
    "-fPIC",
    "-fvisibility=hidden",
    "-fcolor-diagnostics",
    "-UNDEBUG",
    "-fno-stack-protector",
    "-Os",
    "-fexceptions",
    "-frtti",
    "-std=c++17",
]

tu = clang.cindex.TranslationUnit.from_source(
    filename,
    comp_args,
    options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    | clang.cindex.TranslationUnit.PARSE_INCOMPLETE,
)
for d in tu.diagnostics:
    print(d)


def class_visitor(node: clang.cindex.Cursor, parent, userdata):
    if (
        node.kind
        in {
            CursorKind.CXX_METHOD
            # , CursorKind.FUNCTION_TEMPLATE
        }
        and node.access_specifier == AccessSpecifier.PUBLIC
    ):
        fn_name = node.displayname.split("(")[0]
        args = ", ".join(
            [
                f'"{inflection.underscore(a)}"_a'
                for a in map(lambda x: x.displayname, node.get_arguments())
            ]
        )
        if args:
            args = f", {args}"
        print(
            f'.def("{inflection.underscore(fn_name)}", &{parent.displayname}::{fn_name}{args})'
        )
    return 2  # means continue visiting recursively


for node in tu.cursor.walk_preorder():
    node: Cursor
    ty: Type = node.type
    if (
        node.kind == CursorKind.CLASS_DECL
        and node.is_definition()
        and ty.spelling.startswith("mlir")
    ):
        print(f'nb::class_<{node.displayname}>(m, "{node.displayname}")')
        clang.cindex.conf.lib.clang_visitChildren(
            node, clang.cindex.callbacks["cursor_visit"](class_visitor), []
        )
        print(";")
        print()


def get_cursor(source, spelling):
    root_cursor = source if isinstance(source, Cursor) else source.cursor
    for cursor in root_cursor.walk_preorder():
        if cursor.spelling == spelling:
            return cursor

    return None


# src = """
# class test_class_wtf
# class test_class {
# public:
# template <typename T>
# void public_member_function();
# protected:
# void protected_member_function();
# private:
# void private_member_function();
# };
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
# test_class = get_cursor(tu, "test_class")
# print(test_class.kind, test_class.is_definition())
# test_class_wtf = get_cursor(tu, "test_class_wtf")
# print(test_class_wtf.kind, test_class_wtf.is_definition())
#
# public = get_cursor(tu.cursor, "public_member_function")
#
# protected = get_cursor(tu.cursor, "protected_member_function")
#
# private = get_cursor(tu.cursor, "private_member_function")
