import argparse
import re
from pathlib import Path
from textwrap import dedent

import litgen


def preprocess_code(code: str, here, header_f) -> str:
    i = 0

    def replacement(s):
        nonlocal i
        r = f"enum {header_f.stem}_{i} {{"
        i += 1
        return r

    pattern = r"^enum\s*\{"
    transformed_code = re.sub(pattern, replacement, code, flags=re.MULTILINE)

    pattern = r"typedef\s+enum\s*\{(.*?)\}\s*([^;]+)\s*;"
    replacement = r"enum \2 {\1};"
    transformed_code = re.sub(pattern, replacement, transformed_code, flags=re.DOTALL)

    pattern = r"typedef\s+struct\s*\{(.*?)\}\s*([^;]+)\s*;"
    replacement = r"struct \2 {\1};"
    transformed_code = re.sub(pattern, replacement, transformed_code, flags=re.DOTALL)

    pattern = r"typedef struct \w+ (\w+);"
    replacement = r"struct \1;"
    transformed_code = re.sub(pattern, replacement, transformed_code)

    pattern = r"typedef\s+struct\s*\w+\s*\*([^;]+)\s*;"
    replacement = r"struct \1 { void* ptr; };"
    transformed_code = re.sub(pattern, replacement, transformed_code)

    pattern = r'#include "llvm-c/(\w+).h"'
    replacement = rf'#include "{here}/\1.h"'
    transformed_code = re.sub(pattern, replacement, transformed_code)

    transformed_code = transformed_code.replace(
        "typedef const void *LLVMDisasmContextRef;",
        "typedef void *LLVMDisasmContextRef;",
    )
    transformed_code = transformed_code.replace(
        "typedef const void *LLVMErrorTypeId;", "typedef void *LLVMErrorTypeId;"
    )
    transformed_code = transformed_code.replace(
        "extern const void*", "extern void*"
    )
    transformed_code = transformed_code.replace("/**", "/*")

    pattern = "^LLVM_C_EXTERN_C_BEGIN"
    replacement = 'extern "C" {'
    transformed_code = re.sub(pattern, replacement, transformed_code, flags=re.MULTILINE)

    pattern = "^LLVM_C_EXTERN_C_END"
    replacement = "}"
    transformed_code = re.sub(pattern, replacement, transformed_code, flags=re.MULTILINE)

    return transformed_code


def postprocess(code: str) -> str:
    code = code.replace('m, "LLVM', 'm, "')
    code = code.replace('.value("llvm_', '.value("')
    code = code.replace('.value("llvm', '.value("')
    code = code.replace('m.def("llvm_', 'm.def("')
    code = code.replace('m.def("llvm', 'm.def("')

    return code


def generate_header_bindings(cpp_code):
    options = litgen.LitgenOptions()
    options.srcmlcpp_options.preserve_empty_lines = False
    options.use_nanobind()
    options.python_reproduce_cpp_layout = False
    options.python_strip_empty_comment_lines = True
    options.postprocess_pydef_function = postprocess
    # options.comments_exclude = True
    options.fn_exclude_by_name__regex = "LLVMDisposeMessage|LLVMContextGetDiagnosticHandler|LLVMDisasmInstruction|LLVMDisposeErrorMessage|LLVMOrcCreateStaticLibrarySearchGeneratorForPath|LLVMRemarkVersion"
    generated_code = litgen.generate_code(options, cpp_code)
    return generated_code.pydef_code


def main(header_root, output_root):
    pp_dir = output_root / "pp"
    pp_dir.mkdir(parents=True, exist_ok=True)
    for header_f in Path(header_root).rglob("*.h"):
        with open(header_f) as ff:
            orig_code = ff.read()
        pp_header_f = pp_dir / header_f.name
        with open(pp_header_f, "w") as ff:
            ff.write(preprocess_code(orig_code, pp_dir, header_f))

        g = generate_header_bindings(open(pp_header_f).read())
        if not len(g.strip()):
            continue
        with open(output_root / f"{pp_header_f.stem}.cpp", "w") as ff:
            ff.write(
                dedent(
                    f"""
            #include "{pp_header_f}"
            #include "types.h"
            #include <nanobind/nanobind.h>
            #include <nanobind/ndarray.h>
            #include <nanobind/stl/optional.h>
            namespace nb = nanobind;
            """
                )
            )
            ff.write(f"void populate_{pp_header_f.stem}(nb::module_ &m) {{")
            ff.write(g)
            ff.write("}")

    for mod in Path(output_root).glob("*.cpp"):
        print(f"extern void populate_{mod.stem}(nb::module_ &m);")
        print(f"populate_{mod.stem}(m);")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eudsl-llvmpy-generate")
    parser.add_argument("headers_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    main(args.headers_root, args.output_root)
