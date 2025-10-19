import argparse
import platform
import re
from pathlib import Path
from textwrap import dedent

import litgen
from eudsl_tblgen import RecordKeeper, lookup_intrinsic_id, intrinsic_is_overloaded


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
    replacement = rf'#include "{here.as_posix()}/\1.h"'
    transformed_code = re.sub(pattern, replacement, transformed_code)

    transformed_code = transformed_code.replace(
        "typedef const void *LLVMDisasmContextRef;",
        "typedef void *LLVMDisasmContextRef;",
    )
    transformed_code = transformed_code.replace(
        "typedef const void *LLVMErrorTypeId;", "typedef void *LLVMErrorTypeId;"
    )
    transformed_code = transformed_code.replace("extern const void*", "extern void*")
    transformed_code = transformed_code.replace("/**", "/*")

    pattern = "^LLVM_C_EXTERN_C_BEGIN"
    replacement = 'extern "C" {'
    transformed_code = re.sub(
        pattern, replacement, transformed_code, flags=re.MULTILINE
    )

    pattern = "^LLVM_C_EXTERN_C_END"
    replacement = "}"
    transformed_code = re.sub(
        pattern, replacement, transformed_code, flags=re.MULTILINE
    )
    transformed_code = transformed_code.replace(
        "uint8_t buf[LLVM_BLAKE3_BLOCK_LEN];", "uint8_t buf[64];"
    )
    transformed_code = transformed_code.replace(
        "uint8_t cv_stack[(LLVM_BLAKE3_MAX_DEPTH + 1) * LLVM_BLAKE3_OUT_LEN];",
        "uint8_t cv_stack[1760];",
    )

    return transformed_code


def postprocess(code: str) -> str:
    code = code.replace('m, "LLVM', 'm, "')
    code = code.replace('.value("llvm_', '.value("')
    code = code.replace('.value("llvm', '.value("')
    code = code.replace('m.def("llvm_', 'm.def("')
    code = code.replace('m.def("llvm', 'm.def("')
    pattern = r'\.def_rw\("ptr", &(\w+)::ptr, ""\)'
    repl = r'.def_rw("ptr", &\1::ptr, "").def_static("from_capsule", [](nb::capsule caps) -> \1 { void *ptr = PyCapsule_GetPointer(caps.ptr(), "nb_handle"); return {ptr}; })'
    code = re.sub(pattern, repl, code)

    return code


def generate_header_bindings(cpp_code):
    options = litgen.LitgenOptions()
    options.srcmlcpp_options.preserve_empty_lines = False
    options.use_nanobind()
    options.python_reproduce_cpp_layout = False
    options.python_strip_empty_comment_lines = True
    options.postprocess_pydef_function = postprocess
    # options.comments_exclude = True
    excludes = [
        "LLVMContextGetDiagnosticHandler",
        "LLVMDisasmInstruction",
        "LLVMDisposeErrorMessage",
        "LLVMDisposeMessage",
        "LLVMOrcCreateStaticLibrarySearchGeneratorForPath",
        "LLVMRemarkVersion",
    ]

    # APIs with callbacks that nanobind can't deduce/compile correctly
    if platform.system() == "Windows":
        excludes += [
            "LLVMContextSetDiagnosticHandler",
            "LLVMContextSetYieldCallback",
            "LLVMCreateDisasm",
            "LLVMCreateDisasmCPU",
            "LLVMCreateDisasmCPUFeatures",
            "LLVMCreateSimpleMCJITMemoryManager",
            "LLVMInstallFatalErrorHandler",
            "LLVMOrcCreateCustomCAPIDefinitionGenerator",
            "LLVMOrcCreateCustomMaterializationUnit",
            "LLVMOrcCreateDynamicLibrarySearchGeneratorForPath",
            "LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess",
            "LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks",
            "LLVMOrcExecutionSessionLookup",
            "LLVMOrcExecutionSessionSetErrorReporter",
            "LLVMOrcIRTransformLayerSetTransform",
            "LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator",
            "LLVMOrcObjectTransformLayerSetTransform",
            "LLVMOrcThreadSafeModuleWithModuleDo",
        ]

    options.fn_exclude_by_name__regex = "|".join(excludes)
    generated_code = litgen.generate_code(options, cpp_code)
    return generated_code.pydef_code


__normalize_python_kws = {"class": "class_", "if": "if_", "else": "else_"}


def generate_amdgcn_intrinsics(llvm_include_root: Path, llvmpy_module_dir: Path):
    amdgcn_f = open(llvmpy_module_dir / "amdgcn.py", "w")
    print(
        dedent(
            """\
    from typing import NewType, TypeVar, Generic, Literal
    from .util import call_intrinsic
    try:
        from . import ValueRef
    except ImportError:
        class ValueRef:
            pass
    
    any = NewType("any", ValueRef)      
    anyfloat = NewType("anyfloat", ValueRef)
    anyint = NewType("anyint", ValueRef)
    anyptr = NewType("anyptr", ValueRef)
    anyvector = NewType("anyvector", ValueRef)
    bfloat = NewType("bfloat", ValueRef)
    double = NewType("double", ValueRef)
    fp128 = NewType("fp128", ValueRef)
    fp80 = NewType("fp80", ValueRef)
    float = NewType("float", ValueRef)
    half = NewType("half", ValueRef)
    int1 = NewType("int1", ValueRef)
    int128 = NewType("int128", ValueRef)
    int16 = NewType("int16", ValueRef)     
    int32 = NewType("int32", ValueRef)
    int64 = NewType("int64", ValueRef)
    int8 = NewType("int8", ValueRef)
    ppcfp128 = NewType("ppcfp128", ValueRef)
    pointer = NewType("pointer", ValueRef)
    void = NewType("void", ValueRef)
    
    v1i1 = NewType("v1i1", ValueRef)
    v2i1 = NewType("v2i1", ValueRef)
    v3i1 = NewType("v3i1", ValueRef)
    v4i1 = NewType("v4i1", ValueRef)
    v8i1 = NewType("v8i1", ValueRef)
    v16i1 = NewType("v16i1", ValueRef)
    v32i1 = NewType("v32i1", ValueRef)
    v64i1 = NewType("v64i1", ValueRef)
    v128i1 = NewType("v128i1", ValueRef)
    v256i1 = NewType("v256i1", ValueRef)
    v512i1 = NewType("v512i1", ValueRef)
    v1024i1 = NewType("v1024i1", ValueRef)
    v2048i1 = NewType("v2048i1", ValueRef)

    v128i2 = NewType("v128i2", ValueRef)
    v256i2 = NewType("v256i2", ValueRef)

    v64i4 = NewType("v64i4", ValueRef)
    v128i4 = NewType("v128i4", ValueRef)

    v1i8 = NewType("v1i8", ValueRef)
    v2i8 = NewType("v2i8", ValueRef)
    v3i8 = NewType("v3i8", ValueRef)
    v4i8 = NewType("v4i8", ValueRef)
    v8i8 = NewType("v8i8", ValueRef)
    v16i8 = NewType("v16i8", ValueRef)
    v32i8 = NewType("v32i8", ValueRef)
    v64i8 = NewType("v64i8", ValueRef)
    v128i8 = NewType("v128i8", ValueRef)
    v256i8 = NewType("v256i8", ValueRef)
    v512i8 = NewType("v512i8", ValueRef)
    v1024i8 = NewType("v1024i8", ValueRef)

    v1i16 = NewType("v1i16", ValueRef)
    v2i16 = NewType("v2i16", ValueRef)
    v3i16 = NewType("v3i16", ValueRef)
    v4i16 = NewType("v4i16", ValueRef)
    v8i16 = NewType("v8i16", ValueRef)
    v16i16 = NewType("v16i16", ValueRef)
    v32i16 = NewType("v32i16", ValueRef)
    v64i16 = NewType("v64i16", ValueRef)
    v128i16 = NewType("v128i16", ValueRef)
    v256i16 = NewType("v256i16", ValueRef)
    v512i16 = NewType("v512i16", ValueRef)

    v1i32 = NewType("v1i32", ValueRef)
    v2i32 = NewType("v2i32", ValueRef)
    v3i32 = NewType("v3i32", ValueRef)
    v4i32 = NewType("v4i32", ValueRef)
    v5i32 = NewType("v5i32", ValueRef)
    v6i32 = NewType("v6i32", ValueRef)
    v7i32 = NewType("v7i32", ValueRef)
    v8i32 = NewType("v8i32", ValueRef)
    v9i32 = NewType("v9i32", ValueRef)
    v10i32 = NewType("v10i32", ValueRef)
    v11i32 = NewType("v11i32", ValueRef)
    v12i32 = NewType("v12i32", ValueRef)
    v16i32 = NewType("v16i32", ValueRef)
    v32i32 = NewType("v32i32", ValueRef)
    v64i32 = NewType("v64i32", ValueRef)
    v128i32 = NewType("v128i32", ValueRef)
    v256i32 = NewType("v256i32", ValueRef)
    v512i32 = NewType("v512i32", ValueRef)
    v1024i32 = NewType("v1024i32", ValueRef)
    v2048i32 = NewType("v2048i32", ValueRef)

    v1i64 = NewType("v1i64", ValueRef)
    v2i64 = NewType("v2i64", ValueRef)
    v3i64 = NewType("v3i64", ValueRef)
    v4i64 = NewType("v4i64", ValueRef)
    v8i64 = NewType("v8i64", ValueRef)
    v16i64 = NewType("v16i64", ValueRef)
    v32i64 = NewType("v32i64", ValueRef)
    v64i64 = NewType("v64i64", ValueRef)
    v128i64 = NewType("v128i64", ValueRef)
    v256i64 = NewType("v256i64", ValueRef)

    v1i128 = NewType("v1i128", ValueRef)

    v1f16 = NewType("v1f16", ValueRef)
    v2f16 = NewType("v2f16", ValueRef)
    v3f16 = NewType("v3f16", ValueRef)
    v4f16 = NewType("v4f16", ValueRef)
    v8f16 = NewType("v8f16", ValueRef)
    v16f16 = NewType("v16f16", ValueRef)
    v32f16 = NewType("v32f16", ValueRef)
    v64f16 = NewType("v64f16", ValueRef)
    v128f16 = NewType("v128f16", ValueRef)
    v256f16 = NewType("v256f16", ValueRef)
    v512f16 = NewType("v512f16", ValueRef)

    v1bf16 = NewType("v1bf16", ValueRef)
    v2bf16 = NewType("v2bf16", ValueRef)
    v3bf16 = NewType("v3bf16", ValueRef)
    v4bf16 = NewType("v4bf16", ValueRef)
    v8bf16 = NewType("v8bf16", ValueRef)
    v16bf16 = NewType("v16bf16", ValueRef)
    v32bf16 = NewType("v32bf16", ValueRef)
    v64bf16 = NewType("v64bf16", ValueRef)
    v128bf16 = NewType("v128bf16", ValueRef)

    v1f32 = NewType("v1f32", ValueRef)
    v2f32 = NewType("v2f32", ValueRef)
    v3f32 = NewType("v3f32", ValueRef)
    v4f32 = NewType("v4f32", ValueRef)
    v5f32 = NewType("v5f32", ValueRef)
    v6f32 = NewType("v6f32", ValueRef)
    v7f32 = NewType("v7f32", ValueRef)
    v8f32 = NewType("v8f32", ValueRef)
    v9f32 = NewType("v9f32", ValueRef)
    v10f32 = NewType("v10f32", ValueRef)
    v11f32 = NewType("v11f32", ValueRef)
    v12f32 = NewType("v12f32", ValueRef)
    v16f32 = NewType("v16f32", ValueRef)
    v32f32 = NewType("v32f32", ValueRef)
    v64f32 = NewType("v64f32", ValueRef)
    v128f32 = NewType("v128f32", ValueRef)
    v256f32 = NewType("v256f32", ValueRef)
    v512f32 = NewType("v512f32", ValueRef)
    v1024f32 = NewType("v1024f32", ValueRef)
    v2048f32 = NewType("v2048f32", ValueRef)

    v1f64 = NewType("v1f64", ValueRef)
    v2f64 = NewType("v2f64", ValueRef)
    v3f64 = NewType("v3f64", ValueRef)
    v4f64 = NewType("v4f64", ValueRef)
    v8f64 = NewType("v8f64", ValueRef)
    v16f64 = NewType("v16f64", ValueRef)
    v32f64 = NewType("v32f64", ValueRef)
    v64f64 = NewType("v64f64", ValueRef)
    v128f64 = NewType("v128f64", ValueRef)
    v256f64 = NewType("v256f64", ValueRef)
    
    vararg = NewType("vararg", ValueRef)
    metadata = NewType("metadata", ValueRef)
    
    flat_ptr = NewType("flat_ptr", ValueRef)
    
    _T = TypeVar('_T')
    
    class LLVMQualPointerType(Generic[_T]):
        pass
    
    local_ptr = LLVMQualPointerType[Literal[3]]
    global_ptr = LLVMQualPointerType[Literal[1]]
    AMDGPUBufferRsrcTy = LLVMQualPointerType[Literal[8]];
    
    class LLVMMatchType(Generic[_T]):
        pass
    """
        ),
        file=amdgcn_f,
    )
    intrins = RecordKeeper().parse_td(
        str(llvm_include_root / "llvm" / "IR" / "Intrinsics.td"),
        include_dirs=[str(llvm_include_root)],
    )
    int_regex = re.compile(r"_i(\d+)")
    fp_regex = re.compile(r"_f(\d+)")

    defs = intrins.get_defs()
    for d in defs:
        intr = defs[d]
        if (
            intr.get_name().startswith("int_amdgcn")
            and intr.get_type().get_as_string() != "ClangBuiltin"
        ):
            arg_types = []
            ret_types = []
            for p in intr.get_values().ParamTypes.get_value():
                p_s = p.get_as_string()
                if p_s.startswith("anon"):
                    p_s = p.get_type().get_as_string()
                    pdv = p.get_def().get_values()
                    if p_s == "LLVMMatchType":
                        p_s += f"[Literal[{pdv.Number.get_value()}]]"
                    elif p_s == "LLVMQualPointerType":
                        kind, addr_space = pdv.Sig.get_value()
                        p_s += f"[Literal[{addr_space}]]"
                    else:
                        raise NotImplemented(f"unsupported {p_s=}")
                else:
                    p_s = re.sub(int_regex, r"_int\1", p_s)
                    p_s = re.sub(fp_regex, r"_fp\1", p_s)
                    p_s = p_s.replace("llvm_", "").replace("_ty", "")

                if p_s == "ptr":
                    p_s = "pointer"

                arg_types.append(p_s)
            for p in intr.get_values().RetTypes.get_value():
                ret_types.append(p.get_as_string())

            ret_str = ""
            if len(ret_types):
                ret_str = "return "

            intr_name = d.replace("int_amdgcn_", "")
            llvm_intr_name = f"llvm.amdgcn.{intr_name.replace('_', '.')}"
            intr_id = lookup_intrinsic_id(llvm_intr_name)
            is_overloaded = intrinsic_is_overloaded(intr_id)
            arg_names = "abcdefghijklmnopqrstuvwxyz"[: len(arg_types)]
            fn_args_str = ", ".join([f"{n}: {t}" for n, t in zip(arg_names, arg_types)])
            call_args_str = ", ".join(arg_names)
            if fn_args_str:
                fn_args_str = f"{fn_args_str}, "
                call_args_str = f"{call_args_str}, "

            intr_name = __normalize_python_kws.get(intr_name, intr_name)

            print(
                dedent(
                    f"""
                def {intr_name}({fn_args_str}):
                    {ret_str}call_intrinsic({call_args_str}{intr_id=}, {is_overloaded=}, intr_name="{llvm_intr_name}")
            """
                ),
                file=amdgcn_f,
            )

    amdgcn_f.flush()
    amdgcn_f.close()


def generate_nb_bindings(header_root: Path, output_root: Path):
    pp_dir = output_root / "pp"
    pp_dir.mkdir(parents=True, exist_ok=True)
    for header_f in header_root.rglob("*.h"):
        if header_f.name == "lto.h":
            continue
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
            #include "{pp_header_f.as_posix()}"
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

    # for mod in Path(output_root).glob("*.cpp"):
    #     print(f"extern void populate_{mod.stem}(nb::module_ &m);")
    #     print(f"populate_{mod.stem}(m);")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eudsl-llvmpy-generate")
    parser.add_argument("llvm_include_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("llvmpy_module_dir", type=Path)
    args = parser.parse_args()

    generate_nb_bindings(args.llvm_include_root / "llvm-c", args.output_root)
    generate_amdgcn_intrinsics(args.llvm_include_root, args.llvmpy_module_dir)
