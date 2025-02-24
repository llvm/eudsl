#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.
import warnings
from dataclasses import dataclass
import re
from textwrap import dedent

from .. import AttrOrTypeParameter


# stolen from inflection
def underscore(word: str) -> str:
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
    word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
    word = word.replace("-", "_")
    return word.lower()


def map_cpp_to_c_type(t):
    if t in {"unsigned", "bool", "int8_t", "int16_t", "int32_t", "int64_t"}:
        return t
    if t in {"RankedTensorType", "Type"}:
        return "MlirType"
    if t in {"Attribute", "CTALayoutAttr"}:
        return "MlirAttribute"
    warnings.warn(f"unrecognized cpp type {t}")
    return t


element_ty_reg = re.compile(r"ArrayRef<(\w+)>")


@dataclass
class Param:
    class_name: str
    param_name: str
    c_type: str
    cpp_type: str
    param_def: AttrOrTypeParameter

    def c_param_str(self):
        return f"{self.c_type} {self.param_name}"

    @property
    def getter_name(self):
        return f"mlir{self.class_name}Get{self.param_name}"

    # TODO(max): bad heuristic - should look inside param_def
    def needs_wrap_unwrap(self):
        return self.cpp_type != self.c_type


@dataclass
class ArrayRefParam(Param):
    c_element_type: str

    def c_param_str(self):
        return f"{self.c_element_type} *{self.param_name}, unsigned n{self.param_name}s"


def map_params(class_name, params: list[AttrOrTypeParameter]):
    mapped_params = []
    for p in params:
        cpp_ty = p.get_cpp_type()
        p_name = p.get_name()
        if "ArrayRef" in cpp_ty:
            element_ty = element_ty_reg.findall(cpp_ty)
            assert len(element_ty) == 1, f"couldn't find unique element_ty for {cpp_ty}"
            element_ty = element_ty[0]
            mapped_params.append(
                ArrayRefParam(
                    class_name, p_name, None, cpp_ty, p, map_cpp_to_c_type(element_ty)
                )
            )
        else:
            mapped_params.append(
                Param(class_name, p_name, map_cpp_to_c_type(cpp_ty), cpp_ty, p)
            )

    return mapped_params


try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class CClassKind(StrEnum):
    ATTRIBUTE = "MlirAttribute"
    TYPE = "MlirType"


def emit_c_attr_or_type_builder(
    cclass_kind: CClassKind, class_name, params: list[AttrOrTypeParameter]
):
    mapped_params = map_params(class_name, params)
    sig = f"""{cclass_kind} mlir{class_name}{'Attr' if cclass_kind == CClassKind.ATTRIBUTE else 'Type'}Get({', '.join([p.c_param_str() for p in mapped_params])}, MlirContext mlirContext)"""
    decl = f"""MLIR_CAPI_EXPORTED {sig};"""
    defn = dedent(
        f"""
    {sig} {{
      mlir::MLIRContext* context = unwrap(mlirContext);
    """
    )
    for p in mapped_params:
        if isinstance(p, ArrayRefParam):
            defn += f"  {p.cpp_type} {p.param_name}_ = {{{p.param_name}, n{p.param_name}s}};\n"
        else:
            rhs = (
                f"llvm::cast<{p.cpp_type}>(unwrap({p.param_name}))"
                if p.needs_wrap_unwrap()
                else p.param_name
            )
            defn += f"  {p.cpp_type} {p.param_name}_ = {rhs};\n"
    defn += f"  return wrap({class_name}::get(context, {', '.join([p.param_name + '_' for p in mapped_params])}));\n"
    defn += "}"

    return decl, defn


def emit_c_attr_or_type_field_getter(
    cclass_kind: CClassKind, class_name, param: AttrOrTypeParameter
):
    mapped_param = map_params(class_name, [param])[0]
    if isinstance(mapped_param, ArrayRefParam):
        sig = f"""void {mapped_param.getter_name}({cclass_kind} mlir{class_name}, {mapped_param.c_element_type}** {mapped_param.param_name}Ptr, unsigned *n{mapped_param.param_name}s)"""
        decl = f"MLIR_CAPI_EXPORTED {sig};"
        defn = dedent(
            f"""
        {sig} {{
          {mapped_param.param_def.get_cpp_accessor_type()} {mapped_param.param_name} = llvm::cast<{class_name}>(unwrap(mlir{class_name})).{mapped_param.param_def.get_accessor_name()}();
          *n{mapped_param.param_name}s = {mapped_param.param_name}.size();
          *{mapped_param.param_name}Ptr = const_cast<{mapped_param.c_element_type}*>({mapped_param.param_name}.data());
        }}
        """
        )
    else:
        sig = f"""{mapped_param.c_type} {mapped_param.getter_name}({cclass_kind} mlir{class_name})"""
        decl = f"""MLIR_CAPI_EXPORTED {sig};"""
        ret = f"llvm::cast<{class_name}>(unwrap(mlir{class_name})).{mapped_param.param_def.get_accessor_name()}()"
        if mapped_param.needs_wrap_unwrap():
            ret = f"wrap({ret})"
        defn = dedent(
            f"""
        {sig} {{
          return {ret};
        }}
        """
        )

    return decl, defn


def emit_attr_or_type_nanobind_class(
    cclass_kind: CClassKind, class_name, params: list[AttrOrTypeParameter]
):
    mapped_params = map_params(class_name, params)

    mlir_attr_or_mlir_type = (
        "MlirAttribute" if cclass_kind == CClassKind.ATTRIBUTE else "MlirType"
    )

    helper_decls = []
    helper_defns = []
    helper_decls.append(
        f"MLIR_CAPI_EXPORTED MlirTypeID mlir{class_name}GetTypeID(void);"
    )
    helper_defns.append(
        dedent(
            f"""\
    MlirTypeID mlir{class_name}GetTypeID() {{
      return wrap({class_name}::getTypeID());
    }}
    """
        )
    )
    helper_decls.append(
        f"MLIR_CAPI_EXPORTED bool isaMlir{class_name}({mlir_attr_or_mlir_type} thing);"
    )
    helper_defns.append(
        dedent(
            f"""\
    bool isaMlir{class_name}({mlir_attr_or_mlir_type} thing) {{
      return isa<{class_name}>(unwrap(thing));
    }}
    """
        )
    )

    params_str = []
    for mp in mapped_params:
        if isinstance(mp, ArrayRefParam):
            params_str.append(f"std::vector<{mp.c_element_type}> &{mp.param_name}")
        else:
            params_str.append(f"{mp.c_type} {mp.param_name}")
    params_str = ", ".join(params_str)
    s = dedent(
        f"""
        auto nb{class_name} = {'mlir_attribute_subclass' if cclass_kind == CClassKind.ATTRIBUTE else 'mlir_type_subclass'}(m, "{class_name}", isaMlir{class_name}, mlir{class_name}GetTypeID);
        nb{class_name}.def_staticmethod("get", []({params_str}, MlirContext context) {{
        """
    )

    arg_str = []
    help_str = []
    for mp in mapped_params:
        if isinstance(mp, ArrayRefParam):
            arg_str.append(f"{mp.param_name}.data(), {mp.param_name}.size()")
        else:
            arg_str.append(f"{mp.param_name}")
        help_str.append(f'"{underscore(mp.param_name)}"_a')
    arg_str.append("context")
    arg_str = ", ".join(arg_str)

    help_str.append('"context"_a = nb::none()')
    help_str = ", ".join(help_str)

    s += dedent(
        f"""\
        return mlir{class_name}{'Attr' if cclass_kind == CClassKind.ATTRIBUTE else 'Type'}Get({arg_str});
    }}, {help_str});
    """
    )

    for mp in mapped_params:
        if isinstance(mp, ArrayRefParam):
            s += dedent(
                f"""
                nb{class_name}.def_property_readonly("{underscore(mp.param_name)}", []({mlir_attr_or_mlir_type} self) {{
                  unsigned n{mp.param_name}s;
                  {mp.c_element_type}* {mp.param_name}Ptr;
                  {mp.getter_name}(self, &{mp.param_name}Ptr, &n{mp.param_name}s);
                  return std::vector<{mp.c_element_type}>{{{mp.param_name}Ptr, {mp.param_name}Ptr + n{mp.param_name}s}};
                }});
            """
            )
        else:
            s += dedent(
                f"""
                nb{class_name}.def_property_readonly("{underscore(mp.param_name)}", []({'MlirAttribute' if cclass_kind == CClassKind.ATTRIBUTE else 'MlirType'} self) {{
                  return {mp.getter_name}(self);
                }});
            """
            )

    return helper_decls, helper_defns, s


def emit_decls_defns_nbclasses(cclass_kind: CClassKind, defs):
    decls = []
    defns = []
    nbclasses = []
    for d in defs:
        params = list(d.get_parameters())
        if params:
            base_class_name = d.get_cpp_base_class_name()
            assert base_class_name in {"::mlir::Attribute", "::mlir::Type"}
            class_name = d.get_cpp_class_name()
            decl, defn = emit_c_attr_or_type_builder(cclass_kind, class_name, params)
            decls.append(decl)
            defns.append(defn)
            for p in params:
                decl, defn = emit_c_attr_or_type_field_getter(
                    cclass_kind, class_name, p
                )
                decls.append(decl)
                defns.append(defn)
            nbclass = emit_attr_or_type_nanobind_class(cclass_kind, class_name, params)
            nbclasses.append(nbclass)

    return decls, defns, nbclasses
