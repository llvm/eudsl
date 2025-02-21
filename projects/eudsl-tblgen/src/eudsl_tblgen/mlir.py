#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

from dataclasses import dataclass
import re
from textwrap import dedent

from inflection import underscore

from . import AttrOrTypeParameter


def map_cpp_to_c_type(t):
    if t == "unsigned":
        return t
    if t in {"RankedTensorType"}:
        return "MlirType"
    raise NotImplementedError(t)


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


def emit_c_attr_builder(class_name, params: list[AttrOrTypeParameter]):
    mapped_params = map_params(class_name, params)
    sig = f"""MlirAttribute mlir{class_name}AttrGet({', '.join([p.c_param_str() for p in mapped_params])}, MlirContext mlirContext)"""
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
            defn += f"  {p.cpp_type} {p.param_name}_ = unwrap({p.param_name});\n"
    defn += f"  return wrap({class_name}::get(context, {', '.join([p.param_name + '_' for p in mapped_params])}));\n"
    defn += "}"

    return decl, defn


def emit_c_attr_field_getter(class_name, param: AttrOrTypeParameter):
    mapped_param = map_params(class_name, [param])[0]
    if isinstance(mapped_param, ArrayRefParam):
        if not mapped_param.c_element_type == "unsigned":
            raise NotImplementedError(mapped_param.c_element_type)
        sig = f"""void {mapped_param.getter_name}(MlirAttribute mlir{class_name}, {mapped_param.c_element_type}** {mapped_param.param_name}Ptr,  unsigned *n{mapped_param.param_name}s)"""
        decl = f"MLIR_CAPI_EXPORTED {sig};"
        defn = dedent(
            f"""
        {sig} {{
          {mapped_param.param_def.get_cpp_accessor_type()} {mapped_param.param_name} = llvm::cast<{class_name}>(unwrap(mlir{class_name})).{mapped_param.param_def.get_accessor_name()}();
          *n{mapped_param.param_name}s = {mapped_param.param_name}.size();
          *{mapped_param.param_name}Ptr = {mapped_param.param_name}.data();
        }}
        """
        )
    else:
        sig = f"""{mapped_param.c_type} {mapped_param.getter_name}(MlirAttribute mlir{class_name})"""
        decl = f"""MLIR_CAPI_EXPORTED {sig};"""
        defn = dedent(
            f"""
        {sig} {{
          return unwrap(mlir{class_name}).{mapped_param.param_def.get_accessor_name()}();
        }}
        """
        )

    return decl, defn


def emit_attr_nanobind_class(class_name, params: list[AttrOrTypeParameter]):
    mapped_params = map_params(class_name, params)

    params_str = []
    for mp in mapped_params:
        if isinstance(mp, ArrayRefParam):
            params_str.append(f"std::vector<{mp.c_element_type}> &{mp.param_name}")
        else:
            params_str.append(f"{mp.c_type} {mp.param_name}")
    params_str = ", ".join(params_str)
    s = dedent(
        f"""
        auto nb{class_name} = mlir_attribute_subclass(m, "{class_name}", isaMlir{class_name}, mlir{class_name}GetTypeID);
        nb{class_name}.def("get", []({params_str}, MlirContext context) {{
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
        return mlir{class_name}AttrGet({arg_str});
    }}, {help_str});
    """
    )

    for mp in mapped_params:
        if isinstance(mp, ArrayRefParam):
            s += dedent(
                f"""
                nb{class_name}.def_property_readonly("{underscore(mp.param_name)}", [](MlirAttribute self) {{
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
                nb{class_name}.def_property_readonly("{underscore(mp.param_name)}", [](MlirAttribute self) {{
                  return {mp.getter_name}(self);
                }});
            """
            )

    return s


def emit_type_nanobind_class(class_name, params: list[AttrOrTypeParameter]):
    mapped_params = map_params(class_name, params)

    s = f'auto nb{class_name} = mlir_type_subclass(m, "{class_name}", isaMlir{class_name}, mlir{class_name}GetTypeID);'

    return s


def emit_c_type_builder(name, params: list[AttrOrTypeParameter]):
    # print(name)
    # for p in params:
    #     print(p.get_cpp_type())

    return ""
