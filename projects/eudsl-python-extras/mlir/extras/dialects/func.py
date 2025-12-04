# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import inspect
import sys
from functools import update_wrapper
from typing import Optional, List, Union, TypeVar

from .. import types
from ..ast.py_type import PyTypeVarObject, _Ptr, PyObject
from ..ast.util import copy_func
from ..meta import op_region_builder
from ..util import get_user_code_loc, make_maybe_no_args_decorator
from ...dialects._ods_common import get_op_result_or_op_results
from ...dialects.func import *
from ...ir import (
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    OpView,
    Operation,
    OpResultList,
    Type,
    TypeAttr,
    Value,
)


_call = call


def call(
    callee_or_results: Union[FuncOp, List[Type]],
    arguments_or_callee: Union[List[Value], FlatSymbolRefAttr, str],
    arguments: Optional[list] = None,
    *,
    call_op_ctor=CallOp.__base__,
    loc=None,
    ip=None,
):
    if isinstance(callee_or_results, FuncOp.__base__):
        if not isinstance(arguments_or_callee, (list, tuple)):
            raise ValueError(
                "when constructing a call to a function, expected "
                + "the second argument to be a list of call arguments, "
                + f"got {type(arguments_or_callee)}"
            )
        if arguments is not None:
            raise ValueError(
                "unexpected third argument when constructing a call" + "to a function"
            )
        if not all(
            isinstance(a, (Value, Operation, OpView)) for a in arguments_or_callee
        ):
            raise ValueError(
                f"{arguments_or_callee} must all be Value, Operation, or OpView"
            )

        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results.function_type.value.results,
                FlatSymbolRefAttr.get(callee_or_results.sym_name.value),
                arguments_or_callee,
                loc=loc,
                ip=ip,
            )
        )

    if isinstance(arguments_or_callee, list):
        raise ValueError(
            "when constructing a call to a function by name, "
            + "expected the second argument to be a string or a "
            + f"FlatSymbolRefAttr, got {type(arguments_or_callee)}"
        )

    if isinstance(arguments_or_callee, FlatSymbolRefAttr):
        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results, arguments_or_callee, arguments, loc=loc, ip=ip
            )
        )
    elif isinstance(arguments_or_callee, str):
        return get_op_result_or_op_results(
            call_op_ctor(
                callee_or_results,
                FlatSymbolRefAttr.get(arguments_or_callee),
                arguments,
                loc=loc,
                ip=ip,
            )
        )
    else:
        raise ValueError(f"unexpected type {callee_or_results=}")


def isalambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__


def prep_func_types(sig, return_types):
    assert not (
        not sig.return_annotation is inspect.Signature.empty and len(return_types) > 0
    ), f"func can use return annotation or explicit return_types but not both"
    return_types = (
        sig.return_annotation
        if not sig.return_annotation is inspect.Signature.empty
        else return_types
    )
    if not isinstance(return_types, (tuple, list)):
        return_types = [return_types]
    return_types = list(return_types)
    assert all(
        isinstance(r, (str, Type, TypeVar)) or isalambda(r) for r in return_types
    ), f"all return types must be mlir types or strings or TypeVars or lambdas {return_types=}"

    input_types = [
        p.annotation
        for p in sig.parameters.values()
        if not p.annotation is inspect.Signature.empty
    ]
    assert all(
        isinstance(r, (str, Type, TypeVar)) or isalambda(r) for r in input_types
    ), f"all input types must be mlir types or strings or TypeVars or lambdas {input_types=}"
    user_loc = get_user_code_loc()
    # If ir.Context is none (like for deferred func emit)
    if user_loc is None:
        user_locs = None
    else:
        user_locs = [user_loc] * len(sig.parameters)
    return input_types, return_types, user_locs


def get_type_var_default_bound(tvar):
    type_var = PyTypeVarObject.from_object(tvar)
    type_var_bound = type_var.bound
    type_var_default = None
    if sys.version_info >= (3, 13) and tvar.has_default():
        type_var_default = type_var.default_value

    return type_var_default, type_var_bound


class ReifiedTypeParam:
    name: str
    concrete_val: object
    type_name: Optional[type]

    def __init__(
        self,
        tvar,
        concrete_val=None,
        already_reified_type_params: dict[str, object] = None,
    ):
        self.name = tvar.__name__
        if already_reified_type_params is None:
            already_reified_type_params = {}

        type_var_default, type_var_bound = get_type_var_default_bound(tvar)

        if concrete_val is None and not bool(type_var_default):
            raise ValueError(
                "either concrete_val must be provided or typevar must have a default"
            )

        if bool(type_var_bound):
            type_var_bound = maybe_eval_type_data_closure_vals(
                type_var_bound, already_reified_type_params
            )
        elif not bool(type_var_default):
            if isinstance(concrete_val, Type):
                type_var_bound = "type"
            else:
                type_var_bound = type(concrete_val).__name__

        if bool(type_var_default):
            type_var_default = maybe_eval_type_data_closure_vals(
                type_var_default, already_reified_type_params
            )
            if not bool(type_var_bound):
                type_var_bound = type(type_var_default).__name__

        if bool(type_var_default) and concrete_val is None:
            self.concrete_val = type_var_default
        else:
            assert concrete_val is not None, "expected non-null concrete_val"
            self.concrete_val = concrete_val

        self.type_name = type_var_bound
        # TODO(max): implement _some_ kind of type checking to make sure
        # self.concrete_val matches either the type bound or the type of the default

    def add_replace_in_closure(self, fn):
        # only in the closure if used in the body
        if self.name in fn.__code__.co_freevars:
            free_i = fn.__code__.co_freevars.index(self.name)
            fn.__closure__[free_i].cell_contents = self.concrete_val


# For "generics" (i.e. typevars) which are dependent on previous generics (identified by the fact that they have vals in their own closures),
# we collect all such previous generics along with the concrete vals (into already_reified_type_params) and then
# evaluate the typevars in the fully-populated closure. Note, in order to get the unevaled typevar bound and default value
# we access them in the PyTypeVarObject C struct itself instead of the API that python provides.
def maybe_eval_type_data_closure_vals(
    unevaled_type_data: _Ptr[PyObject],
    already_reified_type_params: dict[str, object],
):
    assert type(unevaled_type_data) == _Ptr[PyObject]
    unevaled_type_data = unevaled_type_data.contents.into_object()
    cvrs = inspect.getclosurevars(unevaled_type_data).nonlocals
    if len(cvrs):
        for k, v in cvrs.items():
            if not isinstance(v, TypeVar):
                continue
            if k not in already_reified_type_params:
                raise RuntimeError(
                    f"typevar {k} not reified prior to evaluating dependent typevar {v}"
                )
            cvrs[k] = already_reified_type_params[k]
        unevaled_type_data = copy_func(unevaled_type_data, cvrs)
    return unevaled_type_data()


class FuncBase:
    def __init__(
        self,
        body_builder,
        func_op_ctor,
        return_op_ctor,
        call_op_ctor,
        *,
        return_types=None,
        sym_visibility=None,
        sym_name=None,
        arg_attrs=None,
        res_attrs=None,
        func_attrs=None,
        function_type=None,
        generics: List[Union[TypeVar, ReifiedTypeParam]] = None,
        qualname=None,
        loc=None,
        ip=None,
    ):
        assert inspect.isfunction(body_builder), body_builder
        assert inspect.isclass(func_op_ctor), func_op_ctor
        if return_op_ctor is not None:
            assert inspect.isclass(return_op_ctor), return_op_ctor
        assert inspect.isclass(call_op_ctor), call_op_ctor

        self.body_builder = body_builder
        if sym_name is None:
            sym_name = self.body_builder.__name__
        self.func_name = sym_name
        self.func_op_ctor = func_op_ctor
        self.return_op_ctor = return_op_ctor
        self.call_op_ctor = call_op_ctor
        self.arg_attrs = arg_attrs
        self.res_attrs = res_attrs
        if generics is None:
            generics = []
        self.generics = generics
        self.loc = loc
        self.ip = ip
        self._func_op = None
        # in case this function lives inside a class
        self.qualname = qualname

        self.sym_visibility = sym_visibility
        self.func_attrs = func_attrs
        if self.func_attrs is None:
            self.func_attrs = {}
        self.function_type = function_type

        if return_types is None:
            return_types = []
        sig = inspect.signature(self.body_builder)
        self.input_types, self.return_types, self.arg_locs = prep_func_types(
            sig, return_types
        )

        if self._is_decl():
            assert len(self.input_types) == len(
                sig.parameters
            ), f"func decl needs all input types annotated"
            self.sym_visibility = "private"
            self.emit()

    def _is_decl(self):
        # magic constant found from looking at the code for an empty fn
        if sys.version_info.minor == 14:
            return self.body_builder.__code__.co_code == b"\x80\x00R\x00#\x00"
        if sys.version_info.minor == 13:
            return self.body_builder.__code__.co_code == b"\x95\x00g\x00"
        if sys.version_info.minor == 12:
            return self.body_builder.__code__.co_code == b"\x97\x00y\x00"
        if sys.version_info.minor == 11:
            return self.body_builder.__code__.co_code == b"\x97\x00d\x00S\x00"
        if sys.version_info.minor in {8, 9, 10}:
            return self.body_builder.__code__.co_code == b"d\x00S\x00"
        raise NotImplementedError(f"{sys.version_info.minor} not supported.")

    def __str__(self):
        return str(f"{self.__class__} {self.__dict__}")

    def _build_input_types(self) -> Union[list[Type], OpView]:
        """Either build all input types (if no generics or all generics reified) or return a further specialized funcop thing (the return of __getitem__)."""
        locals = {}
        generics = list(self.generics)
        while generics and isinstance(generics[0], ReifiedTypeParam):
            g = generics.pop(0)
            locals[g.name] = g.concrete_val

        # (potentially) reify generics with defaults (i.e., fully specialize i.e., do __getitem__ with defaults)
        default_vals = []
        while generics:
            g = generics.pop(0)
            if not isinstance(g, TypeVar):
                raise ValueError(f"expected {g=} to be a TypeVar")
            # explicit None means use default
            default_vals.append(None)
        if default_vals:
            return self.__getitem__(tuple(default_vals)).emit()

        # pre-load locals with useful things (TODO(max): this probably should go away)
        if "T" in locals:
            raise ValueError(
                f"T is a reserved generic name; use a different one for {locals['T']}"
            )
        locals["T"] = types
        if "S" in locals:
            raise ValueError(
                f"S is a reserved generic name; use a different one for {locals['S']}"
            )
        locals["S"] = ShapedType.get_dynamic_size()

        # evaluate type annotations (which could be strings or lambdas)
        input_types = self.input_types[:]
        for i, v in enumerate(input_types):
            if isinstance(v, TypeVar):
                v = v.__name__
            if isinstance(v, str):
                input_types[i] = Type(eval(v, self.body_builder.__globals__, locals))
            elif isalambda(v):
                input_types[i] = v()

        return input_types

    def emit(self, *call_args, decl=False, force=False) -> FuncOp:
        if self._func_op and not (decl or force):
            return self._func_op

        if self.function_type is not None:
            input_types = self.function_type.inputs
        elif len(call_args):
            input_types = [a.type for a in call_args]
        else:
            input_types = self._build_input_types()
            assert isinstance(
                input_types, (list, self.func_op_ctor)
            ), "expected self._build_input_types to either return a list of types or a partially specialized funcop thing."
            if isinstance(input_types, self.func_op_ctor):
                return input_types

        if self.function_type is not None:
            function_type = TypeAttr.get(self.function_type)
        else:
            function_type = TypeAttr.get(
                FunctionType.get(inputs=input_types, results=self.return_types)
            )

        self._func_op = self.func_op_ctor(
            self.func_name,
            function_type,
            sym_visibility=self.sym_visibility,
            arg_attrs=self.arg_attrs,
            res_attrs=self.res_attrs,
            loc=self.loc,
            ip=self.ip or InsertionPoint.current,
        )
        for k, v in self.func_attrs.items():
            self._func_op.attributes[k] = v
        # if only a decl, don't build body
        if self._is_decl() or decl:
            return self._func_op

        self._func_op.regions[0].blocks.append(*input_types, arg_locs=self.arg_locs)
        builder_wrapper = op_region_builder(
            self._func_op, self._func_op.regions[0], terminator=self.return_op_ctor
        )

        return_types = []

        # infer result types from returned values in the body_builder
        def grab_results(*args):
            nonlocal return_types
            results = self.body_builder(*args)
            if isinstance(results, (tuple, list, OpResultList)):
                return_types.extend([r.type for r in results])
            elif results is not None:
                return_types.append(results.type)
            return results

        if self.function_type is None:
            builder_wrapper(grab_results)
            function_type = FunctionType.get(inputs=input_types, results=return_types)
            self._func_op.attributes["function_type"] = TypeAttr.get(function_type)
        else:
            builder_wrapper(self.body_builder)

        return self._func_op

    def __call__(self, *call_args):
        return call(self.emit(*call_args), call_args)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if self.generics is None:
            raise RuntimeError(
                "using a generic call requires the func be generic (i.e., have type_params)"
            )
        # this also copies the function so that the original body_builder remains "generic" (via its closure)
        body_builder = copy_func(self.body_builder)
        reified_type_params: list[ReifiedTypeParam] = []
        already_reified_type_params: dict[str, object] = {}
        generics = list(self.generics)

        while len(generics) and isinstance(generics[0], ReifiedTypeParam):
            g = generics.pop(0)
            reified_type_params.append(g)
            already_reified_type_params[g.name] = g.concrete_val

        for it in item:
            tvar = generics.pop(0)
            if tvar.__name__ in body_builder.__globals__:
                raise RuntimeError("global typevars for generics are not supported")
            r = ReifiedTypeParam(tvar, it, already_reified_type_params)
            already_reified_type_params[r.name] = r.concrete_val
            reified_type_params.append(r)

        for r in reified_type_params:
            r.add_replace_in_closure(body_builder)

        name_mangled_generics = []
        for r in reified_type_params:
            tvar, v = r.type_name, r.concrete_val
            if callable(v):
                v = v.__name__
            name_mangled_generics.append(f"{tvar}_{v}")

        return FuncBase(
            body_builder,
            self.func_op_ctor,
            self.return_op_ctor,
            self.call_op_ctor,
            return_types=self.return_types,
            sym_visibility=self.sym_visibility,
            sym_name=(
                self.body_builder.__name__ + "_" + "_".join(name_mangled_generics)
            ),
            arg_attrs=self.arg_attrs,
            res_attrs=self.res_attrs,
            func_attrs=self.func_attrs,
            generics=reified_type_params + generics,
            qualname=self.qualname,
            loc=self.loc,
            ip=self.ip,
        )


@make_maybe_no_args_decorator
def func(
    f,
    *,
    sym_visibility=None,
    sym_name=None,
    arg_attrs=None,
    res_attrs=None,
    func_attrs=None,
    function_type=None,
    emit=False,
    loc=None,
    ip=None,
) -> FuncBase:
    func_ = FuncBase(
        body_builder=f,
        func_op_ctor=FuncOp.__base__,
        return_op_ctor=ReturnOp,
        call_op_ctor=CallOp.__base__,
        sym_visibility=sym_visibility,
        sym_name=sym_name,
        arg_attrs=arg_attrs,
        res_attrs=res_attrs,
        func_attrs=func_attrs,
        function_type=function_type,
        generics=getattr(f, "__type_params__", None),
        loc=loc,
        ip=ip,
    )
    func_ = update_wrapper(func_, f)
    if emit:
        func_.emit()
    return func_
