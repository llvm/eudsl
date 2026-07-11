# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from functools import update_wrapper
from typing import Union, Optional

from ..func import FuncBase
from ...util import infer_mlir_type, make_maybe_no_args_decorator
from ....dialects._ods_common import get_op_result_or_op_results

# noinspection PyUnresolvedReferences
from ....dialects.llvm import *

# noinspection PyUnresolvedReferences
from ....dialects.llvm import FunctionType as LLVMFunctionType
from ....ir import (
    FlatSymbolRefAttr,
    FloatAttr,
    IntegerAttr,
    Type,
    TypeAttr,
    Value,
)

ValueRef = Value


def llvm_ptr_t():
    return Type.parse("!llvm.ptr")


def llvm_void_t():
    return Type.parse("!llvm.void")


try:
    from . import amdgcn
except ImportError:
    pass


class _ReturnOp:
    """Adapts ``FuncBase``'s positional-list return convention to
    ``llvm.ReturnOp(arg=...)`` (which takes a single, optional, keyword value).

    Kept a class so it satisfies ``FuncBase.__init__``'s ``isclass`` assertion.
    """

    def __init__(self, operands, *, loc=None, ip=None):
        ReturnOp(arg=operands[0] if operands else None, loc=loc, ip=ip)


class LLVMFunc(FuncBase):
    def _make_function_type(self, input_types, result_types) -> TypeAttr:
        if len(result_types) == 0:
            ret = llvm_void_t()
        elif len(result_types) == 1:
            ret = result_types[0]
        else:
            raise ValueError(
                f"llvm.func supports at most one result type, got {result_types}"
            )
        return TypeAttr.get(LLVMFunctionType.get(ret, list(input_types)))

    def __call__(self, *call_args):
        op = self.emit(*call_args)
        ftype = LLVMFunctionType(op.function_type.value)
        result = None if ftype.return_type == llvm_void_t() else ftype.return_type
        return get_op_result_or_op_results(
            self.call_op_ctor(
                result,
                list(call_args),
                [],
                [],
                callee=FlatSymbolRefAttr.get(self.func_name),
            )
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
) -> LLVMFunc:
    func_ = LLVMFunc(
        body_builder=f,
        func_op_ctor=LLVMFuncOp,
        return_op_ctor=_ReturnOp,
        call_op_ctor=CallOp,
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


def mlir_constant(
    value: Union[int, float, bool], type: Optional[Type] = None, *, loc=None, ip=None
) -> Value:
    if type is None:
        type = infer_mlir_type(value, vector=False)

    if isinstance(value, int):
        value = IntegerAttr.get(type, value)
    elif isinstance(value, float):
        value = FloatAttr.get(type, value)
    else:
        raise NotImplementedError(f"{value} is not a valid type")

    return get_op_result_or_op_results(
        ConstantOp(res=value.type, value=value, loc=loc, ip=ip)
    )
