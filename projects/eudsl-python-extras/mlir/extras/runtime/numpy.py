# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes
import logging
import os
import platform
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from ... import _mlir_libs
from ...dialects.func import CallOp, FuncOp
from ...ir import InsertionPoint, MemRefType, Module, UnitAttr

from ...execution_engine import ExecutionEngine
from ...runtime import (
    UnrankedMemRefDescriptor,
    get_ranked_memref_descriptor,
    unranked_memref_to_numpy,
    get_unranked_memref_descriptor,
)

try:
    from ...wasm_execution_engine import WasmExecutionEngine

    HAS_WASM_EE = True
except ImportError:
    warnings.warn("Couldn't import WasmExecutionEngine; wasm runtime won't work")
    HAS_WASM_EE = False

from .. import types as T
from ...dialects.memref import cast
from ..runtime.passes import Pipeline, run_pipeline
from ..util import (
    memref_type_to_np_dtype,
    mlir_type_to_ctype,
    np_dtype_to_mlir_type,
)
from ..util import shlib_ext, find_ops, shlib_prefix

logger = logging.getLogger(__name__)

# adapted from https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir_e2e_test/linalg_on_tensors_backends/refbackend.py


CONSUME_RETURN_CALLBACK_ATTR = "refbackend_consume_return_callback"
refback_cb_attr = CONSUME_RETURN_CALLBACK_ATTR

_exec_engine_shared_libs = []


def _try_find_runtime_libraries(local_vars: dict):
    mlir_root_path = Path(_mlir_libs.__file__).parent
    lib_prefix_locations = {
        mlir_root_path,
        mlir_root_path.parent.parent.parent.parent / "lib",
    }
    libraries_to_find = {
        "async_runtime",
        "c_runner_utils",
        "runner_utils",
        "cuda_runtime",
    }
    # TODO(max): for some reason adding cuda runtime lib to execengine
    # causes a segfault (or something)

    def try_find_library(library: str):
        var_name = f"{library.upper()}_LIB_PATH"
        if env_var := os.getenv(var_name):
            local_vars[var_name] = Path(env_var)
            return Path(env_var)

        lib_name = f"{shlib_prefix()}mlir_{library}.{shlib_ext()}"
        for loc in lib_prefix_locations:
            if (loc / lib_name).exists():
                local_vars[var_name] = loc / lib_name
                return loc / lib_name

        logger.debug(
            f"Falling back on wheel path for {library} even though it was not found there"
        )
        local_vars[var_name] = mlir_root_path / lib_name

    for library in libraries_to_find:
        if lib_path := try_find_library(library):
            _exec_engine_shared_libs.append(lib_path)


_try_find_runtime_libraries(locals())


def get_ctype_func(mlir_ret_types):
    ctypes_arg = [None]
    legal_ret_types = []
    for type in mlir_ret_types:
        if ctype := mlir_type_to_ctype(type):
            ctypes_arg.append(ctype)
            legal_ret_types.append(type)
        elif memref_type_to_np_dtype(type):
            ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
            legal_ret_types.append(type)
        else:
            raise ValueError(f"Not supported type for callback return: {type=}")

    return ctypes.CFUNCTYPE(*ctypes_arg), legal_ret_types


# https://stackoverflow.com/a/68198336/9045206
CData = ctypes._SimpleCData.__mro__[-2]


def convert_arg_to_ctype(arg, unranked=True):
    if isinstance(arg, CData) or isinstance(arg, (int, float, bool)):
        return arg
    elif isinstance(arg, np.ndarray):
        assert np_dtype_to_mlir_type(
            arg.dtype.type
        ), f"unsupported numpy array type {arg.dtype}"
        if unranked:
            return ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(arg)))
        else:
            return ctypes.pointer(
                # TODO(max): sometimes these need to be unranked memref descriptors
                ctypes.pointer(get_ranked_memref_descriptor(arg))
            )
    raise ValueError(f"unsupported {arg=} for conversion to ctype")


def convert_returns_from_ctype(args, mlir_types):
    return tuple(
        (
            arg
            if mlir_type_to_ctype(type)
            else unranked_memref_to_numpy(arg, memref_type_to_np_dtype(type))
        )
        for arg, type in zip(args, mlir_types)
    )


_np_dtype_to_mlir_type_ctor = {
    np.int8: T.i8,
    np.int16: T.i16,
    np.int32: T.i32,
    # windows
    np.intc: T.i32,
    np.int64: T.i64,
    # is technically wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map to index type
    np.longlong: T.index,
    np.uintp: T.index,
    np.float16: T.f16,
    np.float32: T.f32,
    np.float64: T.f64,
}

_mlir_type_ctor_to_np_dtype = lambda: {
    v: k for k, v in _np_dtype_to_mlir_type_ctor.items()
}


def np_dtype_to_mlir_type(np_dtype):
    if typ := _np_dtype_to_mlir_type_ctor.get(np_dtype):
        return typ()


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype.get(mlir_type)


_mlir_type_to_ctype = {
    T.bool: ctypes.c_bool,
    T.i8: ctypes.c_byte,
    T.i64: ctypes.c_int,
    T.f32: ctypes.c_float,
    T.f64: ctypes.c_double,
}


def mlir_type_to_ctype(mlir_type):
    __mlir_type_to_ctype = {k(): v for k, v in _mlir_type_to_ctype.items()}
    return _mlir_type_to_ctype.get(mlir_type)
