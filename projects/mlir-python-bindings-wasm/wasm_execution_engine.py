import ctypes
import gc
import sys
from textwrap import dedent

from mlir._mlir_libs import _mlirWasmExecutionEngine
from mlir.ir import *
from mlir.passmanager import *
from mlir.runtime import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


def run(f):
    log("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


def lookup(name, return_type=None):
    func = _mlirWasmExecutionEngine.get_symbol_address(name)
    if not func:
        raise RuntimeError("Unknown function " + name)
    prototype = ctypes.CFUNCTYPE(return_type, ctypes.c_void_p)
    return prototype(func)


def invoke(name, ctypes_args, return_type=None):
    func = lookup(name, return_type)
    packed_args = (ctypes.c_void_p * len(ctypes_args))()
    for argNum in range(len(ctypes_args)):
        packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
    return func(packed_args)


@run
def testapis():
    with Context():
        module = Module.parse(
            dedent(
                r"""
                llvm.func @none(%arg0: i32) -> i32 {
                  %0 = llvm.mlir.constant(333 : i32) : i32
                  %t0 = llvm.add %arg0, %0 : i32
                  llvm.return %t0 : i32
                }
                """
            )
        )
        name = _mlirWasmExecutionEngine.compile(module.operation, "foo")
        print(name)
        _mlirWasmExecutionEngine.link_load(name, "foo.wasm")
        print(_mlirWasmExecutionEngine.get_symbol_address("none"))
        func = _mlirWasmExecutionEngine.get_symbol_address("none")
        func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(func)
        assert func(20) == 353


def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(convert-complex-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


def make_nd_memref_descriptor(rank, dtype):
    class MemRefDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given rank/dtype, where rank>0."""

        _fields_ = [
            ("allocated", ctypes.POINTER(dtype)),
            ("aligned", ctypes.POINTER(dtype)),
            ("offset", ctypes.c_longlong),
            ("shape", ctypes.c_long * rank),
            ("strides", ctypes.c_longlong * rank),
        ]

    return MemRefDescriptor


def make_zero_d_memref_descriptor(dtype):
    class MemRefDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given dtype, where rank=0."""

        _fields_ = [
            ("allocated", ctypes.POINTER(dtype)),
            ("aligned", ctypes.POINTER(dtype)),
            ("offset", ctypes.c_longlong),
        ]

    return MemRefDescriptor


def get_ranked_memref_descriptor(nparray):
    """Returns a ranked memref descriptor for the given numpy array."""
    ctp = as_ctype(nparray.dtype)
    if nparray.ndim == 0:
        x = make_zero_d_memref_descriptor(ctp)()
        # x.allocated = nparray.ctypes.data
        x.allocated = nparray.ctypes.data_as(ctypes.POINTER(ctp))
        x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
        x.offset = ctypes.c_longlong(0)
        return x

    x = make_nd_memref_descriptor(nparray.ndim, ctp)()
    # x.allocated = nparray.ctypes.data
    x.allocated = nparray.ctypes.data_as(ctypes.POINTER(ctp))
    x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
    x.offset = ctypes.c_longlong(0)
    x.shape = nparray.ctypes.shape

    # Numpy uses byte quantities to express strides, MLIR OTOH uses the
    # torch abstraction which specifies strides in terms of elements.
    strides_ctype_t = ctypes.c_longlong * nparray.ndim
    x.strides = strides_ctype_t(*[x // nparray.itemsize for x in nparray.strides])
    return x


@run
def testMemrefAdd():
    with Context():
        module = Module.parse(
            dedent(
                """
                module  {
                  func.func @main(%arg0: memref<1xf32>, %arg1: memref<f32>, %arg2: memref<1xf32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %0 = arith.constant 0 : index
                    %1 = memref.load %arg0[%0] : memref<1xf32>
                    %2 = memref.load %arg1[] : memref<f32>
                    %3 = arith.addf %1, %2 : f32
                    return %1 : f32
                    // memref.store %3, %arg2[%0] : memref<1xf32>
                  }
                  func.func @main2(%arg0: memref<f32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %1 = memref.load %arg0[] : memref<f32>
                    return %1 : f32
                  }
                } """
            )
        )
        arg1 = np.array([32.5]).astype(np.float32)
        arg2 = np.array(6).astype(np.float32)
        res = np.array([0]).astype(np.float32)

        arg1_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg1))
        )
        arg2_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(arg2))
        )
        res_memref_ptr = ctypes.pointer(
            ctypes.pointer(get_ranked_memref_descriptor(res))
        )

        module = lowerToLLVM(module)
        print(module)
        name = _mlirWasmExecutionEngine.compile(module.operation, "bar")
        _mlirWasmExecutionEngine.link_load(name, "bar.wasm")
        print(_mlirWasmExecutionEngine.get_symbol_address("main"))

        ctp = as_ctype(arg2.dtype)
        func = _mlirWasmExecutionEngine.get_symbol_address("main2")
        func = ctypes.CFUNCTYPE(
            ctypes.c_float,
            ctypes.POINTER(ctp),
            ctypes.POINTER(ctp),
            ctypes.c_longlong,
        )(func)
        res_ = func(
            arg2.ctypes.data_as(ctypes.POINTER(ctp)),
            arg2.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
        )
        print(res_)

        res_ = invoke(
            "_mlir_ciface_main",
            [arg1_memref_ptr, arg2_memref_ptr, res_memref_ptr],
            return_type=ctypes.c_float,
        )
        print(res_)
        # CHECK: [32.5] + 6.0 = [38.5]
        print("{0} + {1} = {2}".format(arg1, arg2, res))
