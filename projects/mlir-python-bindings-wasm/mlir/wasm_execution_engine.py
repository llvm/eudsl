import ctypes

from ._mlir_libs import _mlirWasmExecutionEngine
from .ir import Module
from .runtime.np_to_memref import as_ctype


class WasmExecutionEngine:
    def lookup(self, name, return_type=None):
        func = _mlirWasmExecutionEngine.get_symbol_address(name)
        if not func:
            raise RuntimeError("Unknown function " + name)
        prototype = ctypes.CFUNCTYPE(return_type, ctypes.c_void_p)
        return prototype(func)

    def invoke(self, name, ctypes_args, return_type=None):
        func = self.lookup(name, return_type)
        packed_args = (ctypes.c_void_p * len(ctypes_args))()
        for argNum in range(len(ctypes_args)):
            packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
        return func(packed_args)

    def load_module(self, module_op, module_name="foo"):
        if isinstance(module_op, Module):
            module_op = module_op.operation
        object_fn = _mlirWasmExecutionEngine.compile(module_op, module_name)
        binary_fn = f"{module_name}.wasm"
        _mlirWasmExecutionEngine.link_load(object_fn, binary_fn)


def make_nd_memref_descriptor(rank, dtype):
    class MemRefDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given rank/dtype, where rank>0."""

        _fields_ = [
            ("allocated", ctypes.c_long),
            ("aligned", ctypes.POINTER(dtype)),
            ("offset", ctypes.c_long),
            ("shape", ctypes.c_long * rank),
            ("strides", ctypes.c_long * rank),
        ]

    return MemRefDescriptor


def make_zero_d_memref_descriptor(dtype):
    class MemRefDescriptor(ctypes.Structure):
        """Builds an empty descriptor for the given dtype, where rank=0."""

        _fields_ = [
            ("allocated", ctypes.c_long),
            ("aligned", ctypes.POINTER(dtype)),
            ("offset", ctypes.c_long),
        ]

    return MemRefDescriptor


def get_ranked_memref_descriptor(nparray):
    """Returns a ranked memref descriptor for the given numpy array."""
    ctp = as_ctype(nparray.dtype)
    if nparray.ndim == 0:
        x = make_zero_d_memref_descriptor(ctp)()
        x.allocated = nparray.ctypes.data
        x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
        x.offset = ctypes.c_long(0)
        return x

    x = make_nd_memref_descriptor(nparray.ndim, ctp)()
    x.allocated = nparray.ctypes.data
    x.aligned = nparray.ctypes.data_as(ctypes.POINTER(ctp))
    x.offset = ctypes.c_long(0)
    x.shape = nparray.ctypes.shape

    # Numpy uses byte quantities to express strides, MLIR OTOH uses the
    # torch abstraction which specifies strides in terms of elements.
    strides_ctype_t = ctypes.c_long * nparray.ndim
    x.strides = strides_ctype_t(*[x // nparray.itemsize for x in nparray.strides])
    return x
