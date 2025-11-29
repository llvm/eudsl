import ctypes
import random
import string


from ._mlir_libs import _mlirWasmExecutionEngine
from .ir import Module, StringAttr
from .runtime import np_to_memref as _np_to_memref


class WasmExecutionEngine:
    def __init__(
        self,
        module_op,
        opt_level=2,
        shared_libs=None,
        module_name=None,
        target_triple="wasm32-unknown-emscripten",
    ):
        self.shared_libs = shared_libs
        if self.shared_libs is None:
            self.shared_libs = []

        if isinstance(module_op, Module):
            module_op = module_op.operation
        if "llvm.target_triple" not in module_op.attributes:
            module_op.attributes["llvm.target_triple"] = StringAttr.get(target_triple)
        if module_name is None:
            # 8 is the max length?
            module_name = "".join(random.choices(string.ascii_uppercase, k=8)).lower()

        object_fn = _mlirWasmExecutionEngine.compile_module(
            module_op, module_name, opt_level
        )
        self.link_load_module(object_fn, module_name)

        for i, sh in enumerate(self.shared_libs):
            self.shared_libs[i] = ctypes.CDLL(sh, mode=ctypes.RTLD_GLOBAL)

    @staticmethod
    def link_load_module(object_fn, module_name):
        binary_fn = f"{module_name}.wasm"
        _mlirWasmExecutionEngine.link_load_module(object_fn, binary_fn)

    @staticmethod
    def lookup(name, return_type=None):
        if name == "main":
            raise ValueError("functions named `main` are not supported on wasm")
        func = _mlirWasmExecutionEngine.get_symbol_address(name)
        if not func:
            raise RuntimeError("Unknown function " + name)
        prototype = ctypes.CFUNCTYPE(return_type, ctypes.c_void_p)
        return prototype(func)

    def invoke(self, name, *ctypes_args):
        return self.invoke_with_return_type(name, ctypes_args)

    def invoke_with_return_type(self, name, ctypes_args, return_type=None):
        func = self.lookup(name, return_type)
        packed_args = (ctypes.c_void_p * len(ctypes_args))()
        for argNum in range(len(ctypes_args)):
            packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
        return func(packed_args)


# These are copy-pasta from np_to_memref but for 32b arch (wasm32)


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
    ctp = _np_to_memref.as_ctype(nparray.dtype)
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


class UnrankedMemRefDescriptor(ctypes.Structure):
    """Creates a ctype struct for memref descriptor"""

    _fields_ = [("rank", ctypes.c_long), ("descriptor", ctypes.c_void_p)]


def get_unranked_memref_descriptor(nparray):
    """Returns a generic/unranked memref descriptor for the given numpy array."""
    d = UnrankedMemRefDescriptor()
    d.rank = nparray.ndim
    x = get_ranked_memref_descriptor(nparray)
    d.descriptor = ctypes.cast(ctypes.pointer(x), ctypes.c_void_p)
    return d


_np_to_memref.make_nd_memref_descriptor = make_nd_memref_descriptor
_np_to_memref.make_zero_d_memref_descriptor = make_zero_d_memref_descriptor
_np_to_memref.get_ranked_memref_descriptor = get_ranked_memref_descriptor
_np_to_memref.UnrankedMemRefDescriptor = UnrankedMemRefDescriptor
_np_to_memref.get_unranked_memref_descriptor = get_unranked_memref_descriptor
