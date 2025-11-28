import gc
import sys
from textwrap import dedent
import ctypes

from mlir.wasm_execution_engine import (
    _mlirWasmExecutionEngine,
    WasmExecutionEngine,
    get_ranked_memref_descriptor,
    as_ctype
)
from mlir.ir import *
from mlir.passmanager import *


def log(*args):
    print(*args, file=sys.stderr)
    sys.stderr.flush()


def run(f):
    log("\nTEST:", f.__name__)
    f()
    gc.collect()
    assert Context._get_live_count() == 0


wasm_ee = WasmExecutionEngine()


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
        wasm_ee.load_module(module.operation)
        print(_mlirWasmExecutionEngine.get_symbol_address("none"))
        func = _mlirWasmExecutionEngine.get_symbol_address("none")
        func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(func)
        assert func(20) == 353


def lowerToLLVM(module):
    pm = PassManager.parse(
        "builtin.module(convert-complex-to-llvm,finalize-memref-to-llvm{index-bitwidth=32},convert-func-to-llvm{index-bitwidth=32},convert-arith-to-llvm{index-bitwidth=32},convert-cf-to-llvm{index-bitwidth=32},reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


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
                    memref.store %3, %arg2[%0] : memref<1xf32>
                    return %3 : f32
                  }
                  func.func @main2(%arg0: memref<f32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %1 = memref.load %arg0[] : memref<f32>
                    return %1 : f32
                  }
                  func.func @main3(%arg0: memref<1xf32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %0 = arith.constant 0 : index
                    %1 = memref.load %arg0[%0] : memref<1xf32>
                    return %1 : f32
                  }
                  func.func @main4(%arg0: memref<1xf32>, %arg1: memref<1xf32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %0 = arith.constant 0 : index
                    %1 = memref.load %arg0[%0] : memref<1xf32>
                    return %1 : f32
                  }
                  func.func @main5(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: memref<f32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %0 = arith.constant 0 : index
                    %1 = memref.load %arg0[%0] : memref<1xf32>
                    return %1 : f32
                  }
                  func.func @main6(%arg0: memref<1xf32>, %arg2: memref<1xf32>, %arg1: memref<f32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %0 = arith.constant 0 : index
                    %1 = memref.load %arg0[%0] : memref<1xf32>
                    %2 = memref.load %arg1[] : memref<f32>
                    %3 = arith.addf %1, %2 : f32
                    memref.store %3, %arg2[%0] : memref<1xf32>
                    return %3 : f32
                  }
                  func.func @main7(%arg0: memref<1xf32>, %arg1: memref<f32>, %arg2: memref<1xf32>) -> (f32) attributes { llvm.emit_c_interface } {
                    %0 = arith.constant 0 : index
                    %1 = memref.load %arg0[%0] : memref<1xf32>
                    %2 = memref.load %arg1[] : memref<f32>
                    %3 = arith.addf %1, %2 : f32
                    memref.store %3, %arg2[%0] : memref<1xf32>
                    return %3 : f32
                  }
                } """
            )
        )

        module = lowerToLLVM(module)

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

        # print(module)
        name = _mlirWasmExecutionEngine.compile(module.operation, "bar")
        _mlirWasmExecutionEngine.link_load(name, "bar.wasm")
        print(_mlirWasmExecutionEngine.get_symbol_address("main"))

        res_ = wasm_ee.invoke(
            "_mlir_ciface_main",
            [arg1_memref_ptr, arg2_memref_ptr, res_memref_ptr],
            return_type=ctypes.c_float,
        )
        print(res_)
        # CHECK: [32.5] + 6.0 = [38.5]
        print("{0} + {1} = {2}".format(arg1, arg2, res))

        ctp = as_ctype(arg2.dtype)
        func = _mlirWasmExecutionEngine.get_symbol_address("main2")
        func = ctypes.CFUNCTYPE(
            ctypes.c_float,
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
        )(func)
        res_ = func(
            arg2.ctypes.data,
            arg2.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
        )
        print(res_)

        ctp = as_ctype(arg2.dtype)
        func = _mlirWasmExecutionEngine.get_symbol_address("main3")
        func = ctypes.CFUNCTYPE(
            ctypes.c_float,
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
        )(func)
        res_ = func(
            arg1.ctypes.data,
            arg1.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
        )
        print(res_)

        size_of_void_p = ctypes.sizeof(ctypes.c_void_p)
        print(f"The size of ctypes.c_void_p is: {size_of_void_p} bytes")

        size_of_longlong = ctypes.sizeof(ctypes.c_longlong)
        print(f"The size of ctypes.c_longlong is: {size_of_longlong} bytes")

        func = _mlirWasmExecutionEngine.get_symbol_address("main4")
        func = ctypes.CFUNCTYPE(
            ctypes.c_float,
            # arg1
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            # # arg2
            # ctypes.c_long,
            # ctypes.POINTER(ctp),
            # ctypes.c_long,
            # res
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
        )(func)
        res_ = func(
            # arg1
            arg1.ctypes.data,
            arg1.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
            # # # arg2
            # arg2.ctypes.data,
            # arg2.ctypes.data_as(ctypes.POINTER(ctp)),
            # 0,
            # res
            res.ctypes.data,
            res.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
        )
        print(res_)

        func = _mlirWasmExecutionEngine.get_symbol_address("main5")
        func = ctypes.CFUNCTYPE(
            ctypes.c_float,
            # arg1
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            # res
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            # arg2
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
        )(func)
        res_ = func(
            # arg1
            arg1.ctypes.data,
            arg1.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
            # res
            res.ctypes.data,
            res.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
            # arg2
            arg2.ctypes.data,
            arg2.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
        )
        print(res_)

        func = _mlirWasmExecutionEngine.get_symbol_address("main6")
        func = ctypes.CFUNCTYPE(
            ctypes.c_float,
            # arg1
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            # res
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            # arg2
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
        )(func)
        res_ = func(
            # arg1
            arg1.ctypes.data,
            arg1.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
            # res
            res.ctypes.data,
            res.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
            # arg2
            arg2.ctypes.data,
            arg2.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
        )
        print(res_)

        func = _mlirWasmExecutionEngine.get_symbol_address("main7")
        func = ctypes.CFUNCTYPE(
            ctypes.c_float,
            # arg1
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            # arg2
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            # res
            ctypes.c_long,
            ctypes.POINTER(ctp),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
        )(func)
        res_ = func(
            # arg1
            arg1.ctypes.data,
            arg1.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
            # arg2
            arg2.ctypes.data,
            arg2.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            # res
            res.ctypes.data,
            res.ctypes.data_as(ctypes.POINTER(ctp)),
            0,
            1,
            1,
        )
        print(res_)

        wasm_ee.invoke(
            "_mlir_ciface_main7",
            [arg1_memref_ptr, arg2_memref_ptr, res_memref_ptr],
            return_type=ctypes.c_float,
        )
        # CHECK: [32.5] + 6.0 = [38.5]
        print("{0} + {1} = {2}".format(arg1, arg2, res))
