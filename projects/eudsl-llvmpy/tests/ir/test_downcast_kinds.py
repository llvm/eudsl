#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Every bound Value subclass is reached and downcast to the right Python class.

The `type_hook` (Kinds.cpp `valueTypeInfo`) maps an llvm::Value* to its concrete
bound class. Most instruction/constant classes are registered with a bare
`nb::class_<...>` and no methods, so they add no coverage region -- neither line
nor function coverage forces a test to ever obtain one. A wrong entry in the
dispatch (or a missing registration) would go unnoticed. These tests construct
IR of each kind and assert the downcast produces the expected concrete class.
"""

from textwrap import dedent

import llvm
from llvm.testing import assert_no_leaks


def _instruction_kinds(mod):
    kinds = set()
    for f in mod.functions:
        for bb in f.basic_blocks:
            for inst in bb.instructions:
                kinds.add(type(inst).__name__)
    return kinds


# Each source contributes one or more instruction kinds not covered by the
# opcode sweep in test_cpp_coverage.py.
_SOURCES = {
    "casts": """\
        define void @f(i8 %a, float %b, double %c, ptr %p) {
          %1 = sext i8 %a to i32
          %2 = fptoui float %b to i32
          %3 = uitofp i32 %2 to float
          %4 = fptrunc double %c to float
          %5 = fpext float %b to double
          %6 = addrspacecast ptr %p to ptr addrspace(1)
          %7 = fneg float %b
          %8 = freeze i32 %2
          ret void
        }
    """,
    "vector_aggregate": """\
        define void @f(<4 x i32> %v, <4 x i32> %w, i32 %x, {i32, i32} %s) {
          %1 = extractelement <4 x i32> %v, i32 0
          %2 = insertelement <4 x i32> %v, i32 %x, i32 1
          %3 = shufflevector <4 x i32> %v, <4 x i32> %w, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
          %4 = extractvalue {i32, i32} %s, 0
          %5 = insertvalue {i32, i32} %s, i32 %x, 1
          ret void
        }
    """,
    "terminators": """\
        define void @f(i32 %x, ptr %p) {
        entry:
          switch i32 %x, label %d [i32 0, label %a]
        a:
          %v = va_arg ptr %p, i32
          indirectbr ptr %p, [label %d]
        d:
          unreachable
        }
    """,
    "callbr": """\
        define void @f() {
        entry:
          callbr void asm "", ""() to label %n []
        n:
          ret void
        }
    """,
    "atomics": """\
        define void @f(ptr %p) {
          %1 = atomicrmw add ptr %p, i32 1 seq_cst
          %2 = cmpxchg ptr %p, i32 0, i32 1 seq_cst seq_cst
          fence seq_cst
          ret void
        }
    """,
    "itanium_eh": """\
        declare void @g()
        declare i32 @__gxx_personality_v0(...)
        define void @f() personality ptr @__gxx_personality_v0 {
        entry:
          invoke void @g() to label %ok unwind label %lp
        ok:
          ret void
        lp:
          %e = landingpad {ptr, i32} cleanup
          resume {ptr, i32} %e
        }
    """,
    "windows_catch": """\
        declare void @g()
        declare i32 @__CxxFrameHandler3(...)
        define void @f() personality ptr @__CxxFrameHandler3 {
        entry:
          invoke void @g() to label %ok unwind label %cs
        ok:
          ret void
        cs:
          %c = catchswitch within none [label %h] unwind to caller
        h:
          %p = catchpad within %c [ptr null, i32 64, ptr null]
          catchret from %p to label %ok
        }
    """,
    "windows_cleanup": """\
        declare void @g()
        declare i32 @__CxxFrameHandler3(...)
        define void @f() personality ptr @__CxxFrameHandler3 {
        entry:
          invoke void @g() to label %ok unwind label %cp
        ok:
          ret void
        cp:
          %p = cleanuppad within none []
          cleanupret from %p unwind to caller
        }
    """,
}


def _leaf_instruction_classes():
    """Derive the set of leaf Instruction subclasses from the bindings."""
    instr = llvm.ir.Instruction
    all_sub = set()
    for name in dir(llvm.ir):
        obj = getattr(llvm.ir, name)
        if isinstance(obj, type) and issubclass(obj, instr) and obj is not instr:
            all_sub.add(obj)
    leaves = set()
    for cls in all_sub:
        is_parent = any(
            issubclass(other, cls) and other is not cls for other in all_sub
        )
        if not is_parent:
            leaves.add(cls.__name__)
    # BinaryOperator is not a leaf (FPBinaryOperator subclasses it) but the
    # type_hook returns it for integer binops, so include it.
    if hasattr(llvm.ir, "BinaryOperator"):
        leaves.add("BinaryOperator")
    return leaves


def test_instruction_kinds_downcast():
    expected = _leaf_instruction_classes()
    seen = set()
    with llvm.ir.Context() as ctx:
        for name, src in _SOURCES.items():
            mod = llvm.ir.parse_assembly(dedent(src), ctx, name)
            seen |= _instruction_kinds(mod)
            del mod
    # This test covers the "hard" kinds (EH, atomics, vector, etc.) that the
    # per-instruction test in test_cpp_coverage.py doesn't exercise. Together
    # the two files must cover every leaf. Anything missing here means a new
    # source snippet is needed.
    covered_by_opcode_sweep = {
        "BinaryOperator",
        "FPBinaryOperator",
        "TruncInst",
        "ZExtInst",
        "SIToFPInst",
        "FPToSIInst",
        "BitCastInst",
        "PtrToIntInst",
        "IntToPtrInst",
        "ICmpInst",
        "FCmpInst",
        "SelectInst",
        "AllocaInst",
        "StoreInst",
        "LoadInst",
        "GetElementPtrInst",
        "CallInst",
        "BranchInst",
        "PHINode",
        "ReturnInst",
    }
    missing = expected - seen - covered_by_opcode_sweep
    assert not missing, f"instruction kinds not downcast: {sorted(missing)}"
    assert_no_leaks()


_CONSTANTS_SRC = """\
    @g = global i32 0
    @parr = global [2 x ptr] [ptr @g, ptr null]
    @st   = global {i32, i32} {i32 1, i32 2}
    @darr = global [4 x i8] c"abcd"
    @zi   = global [4 x i32] zeroinitializer
    @pvec = global <2 x ptr> <ptr @g, ptr null>
    @dvec = global <4 x i32> <i32 1, i32 2, i32 3, i32 4>
    @ce   = global i64 ptrtoint (ptr @g to i64)
    @ba   = global ptr blockaddress(@f, %b)
    @al   = alias i32, ptr @g
    @res  = global ptr null
    @if   = ifunc i32 (), ptr @res

    declare void @noop()
    declare i32 @__CxxFrameHandler3(...)
    define i32 @f() personality ptr @__CxxFrameHandler3 {
    entry:
      br label %b
    b:
      %parr = load ptr, ptr @parr
      %st   = load i32, ptr @st
      %darr = load i8,  ptr @darr
      %zi   = load i32, ptr @zi
      %pvec = load <2 x ptr>, ptr @pvec
      %dvec = load <4 x i32>, ptr @dvec
      %ce   = load i64, ptr @ce
      %ba   = load ptr, ptr @ba
      %al   = load i32, ptr @al
      %if   = load ptr, ptr @if
      invoke void @noop() to label %ok unwind label %cp
    ok:
      ret i32 0
    cp:
      %pad = cleanuppad within none []
      cleanupret from %pad unwind to caller
    }
"""


def test_constant_and_global_kinds_downcast():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(dedent(_CONSTANTS_SRC), ctx, "m")
        f = mod.get_function("f")
        # Map each load's pointed-at global name -> (global class, init class).
        by_name = {}
        for bb in f.basic_blocks:
            for inst in bb.instructions:
                if type(inst).__name__ == "LoadInst":
                    g = inst.pointer_operand
                    init = getattr(g, "initializer", None)
                    by_name[g.name] = (
                        type(g).__name__,
                        type(init).__name__ if init is not None else None,
                    )
        # Global kinds obtained directly.
        assert by_name["al"][0] == "GlobalAlias"
        assert by_name["if"][0] == "GlobalIFunc"
        # Constant kinds obtained as initializers.
        inits = {v[1] for v in by_name.values()}
        expected = {
            "ConstantArray",
            "ConstantStruct",
            "ConstantDataArray",
            "ConstantAggregateZero",
            "ConstantVector",
            "ConstantDataVector",
            "ConstantExpr",
            "BlockAddress",
        }
        assert expected <= inits, f"missing: {sorted(expected - inits)}"
        # ConstantTokenNone is the `within none` operand of the cleanup pad.
        pad = next(
            i
            for bb in f.basic_blocks
            for i in bb.instructions
            if type(i).__name__ == "CleanupPadInst"
        )
        assert type(pad.operand(0)).__name__ == "ConstantTokenNone"
        del f, mod
    assert_no_leaks()
