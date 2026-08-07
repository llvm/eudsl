#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-style checked-downcast constructors: IntegerType(t), LoadInst(v), ...

Each concrete class accepts a base handle (Type or Value) and returns it
re-typed if the dynamic kind matches, else raises ValueError -- the parity
analogue of MLIR's `IntegerType(t)`.
"""
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    @g = global i32 0
    declare i32 @callee(i32)
    define i32 @f(i1 %c, ptr %p, i32 %x) {
    entry:
      %s = alloca i32
      store i32 7, ptr %s
      %v = load i32, ptr %s
      %e = getelementptr i32, ptr %p, i64 1
      %cmp = icmp eq i32 %x, 0
      %call = call i32 @callee(i32 %x)
      %gl = load i32, ptr @g
      br i1 %c, label %a, label %b
    a:
      br label %join
    b:
      br label %join
    join:
      %phi = phi i32 [ %v, %a ], [ %call, %b ]
      ret i32 %phi
    }
    """
)


def test_type_cast_constructors():
    with llvm.Context() as ctx:
        cases = [
            (llvm.types.IntegerType, llvm.types.i32(ctx)),
            (llvm.types.PointerType, llvm.types.ptr(context=ctx)),
            (llvm.types.StructType, llvm.types.struct([llvm.types.i32(ctx)], context=ctx)),
            (llvm.types.ArrayType, llvm.types.array(llvm.types.i32(ctx), 2)),
            (llvm.types.VectorType, llvm.types.vector(llvm.types.i32(ctx), 2)),
            (llvm.types.FunctionType, llvm.types.function(llvm.types.void(ctx), [])),
        ]
        for cls, ty in cases:
            narrowed = cls(ty)  # cast from the base Type handle
            assert isinstance(narrowed, cls)
            assert narrowed == ty
        # Mismatch raises ValueError (shared throw path for every cast ctor).
        with pytest.raises(ValueError, match="is not a"):
            llvm.types.IntegerType(llvm.types.ptr(context=ctx))
    assert_no_leaks()


def test_value_cast_constructors():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        entry = f.entry_block

        def by(cls):
            return next(i for i in entry.instructions if isinstance(i, cls))

        alloca = by(llvm.AllocaInst)
        store = by(llvm.StoreInst)
        gep = by(llvm.GetElementPtrInst)
        cmp = by(llvm.CmpInst)
        call = by(llvm.CallBase)
        load_v = next(
            i for i in entry.instructions if isinstance(i, llvm.LoadInst)
        )
        gvar = [
            i for i in entry.instructions if isinstance(i, llvm.LoadInst)
        ][-1].pointer_operand
        join = next(b for b in f.basic_blocks if b.name == "join")
        phi = next(i for i in join.instructions if isinstance(i, llvm.PHINode))
        ret = join.terminator
        a_block = next(b for b in f.basic_blocks if b.name == "a")

        # (concrete class, a value that IS one) -> cast round-trips to the class.
        cases = [
            (llvm.Function, f),
            (llvm.Argument, f.arg(0)),
            (llvm.BasicBlock, entry),
            (llvm.Instruction, alloca),
            (llvm.User, alloca),
            (llvm.AllocaInst, alloca),
            (llvm.StoreInst, store),
            (llvm.LoadInst, load_v),
            (llvm.GetElementPtrInst, gep),
            (llvm.CmpInst, cmp),
            (llvm.CallBase, call),
            (llvm.PHINode, phi),
            (llvm.ReturnInst, ret),
            (llvm.CondBrInst, entry.terminator),
            (llvm.UncondBrInst, a_block.terminator),
            (llvm.ConstantInt, llvm.const_int(llvm.types.i32(ctx), 1)),
            (llvm.ConstantFP, llvm.const_fp(llvm.types.f32(ctx), 1.0)),
            (llvm.GlobalVariable, gvar),
        ]
        for cls, val in cases:
            narrowed = cls(val)
            assert isinstance(narrowed, cls)
            assert narrowed == val
        # Mismatch raises ValueError.
        with pytest.raises(ValueError, match="is not a"):
            llvm.LoadInst(store)
        del f, entry, mod
    assert_no_leaks()
