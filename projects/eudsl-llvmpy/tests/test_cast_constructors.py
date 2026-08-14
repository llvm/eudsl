#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""MLIR-style checked-downcast constructors: IntegerType(t), LoadInst(v), ...

Each concrete class accepts a base handle (Type or Value) and returns it
re-typed if the dynamic kind matches. A wrong kind within the same base
hierarchy -- and None -- raise ValueError (the shared throw path). An argument
from the other hierarchy entirely (a Value into a Type ctor, or vice versa) is
rejected by nanobind as a TypeError before the cast body runs. This is the
parity analogue of MLIR's `IntegerType(t)`.
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
        # (concrete class, a value that IS one, a probe reading a subclass-only
        # accessor on the narrowed handle -> proves the cast yields a usable
        # concrete object, not just an object that compares equal).
        cases = [
            (llvm.types.IntegerType, llvm.types.i32(ctx), lambda t: t.bit_width == 32),
            (llvm.types.PointerType, llvm.types.ptr(ctx), lambda t: t.address_space == 0),
            (
                llvm.types.StructType,
                llvm.types.struct(ctx, [llvm.types.i32(ctx)]),
                lambda t: t.num_elements == 1,
            ),
            (
                llvm.types.ArrayType,
                llvm.types.array(llvm.types.i32(ctx), 2),
                lambda t: t.num_elements == 2,
            ),
            (
                llvm.types.VectorType,
                llvm.types.vector(llvm.types.i32(ctx), 2),
                lambda t: t.min_num_elements == 2,
            ),
            (
                llvm.types.FunctionType,
                llvm.types.function(llvm.types.void(ctx), []),
                lambda t: t.num_params == 0,
            ),
        ]
        for cls, ty, probe in cases:
            narrowed = cls(ty)  # cast from the base Type handle
            assert isinstance(narrowed, cls)
            assert narrowed == ty
            assert probe(narrowed)
        # Wrong kind within the Type hierarchy raises ValueError (shared throw
        # path for every cast ctor). A no-op `return v` ctor would NOT raise.
        with pytest.raises(ValueError, match="is not a"):
            llvm.types.IntegerType(llvm.types.ptr(ctx))
        with pytest.raises(ValueError, match="is not a"):
            llvm.types.VectorType(llvm.types.array(llvm.types.i32(ctx), 2))
        # None reaches the dyn_cast_or_null null branch -> ValueError, not a
        # TypeError that would slip past callers catching ValueError.
        with pytest.raises(ValueError, match="is not a"):
            llvm.types.IntegerType(None)
        # A Value handed to a Type ctor is the other hierarchy entirely:
        # nanobind rejects it as a TypeError before the cast body runs.
        with pytest.raises(TypeError):
            llvm.types.IntegerType(llvm.const_int(llvm.types.i32(ctx), 1))
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

        # (concrete class, a value that IS one, a probe reading a subclass-only
        # accessor on the narrowed handle).
        cases = [
            (llvm.Function, f, lambda v: v.num_args == 3),
            (llvm.Argument, f.arg(0), lambda v: v.arg_no == 0),
            (llvm.BasicBlock, entry, lambda v: v.name == "entry"),
            (llvm.Instruction, alloca, lambda v: v.is_terminator is False),
            (llvm.User, alloca, lambda v: isinstance(v.num_operands, int)),
            (llvm.AllocaInst, alloca, lambda v: str(v.allocated_type) == "i32"),
            (llvm.StoreInst, store, lambda v: v.pointer_operand is not None),
            (llvm.LoadInst, load_v, lambda v: v.pointer_operand is not None),
            (
                llvm.GetElementPtrInst,
                gep,
                lambda v: str(v.source_element_type) == "i32",
            ),
            (llvm.CmpInst, cmp, lambda v: v.predicate is not None),
            (llvm.CallBase, call, lambda v: v.num_args == 1),
            (llvm.PHINode, phi, lambda v: v.num_incoming == 2),
            (llvm.ReturnInst, ret, lambda v: v.return_value is not None),
            (llvm.CondBrInst, entry.terminator, lambda v: v.is_conditional is True),
            (
                llvm.UncondBrInst,
                a_block.terminator,
                lambda v: v.is_conditional is False,
            ),
            (
                llvm.ConstantInt,
                llvm.const_int(llvm.types.i32(ctx), 1),
                lambda v: v.value == 1,
            ),
            (
                llvm.ConstantFP,
                llvm.const_fp(llvm.types.f32(ctx), 1.0),
                lambda v: v.double_value == 1.0,
            ),
            (llvm.GlobalVariable, gvar, lambda v: v.is_constant is False),
        ]
        for cls, val, probe in cases:
            narrowed = cls(val)
            assert isinstance(narrowed, cls)
            assert narrowed == val
            assert probe(narrowed)
        # Wrong kind within the Value hierarchy raises ValueError (siblings, not
        # obviously-unrelated). A no-op `return v` ctor would NOT raise.
        with pytest.raises(ValueError, match="is not a"):
            llvm.LoadInst(store)
        with pytest.raises(ValueError, match="is not a"):
            llvm.StoreInst(load_v)
        with pytest.raises(ValueError, match="is not a"):
            llvm.PHINode(ret)
        # None reaches the dyn_cast_or_null null branch -> ValueError.
        with pytest.raises(ValueError, match="is not a"):
            llvm.LoadInst(None)
        # A Type handed to a Value ctor is the other hierarchy: nanobind rejects
        # it as a TypeError before the cast body runs.
        with pytest.raises(TypeError):
            llvm.LoadInst(llvm.types.i32(ctx))
        del f, entry, mod
    assert_no_leaks()
