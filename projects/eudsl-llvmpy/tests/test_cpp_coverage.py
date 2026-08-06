#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Exercises C++ binding paths (llvm-cov) not hit by the behavioral tests:
identity/hash dunders, operand/metadata accessors, and error paths."""
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @f(i32 %x, i32 %y) {
    entry:
      %s = add i32 %x, %y
      ret i32 %s
    }
    """
)


def test_value_eq_hash_and_operands():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        add = f.arg(0).users[0]
        # __eq__ against a non-Value returns False (try_cast fails).
        assert (f.arg(0) == "not a value") is False
        assert (f.arg(0) != "not a value") is True
        # __hash__: usable in a set/dict.
        assert len({f.arg(0), f.arg(1), f.arg(0)}) == 2
        # User.operands materializes the operand list.
        ops = add.operands
        assert len(ops) == 2
        assert ops[0] == f.arg(0)
        del f, add, ops, mod
    assert_no_leaks()


def test_type_eq_and_anonymous_struct_name():
    with llvm.Context() as ctx:
        i32 = llvm.types.i32(ctx)
        assert (i32 == "not a type") is False
        # A literal (anonymous) struct has no name.
        lit = llvm.types.struct(ctx, [i32, i32])
        assert lit.name is None
        # A named struct reports its name.
        named = llvm.types.named_struct(ctx, "S")
        assert named.name == "S"
    assert_no_leaks()


def test_mdnode_operand_accessor():
    with llvm.Context() as ctx:
        s = llvm.md_string(ctx, "x")
        node = llvm.md_node(ctx, [s])
        assert node.num_operands == 1
        assert node.operand(0).string == "x"
    assert_no_leaks()


def test_basic_block_create_static_and_empty_entry_block():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        fn = llvm.Function.create(
            llvm.types.function(llvm.types.void(ctx), []), "f", mod
        )
        # A declaration (no blocks) has no entry block.
        assert fn.entry_block is None
        # BasicBlock.create with an explicit parent.
        bb = llvm.BasicBlock.create(ctx, "entry", fn)
        assert bb.name == "entry"
        assert fn.entry_block == bb
        del fn, bb, mod
    assert_no_leaks()


def test_use_of_released_context_raises():
    ctx = llvm.Context()
    ctx.__exit__(None, None, None)  # release the underlying LLVMContext
    with pytest.raises(RuntimeError, match="released"):
        llvm.types.i32(ctx)
    del ctx


def test_target_machine_bad_triple_raises():
    with pytest.raises(RuntimeError, match="No available targets"):
        llvm.TargetMachine("nonsense-not-a-triple")


def test_linker_conflicting_symbols_raises():
    with llvm.Context() as ctx:
        a = llvm.parse_assembly(
            "define i32 @dup() {\n ret i32 1\n}\n", ctx, "a"
        )
        b = llvm.parse_assembly(
            "define i32 @dup() {\n ret i32 2\n}\n", ctx, "b"
        )
        with pytest.raises(RuntimeError, match="linkModules failed"):
            llvm.link_into(a, b)
        del a, b
    assert_no_leaks()


def test_builder_fcmp_gep_call():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        f32 = llvm.types.f32(ctx)
        i32 = llvm.types.i32(ctx)
        callee = llvm.Function.create(llvm.types.function(i32, [i32]), "callee", mod)
        fn = llvm.Function.create(
            llvm.types.function(llvm.types.i1(ctx), [f32, f32, llvm.types.ptr(ctx), i32]),
            "f",
            mod,
        )
        bb = fn.append_basic_block("entry")
        b = llvm.IRBuilder(ctx)
        with b.at_end_of(bb):
            b.fcmp(llvm.FCmpPredicate.OLT, fn.arg(0), fn.arg(1), "lt")
            # Runtime operand so the icmp isn't constant-folded away.
            b.icmp(llvm.ICmpPredicate.EQ, fn.arg(3), b.i32_const(2), "e")
            b.gep(i32, fn.arg(2), [b.i64_const(1)], "g")
            b.call(callee, [llvm.const_int(i32, 3)], "c")
            b.ret(b.fcmp(llvm.FCmpPredicate.OEQ, fn.arg(0), fn.arg(1)))
        printed = str(mod)
        assert "fcmp olt" in printed and "getelementptr" in printed
        assert "icmp eq" in printed
        assert "call i32 @callee" in printed
        del b, bb, fn, callee, mod
    assert_no_leaks()


def test_constant_int_zext_and_global_initializer():
    src = dedent(
        """\
        @g = global i32 5
        @e = external global i32
        define i32 @f() {
        entry:
          %a = load i32, ptr @g
          %b = load i32, ptr @e
          ret i32 %a
        }
        """
    )
    with llvm.Context() as ctx:
        assert llvm.const_int(llvm.types.i32(ctx), 7).zext_value == 7
        mod = llvm.parse_assembly(src, ctx, "m")
        loads = [
            i
            for i in mod.get_function("f").basic_blocks[0].instructions
            if type(i).__name__ == "LoadInst"
        ]
        # The load pointer operands are the GlobalVariables @g and @e.
        g = loads[0].pointer_operand
        e = loads[1].pointer_operand
        assert type(g).__name__ == "GlobalVariable"
        assert g.initializer.value == 5  # @g's initializer is the constant i32 5
        assert e.initializer is None  # @e is external, no initializer
        del loads, g, e, mod
    assert_no_leaks()



def test_verify_rejects_malformed_module():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        i32 = llvm.types.i32(ctx)
        fn = llvm.Function.create(llvm.types.function(i32, []), "bad", mod)
        # A basic block with no terminator is invalid IR.
        fn.append_basic_block("entry")
        with pytest.raises(llvm.VerifyError):
            mod.verify()
        del fn, mod
    assert_no_leaks()


def test_parse_bitcode_garbage_raises():
    with llvm.Context() as ctx:
        with pytest.raises(llvm.ParseError):
            llvm.parse_bitcode(b"not real bitcode", ctx)
    assert_no_leaks()


def test_callinst_arg_operand_and_gep_source_type():
    src = dedent(
        """\
        declare i32 @g(i32)
        define i32 @f(ptr %p) {
        entry:
          %e = getelementptr i32, ptr %p, i64 2
          %c = call i32 @g(i32 7)
          ret i32 %c
        }
        """
    )
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(src, ctx, "m")
        f = mod.get_function("f")
        insts = f.basic_blocks[0].instructions
        gep = next(i for i in insts if type(i).__name__ == "GetElementPtrInst")
        call = next(i for i in insts if type(i).__name__ == "CallInst")
        assert str(gep.source_element_type) == "i32"
        assert call.num_args == 1
        assert str(call.arg_operand(0)) == "i32 7"
        del f, insts, gep, call, mod
    assert_no_leaks()


def test_jit_lookup_missing_symbol_raises():
    ctx = llvm.Context()
    mod = llvm.parse_assembly(
        "define i32 @present() {\n ret i32 0\n}\n", ctx, "m"
    )
    jit = llvm.LLJIT()
    jit.add_module(mod)
    with pytest.raises(RuntimeError, match="Symbols not found"):
        jit.lookup("absent")
    del jit, mod, ctx


def test_module_context_accessor():
    with llvm.Context() as ctx:
        mod = llvm.Module("m", ctx)
        assert mod.context is ctx
        del mod
    assert_no_leaks()


def test_valuetypeinfo_downcasts_many_opcodes_and_kinds():
    src = dedent(
        """\
        @g = global i32 0
        define i32 @f(i32 %x, ptr %p, float %fp) {
        entry:
          %add = add i32 %x, 1
          %sub = sub i32 %add, 1
          %mul = mul i32 %sub, 2
          %fadd = fadd float %fp, 1.0
          %and = and i32 %x, 3
          %shl = shl i32 %x, 1
          %tr = trunc i32 %x to i16
          %ze = zext i16 %tr to i32
          %si = sitofp i32 %x to float
          %fp2i = fptosi float %fp to i32
          %bc = bitcast i32 %x to float
          %pti = ptrtoint ptr %p to i64
          %itp = inttoptr i64 %pti to ptr
          %cmp = icmp slt i32 %x, 10
          %fcmp = fcmp olt float %fp, 1.0
          %sel = select i1 %cmp, i32 %x, i32 %add
          %al = alloca i32
          store i32 %x, ptr %al
          %ld = load i32, ptr %al
          %ge = getelementptr i32, ptr %p, i64 1
          %ca = call i32 @f(i32 %x, ptr %p, float %fp)
          br label %next
        next:
          %ph = phi i32 [ %add, %entry ]
          ret i32 %ph
        }
        """
    )
    expected_per_name = {
        "add": "BinaryOperator",
        "sub": "BinaryOperator",
        "mul": "BinaryOperator",
        "and": "BinaryOperator",
        "shl": "BinaryOperator",
        "fadd": "FPBinaryOperator",
        "tr": "TruncInst",
        "ze": "ZExtInst",
        "si": "SIToFPInst",
        "fp2i": "FPToSIInst",
        "bc": "BitCastInst",
        "pti": "PtrToIntInst",
        "itp": "IntToPtrInst",
        "cmp": "ICmpInst",
        "fcmp": "FCmpInst",
        "sel": "SelectInst",
        "al": "AllocaInst",
        "ld": "LoadInst",
        "ge": "GetElementPtrInst",
        "ca": "CallInst",
        "ph": "PHINode",
    }
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(src, ctx, "m")
        f = mod.get_function("f")
        named_insts = {}
        for bb in f.basic_blocks:
            for inst in bb.instructions:
                if inst.name:
                    named_insts[inst.name] = type(inst).__name__
            terminator = bb.terminator
            if type(terminator).__name__ == "UncondBrInst":
                named_insts["__br__"] = "UncondBrInst"
            elif type(terminator).__name__ == "ReturnInst":
                named_insts["__ret__"] = "ReturnInst"
        for name, expected_cls in expected_per_name.items():
            actual = named_insts.get(name)
            assert actual == expected_cls, (
                f"%{name}: expected {expected_cls}, got {actual}"
            )
        assert named_insts.get("__br__") == "UncondBrInst"
        assert named_insts.get("__ret__") == "ReturnInst"
        assert type(f).__name__ == "Function"
        assert type(f.arg(0)).__name__ == "Argument"
        # StoreInst has no name; check it separately.
        store_insts = [
            inst for bb in f.basic_blocks for inst in bb.instructions
            if type(inst).__name__ == "StoreInst"
        ]
        assert len(store_insts) == 1
        del f, mod
    assert_no_leaks()


def test_valuetypeinfo_downcasts_constant_kinds():
    with llvm.Context() as ctx:
        i32 = llvm.types.i32(ctx)
        assert type(llvm.const_int(i32, 1)).__name__ == "ConstantInt"
        assert type(llvm.const_fp(llvm.types.f32(ctx), 1.0)).__name__ == "ConstantFP"
        assert type(llvm.undef(i32)).__name__ == "UndefValue"
        assert type(llvm.poison(i32)).__name__ == "PoisonValue"
        assert type(llvm.null(llvm.types.ptr(ctx))).__name__ == "ConstantPointerNull"
        assert type(llvm.null(i32)).__name__ == "ConstantInt"  # zero int
    assert_no_leaks()
