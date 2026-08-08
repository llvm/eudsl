#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import pytest

import llvm
from llvm.testing import assert_no_leaks

_PHI_SRC = dedent(
    """\
    define i32 @f(i1 %c) {
    entry:
      br i1 %c, label %a, label %b
    a:
      br label %join
    b:
      br label %join
    join:
      %p = phi i32 [ 1, %a ], [ 2, %b ]
      %eq = icmp eq i32 %p, 1
      ret i32 %p
    }
    """
)

# Exercises the memory, call, cast and fcmp accessors that _PHI_SRC does not
# reach. Several results (%gep, %fc) are intentionally unused; the parser keeps
# them and that is all these accessor tests need.
_MEM_SRC = dedent(
    """\
    declare i32 @g(i32, i32)

    define i32 @f(i32 %x, ptr %p) {
    entry:
      %slot = alloca i32
      store i32 %x, ptr %slot
      %ld = load i32, ptr %slot
      %gep = getelementptr i32, ptr %p, i32 2
      %cl = call i32 @g(i32 %x, i32 %ld)
      %fc = fcmp ogt float 1.0, 2.0
      ret i32 %cl
    }
    """
)


def _insts_by_class(mod, name):
    out = []
    for f in mod.functions:
        for bb in f.basic_blocks:
            for i in bb.instructions:
                if type(i).__name__ == name:
                    out.append(i)
    return out


def _only(mod, name):
    (inst,) = _insts_by_class(mod, name)
    return inst


def test_instructions_downcast():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_PHI_SRC, ctx, "m")
        # add-style ops become BinaryOperator; the icmp becomes ICmpInst; the
        # phi becomes PHINode; the conditional branch becomes CondBrInst.
        assert len(_insts_by_class(mod, "PHINode")) == 1
        assert len(_insts_by_class(mod, "ICmpInst")) == 1
        assert len(_insts_by_class(mod, "CondBrInst")) == 1
        del mod
    assert_no_leaks()


def test_phi_incoming():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_PHI_SRC, ctx, "m")
        (phi,) = _insts_by_class(mod, "PHINode")
        assert phi.num_incoming == 2
        assert phi.incoming_block(0).name == "a"
        assert phi.incoming_block(1).name == "b"
        assert str(phi.incoming_value(0)) == "i32 1"
        assert str(phi.incoming_value(1)) == "i32 2"
        del phi, mod
    assert_no_leaks()


def test_phi_add_incoming():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_PHI_SRC, ctx, "m")
        phi = _only(mod, "PHINode")
        entry = mod.get_function("f").entry_block
        # add_incoming(value, block) is the write path. Feed back an existing
        # i32 incoming value (the phi's own operand 0) so the added type matches
        # the phi type; a debug LLVM would assert on a mismatch.
        v = phi.incoming_value(0)
        phi.add_incoming(v, entry)
        assert phi.num_incoming == 3
        assert phi.incoming_value(2) == v
        assert phi.incoming_block(2) == entry
        # Out-of-range indices raise rather than crash.
        with pytest.raises(IndexError):
            phi.incoming_value(99)
        with pytest.raises(IndexError):
            phi.incoming_block(99)
        del phi, entry, v, mod
    assert_no_leaks()


def test_icmp_predicate():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_PHI_SRC, ctx, "m")
        (icmp,) = _insts_by_class(mod, "ICmpInst")
        assert icmp.predicate == llvm.ir.ICmpPredicate.EQ
        del icmp, mod
    assert_no_leaks()


def test_conditional_branch():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_PHI_SRC, ctx, "m")
        cbrs = _insts_by_class(mod, "CondBrInst")
        assert len(cbrs) == 1
        assert cbrs[0].is_conditional
        assert cbrs[0].num_successors == 2
        # The condition is the i1 argument %c.
        assert cbrs[0].condition.name == "c"
        del cbrs, mod
    assert_no_leaks()


def test_unconditional_branch():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_PHI_SRC, ctx, "m")
        # The a->join and b->join branches are unconditional.
        ubrs = _insts_by_class(mod, "UncondBrInst")
        assert len(ubrs) == 2
        assert ubrs[0].is_conditional is False
        assert ubrs[0].num_successors == 1
        del ubrs, mod
    assert_no_leaks()


def test_return_value():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_PHI_SRC, ctx, "m")
        ret = _only(mod, "ReturnInst")
        assert ret.return_value.name == "p"
        del ret, mod
    assert_no_leaks()


def test_memory_accessors():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_MEM_SRC, ctx, "m")
        alloca = _only(mod, "AllocaInst")
        assert str(alloca.allocated_type) == "i32"
        load = _only(mod, "LoadInst")
        # pointer_operand downcasts to the alloca it loads from.
        assert type(load.pointer_operand).__name__ == "AllocaInst"
        assert load.pointer_operand.name == "slot"
        store = _only(mod, "StoreInst")
        assert store.pointer_operand.name == "slot"
        gep = _only(mod, "GetElementPtrInst")
        assert str(gep.source_element_type) == "i32"
        del alloca, load, store, gep, mod
    assert_no_leaks()


def test_call_accessors():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_MEM_SRC, ctx, "m")
        call = _only(mod, "CallInst")
        assert call.num_args == 2
        assert call.arg_operand(0).name == "x"
        assert call.arg_operand(1).name == "ld"
        # The called operand is the @g function.
        assert call.called_operand.name == "g"
        with pytest.raises(IndexError):
            call.arg_operand(99)
        del call, mod
    assert_no_leaks()


def test_fcmp_predicate():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_MEM_SRC, ctx, "m")
        fcmp = _only(mod, "FCmpInst")
        assert fcmp.predicate == llvm.ir.FCmpPredicate.OGT
        del fcmp, mod
    assert_no_leaks()
