#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

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


def _insts_by_class(mod, name):
    out = []
    for f in mod.functions:
        for bb in f.basic_blocks:
            for i in bb.instructions:
                if type(i).__name__ == name:
                    out.append(i)
    return out


def test_instructions_downcast():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_PHI_SRC, ctx, "m")
        # add-style ops become BinaryOperator; the icmp becomes ICmpInst; the
        # phi becomes PHINode; the conditional branch becomes CondBrInst.
        assert len(_insts_by_class(mod, "PHINode")) == 1
        assert len(_insts_by_class(mod, "ICmpInst")) == 1
        assert len(_insts_by_class(mod, "CondBrInst")) == 1
        del mod
    assert_no_leaks()


def test_phi_incoming():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_PHI_SRC, ctx, "m")
        (phi,) = _insts_by_class(mod, "PHINode")
        assert phi.num_incoming == 2
        assert phi.incoming_block(0).name == "a"
        assert phi.incoming_block(1).name == "b"
        assert str(phi.incoming_value(0)) == "i32 1"
        del phi, mod
    assert_no_leaks()


def test_icmp_predicate():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_PHI_SRC, ctx, "m")
        (icmp,) = _insts_by_class(mod, "ICmpInst")
        assert icmp.predicate == llvm.ICmpPredicate.EQ
        del icmp, mod
    assert_no_leaks()


def test_conditional_branch():
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(_PHI_SRC, ctx, "m")
        cbrs = _insts_by_class(mod, "CondBrInst")
        assert len(cbrs) == 1
        assert cbrs[0].is_conditional
        assert cbrs[0].num_successors == 2
        del cbrs, mod
    assert_no_leaks()
