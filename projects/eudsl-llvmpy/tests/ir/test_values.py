#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import gc

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent(
    """\
    define i32 @f(i32 %x, i32 %y) {
    entry:
      %sum = add i32 %x, %y
      ret i32 %sum
    }
    """
)


def test_value_and_user_registered():
    # The Value/User classes are registered and module round-tripping works.
    assert hasattr(llvm.ir, "Value")
    assert hasattr(llvm.ir, "User")
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        assert "define i32 @f(i32 %x, i32 %y)" in str(mod)
        del mod
    assert_no_leaks()


def test_function_traversal():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        assert [f.name for f in mod.functions] == ["f"]
        f = mod.get_function("f")
        assert f is not None
        assert mod.get_function("nope") is None
        assert f.num_args == 2
        assert [a.name for a in f.args] == ["x", "y"]
        assert f.arg(0).arg_no == 0
        assert f.arg(1).parent == f
        assert not f.is_declaration
        assert str(f.return_type) == "i32"
        del f, mod
    assert_no_leaks()


def test_basic_block_and_instruction_traversal():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        blocks = f.basic_blocks
        assert [b.name for b in blocks] == ["entry"]
        entry = f.entry_block
        assert entry.name == "entry"
        assert entry == blocks[0]
        insts = entry.instructions
        # add, ret
        assert len(insts) == 2
        assert entry.terminator == insts[-1]
        # The Value type_hook downcasts the add to its concrete class.
        assert isinstance(insts[0], llvm.ir.BinaryOperator)
        del f, entry, blocks, insts, mod
    assert_no_leaks()


def test_value_users_and_operands():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        # %x is used by the add.
        assert x.num_uses == 1
        add = x.users[0]
        assert isinstance(add, llvm.ir.BinaryOperator)
        assert add.num_operands == 2
        assert add.operand(0) == x
        del f, x, add, mod
    assert_no_leaks()


def test_append_basic_block():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly("declare void @g()\n", ctx, "m")
        ft = llvm.types.function(llvm.types.void(ctx), [])
        fn = llvm.ir.Function.create(ft, "h", mod)
        bb = fn.append_basic_block("entry")
        assert bb.name == "entry"
        assert fn.entry_block == bb
        assert bb.parent == fn
        del fn, bb, mod
    assert_no_leaks()


def test_function_create_linkage():
    # Function.create takes a linkage argument (default external); selecting
    # internal linkage is reflected in the printed IR.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.Module("m", ctx)
        ft = llvm.types.function(llvm.types.void(ctx), [])
        fn = llvm.ir.Function.create(ft, "priv", mod, linkage=llvm.ir.Linkage.INTERNAL)
        fn.append_basic_block("entry")
        assert "define internal void @priv" in str(mod)
        del fn, mod
    assert_no_leaks()


def test_entry_block_and_terminator_are_none_when_absent():
    # entry_block (Function -> BasicBlock*) and terminator
    # (BasicBlock -> Instruction*) can both return null on the C++ side; the
    # Value type_hook must map that null to None rather than dereferencing it.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly("declare void @g()\n", ctx, "m")
        g = mod.get_function("g")
        assert g.entry_block is None
        ft = llvm.types.function(llvm.types.void(ctx), [])
        fn = llvm.ir.Function.create(ft, "h", mod)
        bb = fn.append_basic_block("entry")
        assert bb.terminator is None
        del g, fn, bb, mod
    assert_no_leaks()


def test_replace_all_uses_with():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        x, y = f.arg(0), f.arg(1)
        assert x.num_uses == 1
        assert y.num_uses == 1
        x.replace_all_uses_with(y)
        assert x.num_uses == 0
        assert y.num_uses == 2
        del f, x, y, mod
    assert_no_leaks()


def test_value_name_setter():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        x = f.arg(0)
        x.name = "renamed"
        assert x.name == "renamed"
        assert "%renamed" in str(mod)
        del f, x, mod
    assert_no_leaks()


def test_instruction_base_accessors():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        entry = f.entry_block
        add, ret = entry.instructions
        assert not add.is_terminator
        assert ret.is_terminator
        assert add.parent == entry
        assert ret.num_successors == 0
        del f, entry, add, ret, mod
    assert_no_leaks()


def test_function_type_accessor():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        f = mod.get_function("f")
        assert f.function_type.num_params == 2
        assert str(f.function_type.return_type) == "i32"
        del f, mod
    assert_no_leaks()


def test_module_outlives_released_context():
    # A Module holds a shared_ptr to the LLVMContext, so it stays usable after
    # the Python context manager's __exit__ releases the Context. The live count
    # drops at __exit__ (nothing leaked), but the underlying LLVMContext survives
    # until the module itself is gone. Under a unique_ptr Context, dereferencing
    # the module here would be a use-after-free.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
    # Context released; the live count is already back to zero.
    assert llvm.ir.Context._get_live_count() == 0
    # Dereference the still-live module: print it and walk its instructions.
    assert "define i32 @f(i32 %x, i32 %y)" in str(mod)
    for f in mod.functions:
        for bb in f.basic_blocks:
            for inst in bb.instructions:
                assert str(inst)
    del mod
    gc.collect()
    assert_no_leaks()
