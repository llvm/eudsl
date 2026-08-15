#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Lifetime safety: a wrapper returned by any traversal accessor keeps the
owning module alive, so holding it after the module handle is dropped is never a
use-after-free. Each case holds ONLY the accessor result, drops the module, and
asserts the module is still live (== 1) before touching the result, then asserts
dropping the result releases the module (== 0). The count assertion runs before
any dereference, so an accessor that fails to pin fails cleanly rather than
segfaulting."""
import gc

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = """\
@g = global i32 7
@res = global ptr null
@fp = ifunc i32 (), ptr @res
@al = alias i32, ptr @g
@ba = global ptr blockaddress(@f, %b)

define i32 @f(i32 %x, i32 %y) {
entry:
  %gv = load i32, ptr @g
  %a = load i32, ptr @al
  %i = load ptr, ptr @fp
  %bp = load ptr, ptr @ba
  %s = add i32 %x, %y
  br label %b
b:
  ret i32 %s
}
"""


def _add(m):
    # The `add` instruction (a User with two operands) in @f's entry block.
    for inst in m.get_function("f").entry_block.instructions:
        if isinstance(inst, llvm.ir.BinaryOperator):
            return inst
    raise AssertionError("add")


def _load_ptr(m, name):
    # pointer_operand of the load whose pointer is the named global/alias/ifunc.
    for inst in m.get_function("f").entry_block.instructions:
        if isinstance(inst, llvm.ir.LoadInst) and inst.pointer_operand.name == name:
            return inst.pointer_operand
    raise AssertionError(name)


# (id, make(mod) -> wrapper held as the sole reference, touch(wrapper) -> uses it)
CASES = [
    ("get_function", lambda m: m.get_function("f"), lambda v: v.name == "f"),
    ("get_global_variable", lambda m: m.get_global_variable("g"), lambda v: v.name == "g"),
    ("module_getitem", lambda m: m[0], lambda v: bool(v.name)),
    ("module_functions_view", lambda m: m.functions[0], lambda v: bool(v.name)),
    ("function_arg", lambda m: m.get_function("f").arg(0), lambda v: v.arg_no == 0),
    ("function_args_view", lambda m: m.get_function("f").args[0], lambda v: v.arg_no == 0),
    ("function_entry_block", lambda m: m.get_function("f").entry_block, lambda v: bool(v.name)),
    ("function_bbs_view", lambda m: m.get_function("f").basic_blocks[0], lambda v: bool(v.name)),
    ("function_getitem", lambda m: m.get_function("f")[0], lambda v: bool(v.name)),
    ("bb_parent", lambda m: m.get_function("f").entry_block.parent, lambda v: v.name == "f"),
    ("bb_terminator", lambda m: m.get_function("f").entry_block.terminator, lambda v: v.is_terminator),
    ("bb_instructions_view", lambda m: m.get_function("f").entry_block.instructions[0], lambda v: bool(str(v))),
    ("bb_getitem", lambda m: m.get_function("f").entry_block[0], lambda v: bool(str(v))),
    ("inst_parent", lambda m: _add(m).parent, lambda v: bool(v.name)),
    ("inst_operand", lambda m: _add(m).operand(0), lambda v: bool(str(v))),
    ("inst_operands_view", lambda m: _add(m).operands[0], lambda v: bool(str(v))),
    ("user_getitem", lambda m: _add(m)[0], lambda v: bool(str(v))),
    ("arg_parent", lambda m: m.get_function("f").arg(0).parent, lambda v: v.name == "f"),
    ("arg_users_view", lambda m: m.get_function("f").arg(0).users[0], lambda v: bool(str(v))),
    ("arg_uses_view", lambda m: m.get_function("f").arg(0).uses[0], lambda v: v.operand_number == 0),
    ("use_user", lambda m: m.get_function("f").arg(0).uses[0].user, lambda v: bool(str(v))),
    ("gvar_initializer", lambda m: m.get_global_variable("g").initializer, lambda v: bool(str(v))),
    ("load_pointer_operand", lambda m: _load_ptr(m, "g"), lambda v: v.name == "g"),
    ("global_alias_aliasee", lambda m: _load_ptr(m, "al").aliasee, lambda v: v.name == "g"),
    ("global_ifunc_resolver", lambda m: _load_ptr(m, "fp").resolver, lambda v: v.name == "res"),
    ("block_address_function", lambda m: _load_ptr(m, "ba").initializer.function, lambda v: v.name == "f"),
    ("block_address_block", lambda m: _load_ptr(m, "ba").initializer.basic_block, lambda v: v.name == "b"),
]


@pytest.mark.parametrize("make,touch", [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES])
def test_accessor_result_pins_module(make, touch):
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        held = make(mod)
        del mod  # the accessor result is now the only reference to the module
        gc.collect()
        # Assert BEFORE touching: a non-pinning accessor fails here (count 0),
        # so we never dereference freed storage.
        assert llvm.ir.Context._get_live_module_count() == 1
        assert touch(held)
        del held
        gc.collect()
        assert llvm.ir.Context._get_live_module_count() == 0
    assert_no_leaks()


@pytest.mark.parametrize(
    "make",
    [
        lambda m: next(iter(m)),  # Module.__iter__ -> Function
        lambda m: next(iter(m.get_function("f"))),  # Function.__iter__ -> BasicBlock
        lambda m: next(iter(m.get_function("f").entry_block)),  # BB.__iter__ -> Instruction
        lambda m: next(iter(_add(m))),  # User.__iter__ -> operand Value
    ],
    ids=["module_iter", "function_iter", "bb_iter", "user_iter"],
)
def test_iterator_element_pins_module(make):
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        held = make(mod)  # element from an iterator; the iterator is not retained
        del mod
        gc.collect()
        assert llvm.ir.Context._get_live_module_count() == 1
        assert str(held) is not None
        del held
        gc.collect()
        assert llvm.ir.Context._get_live_module_count() == 0
    assert_no_leaks()


def test_no_module_owner_view_is_safe():
    # A context-owned value (a constant) has no owning module, so its views get
    # a none owner (nothing to pin). Accessing them must not crash; it simply
    # isn't module-pinned. Exercises ownerObjectFor's no-module fold.
    with llvm.ir.Context() as ctx:
        c = llvm.ir.const_int(llvm.types.i32(ctx), 5)
        assert len(c.users) == 0
        assert len(c.uses) == 0
        del c
    assert_no_leaks()
