#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Lifetime safety: a wrapper returned by any traversal accessor keeps the
owning module alive *while that module is live*, so dropping the Python module
handle (but not consuming it) never dangles a held wrapper. Each case holds ONLY
the accessor result, drops the module, and asserts the module is still live
(== 1) before touching the result, then asserts dropping the result releases the
module (== 0). The count assertion runs before any dereference, so an accessor
that fails to pin fails cleanly rather than segfaulting. (Consuming the module
via take()/_take() hands its storage to the JIT and is outside this guarantee;
see test_take_consumes_module_and_is_outside_pinning_guarantee.)"""
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
@ce = global i64 ptrtoint (ptr @g to i64)

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


# (id, make(mod) -> wrapper held as the sole reference, touch(wrapper) -> asserts
# concrete content while storage is valid). The touch confirms downcast/identity;
# it is NOT the safety signal (freed LLVM storage reads back plausible garbage,
# e.g. name -> ""), which is the `== 1` count assertion that precedes it.
CASES = [
    ("get_function", lambda m: m.get_function("f"), lambda v: v.name == "f"),
    ("get_global_variable", lambda m: m.get_global_variable("g"), lambda v: v.name == "g"),
    ("module_getitem", lambda m: m[0], lambda v: v.name == "f"),
    ("module_functions_view", lambda m: m.functions[0], lambda v: v.name == "f"),
    ("function_arg", lambda m: m.get_function("f").arg(0), lambda v: v.arg_no == 0),
    ("function_args_view", lambda m: m.get_function("f").args[0], lambda v: v.arg_no == 0),
    ("function_entry_block", lambda m: m.get_function("f").entry_block, lambda v: v.name == "entry"),
    ("function_bbs_view", lambda m: m.get_function("f").basic_blocks[0], lambda v: v.name == "entry"),
    ("function_getitem", lambda m: m.get_function("f")[0], lambda v: v.name == "entry"),
    ("bb_parent", lambda m: m.get_function("f").entry_block.parent, lambda v: v.name == "f"),
    ("bb_terminator", lambda m: m.get_function("f").entry_block.terminator, lambda v: v.is_terminator),
    ("bb_instructions_view", lambda m: m.get_function("f").entry_block.instructions[0], lambda v: v.name == "gv"),
    ("bb_getitem", lambda m: m.get_function("f").entry_block[0], lambda v: v.name == "gv"),
    ("inst_parent", lambda m: _add(m).parent, lambda v: v.name == "entry"),
    ("inst_operand", lambda m: _add(m).operand(0), lambda v: v.name == "x"),
    ("inst_operands_view", lambda m: _add(m).operands[0], lambda v: v.name == "x"),
    ("user_getitem", lambda m: _add(m)[0], lambda v: v.name == "x"),
    ("arg_parent", lambda m: m.get_function("f").arg(0).parent, lambda v: v.name == "f"),
    ("arg_users_view", lambda m: m.get_function("f").arg(0).users[0], lambda v: v.name == "s"),
    ("arg_uses_view", lambda m: m.get_function("f").arg(0).uses[0], lambda v: v.operand_number == 0),
    ("use_user", lambda m: m.get_function("f").arg(0).uses[0].user, lambda v: v.name == "s"),
    ("gvar_initializer", lambda m: m.get_global_variable("g").initializer, lambda v: str(v) == "i32 7"),
    ("load_pointer_operand", lambda m: _load_ptr(m, "g"), lambda v: v.name == "g"),
    ("global_alias_aliasee", lambda m: _load_ptr(m, "al").aliasee, lambda v: v.name == "g"),
    ("global_ifunc_resolver", lambda m: _load_ptr(m, "fp").resolver, lambda v: v.name == "res"),
    ("block_address_function", lambda m: _load_ptr(m, "ba").initializer.function, lambda v: v.name == "f"),
    ("block_address_block", lambda m: _load_ptr(m, "ba").initializer.basic_block, lambda v: v.name == "b"),
    # A none-owner container (the ConstantExpr `ptrtoint (ptr @g to i64)`, which
    # has no module) yielding a module-owned element (@g): the element must pin
    # its OWN module, not the container's. Before the per-element owner fix this
    # failed cleanly at count 0.
    ("const_expr_operand_module_owned", lambda m: m.get_global_variable("ce").initializer.operands[0], lambda v: v.name == "g"),
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


_ITER_CASES = [
    ("module_iter", lambda m: next(iter(m)), lambda v: v.name == "f"),  # -> Function
    ("function_iter", lambda m: next(iter(m.get_function("f"))), lambda v: v.name == "entry"),  # -> BasicBlock
    ("bb_iter", lambda m: next(iter(m.get_function("f").entry_block)), lambda v: v.name == "gv"),  # -> Instruction
    ("user_iter", lambda m: next(iter(_add(m))), lambda v: v.name == "x"),  # -> operand Value
]


@pytest.mark.parametrize(
    "make,check",
    [(c[1], c[2]) for c in _ITER_CASES],
    ids=[c[0] for c in _ITER_CASES],
)
def test_iterator_element_pins_module(make, check):
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        held = make(mod)  # element from an iterator; the iterator is not retained
        del mod
        gc.collect()
        assert llvm.ir.Context._get_live_module_count() == 1
        # Concrete content, not `str(held) is not None` (always true): confirms
        # the yielded element is the expected wrapper while its storage is valid.
        assert check(held)
        del held
        gc.collect()
        assert llvm.ir.Context._get_live_module_count() == 0
    assert_no_leaks()


def test_missing_lookup_returns_none():
    # get_function/get_global_variable return a null llvm pointer for a missing
    # name; reference_internal on null yields None (keep_alive no-ops on None).
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        assert mod.get_function("missing") is None
        assert mod.get_global_variable("missing") is None
        del mod
    assert_no_leaks()


def test_take_consumes_module_and_is_outside_pinning_guarantee():
    # take()/_take() hand the llvm::Module to the JIT (here, drop it), which then
    # owns its storage. The pinning guarantee covers a *live* module; a wrapper
    # obtained before the module is consumed is not covered. Document that the
    # wrapper is consumed and further module access raises cleanly (rather than
    # dereferencing a held pre-take value, which is now freed).
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        mod._take()
        with pytest.raises(RuntimeError, match="consumed"):
            mod.get_function("f")
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
