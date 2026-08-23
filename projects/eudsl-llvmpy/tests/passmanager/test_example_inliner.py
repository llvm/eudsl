#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""A minimal inliner written as a Python module pass, exercising the
object-level bindings end to end: for a call to a single-block, defined callee
it clones each callee instruction into the caller before the call, remaps
operands (callee args -> actual args, callee values -> their clones) with
set_operand, rewires the call result to the callee's returned value, and erases
the call. No LLVM analyses; the (trivial) inlinability check is done in Python."""

import ctypes
from textwrap import dedent

import pytest

import llvm
from llvm import ir, jit
from llvm.testing import assert_no_leaks

_SRC = dedent("""\
    define i32 @callee(i32 %a, i32 %b) {
      %t = add i32 %a, %b
      %u = mul i32 %t, %a
      ret i32 %u
    }
    define i32 @caller(i32 %x) {
      %c = call i32 @callee(i32 %x, i32 3)
      %d = add i32 %c, 100
      ret i32 %d
    }
    """)


def _inline_single_block(call, callee):
    # Map callee formals to the actual call arguments.
    vmap = {callee.arg(i): call.arg_operand(i) for i in range(callee.num_args)}
    ret_val = None
    for ci in callee.entry_block.instructions:
        if ci.opcode_name == "ret":
            ret_val = (
                vmap.get(ci.operand(0), ci.operand(0)) if ci.num_operands else None
            )
            break
        clone = ci.clone()
        # Remap operands that refer to callee-local values; leave constants /
        # globals (not in the map) pointing at the shared objects.
        for k in range(clone.num_operands):
            op = clone.operand(k)
            if op in vmap:
                clone.set_operand(k, vmap[op])
        clone.insert_before(call)
        vmap[ci] = clone
    if ret_val is not None:
        call.replace_all_uses_with(ret_val)
    call.erase_from_parent()


def inline_calls(module):
    """Inline every call to a defined, single-block function (once)."""
    changed = False
    for fn in module.functions:
        if fn.is_declaration:
            continue
        for inst in list(fn.entry_block.instructions):
            if inst.opcode_name != "call":
                continue
            callee = inst.called_operand
            if not isinstance(callee, ir.Function) or callee.is_declaration:
                continue
            if len(list(callee.basic_blocks)) != 1:  # minimal: single-block callee
                continue
            _inline_single_block(inst, callee)
            changed = True
    return changed


def test_inliner_removes_call_and_splices_body():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_python_pass_on_module(mod, inline_calls)
        caller = str(mod).split("define i32 @caller")[1]
        assert "call i32 @callee" not in caller  # the call was removed
        assert "add i32 %x, 3" in caller  # callee body cloned + args remapped
        assert "mul i32" in caller
        mod.verify()
        del mod
    assert_no_leaks()


def test_inliner_preserves_semantics():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        llvm.passmanager.run_python_pass_on_module(mod, inline_calls)
        mod.verify()
        j = jit.LLJIT()
        j.add_module(mod)  # consumes the module
        caller = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32)(j.lookup("caller"))
        assert caller(5) == (5 + 3) * 5 + 100  # 140
    assert_no_leaks()


def test_inliner_leaves_external_calls_alone():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(
            dedent("""\
                declare i32 @ext(i32)
                define i32 @caller(i32 %x) {
                  %c = call i32 @ext(i32 %x)
                  ret i32 %c
                }
                """),
            ctx,
            "m",
        )
        llvm.passmanager.run_python_pass_on_module(mod, inline_calls)
        assert "call i32 @ext" in str(mod)  # declaration -> not inlined
        mod.verify()
        del mod
    assert_no_leaks()
