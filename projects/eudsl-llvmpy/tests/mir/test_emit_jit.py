#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Route B capstone: emit an object from hand-built selected MIR and JIT it.

emit_object runs the back half of codegen (register allocation, prologue/
epilogue, object emission) over already-selected MIR via -start-after=
finalize-isel, so no instruction selection runs. LLJIT.add_object then loads the
object so the DSL-built function can be called. The instructions are AArch64
(ADDWrr/COPY/RET_ReallyLR), so actually executing the result needs an AArch64
host; object emission itself is host-independent (cross ELF).
"""

import ctypes
import platform
from textwrap import dedent

import pytest

import llvm
from llvm import ir, jit, mir, types
from llvm.testing import assert_no_leaks

# Object emission uses an AArch64 target (cross ELF), so needs the AArch64
# backend linked; JIT-executing additionally needs an AArch64 host (below).
pytestmark = pytest.mark.skipif(
    "aarch64" not in llvm.jit.registered_targets(),
    reason="AArch64 backend not linked (EUDSL_LLVMPY_TARGETS)",
)

_AARCH64_LINUX = "aarch64-unknown-linux-gnu"
_IS_AARCH64 = platform.machine() in ("arm64", "aarch64")


def _build_selected_add(mmi, declare_liveins=True, name="add", opcode="ADDWrr"):
    """Hand-build a fully-selected AArch64 add(i32,i32)->i32 MachineFunction:
    liveins $w0/$w1, two COPYs in, ADDWrr, a COPY to $w0, RET_ReallyLR. With
    declare_liveins=False the live-ins are omitted, so the MIR fails verify()."""
    mf = mmi.machine_function(name)
    b = mir.MachineIRBuilder(mf)
    entry = mf.blocks[0]
    gpr32 = mf.reg_class("GPR32")
    w0, w1 = mf.physreg("W0"), mf.physreg("W1")
    if declare_liveins:
        entry.add_livein(w0)
        entry.add_livein(w1)
    v0, v1, v2 = (mf.reg_info.create_virtual_register(gpr32) for _ in range(3))
    copy = mf.opcode("COPY")
    for dst, src in ((v0, w0), (v1, w1)):
        c = b.build_instr(copy)
        c.add_reg(dst, is_def=True)
        c.add_reg(src)
    add = b.build_instr(mf.opcode(opcode))
    add.add_reg(v2, is_def=True)
    add.add_reg(v0)
    add.add_reg(v1)
    ret_copy = b.build_instr(copy)
    ret_copy.add_reg(w0, is_def=True)
    ret_copy.add_reg(v2)
    ret = b.build_instr(mf.opcode("RET_ReallyLR"))
    ret.add_reg(w0, implicit=True)
    for prop in ("IsSSA", "TracksLiveness", "NoPHIs"):
        mf.set_property(getattr(mir.MachineFunctionProperty, prop))
    return mf


def test_emit_object_produces_a_relocatable_object():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)  # cross: ELF, any host
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object()
        assert len(obj) > 0
        assert obj[:4] == b"\x7fELF"  # a real ELF relocatable object
        # Host-independent guard that the object actually defines `add`: the
        # symbol name lives null-terminated in the ELF string table. (Executing
        # it is checked separately on AArch64 hosts.)
        assert b"add\x00" in obj
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64, reason="executing AArch64 MIR needs an AArch64 host"
)
@pytest.mark.parametrize("custom_regalloc", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_jit_executes_two_functions_from_one_emission(custom_regalloc, typed):
    instances = []

    class Recorder(mir.BasicRegAlloc):
        def __init__(self):
            super().__init__()
            instances.append(self)

        def post_optimization(self):
            self.function_name = self.machine_function.name

    with ir.Context() as ctx:
        tm = jit.TargetMachine()
        i32 = types.i32(ctx)
        fn_ty = types.FunctionType.get(i32, [i32, i32], context=ctx) if typed else None
        mmi = mir.create_machine_function(
            ir.Module("m", ctx), tm, "add", function_type=fn_ty
        )
        mmi.add_machine_function("sub", function_type=fn_ty)
        _build_selected_add(mmi)
        _build_selected_add(mmi, name="sub", opcode="SUBWrr")
        if custom_regalloc:
            mir.register_regalloc("ra-two-functions", Recorder)
            obj = mmi.emit_object(regalloc="ra-two-functions")
            assert len(instances) == 2
            assert instances[0] is not instances[1]
            assert [instance.function_name for instance in instances] == ["add", "sub"]
        else:
            obj = mmi.emit_object()

        j = jit.LLJIT()
        j.add_object(obj)
        signature = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
        add, sub = signature(j.lookup("add")), signature(j.lookup("sub"))
        for a, b in ((2, 3), (40, 2), (-5, 5), (2**30, -1)):
            assert add(a, b) == a + b
            assert sub(a, b) == a - b
        del j
        assert list(mmi.machine_functions) == []
        with pytest.raises(RuntimeError, match="un-emitted"):
            mmi.add_machine_function("late")
    assert_no_leaks()


@pytest.mark.parametrize("operation", ["emit_object", "regalloc_assignments"])
def test_add_machine_function_rejected_during_codegen(operation):
    modules = []
    rejected = []

    class Recorder(mir.BasicRegAlloc):
        def post_optimization(self):
            with pytest.raises(RuntimeError, match="codegen is already running"):
                modules[0].add_machine_function("late")
            for query in (
                lambda: modules[0].emit_object(),
                lambda: modules[0].regalloc_assignments(),
                lambda: modules[0].to_mir(),
                lambda: modules[0].machine_functions,
                lambda: modules[0].machine_function("add"),
            ):
                with pytest.raises(RuntimeError, match="codegen is already running"):
                    query()
            rejected.append(self.machine_function.name)

    mir.register_regalloc("ra-add-during-codegen", Recorder)
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(ir.Module("m", ctx), tm, "add")
        _build_selected_add(mmi)
        modules.append(mmi)
        try:
            getattr(mmi, operation)(regalloc="ra-add-during-codegen")
        finally:
            modules.clear()
        assert rejected == ["add"]
        assert "@late" not in mmi.to_mir()
    assert_no_leaks()


@pytest.mark.parametrize("failure", ["init", "select"])
def test_multi_function_allocator_preserves_first_error(failure):
    calls = []

    class Broken(mir.BasicRegAlloc):
        def __init__(self):
            super().__init__()
            calls.append("init")
            if len(calls) > 1:
                raise RuntimeError("later function must not re-enter Python")
            if failure == "init":
                raise ValueError("first allocator error")

        def select_or_split(self, li):
            calls.append("select")
            raise ValueError("first allocator error")

    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(ir.Module("m", ctx), tm, "add")
        mmi.add_machine_function("sub")
        _build_selected_add(mmi)
        _build_selected_add(mmi, name="sub", opcode="SUBWrr")
        mir.register_regalloc("ra-multi-error", Broken)
        with pytest.raises(ValueError, match="first allocator error"):
            mmi.emit_object(regalloc="ra-multi-error")
        assert calls == (["init"] if failure == "init" else ["init", "select"])
        with pytest.raises(RuntimeError, match="un-emitted"):
            mmi.add_machine_function("late")

        fresh = mir.create_machine_function(ir.Module("fresh", ctx), tm, "add")
        _build_selected_add(fresh)
        assert fresh.emit_object()[:4] == b"\x7fELF"
    assert_no_leaks()


def test_add_machine_function_rejected_after_regalloc_assignments():
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mod = ir.parse_assembly("declare void @external()", ctx, "m")
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.regalloc_assignments("greedy")
        with pytest.raises(RuntimeError, match="un-emitted"):
            mmi.add_machine_function("late")
    assert_no_leaks()


@pytest.mark.parametrize("operation", ["emit_object", "regalloc_assignments"])
@pytest.mark.parametrize("callback_error", [False, True])
def test_codegen_invalidates_saved_machine_handles(operation, callback_error):
    saved = []
    callback_handles = []

    class Recorder(mir.BasicRegAlloc):
        def post_optimization(self):
            for previous, _ in saved[:2]:
                with pytest.warns(RuntimeWarning, match="uninitialized instance"):
                    with pytest.raises(TypeError):
                        previous.name
            mf = self.machine_function
            callback_handles.append(mf)
            saved.extend(
                [
                    (mf, "name"),
                    (mf.blocks[0], "number"),
                    (mir.MachineIRBuilder(mf), "machine_function"),
                ]
            )
            if callback_error:
                raise ValueError("handle lifetime regression")

    mir.register_regalloc("ra-handle-lifetime", Recorder)
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(ir.Module("m", ctx), tm, "add")
        added = mmi.add_machine_function("sub")
        first = _build_selected_add(mmi)
        _build_selected_add(mmi, name="sub", opcode="SUBWrr")
        block = first.blocks[0]
        instruction = block.instructions[0]
        operand = instruction.operand(0)
        mri = first.reg_info
        reg_class = first.reg_class("GPR32")
        builder = mir.MachineIRBuilder(first)
        ir_function = first.function
        numbered = first.block_numbered(block.number)
        assert numbered is block
        assert added is mmi.machine_function("sub")
        assert first is mmi.machine_functions[0]
        saved.extend(
            [
                (first, "name"),
                (added, "name"),
                (block, "number"),
                (instruction, "opcode"),
                (operand, "is_reg"),
                (builder, "insert_block"),
            ]
        )
        with builder:
            if callback_error:
                with pytest.raises(ValueError, match="handle lifetime regression"):
                    getattr(mmi, operation)(regalloc="ra-handle-lifetime")
            else:
                result = getattr(mmi, operation)(regalloc="ra-handle-lifetime")
                if operation == "regalloc_assignments":
                    assert list(result) == ["add", "sub"]
        assert mmi.machine_functions == []
        with pytest.raises(KeyError, match="has no MachineFunction"):
            mmi.machine_function("add")
        assert ir_function.name == "add"
        for value, attribute in saved:
            with pytest.warns(RuntimeWarning, match="uninitialized instance"):
                with pytest.raises(TypeError):
                    getattr(value, attribute)
        with pytest.warns(RuntimeWarning, match="uninitialized instance"):
            with pytest.raises(TypeError):
                mri.create_virtual_register(reg_class)
        with pytest.raises(RuntimeError, match="no current MachineIRBuilder"):
            mir.current_machine_builder()
    saved.clear()
    callback_handles.clear()
    assert_no_leaks()


@pytest.mark.parametrize("operation", ["emit_object", "regalloc_assignments"])
def test_codegen_validation_failure_preserves_machine_handles(operation):
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(ir.Module("m", ctx), tm, "add")
        mf = _build_selected_add(mmi)
        block = mf.blocks[0]
        with pytest.raises(RuntimeError, match="unknown regalloc"):
            getattr(mmi, operation)(regalloc="ra-no-such-allocator")
        assert mf.name == "add"
        assert block.parent is mf
        assert mf.verify()
        getattr(mmi, operation)(
            regalloc="greedy" if operation != "emit_object" else None
        )
    assert_no_leaks()


def test_codegen_does_not_reuse_invalidated_wrappers_for_new_objects():
    saved = []
    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        for i in range(32):
            mmi = mir.create_machine_function(ir.Module("m", ctx), tm, "add")
            mf = _build_selected_add(mmi)
            block = mf.blocks[0]
            instruction = block.instructions[0]
            operand = instruction.operand(0)
            builder = mir.MachineIRBuilder(mf)
            assert mmi.machine_function("add") is mf
            assert mmi.machine_functions[0] is mf
            assert mf.blocks[0] is block
            assert block.instructions[0] is instruction
            assert instruction.operand(0) is operand
            assert mf.reg_info is mf.reg_info
            assert builder.machine_function is mf
            saved.extend([(mf, "name"), (block, "number"), (instruction, "opcode")])
            if i % 2:
                assert mmi.emit_object()[:4] == b"\x7fELF"
            else:
                assert list(mmi.regalloc_assignments()) == ["add"]
        for value, attribute in saved:
            with pytest.warns(RuntimeWarning, match="uninitialized instance"):
                with pytest.raises(TypeError):
                    getattr(value, attribute)
    saved.clear()
    assert_no_leaks()


def test_emit_object_twice_raises():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object()
        with pytest.raises(RuntimeError, match="already emitted"):
            mmi.emit_object()
    assert_no_leaks()


def test_mir_module_is_queryable_after_emit():
    """After emit_object the MirModule stays valid to query: to_mir() still
    prints the IR module. The emission pipeline appends FreeMachineFunctionPass,
    so the MachineFunctions themselves are gone -- machine_functions is empty and
    machine_function() reports the missing MachineFunction rather than crashing."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object()
        assert "@add" in mmi.to_mir()
        assert list(mmi.machine_functions) == []
        # Routes through the same (post-emit) MMI handle: the IR Function still
        # exists but its MachineFunction was freed.
        with pytest.raises(KeyError, match="has no MachineFunction"):
            mmi.machine_function("add")
    assert_no_leaks()


def test_emit_object_requires_create_machine_function():
    src = dedent("""\
        define i32 @f(i32 %a) {
          ret i32 %a
        }
        """)
    with ir.Context() as ctx:
        mod = ir.parse_assembly(src, ctx, "m")
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.run_codegen_to_mir(mod, tm)  # no build-path MMIWrapperPass
        with pytest.raises(RuntimeError, match="create_machine_function"):
            mmi.emit_object()
    assert_no_leaks()


def test_emit_object_rejects_parse_path():
    """A parse_mir result is ParseOwned (no build-path wrapper either), so
    emit_object rejects it with the same message as the codegen path."""
    src = dedent("""\
        define i32 @f(i32 %a, i32 %b) {
          %s = add i32 %a, %b
          ret i32 %s
        }
        """)
    with ir.Context() as ctx:
        mod = ir.parse_assembly(src, ctx, "m")
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        text = mir.run_codegen_to_mir(mod, tm).to_mir()
    assert_no_leaks()

    with ir.Context() as ctx:
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.parse_mir(text, ctx, tm)
        with pytest.raises(RuntimeError, match="create_machine_function"):
            mmi.emit_object()
    assert_no_leaks()


def test_emit_object_rejects_malformed_mir():
    """emit_object verifies first, so malformed hand-built MIR (here: physregs
    read without live-in declarations) raises instead of muddling through the
    emission pipeline to a garbage object or a fatal codegen abort."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi, declare_liveins=False)
        with pytest.raises(RuntimeError, match="failed verification"):
            mmi.emit_object()
    assert_no_leaks()


def test_add_object_rejects_non_object_bytes():
    """add_object parses eagerly, so bad bytes raise here rather than resurfacing
    as an opaque materialization error at a later lookup."""
    j = jit.LLJIT()
    with pytest.raises(RuntimeError):
        j.add_object(b"not an object file")
    del j
    assert_no_leaks()


def test_start_after_is_restored_after_emit():
    """emit_object mutates the process-global -start-after option and must
    restore it, so a subsequent run_codegen_to_mir still runs full instruction
    selection (rather than skipping to finalize-isel and producing empty MIR)."""
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        mmi.emit_object()
    assert_no_leaks()

    src = dedent("""\
        define i32 @f(i32 %a, i32 %b) {
          %s = add i32 %a, %b
          ret i32 %s
        }
        """)
    with ir.Context() as ctx:
        mod = ir.parse_assembly(src, ctx, "m")
        tm = jit.TargetMachine(triple=_AARCH64_LINUX)
        mmi = mir.run_codegen_to_mir(mod, tm)
        mf = mmi.machine_function("f")
        # Full ISel ran: the function has real selected instructions and
        # verifies. A leaked -start-after would skip ISel and leave it empty.
        assert mf.verify() is True
        assert list(mf.blocks[0].instructions)
    assert_no_leaks()


@pytest.mark.skipif(
    not _IS_AARCH64,
    reason="hand-built MIR is AArch64; executing it needs an AArch64 host",
)
def test_jit_executes_hand_built_add():
    with ir.Context() as ctx:
        mod = ir.Module("m", ctx)
        tm = jit.TargetMachine()  # host triple -> object loadable in-process
        mmi = mir.create_machine_function(mod, tm, "add")
        _build_selected_add(mmi)
        obj = mmi.emit_object()

        j = jit.LLJIT()
        j.add_object(obj)
        add = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(
            j.lookup("add")
        )
        assert add(2, 3) == 5
        assert add(40, 2) == 42
        assert add(-5, 5) == 0
        del j
    assert_no_leaks()
