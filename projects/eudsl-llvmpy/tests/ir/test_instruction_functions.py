#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Free-function instruction emitters (llvm.instructions): they delegate to the
contextual builder (`with builder:` / `with InsertPoint(...):`) when no `builder`
is passed, or to an explicit `builder=`."""
import pytest

import llvm
from llvm import instructions as I
from llvm.ir import InsertPoint, IRBuilder
from llvm.testing import assert_no_leaks, filecheck_with_comments


def _fn(ctx, ret, args, name="f"):
    mod = llvm.ir.Module("m", ctx)
    fn = llvm.ir.Function.create(llvm.types.function(ret, args), name, mod)
    return mod, fn


def test_int_arithmetic_free_functions():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, y = fn.arg(0), fn.arg(1)
            s = I.add(x, y, "s")
            I.sub(x, y, "d")
            I.mul(x, y, "m")
            I.sdiv(x, y, "q")
            I.udiv(x, y, "uq")
            I.ret(s)
        p = str(mod)
        assert "%s = add i32 %0, %1" in p
        assert "%d = sub i32 %0, %1" in p
        assert "%m = mul i32 %0, %1" in p
        assert "%q = sdiv i32 %0, %1" in p
        assert "%uq = udiv i32 %0, %1" in p
        del b, fn, mod
    assert_no_leaks()


def test_float_arithmetic_and_fcmp_free_functions():
    with llvm.ir.Context() as ctx:
        f32 = llvm.types.f32()
        mod, fn = _fn(ctx, f32, [f32, f32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, y = fn.arg(0), fn.arg(1)
            I.fsub(x, y, "d")
            I.fmul(x, y, "m")
            I.fdiv(x, y, "q")
            I.fcmp(llvm.ir.CmpPredicate.OGT, x, y, "c")
            I.ret(I.fadd(x, y, "s"))
        p = str(mod)
        assert "%s = fadd float %0, %1" in p
        assert "%d = fsub float %0, %1" in p
        assert "%m = fmul float %0, %1" in p
        assert "%q = fdiv float %0, %1" in p
        assert "%c = fcmp ogt float %0, %1" in p
        del b, fn, mod
    assert_no_leaks()


def test_icmp_free_function():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, llvm.types.i1(), [i32, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.ret(I.icmp(llvm.ir.CmpPredicate.SLT, fn.arg(0), fn.arg(1), "c"))
        assert "%c = icmp slt i32 %0, %1" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_memory_free_functions():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [llvm.types.ptr()])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            slot = I.alloca(i32, "slot")
            I.store(I.i32_const(5), slot)
            I.gep(i32, fn.arg(0), [I.i64_const(2)], "g")
            I.ret(I.load(i32, slot, "ld"))
        # CHECK: %slot = alloca i32
        # CHECK: store i32 5, ptr %slot
        # CHECK: %g = getelementptr i32, ptr %0, i64 2
        # CHECK: %ld = load i32, ptr %slot
        # CHECK: ret i32 %ld
        filecheck_with_comments(mod)
        del b, fn, mod
    assert_no_leaks()


def test_call_and_aggregate_free_functions():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        st = llvm.types.struct([i32, i32])
        mod = llvm.ir.Module("m", ctx)
        callee = llvm.ir.Function.create(llvm.types.function(i32, [i32]), "callee", mod)
        fn = llvm.ir.Function.create(llvm.types.function(i32, [st, i32]), "f", mod)
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            upd = I.insert_value(fn.arg(0), fn.arg(1), 1, "upd")
            first = I.extract_value(upd, 0, "first")
            I.ret(I.call(callee, [first], "c"))
        p = str(mod)
        assert "%upd = insertvalue { i32, i32 } %0, i32 %1, 1" in p
        assert "%first = extractvalue { i32, i32 } %upd, 0" in p
        assert "%c = call i32 @callee(i32 %first)" in p
        del b, fn, mod
    assert_no_leaks()


def test_terminators_and_phi_free_functions():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [llvm.types.i1()])
        entry = fn.append_basic_block("entry")
        a = fn.append_basic_block("a")
        bb = fn.append_basic_block("b")
        join = fn.append_basic_block("join")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b):
            I.cond_br(fn.arg(0), a, bb)
        with InsertPoint(a, builder=b):
            I.br(join)
        with InsertPoint(bb, builder=b):
            I.br(join)
        with InsertPoint(join, builder=b):
            ph = I.phi(i32, "p")
            ph.add_incoming(I.i32_const(1), a)
            ph.add_incoming(I.i32_const(2), bb)
            I.ret(ph)
        p = str(mod)
        assert "br i1 %0, label %a, label %b" in p
        assert "%p = phi i32 [ 1, %a ], [ 2, %b ]" in p
        assert "ret i32 %p" in p
        del b, fn, entry, a, bb, join, ph, mod
    assert_no_leaks()


def test_free_function_via_with_builder():
    # The contextual builder can also come from a bare `with builder:`.
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32])
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        with b:
            b.set_insert_point(entry)
            I.ret(I.add(fn.arg(0), I.i32_const(1), "s"))
        assert "%s = add i32 %0, 1" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_explicit_builder_without_any_context():
    # No `with builder:` / InsertPoint: pass builder= explicitly.
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32])
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        b.set_insert_point(entry)
        s = I.add(fn.arg(0), I.i32_const(1, builder=b), "s", builder=b)
        I.ret(s, builder=b)
        assert "%s = add i32 %0, 1" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_free_function_without_builder_or_context_raises():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32])
        with pytest.raises(RuntimeError, match="no current IRBuilder"):
            I.add(fn.arg(0), fn.arg(0), "s")  # no builder=, no context
        del mod
    assert_no_leaks()


def test_int_bitwise_rem_and_unary_free_functions():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, y = fn.arg(0), fn.arg(1)
            I.srem(x, y, "srem")
            I.urem(x, y, "urem")
            I.shl(x, y, "shl")
            I.lshr(x, y, "lshr")
            I.ashr(x, y, "ashr")
            I.and_(x, y, "andv")
            I.or_(x, y, "orv")
            I.xor(x, y, "xorv")
            I.neg(x, "neg")
            I.not_(x, "notv")
            I.ret(x)
        p = str(mod)
        assert "%srem = srem i32 %0, %1" in p
        assert "%urem = urem i32 %0, %1" in p
        assert "%shl = shl i32 %0, %1" in p
        assert "%lshr = lshr i32 %0, %1" in p
        assert "%ashr = ashr i32 %0, %1" in p
        assert "%andv = and i32 %0, %1" in p
        assert "%orv = or i32 %0, %1" in p
        assert "%xorv = xor i32 %0, %1" in p
        assert "%neg = sub i32 0, %0" in p
        assert "%notv = xor i32 %0, -1" in p
        del b, fn, mod
    assert_no_leaks()


def test_float_rem_and_fneg_free_functions():
    with llvm.ir.Context() as ctx:
        f32 = llvm.types.f32()
        mod, fn = _fn(ctx, f32, [f32, f32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, y = fn.arg(0), fn.arg(1)
            I.frem(x, y, "frem")
            I.fneg(x, "fneg")
            I.ret(x)
        p = str(mod)
        assert "%frem = frem float %0, %1" in p
        assert "%fneg = fneg float %0" in p
        del b, fn, mod
    assert_no_leaks()


def test_integer_and_pointer_cast_free_functions():
    with llvm.ir.Context() as ctx:
        i64, i32, ptr, f32 = (
            llvm.types.i64(), llvm.types.i32(), llvm.types.ptr(), llvm.types.f32()
        )
        mod, fn = _fn(ctx, i64, [i64, i32, ptr])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            a, n, p_ = fn.arg(0), fn.arg(1), fn.arg(2)
            I.trunc(a, i32, "tr")
            I.zext(n, i64, "ze")
            I.sext(n, i64, "se")
            I.ptrtoint(p_, i64, "p2i")
            I.inttoptr(a, ptr, "i2p")
            I.bitcast(n, f32, "bc")
            I.ret(a)
        p = str(mod)
        assert "%tr = trunc i64 %0 to i32" in p
        assert "%ze = zext i32 %1 to i64" in p
        assert "%se = sext i32 %1 to i64" in p
        assert "%p2i = ptrtoint ptr %2 to i64" in p
        assert "%i2p = inttoptr i64 %0 to ptr" in p
        assert "%bc = bitcast i32 %1 to float" in p
        del b, fn, mod
    assert_no_leaks()


def test_fp_cast_free_functions():
    with llvm.ir.Context() as ctx:
        f32, f64, i32 = llvm.types.f32(), llvm.types.f64(), llvm.types.i32()
        mod, fn = _fn(ctx, f32, [f32, f64, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, d, n = fn.arg(0), fn.arg(1), fn.arg(2)
            I.fptrunc(d, f32, "ft")
            I.fpext(x, f64, "fe")
            I.fptoui(x, i32, "fu")
            I.fptosi(x, i32, "fs")
            I.uitofp(n, f32, "uf")
            I.sitofp(n, f32, "sf")
            I.ret(x)
        p = str(mod)
        assert "%ft = fptrunc double %1 to float" in p
        assert "%fe = fpext float %0 to double" in p
        assert "%fu = fptoui float %0 to i32" in p
        assert "%fs = fptosi float %0 to i32" in p
        assert "%uf = uitofp i32 %2 to float" in p
        assert "%sf = sitofp i32 %2 to float" in p
        del b, fn, mod
    assert_no_leaks()


def test_addrspacecast_free_function():
    with llvm.ir.Context() as ctx:
        ptr = llvm.types.ptr()
        mod, fn = _fn(ctx, ptr, [ptr])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.addrspacecast(fn.arg(0), llvm.types.ptr(1), "asc")
            I.ret(fn.arg(0))
        assert "%asc = addrspacecast ptr %0 to ptr addrspace(1)" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_select_and_freeze_free_functions():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [llvm.types.i1(), i32, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            c, t, f = fn.arg(0), fn.arg(1), fn.arg(2)
            I.freeze(t, "fr")
            I.ret(I.select(c, t, f, "sel"))
        p = str(mod)
        assert "%sel = select i1 %0, i32 %1, i32 %2" in p
        assert "%fr = freeze i32 %1" in p
        del b, fn, mod
    assert_no_leaks()


def test_vector_free_functions():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        vec = llvm.types.vector(i32, 4)
        mod, fn = _fn(ctx, vec, [vec, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            v, e = fn.arg(0), fn.arg(1)
            idx = I.i32_const(0)
            I.extract_element(v, idx, "ee")
            I.insert_element(v, e, idx, "ie")
            I.shuffle_vector(v, v, [0, 1, 2, 3], "sh")
            I.ret(v)
        p = str(mod)
        assert "%ee = extractelement <4 x i32> %0, i32 0" in p
        assert "%ie = insertelement <4 x i32> %0, i32 %1, i32 0" in p
        assert (
            "%sh = shufflevector <4 x i32> %0, <4 x i32> %0, "
            "<4 x i32> <i32 0, i32 1, i32 2, i32 3>" in p
        )
        del b, fn, mod
    assert_no_leaks()


def test_va_arg_free_function():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [llvm.types.ptr()])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.ret(I.va_arg(fn.arg(0), i32, "va"))
        assert "%va = va_arg ptr %0, i32" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_unreachable_free_function():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx, llvm.types.void(), [])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.unreachable()
        assert "unreachable" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_ptrtoaddr_free_function():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx, llvm.types.i64(), [llvm.types.ptr()])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.ret(I.ptrtoaddr(fn.arg(0), "pa"))
        assert "%pa = ptrtoaddr ptr %0 to i64" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_switch_free_function():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32])
        entry = fn.append_basic_block("entry")
        c1 = fn.append_basic_block("c1")
        c2 = fn.append_basic_block("c2")
        default = fn.append_basic_block("default")
        b = IRBuilder(ctx)
        with InsertPoint(entry, builder=b):
            sw = I.switch_(fn.arg(0), default)
            sw.add_case(I.i32_const(1), c1)
            sw.add_case(I.i32_const(2), c2)
        with InsertPoint(c1, builder=b):
            I.ret(I.i32_const(10))
        with InsertPoint(c2, builder=b):
            I.ret(I.i32_const(20))
        with InsertPoint(default, builder=b):
            I.ret(I.i32_const(0))
        p = str(mod)
        assert "switch i32 %0, label %default [" in p
        assert "i32 1, label %c1" in p
        assert "i32 2, label %c2" in p
        del b, fn, entry, c1, c2, default, sw, mod
    assert_no_leaks()


def test_indirect_br_free_function():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx, llvm.types.void(), [])
        entry = fn.append_basic_block("entry")
        t1 = fn.append_basic_block("t1")
        t2 = fn.append_basic_block("t2")
        b = IRBuilder(ctx)
        addr = llvm.ir.BlockAddress.get(fn, t1)
        with InsertPoint(entry, builder=b):
            ibr = I.indirect_br(addr)
            ibr.add_destination(t1)
            ibr.add_destination(t2)
        with InsertPoint(t1, builder=b):
            I.ret()
        with InsertPoint(t2, builder=b):
            I.ret()
        p = str(mod)
        assert (
            "indirectbr ptr blockaddress(@f, %t1), [label %t1, label %t2]" in p
        )
        del b, fn, entry, t1, t2, ibr, mod
    assert_no_leaks()


def test_resume_free_function():
    with llvm.ir.Context() as ctx:
        i32, ptr = llvm.types.i32(), llvm.types.ptr()
        exn_ty = llvm.types.struct([ptr, i32])
        mod, fn = _fn(ctx, llvm.types.void(), [exn_ty])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.resume(fn.arg(0))
        assert "resume { ptr, i32 } %0" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_fence_free_function():
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx, llvm.types.void(), [])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.fence(llvm.ir.AtomicOrdering.SequentiallyConsistent)
            I.fence(llvm.ir.AtomicOrdering.Acquire, single_thread=True)
            I.ret()
        p = str(mod)
        assert "fence seq_cst" in p
        assert 'fence syncscope("singlethread") acquire' in p
        del b, fn, mod
    assert_no_leaks()


def test_atomic_rmw_free_function():
    with llvm.ir.Context() as ctx:
        i32, ptr = llvm.types.i32(), llvm.types.ptr()
        mod, fn = _fn(ctx, i32, [ptr, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.atomic_rmw(
                llvm.ir.AtomicRMWBinOp.Add,
                fn.arg(0),
                fn.arg(1),
                llvm.ir.AtomicOrdering.SequentiallyConsistent,
                name="rmw",
            )
            # a value added in the extended BinOp set, single-threaded scope
            r = I.atomic_rmw(
                llvm.ir.AtomicRMWBinOp.UIncWrap,
                fn.arg(0),
                fn.arg(1),
                llvm.ir.AtomicOrdering.Monotonic,
                single_thread=True,
                name="inc",
            )
            I.ret(r)
        p = str(mod)
        assert "%rmw = atomicrmw add ptr %0, i32 %1 seq_cst, align 4" in p
        assert (
            '%inc = atomicrmw uinc_wrap ptr %0, i32 %1 syncscope("singlethread") '
            "monotonic, align 4" in p
        )
        del b, fn, mod
    assert_no_leaks()


def test_atomic_cmpxchg_free_function():
    with llvm.ir.Context() as ctx:
        i32, ptr = llvm.types.i32(), llvm.types.ptr()
        mod, fn = _fn(ctx, i32, [ptr, i32, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            cx = I.atomic_cmpxchg(
                fn.arg(0),
                fn.arg(1),
                fn.arg(2),
                llvm.ir.AtomicOrdering.SequentiallyConsistent,
                llvm.ir.AtomicOrdering.Monotonic,
                name="cx",
            )
            I.ret(I.extract_value(cx, 0, "old"))
        p = str(mod)
        assert "%cx = cmpxchg ptr %0, i32 %1, i32 %2 seq_cst monotonic, align 4" in p
        del b, fn, mod
    assert_no_leaks()


def test_atomic_cmpxchg_invalid_orderings_raise():
    # Release/AcquireRelease/Unordered/NotAtomic are invalid failure orderings,
    # and Unordered/NotAtomic invalid success orderings; the binding checks
    # before the AtomicCmpXchgInst ctor would assert, so these raise cleanly
    # instead of aborting the interpreter.
    with llvm.ir.Context() as ctx:
        i32, ptr = llvm.types.i32(), llvm.types.ptr()
        mod, fn = _fn(ctx, i32, [ptr, i32, i32])
        b = IRBuilder(ctx)
        AO = llvm.ir.AtomicOrdering
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            with pytest.raises(ValueError, match="failure ordering"):
                I.atomic_cmpxchg(
                    fn.arg(0), fn.arg(1), fn.arg(2), AO.SequentiallyConsistent, AO.Release
                )
            with pytest.raises(ValueError, match="success ordering"):
                I.atomic_cmpxchg(
                    fn.arg(0), fn.arg(1), fn.arg(2), AO.Unordered, AO.Monotonic
                )
            I.ret(I.i32_const(0))
        del b, fn, mod
    assert_no_leaks()


def test_call_intrinsic_free_function():
    with llvm.ir.Context() as ctx:
        f32 = llvm.types.f32()
        mod, fn = _fn(ctx, f32, [f32])
        b = IRBuilder(ctx)
        sqrt_id = llvm.intrinsics.lookup_intrinsic_id("llvm.sqrt")
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.ret(I.call_intrinsic(sqrt_id, [f32], [fn.arg(0)], "ci"))
        assert "%ci = call float @llvm.sqrt.f32(float %0)" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_python_number_operands_become_constants():
    # A Python number in either operand slot becomes a constant of the sibling
    # SSA operand's type -- integer sibling -> int constant, float -> fp.
    with llvm.ir.Context() as ctx:
        i32, f32 = llvm.types.i32(), llvm.types.f32()
        mod, fn = _fn(ctx, i32, [i32, f32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, y = fn.arg(0), fn.arg(1)
            I.add(x, 1, "r")  # literal rhs
            I.sub(10, x, "l")  # literal lhs
            I.and_(x, -1, "m")  # negative literal (signed)
            I.fadd(y, 2, "f")  # int literal coerced to float type
            I.fmul(y, 1.5, "g")
            I.ret(x)
        p = str(mod)
        assert "%r = add i32 %0, 1" in p
        assert "%l = sub i32 10, %0" in p
        assert "%m = and i32 %0, -1" in p
        assert "%f = fadd float %1, 2.000000e+00" in p
        assert "%g = fmul float %1, 1.500000e+00" in p
        del b, fn, mod
    assert_no_leaks()


def test_python_number_operands_in_compares_and_select():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        f32 = llvm.types.f32()
        mod, fn = _fn(ctx, i32, [i32, f32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, y = fn.arg(0), fn.arg(1)
            c = I.icmp(llvm.ir.CmpPredicate.SLT, x, 5, "c")
            I.fcmp(llvm.ir.CmpPredicate.OGT, y, 0, "fc")
            I.ret(I.select(c, x, 0, "sel"))  # literal false value
        p = str(mod)
        assert "%c = icmp slt i32 %0, 5" in p
        assert "%fc = fcmp ogt float %1, 0.000000e+00" in p
        assert "%sel = select i1 %c, i32 %0, i32 0" in p
        del b, fn, mod
    assert_no_leaks()


def test_python_int_gep_and_vector_indices():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        vec = llvm.types.vector(i32, 4)
        mod, fn = _fn(ctx, i32, [llvm.types.ptr(), vec])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            p, v = fn.arg(0), fn.arg(1)
            I.gep(i32, p, [2], "g")  # int index -> i32 constant
            I.extract_element(v, 1, "ee")  # int index -> i32 constant
            I.insert_element(v, I.i32_const(9), 0, "ie")
            I.ret(I.i32_const(0))
        p_ = str(mod)
        assert "%g = getelementptr i32, ptr %0, i32 2" in p_
        assert "%ee = extractelement <4 x i32> %1, i32 1" in p_
        assert "%ie = insertelement <4 x i32> %1, i32 9, i32 0" in p_
        del b, fn, mod
    assert_no_leaks()


def test_python_number_ret_and_call_args():
    with llvm.ir.Context() as ctx:
        i32, f32 = llvm.types.i32(), llvm.types.f32()
        mod = llvm.ir.Module("m", ctx)
        callee = llvm.ir.Function.create(
            llvm.types.function(i32, [i32, f32]), "callee", mod
        )
        fn = llvm.ir.Function.create(llvm.types.function(i32, []), "f", mod)
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            # call args coerced against the callee's parameter types
            I.call(callee, [3, 1.5], "c")
            I.ret(0)  # coerced against the function's return type
        p = str(mod)
        assert "%c = call i32 @callee(i32 3, float 1.500000e+00)" in p
        assert "ret i32 0" in p
        del b, fn, mod
    assert_no_leaks()


def test_call_varargs_prebuilt_value_arg_passes_through():
    # A vararg operand given as an already-built Value (beyond the callee's
    # fixed params) passes through untouched; the fixed number arg is coerced.
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod = llvm.ir.Module("m", ctx)
        callee = llvm.ir.Function.create(
            llvm.types.function(i32, [i32], var_arg=True), "callee", mod
        )
        fn = llvm.ir.Function.create(llvm.types.function(i32, []), "f", mod)
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            I.ret(I.call(callee, [1, I.i32_const(2)], "c"))
        assert "%c = call i32 (i32, ...) @callee(i32 1, i32 2)" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_call_varargs_number_beyond_fixed_params_raises():
    # A number in a vararg slot has no declared type to coerce against.
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod = llvm.ir.Module("m", ctx)
        callee = llvm.ir.Function.create(
            llvm.types.function(i32, [i32], var_arg=True), "callee", mod
        )
        fn = llvm.ir.Function.create(llvm.types.function(i32, []), "f", mod)
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            with pytest.raises(TypeError, match="beyond the callee's 1 fixed"):
                I.call(callee, [1, 2], "c")
            I.ret(I.i32_const(0))
        del b, fn, mod
    assert_no_leaks()


def test_unsigned_range_literal_coerces_by_bit_pattern():
    # A non-negative literal that overflows the signed range but fits unsigned
    # (a bitmask like 0xFF into i8, or a >2**31 value into i32) is accepted and
    # printed by its signed 2's-complement form.
    with llvm.ir.Context() as ctx:
        i8, i32 = llvm.types.i8(), llvm.types.i32()
        mod, fn = _fn(ctx, i8, [i8, i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            x, y = fn.arg(0), fn.arg(1)
            I.and_(x, 0xFF, "m")  # 255 -> i8 all-ones == -1
            I.udiv(y, 4000000000, "d")  # > 2**31, fits u32
            I.ret(x)
        p = str(mod)
        assert "%m = and i8 %0, -1" in p
        assert "%d = udiv i32 %1, -294967296" in p
        del b, fn, mod
    assert_no_leaks()


def test_out_of_range_literal_raises_valueerror():
    # A literal that fits neither signed nor unsigned range surfaces as a
    # catchable ValueError (const_int guards the LLVM assert), not a crash.
    with llvm.ir.Context() as ctx:
        i8 = llvm.types.i8()
        mod, fn = _fn(ctx, i8, [i8])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            with pytest.raises(ValueError, match="does not fit"):
                I.add(fn.arg(0), 5000)  # 5000 fits neither i8 signed nor unsigned
            I.ret(fn.arg(0))
        del b, fn, mod
    assert_no_leaks()


def test_float_literal_into_integer_op_raises():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            with pytest.raises(TypeError, match="float 1.5 as an integer"):
                I.add(fn.arg(0), 1.5)
            I.ret(fn.arg(0))
        del b, fn, mod
    assert_no_leaks()


def test_number_coercion_on_explicit_builder_without_context():
    # Coercion must work off the explicit builder= too, with no active context.
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32])
        entry = fn.append_basic_block("entry")
        b = IRBuilder(ctx)
        b.set_insert_point(entry)
        s = I.add(fn.arg(0), 1, "s", builder=b)
        I.ret(s, builder=b)
        assert "%s = add i32 %0, 1" in str(mod)
        del b, fn, mod
    assert_no_leaks()


def test_ret_number_into_void_function_raises():
    # ret(<number>) routes through the return type; a void return has no
    # constant form.
    with llvm.ir.Context() as ctx:
        mod, fn = _fn(ctx, llvm.types.void(), [])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            with pytest.raises(TypeError, match="cannot build a constant"):
                I.ret(0)
            I.ret()
        del b, fn, mod
    assert_no_leaks()


def test_two_number_operands_raise():
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        mod, fn = _fn(ctx, i32, [i32])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            with pytest.raises(TypeError, match="at least one operand"):
                I.add(1, 2)
            I.ret(fn.arg(0))
        del b, fn, mod
    assert_no_leaks()


def test_number_against_non_scalar_type_raises():
    # A number opposite a vector operand has no unambiguous constant form.
    with llvm.ir.Context() as ctx:
        i32 = llvm.types.i32()
        vec = llvm.types.vector(i32, 4)
        mod, fn = _fn(ctx, vec, [vec])
        b = IRBuilder(ctx)
        with InsertPoint(fn.append_basic_block("entry"), builder=b):
            with pytest.raises(TypeError, match="cannot build a constant"):
                I.add(fn.arg(0), 1)
            I.ret(fn.arg(0))
        del b, fn, mod
    assert_no_leaks()
