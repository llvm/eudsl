#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

from . import (
    AtomicOrdering,
    AtomicRMWBinOp,
    BasicBlockRef,
    IntPredicate,
    Opcode,
    OperandBundleRef,
    RealPredicate,
    TypeRef,
    ValueRef,
    build_a_shr,
    build_add,
    build_addr_space_cast,
    build_aggregate_ret,
    build_alloca,
    build_and,
    build_array_alloca,
    build_array_malloc,
    build_atomic_cmp_xchg,
    build_atomic_cmp_xchg_sync_scope,
    build_atomic_rmw,
    build_atomic_rmw_sync_scope,
    build_bin_op,
    build_bit_cast,
    build_br,
    build_call2,
    build_call_br,
    build_call_with_operand_bundles,
    build_cast,
    build_catch_pad,
    build_catch_ret,
    build_catch_switch,
    build_cleanup_pad,
    build_cleanup_ret,
    build_cond_br,
    build_exact_s_div,
    build_exact_u_div,
    build_extract_element,
    build_extract_value,
    build_f_add,
    build_f_cmp,
    build_f_div,
    build_f_mul,
    build_f_neg,
    build_f_rem,
    build_f_sub,
    build_fence,
    build_fence_sync_scope,
    build_fp_cast,
    build_fp_ext,
    build_fp_to_si,
    build_fp_to_ui,
    build_fp_trunc,
    build_free,
    build_freeze,
    build_gep2,
    build_gep_with_no_wrap_flags,
    build_global_string,
    build_global_string_ptr,
    build_i_cmp,
    build_in_bounds_gep2,
    build_indirect_br,
    build_insert_element,
    build_insert_value,
    build_int_cast2,
    build_int_to_ptr,
    build_invoke2,
    build_invoke_with_operand_bundles,
    build_is_not_null,
    build_is_null,
    build_l_shr,
    build_landing_pad,
    build_load2,
    build_malloc,
    build_mem_cpy,
    build_mem_move,
    build_mul,
    build_neg,
    build_not,
    build_nsw_add,
    build_nsw_mul,
    build_nsw_neg,
    build_nsw_sub,
    build_nuw_add,
    build_nuw_mul,
    build_nuw_sub,
    build_or,
    build_phi,
    build_pointer_cast,
    build_ptr_diff2,
    build_ptr_to_int,
    build_resume,
    build_ret,
    build_ret_void,
    build_s_div,
    build_s_ext,
    build_s_ext_or_bit_cast,
    build_s_rem,
    build_select,
    build_shl,
    build_shuffle_vector,
    build_si_to_fp,
    build_store,
    build_struct_gep2,
    build_sub,
    build_switch,
    build_trunc,
    build_trunc_or_bit_cast,
    build_u_div,
    build_u_rem,
    build_ui_to_fp,
    build_unreachable,
    build_va_arg,
    build_xor,
    build_z_ext,
    build_z_ext_or_bit_cast,
)
from .context import current_context


def ashr(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_a_shr(current_context().builder, lhs, rhs, name)


def add(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_add(current_context().builder, lhs, rhs, name)


def addr_space_cast(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_addr_space_cast(current_context().builder, val, dest_ty, name)


def aggregate_ret(ret_vals: ValueRef, n: int):
    return build_aggregate_ret(current_context().builder, ret_vals, n)


def alloca(ty: TypeRef, name: str = ""):
    return build_alloca(current_context().builder, ty, name)


def and_(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_and(current_context().builder, lhs, rhs, name)


def array_alloca(ty: TypeRef, val: ValueRef, name: str = ""):
    return build_array_alloca(current_context().builder, ty, val, name)


def array_malloc(ty: TypeRef, val: ValueRef, name: str = ""):
    return build_array_malloc(current_context().builder, ty, val, name)


def atomic_cmp_xchg(
    ptr: ValueRef,
    cmp: ValueRef,
    new: ValueRef,
    success_ordering: AtomicOrdering,
    failure_ordering: AtomicOrdering,
    single_thread: int,
):
    return build_atomic_cmp_xchg(
        current_context().builder,
        ptr,
        cmp,
        new,
        success_ordering,
        failure_ordering,
        single_thread,
    )


def atomic_cmp_xchg_sync_scope(
    ptr: ValueRef,
    cmp: ValueRef,
    new: ValueRef,
    success_ordering: AtomicOrdering,
    failure_ordering: AtomicOrdering,
    ssid: int,
):
    return build_atomic_cmp_xchg_sync_scope(
        current_context().builder,
        ptr,
        cmp,
        new,
        success_ordering,
        failure_ordering,
        ssid,
    )


def atomic_rmw(
    op: AtomicRMWBinOp,
    ptr: ValueRef,
    val: ValueRef,
    ordering: AtomicOrdering,
    single_thread: int,
):
    return build_atomic_rmw(
        current_context().builder, op, ptr, val, ordering, single_thread
    )


def atomic_rmw_sync_scope(
    op: AtomicRMWBinOp,
    ptr: ValueRef,
    val: ValueRef,
    ordering: AtomicOrdering,
    ssid: int,
):
    return build_atomic_rmw_sync_scope(
        current_context().builder, op, ptr, val, ordering, ssid
    )


def bin_op(op: Opcode, lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_bin_op(current_context().builder, op, lhs, rhs, name)


def bit_cast(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_bit_cast(current_context().builder, val, dest_ty, name)


def br(dest: BasicBlockRef):
    return build_br(current_context().builder, dest)


def call(param_1: TypeRef, fn: ValueRef, args: ValueRef, num_args: int, name: str = ""):
    return build_call2(current_context().builder, param_1, fn, args, num_args, name)


def call_br(
    ty: TypeRef,
    fn: ValueRef,
    default_dest: BasicBlockRef,
    indirect_dests: BasicBlockRef,
    num_indirect_dests: int,
    args: ValueRef,
    num_args: int,
    bundles: OperandBundleRef,
    num_bundles: int,
    name: str = "",
):
    return build_call_br(
        current_context().builder,
        ty,
        fn,
        default_dest,
        indirect_dests,
        num_indirect_dests,
        args,
        num_args,
        bundles,
        num_bundles,
        name,
    )


def call_with_operand_bundles(
    param_1: TypeRef,
    fn: ValueRef,
    args: ValueRef,
    num_args: int,
    bundles: OperandBundleRef,
    num_bundles: int,
    name: str = "",
):
    return build_call_with_operand_bundles(
        current_context().builder,
        param_1,
        fn,
        args,
        num_args,
        bundles,
        num_bundles,
        name,
    )


def cast(op: Opcode, val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_cast(current_context().builder, op, val, dest_ty, name)


def catch_pad(parent_pad: ValueRef, args: ValueRef, num_args: int, name: str = ""):
    return build_catch_pad(current_context().builder, parent_pad, args, num_args, name)


def catch_ret(catch_pad: ValueRef, bb: BasicBlockRef):
    return build_catch_ret(current_context().builder, catch_pad, bb)


def catch_switch(
    parent_pad: ValueRef, unwind_bb: BasicBlockRef, num_handlers: int, name: str = ""
):
    return build_catch_switch(
        current_context().builder, parent_pad, unwind_bb, num_handlers, name
    )


def cleanup_pad(parent_pad: ValueRef, args: ValueRef, num_args: int, name: str = ""):
    return build_cleanup_pad(
        current_context().builder, parent_pad, args, num_args, name
    )


def cleanup_ret(catch_pad: ValueRef, bb: BasicBlockRef):
    return build_cleanup_ret(current_context().builder, catch_pad, bb)


def cond_br(if_: ValueRef, then: BasicBlockRef, else_: BasicBlockRef):
    return build_cond_br(current_context().builder, if_, then, else_)


def exact_sdiv(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_exact_s_div(current_context().builder, lhs, rhs, name)


def exact_udiv(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_exact_u_div(current_context().builder, lhs, rhs, name)


def extract_element(vec_val: ValueRef, index: ValueRef, name: str = ""):
    return build_extract_element(current_context().builder, vec_val, index, name)


def extract_value(agg_val: ValueRef, index: int, name: str = ""):
    return build_extract_value(current_context().builder, agg_val, index, name)


def fadd(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_f_add(current_context().builder, lhs, rhs, name)


def fcmp(op: RealPredicate, lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_f_cmp(current_context().builder, op, lhs, rhs, name)


def fdiv(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_f_div(current_context().builder, lhs, rhs, name)


def fmul(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_f_mul(current_context().builder, lhs, rhs, name)


def fneg(v: ValueRef, name: str = ""):
    return build_f_neg(current_context().builder, v, name)


def frem(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_f_rem(current_context().builder, lhs, rhs, name)


def fsub(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_f_sub(current_context().builder, lhs, rhs, name)


def fence(ordering: AtomicOrdering, single_thread: int, name: str = ""):
    return build_fence(current_context().builder, ordering, single_thread, name)


def fence_sync_scope(ordering: AtomicOrdering, ssid: int, name: str = ""):
    return build_fence_sync_scope(current_context().builder, ordering, ssid, name)


def fp_cast(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_fp_cast(current_context().builder, val, dest_ty, name)


def fp_ext(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_fp_ext(current_context().builder, val, dest_ty, name)


def fp_to_si(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_fp_to_si(current_context().builder, val, dest_ty, name)


def fp_to_ui(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_fp_to_ui(current_context().builder, val, dest_ty, name)


def fp_trunc(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_fp_trunc(current_context().builder, val, dest_ty, name)


def free(pointer_val: ValueRef):
    return build_free(current_context().builder, pointer_val)


def freeze(val: ValueRef, name: str = ""):
    return build_freeze(current_context().builder, val, name)


def gep(
    ty: TypeRef, pointer: ValueRef, indices: ValueRef, num_indices: int, name: str = ""
):
    return build_gep2(
        current_context().builder, ty, pointer, indices, num_indices, name
    )


def gep_with_no_wrap_flags(
    ty: TypeRef,
    pointer: ValueRef,
    indices: ValueRef,
    num_indices: int,
    name: str,
    no_wrap_flags: int,
):
    return build_gep_with_no_wrap_flags(
        current_context().builder,
        ty,
        pointer,
        indices,
        num_indices,
        name,
        no_wrap_flags,
    )


def global_string(str: str, name: str = ""):
    return build_global_string(current_context().builder, str, name)


def global_string_ptr(str: str, name: str = ""):
    return build_global_string_ptr(current_context().builder, str, name)


def icmp(op: IntPredicate, lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_i_cmp(current_context().builder, op, lhs, rhs, name)


def in_bounds_gep2(
    ty: TypeRef, pointer: ValueRef, indices: ValueRef, num_indices: int, name: str = ""
):
    return build_in_bounds_gep2(
        current_context().builder, ty, pointer, indices, num_indices, name
    )


def indirect_br(addr: ValueRef, num_dests: int):
    return build_indirect_br(current_context().builder, addr, num_dests)


def insert_element(
    vec_val: ValueRef, elt_val: ValueRef, index: ValueRef, name: str = ""
):
    return build_insert_element(
        current_context().builder, vec_val, elt_val, index, name
    )


def insert_value(agg_val: ValueRef, elt_val: ValueRef, index: int, name: str = ""):
    return build_insert_value(current_context().builder, agg_val, elt_val, index, name)


def int_cast(val: ValueRef, dest_ty: TypeRef, is_signed: int, name: str = ""):
    return build_int_cast2(current_context().builder, val, dest_ty, is_signed, name)


def int_to_ptr(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_int_to_ptr(current_context().builder, val, dest_ty, name)


def invoke(
    ty: TypeRef,
    fn: ValueRef,
    args: ValueRef,
    num_args: int,
    then: BasicBlockRef,
    catch: BasicBlockRef,
    name: str = "",
):
    return build_invoke2(
        current_context().builder, ty, fn, args, num_args, then, catch, name
    )


def invoke_with_operand_bundles(
    ty: TypeRef,
    fn: ValueRef,
    args: ValueRef,
    num_args: int,
    then: BasicBlockRef,
    catch: BasicBlockRef,
    bundles: OperandBundleRef,
    num_bundles: int,
    name: str = "",
):
    return build_invoke_with_operand_bundles(
        current_context().builder,
        ty,
        fn,
        args,
        num_args,
        then,
        catch,
        bundles,
        num_bundles,
        name,
    )


def is_not_null(val: ValueRef, name: str = ""):
    return build_is_not_null(current_context().builder, val, name)


def is_null(val: ValueRef, name: str = ""):
    return build_is_null(current_context().builder, val, name)


def lshr(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_l_shr(current_context().builder, lhs, rhs, name)


def landing_pad(ty: TypeRef, pers_fn: ValueRef, num_clauses: int, name: str = ""):
    return build_landing_pad(current_context().builder, ty, pers_fn, num_clauses, name)


def load(ty: TypeRef, pointer_val: ValueRef, name: str = ""):
    return build_load2(current_context().builder, ty, pointer_val, name)


def malloc(ty: TypeRef, name: str = ""):
    return build_malloc(current_context().builder, ty, name)


def mem_cpy(
    dst: ValueRef,
    dst_align: int,
    src: ValueRef,
    src_align: int,
    size: ValueRef,
) -> ValueRef:
    return build_mem_cpy(
        current_context().builder,
        dst,
        dst_align,
        src,
        src_align,
        size,
    )


def mem_move(
    dst: ValueRef,
    dst_align: int,
    src: ValueRef,
    src_align: int,
    size: ValueRef,
) -> ValueRef:
    return build_mem_move(
        current_context().builder,
        dst,
        dst_align,
        src,
        src_align,
        size,
    )


def build_mem_set(ptr: ValueRef, val: ValueRef, len: ValueRef, align: int) -> ValueRef:
    return build_mem_set(current_context().builder, ptr, val, len, align)


def mul(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_mul(current_context().builder, lhs, rhs, name)


def neg(v: ValueRef, name: str = ""):
    return build_neg(current_context().builder, v, name)


def not_(v: ValueRef, name: str = ""):
    return build_not(current_context().builder, v, name)


def nsw_add(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_nsw_add(current_context().builder, lhs, rhs, name)


def nsw_mul(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_nsw_mul(current_context().builder, lhs, rhs, name)


def nsw_neg(v: ValueRef, name: str = ""):
    return build_nsw_neg(current_context().builder, v, name)


def nsw_sub(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_nsw_sub(current_context().builder, lhs, rhs, name)


def nuw_add(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_nuw_add(current_context().builder, lhs, rhs, name)


def nuw_mul(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_nuw_mul(current_context().builder, lhs, rhs, name)


def nuw_sub(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_nuw_sub(current_context().builder, lhs, rhs, name)


def or_(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_or(current_context().builder, lhs, rhs, name)


def phi(ty: TypeRef, name: str = ""):
    return build_phi(current_context().builder, ty, name)


def pointer_cast(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_pointer_cast(current_context().builder, val, dest_ty, name)


def ptr_diff2(elem_ty: TypeRef, lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_ptr_diff2(current_context().builder, elem_ty, lhs, rhs, name)


def ptr_to_int(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_ptr_to_int(current_context().builder, val, dest_ty, name)


def resume(exn: ValueRef):
    return build_resume(current_context().builder, exn)


def ret(v: ValueRef):
    return build_ret(current_context().builder, v)


def ret_void():
    return build_ret_void(current_context().builder)


def sdiv(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_s_div(current_context().builder, lhs, rhs, name)


def sext(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_s_ext(current_context().builder, val, dest_ty, name)


def sext_or_bit_cast(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_s_ext_or_bit_cast(current_context().builder, val, dest_ty, name)


def srem(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_s_rem(current_context().builder, lhs, rhs, name)


def select(if_: ValueRef, then: ValueRef, else_: ValueRef, name: str = ""):
    return build_select(current_context().builder, if_, then, else_, name)


def shl(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_shl(current_context().builder, lhs, rhs, name)


def shuffle_vector(v1: ValueRef, v2: ValueRef, mask: ValueRef, name: str = ""):
    return build_shuffle_vector(current_context().builder, v1, v2, mask, name)


def si_to_fp(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_si_to_fp(current_context().builder, val, dest_ty, name)


def store(val: ValueRef, ptr: ValueRef):
    return build_store(current_context().builder, val, ptr)


def struct_gep(ty: TypeRef, pointer: ValueRef, idx: int, name: str = ""):
    return build_struct_gep2(current_context().builder, ty, pointer, idx, name)


def sub(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_sub(current_context().builder, lhs, rhs, name)


def switch(v: ValueRef, else_: BasicBlockRef, num_cases: int):
    return build_switch(current_context().builder, v, else_, num_cases)


def trunc(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_trunc(current_context().builder, val, dest_ty, name)


def trunc_or_bit_cast(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_trunc_or_bit_cast(current_context().builder, val, dest_ty, name)


def udiv(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_u_div(current_context().builder, lhs, rhs, name)


def urem(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_u_rem(current_context().builder, lhs, rhs, name)


def ui_to_fp(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_ui_to_fp(current_context().builder, val, dest_ty, name)


def unreachable():
    return build_unreachable(current_context().builder)


def va_arg(list: ValueRef, ty: TypeRef, name: str = ""):
    return build_va_arg(current_context().builder, list, ty, name)


def xor(lhs: ValueRef, rhs: ValueRef, name: str = ""):
    return build_xor(current_context().builder, lhs, rhs, name)


def zext(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_z_ext(current_context().builder, val, dest_ty, name)


def zext_or_bit_cast(val: ValueRef, dest_ty: TypeRef, name: str = ""):
    return build_z_ext_or_bit_cast(current_context().builder, val, dest_ty, name)
