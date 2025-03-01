#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2024.

from pathlib import Path

import pytest
from eudsl_tblgen import (
    RecordKeeper,
    collect_all_defs,
    collect_all_attr_or_type_defs,
)
from eudsl_tblgen.mlir import (
    emit_c_attr_or_type_builder,
    emit_c_attr_or_type_field_getter,
    emit_attr_or_type_nanobind_class,
    emit_decls_defns_nbclasses,
    CClassKind,
)


@pytest.fixture(scope="function")
def record_keeper_triton_gpu_attrs():
    here = Path(__file__).parent
    return RecordKeeper().parse_td(
        str(here / "td" / "TritonGPUAttrDefs.td"), [str(here / "td")]
    )


@pytest.fixture(scope="function")
def record_keeper_triton_gpu_types():
    here = Path(__file__).parent
    return RecordKeeper().parse_td(
        str(here / "td" / "TritonGPUTypes.td"), [str(here / "td")]
    )


def test_attrs(record_keeper_triton_gpu_attrs):
    all_defs = collect_all_attr_or_type_defs(
        collect_all_defs(record_keeper_triton_gpu_attrs)
    )
    decls, defns, nbclasses = emit_decls_defns_nbclasses(CClassKind.ATTRIBUTE, all_defs)

    print()
    for d in decls:
        print(d)
    for d in defns:
        print(d)
    for hdecl, hdefn, n in nbclasses:
        for h in hdecl:
            print(h)
        for h in hdefn:
            print(h)
        print(n)


def test_types(record_keeper_triton_gpu_types):
    all_defs = collect_all_attr_or_type_defs(
        collect_all_defs(record_keeper_triton_gpu_types)
    )
    decls, defns, nbclasses = emit_decls_defns_nbclasses(CClassKind.TYPE, all_defs)

    print()
    for d in decls:
        print(d)
    for d in defns:
        print(d)
    for hdecl, hdefn, n in nbclasses:
        for h in hdecl:
            print(h)
        for h in hdefn:
            print(h)
        print(n)


@pytest.fixture(scope="function")
def record_keeper_builtin_attributes():
    here = Path(__file__).parent
    return RecordKeeper().parse_td(
        str(here / "td" / "BuiltinLocationAttributes.td"), [str(here / "td")]
    )


def test_builtin_attributes(record_keeper_builtin_attributes):
    all_defs = collect_all_attr_or_type_defs(
        collect_all_defs(record_keeper_builtin_attributes)
    )
    decls, defns, nbclasses = emit_decls_defns_nbclasses(CClassKind.ATTRIBUTE, all_defs)

    print()
    for d in decls:
        print(d)
    for hdecl, _hdefn, n in nbclasses:
        for h in hdecl:
            print(h)

    for d in defns:
        print(d)

    for _hdecl, hdefn, n in nbclasses:
        for h in hdefn:
            print(h)
        print(n)
