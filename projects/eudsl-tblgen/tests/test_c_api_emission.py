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
    emit_decls_defns_nbclasses,
    CClassKind,
)


@pytest.fixture(scope="function")
def record_keeper_triton_gpu_attrs():
    here = Path(__file__).parent
    rk = RecordKeeper()
    rk.parse_td(str(here / "td" / "TritonGPUAttrDefs.td"), [str(here / "td")])
    return rk


@pytest.fixture(scope="function")
def record_keeper_triton_gpu_types():
    here = Path(__file__).parent
    rk = RecordKeeper()
    rk.parse_td(str(here / "td" / "TritonGPUTypes.td"), [str(here / "td")])
    return rk


def test_attrs(record_keeper_triton_gpu_attrs):
    all_defs = collect_all_attr_or_type_defs(
        collect_all_defs(record_keeper_triton_gpu_attrs)
    )
    decls, defns, nbclasses = emit_decls_defns_nbclasses(
        CClassKind.ATTRIBUTE,
        all_defs,
        exclude={"BlockedEncodingAttr", "SliceEncodingAttr"},
    )

    dump_dir = Path(__file__).parent

    with open(f"{dump_dir}/TritonGPUAttrDefs_MlirAttribute_decls.h.inc", "w") as f:
        for d in decls:
            print(d, file=f)
    with open(f"{dump_dir}/TritonGPUAttrDefs_MlirAttribute_defns.cpp.inc", "w") as f:
        for d in defns:
            print(d, file=f)
    for hdecl, hdefn, n in nbclasses:
        with open(f"{dump_dir}/TritonGPUAttrDefs_MlirAttribute_decls.h.inc", "a") as f:
            for h in hdecl:
                print(h, file=f)
        with open(
            f"{dump_dir}/TritonGPUAttrDefs_MlirAttribute_defns.cpp.inc", "a"
        ) as f:
            for h in hdefn:
                print(h, file=f)
    with open(
        f"{dump_dir}/TritonGPUAttrDefs_MlirAttribute_nbclasses.cpp.inc", "w"
    ) as f:
        for *_, n in nbclasses:
            print(n, file=f)


def test_types(record_keeper_triton_gpu_types):
    all_defs = collect_all_attr_or_type_defs(
        collect_all_defs(record_keeper_triton_gpu_types)
    )
    decls, defns, nbclasses = emit_decls_defns_nbclasses(CClassKind.TYPE, all_defs)
    dump_dir = Path(__file__).parent

    with open(f"{dump_dir}/TritonGPUTypes_MlirType_decls.h.inc", "w") as f:
        for d in decls:
            print(d, file=f)
    with open(f"{dump_dir}/TritonGPUTypes_MlirType_defns.cpp.inc", "w") as f:
        for d in defns:
            print(d, file=f)
    for hdecl, hdefn, n in nbclasses:
        with open(f"{dump_dir}/TritonGPUTypes_MlirType_decls.h.inc", "a") as f:
            for h in hdecl:
                print(h, file=f)
        with open(f"{dump_dir}/TritonGPUTypes_MlirType_defns.cpp.inc", "a") as f:
            for h in hdefn:
                print(h, file=f)
    with open(f"{dump_dir}/TritonGPUTypes_MlirType_nbclasses.cpp.inc", "w") as f:
        for *_, n in nbclasses:
            print(n, file=f)
