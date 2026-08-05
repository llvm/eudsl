#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import gc
from textwrap import dedent

import llvm


def test_symbol_collision():
    # eudsl-tblgen is a separate extension in a different nanobind domain;
    # importing both must not clash.
    import eudsl_tblgen  # noqa: F401

    import llvm  # noqa: F401


def test_smoke():
    src = dedent(
        """\
        declare i32 @foo()
        declare i32 @bar()
        define i32 @entry(i32 %argc) {
        entry:
          %and = and i32 %argc, 1
          %tobool = icmp eq i32 %and, 0
          br i1 %tobool, label %if.end, label %if.then
        if.then:
          %call = tail call i32 @foo()
          br label %return
        if.end:
          %call1 = tail call i32 @bar()
          br label %return
        return:
          %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.end ]
          ret i32 %retval.0
        }
        """
    )
    with llvm.Context() as ctx:
        mod = llvm.parse_assembly(src, ctx, "test_smoke")
        assert mod.name == "test_smoke"
        printed = str(mod)
        assert "define i32 @entry(i32 %argc)" in printed
        assert "phi i32" in printed
        del mod
    gc.collect()
    assert llvm.Context._get_live_count() == 0
