#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from textwrap import dedent

import gc
import weakref

import pytest

import llvm
from llvm.testing import assert_no_leaks

_SRC = dedent("""\
    define i32 @f(i32 %x) {
    entry:
      ret i32 %x
    }
    """)

_SRC2 = dedent("""\
    define i32 @f(i32 %x) {
    entry:
      ret i32 %x
    }
    define i32 @g(i32 %y) {
    entry:
      ret i32 %y
    }
    """)

# One defined function plus a bare declaration -- the function-pass adaptor
# should visit only the defined one.
_SRC_DECL = dedent("""\
    declare i32 @ext(i32)
    define i32 @f(i32 %x) {
    entry:
      %r = call i32 @ext(i32 %x)
      ret i32 %r
    }
    """)


def test_module_pass_is_invoked_once_with_the_module():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")
        seen = []
        llvm.passmanager.run_python_pass_on_module(mod, seen.append)
        assert len(seen) == 1
        # The callback receives the very same Module object, not a fresh wrapper.
        assert seen[0] is mod
        del mod
    assert_no_leaks()


def test_module_pass_can_mutate_ir():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        def rename(m):
            m.get_function("f").name = "g"

        llvm.passmanager.run_python_pass_on_module(mod, rename)
        assert "@g(" in str(mod)
        assert "@f(" not in str(mod)
        del mod
    assert_no_leaks()


def test_module_pass_can_mutate_ir_and_report_changed():
    # A callback that returns a truthy value drives the "IR changed" branch
    # (PreservedAnalyses::none(), invalidating analyses). That invalidation is
    # not observable across independent runs, so this pins the observable part:
    # a truthy return is accepted, the pass runs, and its mutation persists.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        def rename_changed(m):
            m.get_function("f").name = "g"
            return True  # truthy -> reported changed

        llvm.passmanager.run_python_pass_on_module(mod, rename_changed)
        assert "@g(" in str(mod)
        assert "@f(" not in str(mod)
        del mod
    assert_no_leaks()


def test_module_pass_forwards_tuning_and_flags():
    # The tuning/debug/verify_each arguments are accepted and threaded through
    # to the pipeline environment; the pass still runs and can mutate the IR.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        def rename(m):
            m.get_function("f").name = "g"

        tuning = llvm.passmanager.PipelineTuningOptions()
        tuning.slp_vectorization = False
        llvm.passmanager.run_python_pass_on_module(
            mod, rename, tuning=tuning, debug=False, verify_each=True
        )
        assert "@g(" in str(mod)
        del mod
    assert_no_leaks()


def test_exception_in_pass_propagates_and_leaves_module_usable():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        def boom(m):
            raise ValueError("boom in pass")

        with pytest.raises(ValueError, match="boom in pass"):
            llvm.passmanager.run_python_pass_on_module(mod, boom)
        # The trampoline captured the exception and re-raised it after the
        # pipeline returned; the module is still usable afterward.
        assert "@f(" in str(mod)
        del mod
    assert_no_leaks()


def test_exception_from_unboolable_return_propagates():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        class Unboolable:
            def __bool__(self):
                raise ValueError("no truth value")

        # A truthiness that raises exercises the PyObject_IsTrue(<0) branch,
        # which must propagate rather than be silently coerced or lost. The
        # trampoline wraps it with a descriptive message and chains the
        # original ValueError as the cause.
        with pytest.raises(ValueError, match="truthiness") as excinfo:
            llvm.passmanager.run_python_pass_on_module(mod, lambda m: Unboolable())
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "no truth value" in str(excinfo.value.__cause__)
        assert "@f(" in str(mod)
        del mod
    assert_no_leaks()


def test_callback_not_retained_after_exception():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        class Boom:
            def __call__(self, m):
                raise ValueError("boom")

        cb = Boom()
        ref = weakref.ref(cb)
        with pytest.raises(ValueError):
            llvm.passmanager.run_python_pass_on_module(mod, cb)
        del cb
        gc.collect()
        # The trampoline released the callable even on the error path. (This
        # relies on the raised exception -- whose traceback holds Boom.__call__'s
        # frame, and thus cb -- not outliving the pytest.raises block above.)
        assert ref() is None
        del mod
    assert_no_leaks()


def test_none_returning_pass_runs_repeatedly_without_error():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC, ctx, "m")

        def noop(m):
            return None  # falsy -> reported unchanged

        # Each call builds its own pipeline, so this pins that a None-returning
        # pass runs repeatedly without error (not analysis preservation, which
        # is not observable across independent runs).
        llvm.passmanager.run_python_pass_on_module(mod, noop)
        llvm.passmanager.run_python_pass_on_module(mod, noop)
        assert "@f(" in str(mod)
        del mod
    assert_no_leaks()


def test_function_pass_runs_once_per_function():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC2, ctx, "m")
        names = []
        llvm.passmanager.run_python_pass_on_function(
            mod, lambda fn: names.append(fn.name)
        )
        assert sorted(names) == ["f", "g"]
        del mod
    assert_no_leaks()


def test_function_pass_can_mutate_each_function():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC2, ctx, "m")

        def suffix(fn):
            fn.name = fn.name + "_x"

        llvm.passmanager.run_python_pass_on_function(mod, suffix)
        assert "@f_x(" in str(mod)
        assert "@g_x(" in str(mod)
        del mod
    assert_no_leaks()


def test_exception_in_function_pass_propagates():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC2, ctx, "m")
        seen = []

        def boom(fn):
            seen.append(fn.name)
            raise ValueError("boom in " + fn.name)

        # Exceptions unwind through the adaptor's extra -fno-exceptions frames;
        # the trampoline must still capture and re-raise them (not std::terminate).
        # The first function's error propagates and the second function is
        # skipped (the ShouldRun callback gates the adaptor's inner passes).
        with pytest.raises(ValueError, match="boom in f"):
            llvm.passmanager.run_python_pass_on_function(mod, boom)
        assert seen == ["f"]  # g was skipped after f raised
        assert "@f(" in str(mod)
        del mod
    assert_no_leaks()


def test_function_pass_unboolable_return_propagates():
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC2, ctx, "m")

        class Unboolable:
            def __bool__(self):
                raise ValueError("no truth value")

        # The trampoline wraps the __bool__ failure with a descriptive message
        # and chains the original ValueError as the cause.
        with pytest.raises(ValueError, match="truthiness") as excinfo:
            llvm.passmanager.run_python_pass_on_function(mod, lambda fn: Unboolable())
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "no truth value" in str(excinfo.value.__cause__)
        del mod
    assert_no_leaks()


def test_function_pass_skips_declarations():
    # The ModuleToFunctionPassAdaptor visits only defined functions, so a bare
    # declaration must not be handed to the callback.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC_DECL, ctx, "m")
        names = []
        llvm.passmanager.run_python_pass_on_function(
            mod, lambda fn: names.append(fn.name)
        )
        assert names == ["f"]  # @ext (declaration) was not visited
        del mod
    assert_no_leaks()


def test_function_pass_forwards_tuning_and_flags():
    # tuning/debug/verify_each are accepted and threaded through to the pipeline
    # environment; the pass still runs once per defined function.
    with llvm.ir.Context() as ctx:
        mod = llvm.ir.parse_assembly(_SRC2, ctx, "m")
        names = []
        tuning = llvm.passmanager.PipelineTuningOptions()
        tuning.slp_vectorization = False
        llvm.passmanager.run_python_pass_on_function(
            mod, lambda fn: names.append(fn.name), tuning=tuning, verify_each=True
        )
        assert sorted(names) == ["f", "g"]
        del mod
    assert_no_leaks()
