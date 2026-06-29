# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import difflib
import inspect
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path
from subprocess import PIPE, Popen
from textwrap import dedent, indent

import pytest

from .generate_test_checks import main
from ..context import MLIRContext, mlir_mod_ctx
from ..runtime.refbackend import LLVMJITBackend
from ...ir import Module, Operation, Context


def replace_correct_str_with_comments(fun, correct_with_checks):  # pragma: no cover
    # fun = inspect.currentframe().f_back.f_code
    lines, lnum = inspect.findsource(fun)
    fun_src = inspect.getsource(fun)
    fun_src = re.sub(
        r'dedent\(\s+""".*"""\s+\)',
        "#####"
        + indent(correct_with_checks, "    ")
        + "\n    filecheck_with_comments(ctx.module)\n#####",
        fun_src,
        flags=re.DOTALL,
    )
    fun_src = fun_src.splitlines(keepends=True)
    lines[lnum : lnum + len(fun_src)] = fun_src

    with open(inspect.getfile(fun), "w") as f:
        f.writelines(lines)


def get_filecheck_path():
    filecheck_name = "FileCheck"
    if platform.system() == "Windows":  # pragma: no cover
        filecheck_name += ".exe"

    # try from mlir-native-tools
    filecheck_path = Path(sys.prefix) / "bin" / filecheck_name
    # try to find using which
    if not filecheck_path.exists():  # pragma: no cover
        filecheck_path = shutil.which(filecheck_name)
    assert (
        filecheck_path is not None and Path(filecheck_path).exists() is not None
    ), "couldn't find FileCheck"

    return filecheck_path


def _get_module_str(module):
    if isinstance(module, Module):
        module = module.operation
    if isinstance(module, Operation):
        assert module.verify()
    op = str(module).strip()
    op = "\n".join(filter(None, op.splitlines()))
    return dedent(op)


def _run_filecheck(check_content, input_str, *, env=None, source_file=None):
    filecheck_path = get_filecheck_path()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(check_content.encode())
        tmp.flush()
        p = Popen(
            [filecheck_path, tmp.name],
            stdout=PIPE,
            stdin=PIPE,
            stderr=PIPE,
            env=env,
        )
        out, err = map(lambda o: o.decode(), p.communicate(input=input_str.encode()))
        if p.returncode:
            if source_file:
                err = err.replace(tmp.name, source_file)
            return err
    return None


def filecheck(correct: str, module):
    op = _get_module_str(module)

    if platform.system().lower() == "emscripten":
        return  # pragma: no cover

    correct = "\n".join(filter(None, correct.splitlines()))
    correct = dedent(correct)
    correct_with_checks = main(correct).strip().splitlines()
    correct_with_checks = "\n".join(
        [
            (line.replace("CHECK:", "CHECK-NEXT:") if i > 0 else line)
            for i, line in enumerate(correct_with_checks)
        ]
    )

    err = _run_filecheck(correct_with_checks, op)
    if err:
        diff = list(
            difflib.unified_diff(
                op.splitlines(),
                correct.splitlines(),
                lineterm="",
            )
        )
        diff.insert(1, "delta from module to correct")
        print("lit report:", err, file=sys.stderr)
        raise ValueError("\n" + "\n".join(diff))


def filecheck_with_comments(module):
    op = _get_module_str(module)

    if platform.system().lower() == "emscripten":
        return  # pragma: no cover

    fun = inspect.currentframe().f_back.f_code
    _, lnum = inspect.findsource(fun)
    fun_with_checks = inspect.getsource(fun)

    err = _run_filecheck(
        "\n" * lnum + fun_with_checks,
        op,
        env={"FILECHECK_OPTS": "-dump-input-filter=annotation -vv -color"},
        source_file=inspect.getfile(fun),
    )
    if err:
        raise ValueError(f"\n{err}")


@pytest.fixture(scope="function")
def mlir_ctx() -> MLIRContext:
    with mlir_mod_ctx(allow_unregistered_dialects=True) as ctx:
        yield ctx
    # TODO(max): why is context.current being retained now?
    # assert Context.current is None


@pytest.fixture(scope="function")
def backend() -> LLVMJITBackend:
    return LLVMJITBackend()
