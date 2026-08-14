#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Test helpers. Not imported by the llvm package itself."""

import gc
import inspect
import os
import shutil
import subprocess
import sys
import tempfile

from . import Context


def assert_no_leaks():
    """Assert every Context has been released at this point.

    This is a cheap in-body check: `Context.__exit__` calls release(), so after
    a `with` block the context count is back to zero. It does NOT prove the
    underlying objects were destroyed — a Module (and the LLVMContext it keeps
    alive) can still be referenced by a live local. The authoritative leak gate
    is the autouse fixture in tests/conftest.py, which checks both the context
    and module counts after the test frame (and its locals) is gone.
    """
    gc.collect()
    live = Context._get_live_count()
    assert live == 0, f"{live} Context object(s) still alive"


def _find_filecheck():
    """Locate the LLVM FileCheck binary. It ships in mlir-native-tools (into
    sys.prefix/bin, or sys.prefix/Scripts on Windows) and in the mlir_wheel LLVM
    distro (LLVM_BINDIR)."""
    exe = "FileCheck.exe" if sys.platform == "win32" else "FileCheck"
    for cand in (
        os.environ.get("FILECHECK"),
        os.path.join(os.environ.get("LLVM_BINDIR", ""), exe),
        os.path.join(sys.prefix, "bin", exe),
        os.path.join(sys.prefix, "Scripts", exe),  # Windows console-script dir
        shutil.which("FileCheck"),
    ):
        if cand and os.path.isfile(cand):
            return cand
    raise RuntimeError(  # pragma: no cover
        "FileCheck not found; install mlir-native-tools or set LLVM_BINDIR"
    )


def filecheck_with_comments(module):
    """Validate a module's IR against the `# CHECK:` comments in the caller.

    Mirrors eudsl-python-extras' filecheck_with_comments: the calling test
    function's own source is the FileCheck check-file, so `# CHECK:` /
    `# CHECK-NEXT:` / `# CHECK-NOT:` comments are matched (in order, and with
    `[[name:...]]`/`[[name]]` capture-variable binding) against the printed IR --
    catching what a substring `in` check cannot. Uses the real LLVM FileCheck
    binary.
    """
    printed = str(module)
    caller = inspect.currentframe().f_back.f_code
    fn_source = inspect.getsource(caller)
    _, lnum = inspect.findsource(caller)
    # Prepend blank lines so FileCheck's reported line numbers match the source.
    check_content = "\n" * lnum + fn_source
    filecheck = _find_filecheck()
    # Write the check-file and close it *before* invoking FileCheck: on Windows a
    # still-open NamedTemporaryFile is locked exclusively and the subprocess
    # cannot reopen the path. mkstemp + explicit close/unlink works everywhere.
    fd, check_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as check_file:
            check_file.write(check_content)
        proc = subprocess.run(
            [filecheck, check_path],
            input=printed,
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(check_path)
    if proc.returncode != 0:
        raise ValueError(f"FileCheck failed:\n{proc.stdout}\n{proc.stderr}")
