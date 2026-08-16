#!/usr/bin/env bash
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# C++ (llvm-cov) coverage for the eudslllvm_ext bindings, analogous to the
# --cov gate for the Python surface. Builds the extension instrumented, runs
# the pytest suite (which imports the .so and thereby exercises the C++), merges
# the profraw, and enforces a src/IR line-coverage threshold.
#
# Env:
#   CMAKE_PREFIX_PATH  path to the LLVM install/build (for LLVMConfig.cmake)
#   LLVM_COV / LLVM_PROFDATA  override the tools (default: from the LLVM bin dir
#                             next to llvm-config on PATH, else $LLVM_BINDIR)
#   PYTHON             python interpreter (default: python3)
#   COVERAGE_THRESHOLD line-coverage percent to require (default: 90)
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
THRESHOLD="${COVERAGE_THRESHOLD:-90}"
COV_DIR="${PROJ_DIR}/build/coverage"

# Locate llvm-cov / llvm-profdata.
LLVM_BINDIR="${LLVM_BINDIR:-$(dirname "$(command -v llvm-config 2>/dev/null || true)")}"
LLVM_COV="${LLVM_COV:-${LLVM_BINDIR}/llvm-cov}"
LLVM_PROFDATA="${LLVM_PROFDATA:-${LLVM_BINDIR}/llvm-profdata}"
if [[ ! -x "$LLVM_COV" || ! -x "$LLVM_PROFDATA" ]]; then
  echo "error: llvm-cov/llvm-profdata not found (set LLVM_COV/LLVM_PROFDATA or LLVM_BINDIR)" >&2
  exit 1
fi

echo ">> building eudslllvm_ext with coverage instrumentation"
BUILD_LOG="${PROJ_DIR}/build/coverage-build.log"
mkdir -p "$(dirname "$BUILD_LOG")"
if ! "$PY" -m pip install -e "$PROJ_DIR" --no-build-isolation \
  --config-settings=cmake.define.EUDSL_LLVMPY_ENABLE_COVERAGE=ON >"$BUILD_LOG" 2>&1; then
  echo "error: instrumented build failed; full output:" >&2
  cat "$BUILD_LOG" >&2
  exit 1
fi
echo ">> build OK"

echo ">> running the test suite under LLVM_PROFILE_FILE"
rm -rf "$COV_DIR"
mkdir -p "$COV_DIR"
PYTEST_LOG="${COV_DIR}/pytest.log"
# %p (pid) + %m (per-image id) so parallel/forked runs don't clobber counters.
if ! LLVM_PROFILE_FILE="${COV_DIR}/pytest-%p-%m.profraw" \
  "$PY" -m pytest "${PROJ_DIR}/tests" -q -p no:cacheprovider --no-cov >"$PYTEST_LOG" 2>&1; then
  echo "error: pytest failed; full output:" >&2
  cat "$PYTEST_LOG" >&2
  exit 1
fi
echo ">> test suite passed"

echo ">> merging profraw -> profdata"
# shellcheck disable=SC2086
"$LLVM_PROFDATA" merge -sparse ${COV_DIR}/*.profraw -o "${COV_DIR}/eudslllvm.profdata"

# The instrumented shared object actually imported by Python (its location
# depends on the editable mode, so ask the interpreter).
SO="$("$PY" -c 'import llvm.eudslllvm_ext as m; print(m.__file__)')"
if [[ -z "$SO" || ! -f "$SO" ]]; then
  echo "error: could not locate the imported eudslllvm_ext .so" >&2
  exit 1
fi

echo ">> checking src/IR coverage (threshold=${THRESHOLD}%)"
"$PY" "${PROJ_DIR}/scripts/check_coverage.py" \
  --llvm-cov "$LLVM_COV" \
  --profdata "${COV_DIR}/eudslllvm.profdata" \
  --objects "$SO" \
  --sources "${PROJ_DIR}/src/IR" \
  --threshold="${THRESHOLD}"
