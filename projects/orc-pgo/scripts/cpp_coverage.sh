#!/usr/bin/env bash
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Build orc-pgo with source-based coverage, run ctest, and enforce 100%
# line/function coverage over include/, lib/, tools/ (eudsl-llvmpy model).
#
# Env:
#   LLVM_INSTALL_DIR          LLVM distro prefix (default: <repo>/mlir_wheel)
#   BUILD_DIR                 build dir (default: <project>/build-coverage)
#   CC / CXX                  compilers (default: xcrun clang on macOS, clang on Linux)
#   LLVM_COV / LLVM_PROFDATA  override the coverage tools
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
REPO="$(cd "$PROJ/../.." && pwd)"
BUILD="${BUILD_DIR:-$PROJ/build-coverage}"
LLVM_PREFIX="${LLVM_INSTALL_DIR:-$REPO/mlir_wheel}"
# The coverage tools must match the instrumenting clang: raw .profraw formats
# are version-locked, so the distro's (newer) llvm-profdata can't merge them.
# macOS: Apple clang + xcrun llvm-cov/llvm-profdata (as eudsl-llvmpy does).
# Linux: system clang + the system llvm-cov/llvm-profdata on PATH.
if [ "$(uname)" = Darwin ]; then
  CC="${CC:-$(xcrun -f clang)}"
  CXX="${CXX:-$(xcrun -f clang++)}"
  LLVM_COV="${LLVM_COV:-$(xcrun -f llvm-cov)}"
  LLVM_PROFDATA="${LLVM_PROFDATA:-$(xcrun -f llvm-profdata)}"
else
  CC="${CC:-clang}"
  CXX="${CXX:-clang++}"
  LLVM_COV="${LLVM_COV:-$(command -v llvm-cov)}"
  LLVM_PROFDATA="${LLVM_PROFDATA:-$(command -v llvm-profdata)}"
fi

cmake -G Ninja -S "$PROJ" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH="$LLVM_PREFIX" \
  -DORC_PGO_ENABLE_COVERAGE=ON
cmake --build "$BUILD"

rm -rf "$BUILD/profraw"
mkdir -p "$BUILD/profraw"
LLVM_PROFILE_FILE="$BUILD/profraw/%p-%m.profraw" \
  ctest --test-dir "$BUILD" --output-on-failure

"$LLVM_PROFDATA" merge -sparse "$BUILD"/profraw/*.profraw -o "$BUILD/coverage.profdata"

# macOS bash 3.2 has no mapfile.
OBJECTS=()
while IFS= read -r line; do [ -n "$line" ] && OBJECTS+=("$line"); done < "$BUILD/coverage-objects.txt"
# tools/ appears with the driver; only pass directories that exist.
SOURCES=()
for d in include lib tools; do [ -d "$PROJ/$d" ] && SOURCES+=("$PROJ/$d"); done

python3 "$REPO/scripts/check_coverage.py" \
  --llvm-cov "$LLVM_COV" \
  --profdata "$BUILD/coverage.profdata" \
  --objects "${OBJECTS[@]}" \
  --sources "${SOURCES[@]}" \
  --threshold 100 \
  --function-threshold 100 \
  --ignore-filename-regex '(unittests|_deps|test/multi-tu)'
