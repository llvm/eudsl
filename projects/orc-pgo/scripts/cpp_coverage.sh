#!/usr/bin/env bash
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Build orc-pgo with source-based coverage, run ctest, and enforce 100%
# line/function coverage over include/ and lib/.
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

die() { echo "error: $*" >&2; exit 1; }

# The coverage tools must come from the same LLVM version as the instrumenting
# clang: raw .profraw formats are version-locked, so e.g. the distro's
# llvm-profdata can't merge them. macOS: Apple clang + its xcrun tools. Linux:
# the tools next to $CC.
if [ "$(uname)" = Darwin ]; then
  CC="${CC:-$(xcrun -f clang)}"
  CXX="${CXX:-$(xcrun -f clang++)}"
  LLVM_COV="${LLVM_COV:-$(xcrun -f llvm-cov)}"
  LLVM_PROFDATA="${LLVM_PROFDATA:-$(xcrun -f llvm-profdata)}"
else
  CC="${CC:-clang}"
  CXX="${CXX:-clang++}"
  CC_PATH="$(command -v "$CC")" || die "C compiler '$CC' not found"
  CC_BINDIR="$(dirname "$(readlink -f "$CC_PATH")")"
  LLVM_COV="${LLVM_COV:-$CC_BINDIR/llvm-cov}"
  LLVM_PROFDATA="${LLVM_PROFDATA:-$CC_BINDIR/llvm-profdata}"
fi
for tool in "$LLVM_COV" "$LLVM_PROFDATA"; do
  [ -x "$tool" ] || die "$tool not found (set LLVM_COV/LLVM_PROFDATA to match $CC)"
done

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
  ctest --test-dir "$BUILD" --output-on-failure --no-tests=error

PROFRAWS=()
for f in "$BUILD"/profraw/*.profraw; do [ -e "$f" ] && PROFRAWS+=("$f"); done
[ ${#PROFRAWS[@]} -gt 0 ] || die "the tests produced no .profraw files"
"$LLVM_PROFDATA" merge -sparse "${PROFRAWS[@]}" -o "$BUILD/coverage.profdata"

# macOS bash 3.2 has no mapfile.
OBJECTS=()
while IFS= read -r line || [ -n "$line" ]; do
  [ -n "$line" ] && OBJECTS+=("$line")
done < "$BUILD/coverage-objects.txt"
[ ${#OBJECTS[@]} -gt 0 ] || die "no coverage objects in $BUILD/coverage-objects.txt (BUILD_TESTING off?)"

python3 "$REPO/scripts/check_coverage.py" \
  --label orc-pgo \
  --llvm-cov "$LLVM_COV" \
  --profdata "$BUILD/coverage.profdata" \
  --objects "${OBJECTS[@]}" \
  --sources "$PROJ/include" "$PROJ/lib" \
  --threshold 100 \
  --function-threshold 100 \
  --ignore-filename-regex '(unittests|_deps)'
