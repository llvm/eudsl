#!/usr/bin/env bash

TD="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd $TD/.. && pwd)"

LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${REPO_ROOT}/third_party/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${REPO_ROOT}/llvm-build}"
LLVM_INSTALL_DIR="${LLVM_INSTALL_DIR:-${REPO_ROOT}/llvm-install}"

mkdir -p $LLVM_BUILD_DIR
mkdir -p $LLVM_INSTALL_DIR

LLVM_SOURCE_DIR="$(realpath "${LLVM_SOURCE_DIR}")"
LLVM_BUILD_DIR="$(realpath "${LLVM_BUILD_DIR}")"
LLVM_INSTALL_DIR="$(realpath "${LLVM_INSTALL_DIR}")"

echo "Paths canonicalized as:"
echo "LLVM_SOURCE_DIR=${LLVM_SOURCE_DIR}"
echo "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
echo "LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}"

python3_command=""
if (command -v python3 &> /dev/null); then
  python3_command="python3"
elif (command -v python &> /dev/null); then
  python3_command="python"
fi

Python3_EXECUTABLE="${Python3_EXECUTABLE:-$(which $python3_command)}"

set -euo pipefail

echo "*********************** BUILDING LLVM *********************************"

cmake_options=(
  -GNinja
  -S "${LLVM_SOURCE_DIR}/llvm"
  -B "${LLVM_BUILD_DIR}"
  -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-host}"
  -DCMAKE_BUILD_TYPE=Release
  -DPython3_EXECUTABLE="$Python3_EXECUTABLE"
  -C "$TD/cmake/llvm_cache.cmake"
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}"
)
echo "ostype $OSTYPE"
if [[ "$OSTYPE" == "msys"* ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_C_FLAGS="/MT"
    -DCMAKE_CXX_FLAGS="/MT"
  )
fi

echo "Source Directory: ${LLVM_SOURCE_DIR}"
echo "Build Directory: ${LLVM_BUILD_DIR}"
echo "CMake Options: ${cmake_options[*]}"

cmake "${cmake_options[@]}"
cmake --build "${LLVM_BUILD_DIR}" \
  --target install-mlirdevelopment-distribution
