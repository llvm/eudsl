#!/usr/bin/env bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

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

python3_command="python"
if (command -v python3 &> /dev/null); then
  python3_command="python3"
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
  # this flag keeps changing names in CMake...
  -DPython3_EXECUTABLE="$Python3_EXECUTABLE"
  -DPython_EXECUTABLE="$Python3_EXECUTABLE"
  -DPYTHON_EXECUTABLE="$Python3_EXECUTABLE"
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}"
)

if [ -x "$(command -v ccache)" ]; then
  echo 'using ccache' >&2
  export CCACHE_SLOPPINESS=include_file_ctime,include_file_mtime,time_macros
  export CCACHE_CPP2=true
  export CCACHE_UMASK=002
  cmake_options+=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi

# last so that C/CXX flags get set first
cmake_options+=(-C "$TD/cmake/llvm_cache.cmake")

echo "Source Directory: ${LLVM_SOURCE_DIR}"
echo "Build Directory: ${LLVM_BUILD_DIR}"
echo "CMake Options: ${cmake_options[*]}"

cmake "${cmake_options[@]}"
cmake --build "${LLVM_BUILD_DIR}" \
  --target install-mlirdevelopment-distribution
