#!/usr/bin/env bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

TD="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd $TD/.. && pwd)"

LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${REPO_ROOT}/third_party/llvm-project}"
MLIR_NATIVE_TOOLS_BUILD_DIR="${MLIR_NATIVE_TOOLS_BUILD_DIR:-${REPO_ROOT}/mlir-native-tools-build}"
MLIR_NATIVE_TOOLS_INSTALL_DIR="${MLIR_NATIVE_TOOLS_INSTALL_DIR:-${REPO_ROOT}/mlir-native-tools-install}"

mkdir -p $MLIR_NATIVE_TOOLS_BUILD_DIR
mkdir -p $MLIR_NATIVE_TOOLS_INSTALL_DIR

LLVM_SOURCE_DIR="$(realpath "${LLVM_SOURCE_DIR}")"
MLIR_NATIVE_TOOLS_BUILD_DIR="$(realpath "${MLIR_NATIVE_TOOLS_BUILD_DIR}")"
MLIR_NATIVE_TOOLS_INSTALL_DIR="$(realpath "${MLIR_NATIVE_TOOLS_INSTALL_DIR}")"

echo "Paths canonicalized as:"
echo "LLVM_SOURCE_DIR=${LLVM_SOURCE_DIR}"
echo "MLIR_NATIVE_TOOLS_BUILD_DIR=${MLIR_NATIVE_TOOLS_BUILD_DIR}"
echo "MLIR_NATIVE_TOOLS_INSTALL_DIR=${MLIR_NATIVE_TOOLS_INSTALL_DIR}"

set -euo pipefail

echo "*********************** BUILDING MLIR Native tools *********************************"

cmake_options=(
  -GNinja
  -S "${LLVM_SOURCE_DIR}/llvm"
  -B "${MLIR_NATIVE_TOOLS_BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${MLIR_NATIVE_TOOLS_INSTALL_DIR}"
  -DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
  -DLLVM_ENABLE_PROJECTS="mlir;llvm"
  -DLLVM_TARGETS_TO_BUILD=host
  -DLLVM_OPTIMIZED_TABLEGEN=ON
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

echo "Source Directory: ${LLVM_SOURCE_DIR}"
echo "Build Directory: ${MLIR_NATIVE_TOOLS_BUILD_DIR}"
echo "CMake Options: ${cmake_options[*]}"

cmake "${cmake_options[@]}"
cmake --build "${MLIR_NATIVE_TOOLS_BUILD_DIR}" --target \
  install-llvm-tblgen \
  install-llvm-config \
  install-mlir-tblgen \
  install-mlir-linalg-ods-yaml-gen \
  install-mlir-pdll

