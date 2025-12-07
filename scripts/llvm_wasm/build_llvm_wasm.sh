#!/usr/bin/env bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2024.

TD="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd $TD/../.. && pwd)"

LLVM_NATIVE_TOOL_DIR="${LLVM_NATIVE_TOOL_DIR:-$REPO_ROOT/mlir-native-tools-install/bin}"
LLVM_TABLEGEN="${LLVM_TABLEGEN:-$LLVM_NATIVE_TOOL_DIR/llvm-tblgen}"
MLIR_LINALG_ODS_YAML_GEN="${MLIR_LINALG_ODS_YAML_GEN:-$LLVM_NATIVE_TOOL_DIR/mlir-linalg-ods-yaml-gen}"
MLIR_TABLEGEN="${MLIR_TABLEGEN:-$LLVM_NATIVE_TOOL_DIR/mlir-tblgen}"

export LLVM_NATIVE_TOOL_DIR="$(realpath "${LLVM_NATIVE_TOOL_DIR}")"
export LLVM_TABLEGEN="$(realpath "${LLVM_TABLEGEN}")"
export MLIR_LINALG_ODS_YAML_GEN="$(realpath "${MLIR_LINALG_ODS_YAML_GEN}")"
export MLIR_TABLEGEN="$(realpath "${MLIR_TABLEGEN}")"
EMSDK="$(realpath "${EMSDK}")"

echo "Paths canonicalized as:"
echo "LLVM_NATIVE_TOOL_DIR=${LLVM_NATIVE_TOOL_DIR}"
echo "LLVM_TABLEGEN=${LLVM_TABLEGEN}"
echo "MLIR_LINALG_ODS_YAML_GEN=${MLIR_LINALG_ODS_YAML_GEN}"
echo "LLVM_TABLEGEN=${LLVM_TABLEGEN}"

set -euo pipefail

echo "*********************** BUILDING LLVM *********************************"

if [ -x "$(command -v $EMSDK/ccache/git-emscripten_64bit/bin/ccache)" ]; then
  echo 'using ccache' >&2
  export CCACHE_SLOPPINESS=include_file_ctime,include_file_mtime,time_macros
  export CCACHE_CPP2=true
  export CCACHE_UMASK=002
  export CCACHE_COMPILERCHECK="string:$($EMSDK/upstream/bin/clang --version | head -n 1)"
  export CCACHE="$EMSDK/ccache/git-emscripten_64bit/bin/ccache"
  export CMAKE_C_COMPILER_LAUNCHER="$CCACHE"
  export CMAKE_CXX_COMPILER_LAUNCHER="$CCACHE"
fi

pyodide build $TD -o wheelhouse --compression-level 10
