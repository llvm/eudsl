#!/bin/bash


if ! command -v pyodide >/dev/null 2>&1
then
  pip install pyodide-build
fi
# pyodide venv .venv-pyodide
# pip-compile --all-build-deps --only-build-deps -o ./build-reqs.txt ./pyproject.toml

if [ ! -d mlir_native_tools ]; then
  pip download mlir_native_tools -f https://makslevental.github.io/wheels
  unzip -o -j mlir_native_tools-*whl -d mlir_native_tools
  rm -rf mlir_native_tools-*whl
fi
if command -v ccache >/dev/null 2>&1
then
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
fi
export LLVM_NATIVE_TOOL_DIR="$PWD/mlir_native_tools"
export LLVM_TABLEGEN="$PWD/mlir_native_tools/llvm-tblgen"
export MLIR_TABLEGEN="$PWD/mlir_native_tools/mlir-tblgen"
export MLIR_LINALG_ODS_YAML_GEN="$PWD/mlir_native_tools/mlir-linalg-ods-yaml-gen"
export PATH=$EMSDK/upstream/bin:$PATH

pyodide build . -o wheelhouse --compression-level 10
