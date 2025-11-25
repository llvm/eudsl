#!/bin/bash


if ! command -v pyodide >/dev/null 2>&1
then
  pip install pyodide-build
fi
# pyodide venv .venv-pyodide
# pip-compile --all-build-deps --only-build-deps -o ./build-reqs.txt ./pyproject.toml

if [ ! -d mlir_native_tools ]; then
  pip download mlir_native_tools -f https://llvm.github.io/eudsl
  unzip -o -j mlir_native_tools-*whl -d mlir_native_tools
fi
if command -v ccache >/dev/null 2>&1
then
  export LLVM_CCACHE_BUILD=ON
fi
export LLVM_NATIVE_TOOL_DIR="$PWD/mlir_native_tools"
export LLVM_TABLEGEN="$PWD/mlir_native_tools/llvm-tblgen"
export MLIR_TABLEGEN="$PWD/mlir_native_tools/mlir-tblgen"
export MLIR_LINALG_ODS_YAML_GEN="$PWD/mlir_native_tools/mlir-linalg-ods-yaml-gen"
export PATH=$EMSDK/upstream/bin:$PATH
#export CMAKE_BUILD_TYPE=Debug

# note you have comment out the build-system.requires in pyproject.toml for --no-isolation to work (for some reason...)
# https://github.com/scikit-build/scikit-build-core/issues/920
WHEEL_TAG_FP=$(python -c "import scikit_build_core.builder.wheel_tag; print(scikit_build_core.builder.wheel_tag.__file__)")
sed -i.bak 's/__all__ = \["WheelTag"\]/import os/g' $WHEEL_TAG_FP
sed -i.bak "s/# Remove duplicates (e.g. universal2 if macOS > 11.0 and expanded)/plats = [os.environ['_PYTHON_HOST_PLATFORM']] if '_PYTHON_HOST_PLATFORM' in os.environ else plats/g" $(python -c "import scikit_build_core.builder.wheel_tag; print(scikit_build_core.builder.wheel_tag.__file__)")

pyodide build . -o wheelhouse --compression-level 10 --no-isolation
