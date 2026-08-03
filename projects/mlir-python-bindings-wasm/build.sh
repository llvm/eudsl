#!/bin/bash


if ! command -v pyodide >/dev/null 2>&1
then
  pip install "pyodide-build>=0.28.0"
fi

# Pin the cross-build environment; its python version becomes the wheel's ABI
# tag. Unpinned, pyodide-build takes the latest compatible one, which drifts ahead
# of the Pyodide the console and jupyterlite kernel actually run, and a wheel whose
# ABI tag does not match will not load. Keep in sync with the same constant in
# .github/actions/setup_base/action.yml.
#
# NOTE this needs a python 3.13 interpreter: the compatibility check compares the
# HOST python's major.minor against the cross-build env's and refuses a mismatch,
# so from 3.14 (or on 3.12) this fails with "not compatible with the current
# environment".
PYODIDE_VERSION=${PYODIDE_VERSION:-0.29.4}
pyodide xbuildenv install "$PYODIDE_VERSION"
echo "pyodide $PYODIDE_VERSION -> python $(pyodide config get python_version)"

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

if [[ "$OSTYPE" == "darwin"* ]]; then
  # note you have comment out the build-system.requires in pyproject.toml for --no-isolation to work (for some reason...)
  # https://github.com/scikit-build/scikit-build-core/issues/920
  WHEEL_TAG_FP=$(python -c "import scikit_build_core.builder.wheel_tag; print(scikit_build_core.builder.wheel_tag.__file__)")
  sed -i.bak 's/__all__ = \["WheelTag"\]/import os/g' $WHEEL_TAG_FP
  sed -i.bak "s/# Remove duplicates (e.g. universal2 if macOS > 11.0 and expanded)/plats = [os.environ['_PYTHON_HOST_PLATFORM']] if '_PYTHON_HOST_PLATFORM' in os.environ else plats/g" $(python -c "import scikit_build_core.builder.wheel_tag; print(scikit_build_core.builder.wheel_tag.__file__)")
  # the above doesn't work so you need to run in docker
fi

# Without this, linking every extension fails with
#   em++: error: undefined exported symbol: "_LLVMAddSymbol" [-Wundefined] [-Werror]
# (emscripten-core/emscripten#25911). pyodide-build computes an export list and
# passes it via -sSIDE_MODULE=2 -sEXPORTED_FUNCTIONS=@file; "whole_archive" makes
# get_export_flags() skip both and pass -Wl,--whole-archive instead.
export PYODIDE_BUILD_EXPORTS="${PYODIDE_BUILD_EXPORTS:-whole_archive}"

pyodide build . -o wheelhouse --compression-level 10
