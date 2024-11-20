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
EMSDK="$(realpath "${EMSDK}")"

echo "Paths canonicalized as:"
echo "LLVM_SOURCE_DIR=${LLVM_SOURCE_DIR}"
echo "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
echo "LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}"

$EMSDK/emsdk activate
source $EMSDK/emsdk_env.sh
export CCACHE_COMPILERCHECK="string:$(emcc --version | head -n 1)"

set -euo pipefail

echo "*********************** BUILDING LLVM *********************************"

# hack to emit html wrappers
# https://stackoverflow.com/a/75596433/9045206
sed -i.bak 's/CMAKE_EXECUTABLE_SUFFIX ".js"/CMAKE_EXECUTABLE_SUFFIX ".html"/g' "$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"

# all emscripten settings here https://github.com/emscripten-core/emscripten/blob/main/src/settings.js
# wasm-ld exports https://lld.llvm.org/WebAssembly.html#exports
# -sLINKABLE=1 combined with CMAKE_CXX_VISIBILITY_PRESET=default will cause all symbols to be exported
# -sEXPORT_ALL=1 put all of the exported function in the wasm on the JS module
cmake_options=(
  -GNinja
  -S "${LLVM_SOURCE_DIR}/llvm"
  -B "${LLVM_BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_EXE_LINKER_FLAGS="--emit-symbol-map -sSTANDALONE_WASM=1 -sWASM=1 -sWASM_BIGINT=1 -sLINKABLE=1 -sEXPORT_ALL=1 -sEXPORTED_RUNTIME_METHODS=cwrap,ccall,getValue,setValue,writeAsciiToMemory,wasmTable --minify 0 -lembind"
  -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_DIR}"
  -DCMAKE_SYSTEM_NAME=Emscripten
  -DCMAKE_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
  -DCROSS_TOOLCHAIN_FLAGS_NATIVE="-DCMAKE_C_COMPILER=$CC;-DCMAKE_CXX_COMPILER=$CXX"
  -C "$TD/cmake/llvm_wasm_cache.cmake"
  -DCMAKE_PROJECT_INCLUDE="$TD/cmake/llvm_wasm_project_include.cmake"
)

echo "Source Directory: ${LLVM_SOURCE_DIR}"
echo "Build Directory: ${LLVM_BUILD_DIR}"
echo "CMake Options: ${cmake_options[*]}"

cmake "${cmake_options[@]}"
cmake --build "${LLVM_BUILD_DIR}" \
  --target install-mlirdevelopment-distribution

sed -i.bak 's/CMAKE_EXECUTABLE_SUFFIX ".html"/CMAKE_EXECUTABLE_SUFFIX ".js"/g' "$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"

# wasi files aren't installed for some reason
cp "${LLVM_BUILD_DIR}"/bin/mlir-opt.* "${LLVM_INSTALL_DIR}/bin"
# cp "${LLVM_BUILD_DIR}"/lib/*.symbols "${LLVM_INSTALL_DIR}/lib"

# prevent symbol collisions
# sed -i.bak 's/if(LLVM_ENABLE_PIC)/if(LLVM_ENABLE_PIC)\nreturn()\nendif()\nif(LLVM_ENABLE_PIC)/g' "${LLVM_SOURCE_DIR}/mlir/lib/ExecutionEngine/CMakeLists.txt"
# https://stackoverflow.com/a/1252191/9045206
# sed -i.bak -e ':a' -e 'N' -e '$!ba' -e 's/if(LLVM_ENABLE_PIC)\nreturn()\nendif()\n//g' "${LLVM_SOURCE_DIR}/mlir/lib/ExecutionEngine/CMakeLists.txt"
