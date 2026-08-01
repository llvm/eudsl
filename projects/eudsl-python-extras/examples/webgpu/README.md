# MLIR → SPIR-V → WGSL → WebGPU

Work in progress toward authoring GPU kernels in MLIR from Python **in the browser**,
compiling them client-side, and dispatching them via WebGPU.

Two of the three links are proven; the middle one is not built yet.

```
  gpu.func MLIR ──► SPIR-V              ✅ mlir_to_spirv.py  (shipped wasm wheel, no new C++)
                       │
                       ▼
                    Tint                ❌ not built yet
                       │
                       ▼
                    WGSL ──► navigator.gpu   ✅ webgpu_matmul.py  (verified in Chrome)
```

## What works

**`mlir_to_spirv.py`** — MLIR → SPIR-V using only passes already present in the shipped
`mlir-python-bindings` wasm wheel. No new C API, no LLVM patch. `gpu-module-to-binary`
calls `spirv::serialize` internally and stores the blob in a `gpu.binary` op, which
Python can read.

Verified natively and in the browser. Output for `matmul.mlir`: 1332 bytes, magic
`0x07230203`, SPIR-V 1.0, `Logical GLSL450`, caps `[Matrix, Shader]`, entry point
`matmul`, LocalSize `[8, 8, 1]`, and:

```
@group(0) @binding(0)  matmul_arg_0
@group(0) @binding(1)  matmul_arg_1
@group(0) @binding(2)  matmul_arg_2
```

**`webgpu_matmul.py`** — the runtime half. Pyodide reaches `navigator.gpu` through JS
interop, uploads buffers, dispatches a WGSL matmul, reads back, and matches `A @ B`
(`max_abs_err = 2.4e-6` in Chrome 151). Note this needs **no Dawn**: the browser already
has a WebGPU implementation, so Dawn/emdawnwebgpu are only relevant to native builds.

The `@group(0) @binding(i)` layout the compiler emits is exactly what this host code
binds, so the two ends already agree.

**`run_in_browser.py`** — drives the compile through `mlir.passmanager` inside Pyodide.

## What's missing

SPIR-V → WGSL, which needs Tint (`third_party/dawn`). Compare
`compiler/plugins/target/WebGPUSPIRV/SPIRVToWGSL.cpp` in IREE, which is the same path and
is tested: `tint::spirv::reader::ReadIR` → `tint::wgsl::writer::ProgramFromIR` →
`Generate`. IREE builds Tint with everything off except `TINT_BUILD_SPV_READER` and
`TINT_BUILD_WGSL_WRITER`.

## Known rough edge

`nest_spirv_module()` moves the generated `spirv.module` inside the `gpu.module` by
string manipulation. This is needed because `convert-gpu-to-spirv` hoists `spirv.module`
to the top level as a *sibling* of the emptied `gpu.module`, while
`SPIRVTargetAttrImpl::serializeToObject` only looks *inside* a `gpu.module`.

Upstream's Vulkan runner pipeline avoids this with
`test-convert-to-spirv{nest-in-gpu-module=true}`, but that pass is test-only and is not
in the shipped wheel (verified against `libMLIRPythonWasmCAPI.so`). Doing the move
through the GPU dialect Python bindings, or upstreaming a non-test `convert-to-spirv`,
would both be cleaner.

`spirv-update-vce` is mandatory, not cosmetic — `spirv::serialize` hard-fails without a
`vce_triple` attribute.

## Not yet done

The SPIR-V has not been through `spirv-val`. The header, entry point, execution mode and
binding decorations were decoded and checked by hand, which is weaker than validation.
`third_party/SPIRV-Tools` is checked out but unbuilt.

## Running it

Native:

```bash
python3 - <<'EOF'
import subprocess, sys
sys.path.insert(0, "projects/eudsl-python-extras/examples/webgpu")
from mlir_to_spirv import LOWER_PIPELINE, SERIALIZE_PIPELINE, nest_spirv_module, extract_binary, describe

MO = "path/to/mlir-opt"
src = open("projects/eudsl-python-extras/examples/webgpu/matmul.mlir").read()
lowered = subprocess.run([MO, "-pass-pipeline=" + LOWER_PIPELINE], input=src,
                         capture_output=True, text=True, check=True).stdout
binmlir = subprocess.run([MO, "-pass-pipeline=" + SERIALIZE_PIPELINE],
                         input=nest_spirv_module(lowered),
                         capture_output=True, text=True, check=True).stdout
print(describe(extract_binary(binmlir)))
EOF
```

Browser: serve a directory containing `run_in_browser.py`, `mlir_to_spirv.py` and the
wasm wheel over HTTP, load Pyodide, `loadPackage` the wheel, then call
`run_in_browser.main()`. The wheel must be served over HTTP; `loadPackage` will not fetch
it over `file://`.
