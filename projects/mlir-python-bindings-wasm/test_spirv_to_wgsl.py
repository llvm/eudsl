#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""MLIR -> SPIR-V -> WGSL, run inside pyodide.

Guards the pieces that are easy to break from the build side: that every
extension in the wheel actually loads (emscripten-core/emscripten#25911 made
them build fine and then fail to dlopen), and that the SPIR-V comes out with the
@group(0) @binding(i) layout the WebGPU host code binds against.

Fails if mlir.wgsl is missing. Set MLIR_PYTHON_BINDINGS_WGSL=OFF to skip on a
wheel deliberately built without the extension.
"""

import os
import re
import sys

MATMUL = """
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @matmul(%A: memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [8, 8, 1]>} {
      %row = gpu.global_id x
      %col = gpu.global_id y
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %zero = arith.constant 0.0 : f32
      %sum = scf.for %k = %c0 to %c32 step %c1 iter_args(%acc = %zero) -> (f32) {
        %a = memref.load %A[%row, %k] : memref<32x32xf32>
        %b = memref.load %B[%k, %col] : memref<32x32xf32>
        %m = arith.mulf %a, %b : f32
        %n = arith.addf %acc, %m : f32
        scf.yield %n : f32
      }
      memref.store %sum, %C[%row, %col] : memref<32x32xf32>
      gpu.return
    }
  }
}
"""

# convert-gpu-to-spirv hoists the spirv.module to the top level as a sibling of
# the emptied gpu.module, but gpu-module-to-binary only serializes a spirv.module
# nested *inside* a gpu.module, so the two run separately with a re-nest between.
LOWER = (
    "builtin.module("
    "spirv-attach-target{ver=v1.0 caps=Shader "
    "exts=SPV_KHR_storage_buffer_storage_class client_api=Vulkan},"
    "convert-gpu-to-spirv,"
    "spirv.module(spirv-lower-abi-attrs,spirv-update-vce,spirv-webgpu-prepare))"
)
SERIALIZE = "builtin.module(gpu-module-to-binary)"

SPIRV_MAGIC = 0x07230203

# spirv-lower-abi-attrs names the interface variables after the kernel and the
# argument index, and the host binds buffers in that same order. Checking the
# mapping per-variable (rather than just the set of numbers) is what catches an
# argument/binding swap, which would run fine and compute the wrong answer.
EXPECTED_BINDINGS = {
    "matmul_arg_0": (0, 0),
    "matmul_arg_1": (0, 1),
    "matmul_arg_2": (0, 2),
}


def nest_spirv_module(asm):
    lines = asm.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("  spirv.module"))
    depth = 0
    end = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        # A single-line spirv.module balances at i == start, so close there too.
        if depth == 0:
            end = i
            break
    assert end is not None, "unterminated spirv.module"
    spv = lines[start : end + 1]
    rest = lines[:start] + lines[end + 1 :]
    g = next(i for i, l in enumerate(rest) if l.startswith("  gpu.module"))
    return "\n".join(rest[: g + 1] + ["  " + l for l in spv] + rest[g + 1 :])


def extract_binary(asm):
    """Pull the object blob out of gpu.binary. MLIR escapes bytes as \\XX."""
    i = asm.find("gpu.binary")
    assert i >= 0, "no gpu.binary op -- did gpu-module-to-binary run?"
    assert asm.find("gpu.binary", i + 1) < 0, "multiple gpu.binary ops; expected one"
    j = asm.index('"', i) + 1
    out = bytearray()
    while True:
        assert j < len(asm), "unterminated gpu.binary blob"
        c = asm[j]
        if c == '"':
            break
        if c == "\\":
            # MLIR emits \XX hex pairs, but also \" and \\ for those two chars.
            nxt = asm[j + 1]
            if nxt in ('"', "\\"):
                out.append(ord(nxt))
                j += 2
            else:
                out.append(int(asm[j + 1 : j + 3], 16))
                j += 3
        else:
            out.append(ord(c))
            j += 1
    return bytes(out)


def decode_bindings(spv):
    """Decode the entry point, workgroup size, and per-variable bindings.

    Returns bindings as {variable_name: (descriptor_set, binding)}. Keying by
    name matters: the host binds buffers in kernel-argument order, so a set of
    binding numbers that is correct but attached to the wrong variables is a
    silent wrong-answer bug, not a crash.
    """
    import struct

    w = struct.unpack("<%dI" % (len(spv) // 4), spv)
    assert w[0] == SPIRV_MAGIC, f"bad magic {w[0]:#x}"

    def string_at(a, b):
        return b"".join(struct.pack("<I", x) for x in w[a:b]).split(b"\0")[0].decode()

    entry, workgroup, decos, names = None, None, {}, {}
    i = 5
    while i < len(w):
        wc, op = w[i] >> 16, w[i] & 0xFFFF
        assert wc != 0, f"zero word count at word {i}; truncated SPIR-V?"
        if op == 5:  # OpName
            names[w[i + 1]] = string_at(i + 2, i + wc)
        elif op == 15:  # OpEntryPoint
            entry = string_at(i + 3, i + wc)
        elif op == 16:  # OpExecutionMode
            workgroup = list(w[i + 3 : i + wc])
        elif op == 71 and w[i + 2] in (33, 34):  # OpDecorate Binding/DescriptorSet
            decos.setdefault(w[i + 1], {})[w[i + 2]] = w[i + 3]
        i += wc

    bindings = {}
    for target, d in decos.items():
        name = names.get(target, f"<id {target}>")
        assert name not in bindings, f"two variables named {name}"
        bindings[name] = (d.get(34), d.get(33))
    return entry, workgroup, bindings


def main():
    # Only skip when the wheel was deliberately built without the extension.
    # Importing mlir.wgsl exercises the whole side-module graph, so treating an
    # ImportError as a pass would hide exactly the failure this test exists to
    # catch (emscripten#25911 built fine and then would not dlopen).
    expect_wgsl = os.environ.get("MLIR_PYTHON_BINDINGS_WGSL", "ON").upper() not in (
        "OFF",
        "0",
        "FALSE",
        "NO",
    )
    try:
        from mlir.wgsl import spirv_to_wgsl
    except ImportError as e:
        if expect_wgsl:
            print(
                f"FAIL: cannot import mlir.wgsl ({e}).\n"
                "The wheel is expected to carry the SPIR-V -> WGSL extension. "
                "Set MLIR_PYTHON_BINDINGS_WGSL=OFF to skip this test on a wheel "
                "built without it.",
                file=sys.stderr,
            )
            return 1
        print(f"SKIP: MLIR_PYTHON_BINDINGS_WGSL is off and mlir.wgsl is absent ({e})")
        return 0

    from mlir.ir import Context, Module
    from mlir.passmanager import PassManager

    with Context():
        module = Module.parse(MATMUL)
        PassManager.parse(LOWER).run(module.operation)
        lowered = str(module)

    with Context():
        module = Module.parse(nest_spirv_module(lowered))
        PassManager.parse(SERIALIZE).run(module.operation)
        spv = extract_binary(str(module))

    entry, workgroup, bindings = decode_bindings(spv)
    print(f"SPIR-V: {len(spv)} bytes, entry {entry!r}, workgroup_size {workgroup}")
    print(f"bindings: {bindings}")
    assert entry == "matmul", entry
    assert workgroup == [8, 8, 1], workgroup
    # Per-variable, not just the set of numbers: the host binds buffers in
    # kernel-argument order, so arg i must land on binding i.
    assert bindings == EXPECTED_BINDINGS, bindings

    wgsl = spirv_to_wgsl(spv)
    print(f"WGSL: {len(wgsl)} bytes")
    assert "@compute" in wgsl, wgsl[:400]
    assert "@workgroup_size(8" in wgsl, wgsl[:400]
    # Check @group/@binding on the declaration of each specific variable, not
    # just that the numbers appear somewhere in the file.
    for name, (group, binding) in EXPECTED_BINDINGS.items():
        pattern = (
            rf"@group\({group}u?\)\s*@binding\({binding}u?\)"
            rf"\s*var<[^>]*>\s*{re.escape(name)}\s*:"
        )
        assert re.search(pattern, wgsl), (
            f"expected {name} at @group({group}) @binding({binding})\n{wgsl[:600]}"
        )

    print("OK: MLIR -> SPIR-V -> WGSL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
