# eudsl

Project-specific guidance lives next to each project (e.g.
`projects/eudsl-llvmpy/CLAUDE.md`).

## Submodules: do not initialize dawn's nested submodules

`third_party/dawn` is only used for Tint in the wasm/WebGPU build, which takes
SPIRV-Tools, SPIRV-Headers, and abseil-cpp from this repo's own submodules. None
of dawn's ~61 nested submodules (angle, swiftshader, mesa, libc++, catapult,
...) are needed. CI initializes submodules non-recursively
(`git submodule update --init --depth=1`), so they are never checked out there.

Keep them uninitialized locally too. Partially checked-out nested submodules
are invisible to `git status` (`.gitmodules` sets `ignore = dirty` on dawn, and
dawn's own status shows each nested one as a single ` M` line), but editors that
walk nested repos report every missing file -- once over 221k "deleted" files in
`third_party`.

`.gitmodules` cannot prevent this: it only configures this repo's direct
submodules, and dawn's nested ones are declared in dawn's own `.gitmodules`.
Instead, use `scripts/init_submodules.sh` (CI uses it too) rather than
`git submodule update --init` directly. It does the same shallow,
non-recursive init, then deinitializes dawn's nested submodules and marks them
all inactive in dawn's local config (`submodule.active ':(exclude)*'`), so a
later `git submodule update --init --recursive` skips them. Pass paths to init
only some submodules, e.g. `scripts/init_submodules.sh third_party/llvm-project`.

The inactive setting lives in `.git/modules/third_party/dawn/config`, so rerun
the script after a fresh clone. It also works around `deinit` failing to
unregister names containing regex characters (`libc++/src`, `libc++abi/src`).
