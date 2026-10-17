#!/usr/bin/env bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2026.

# Initialize this repo's submodules the way CI does (shallow, non-recursive)
# and keep dawn's nested submodules uninitialized. Tint only needs dawn's own
# tree; the nested ones are never used, and partially checked-out copies show
# up in editors as hundreds of thousands of deleted files. Extra arguments are
# passed through as submodule paths, e.g. `init_submodules.sh third_party/llvm-project`.

set -euo pipefail

TD="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$TD/.." && pwd)"
DAWN="$REPO_ROOT/third_party/dawn"

git -C "$REPO_ROOT" submodule update --init --depth=1 "$@"

if [ -e "$DAWN/.git" ]; then
  # A pathspec that matches nothing makes every nested submodule inactive, so a
  # later `git submodule update --init --recursive` skips them.
  git -C "$DAWN" config submodule.active ':(exclude)*'
  git -C "$DAWN" submodule deinit --all -f >/dev/null 2>&1 || true
  # deinit builds a regex from each name and fails to unregister names with
  # regex characters (third_party/libc++/src), so drop leftovers literally.
  { git -C "$DAWN" config --name-only --get-regexp \
      '^submodule\..*\.(url|active|update|branch)$' || true; } |
    sed -E 's/\.[^.]+$//' | sort -u |
    while read -r section; do
      git -C "$DAWN" config --remove-section "$section" 2>/dev/null || true
    done
fi
