#!/usr/bin/env python3
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
from pathlib import Path
from difflib import unified_diff

LICENSE = """
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
""".strip().splitlines()


def main(file: str):
    filepath = Path(file)
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist")

    lines = filepath.read_text().splitlines()
    if "#!" in lines[0]:
        lines_to_check = lines[1:4]
    else:
        lines_to_check = lines[:3]

    diff = list(unified_diff(lines_to_check, LICENSE))
    if diff:
        sys.stderr.write(f"Expected license in file {filepath}!\n")
        sys.stderr.write("\n".join(diff) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
