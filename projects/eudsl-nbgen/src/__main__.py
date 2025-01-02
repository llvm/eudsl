#  Copyright (c) 2025.
#
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import sys

from . import cmake_dir


def main() -> None:
    parser = argparse.ArgumentParser("eudsl-nbgen")
    parser.add_argument(
        "--cmake_dir",
        action="store_true",
        help="Print the path to the eudsl-nbgen CMake module directory.",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.cmake_dir:
        print(cmake_dir())


if __name__ == "__main__":
    main()
