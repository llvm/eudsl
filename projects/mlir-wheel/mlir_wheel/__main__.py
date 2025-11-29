import argparse
import sys

from . import bin_dir, cmake_dir, include_dir, lib_dir, root_dir


def main() -> None:
    parser = argparse.ArgumentParser("mlir_wheel")
    parser.add_argument(
        "--bin-dir",
        action="store_true",
        help="Print the path to the LLVM/MLIR distribution binary directory.",
    )
    parser.add_argument(
        "--cmake-dir",
        action="store_true",
        help="Print the path to the LLVM/MLIR distribution CMake module directory.",
    )
    parser.add_argument(
        "--include-dir",
        action="store_true",
        help="Print the path to the LLVM/MLIR distribution header directory.",
    )
    parser.add_argument(
        "--lib-dir",
        action="store_true",
        help="Print the path to the LLVM/MLIR distribution library directory.",
    )
    parser.add_argument(
        "--root-dir",
        action="store_true",
        help="Print the path to the LLVM/MLIR distribution root directory.",
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.bin_dir:
        print(bin_dir())
    if args.cmake_dir:
        print(cmake_dir())
    if args.include_dir:
        print(include_dir())
    if args.lib_dir:
        print(lib_dir())
    if args.root_dir:
        print(root_dir())


if __name__ == "__main__":
    main()
