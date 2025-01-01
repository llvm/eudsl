#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2024.

import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent


def make_source_shards(filename: Path, target, extra_includes, max_num_shards):
    assert filename.name.endswith("cpp.gen"), "expected .cpp.gen file"
    with open(filename) as f:
        source = f.read()
    shards = re.split(r"// eudslpy-gen-shard \d+", source)
    if target is None:
        target = filename.stem.split(".")[0]
    for i, shar in enumerate(shards):
        with open(f"{filename}.shard.{i}.cpp", "w") as f:
            if extra_includes is not None:
                for inc in extra_includes:
                    print(f'#include "{inc}"', file=f)
            print(
                dedent(
                    f"""\
            #include <nanobind/nanobind.h>
            namespace nb = nanobind;
            using namespace nb::literals;
            using namespace mlir;
            using namespace llvm;
            #include "type_casters.h"
            
            void populate{target}{i}Module(nb::module_ &m) {{"""
                ),
                file=f,
            )
            print(shar, file=f)
            print("}", file=f)

    if len(shards) > max_num_shards:
        raise RuntimeError("expected less than 20 shards")
    for i in range(len(shards), max_num_shards):
        with open(f"{filename}.shard.{i}.cpp", "w") as f:
            print(f"// dummy shard {i}", file=f)

    with open(f"{filename}.sharded.cpp", "w") as f:
        print(
            dedent(
                f"""\
            #include <nanobind/nanobind.h>
            namespace nb = nanobind;
            using namespace nb::literals;
            void populate{target}Module(nb::module_ &m) {{"""
            ),
            file=f,
        )
        for i in range(len(shards)):
            print(
                dedent(f"extern void populate{target}{i}Module(nb::module_ &m);"),
                file=f,
            )
        for i in range(len(shards)):
            print(dedent(f"populate{target}{i}Module(m);"), file=f)

        print("}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-t", "--target")
    parser.add_argument("-I", "--extra_includes", nargs="*")
    parser.add_argument("-m", "--max-num-shards", type=int, default=20)
    args = parser.parse_args()
    make_source_shards(
        Path(args.filename), args.target, args.extra_includes, args.max_num_shards + 1
    )
