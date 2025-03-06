#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.
import argparse
import enum
from pathlib import Path

from .. import (
    RecordKeeper,
    collect_all_defs,
    collect_all_attr_or_type_defs,
)
from . import emit_decls_defns_nbclasses, CClassKind


# https://stackoverflow.com/a/60750535
class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


def emit_attrs_or_types(
    kind, rk, output_dir, output_prefix, include=None, exclude=None
):
    all_defs = collect_all_attr_or_type_defs(collect_all_defs(rk))
    decls, defns, nbclasses = emit_decls_defns_nbclasses(
        kind, all_defs, include, exclude
    )

    attr_decls = open(output_dir / f"{output_prefix}_{kind}_decls.h.inc", "w")
    attr_defns = open(output_dir / f"{output_prefix}_{kind}_defns.cpp.inc", "w")
    attr_nbclasses = open(output_dir / f"{output_prefix}_{kind}_nbclasses.cpp.inc", "w")
    for d in decls:
        if "LinearLayout" in d:
            continue
        print(d, file=attr_decls)
    for d in defns:
        if "LinearLayout" in d:
            continue
        print(d, file=attr_defns)
    for hdecls, hdefns, n in nbclasses:
        if "LinearLayout" in n:
            continue
        for h in hdecls:
            print(h, file=attr_decls)
        for h in hdefns:
            print(h, file=attr_defns)

        print(n, file=attr_nbclasses)


def main(args):
    defs_rk = RecordKeeper().parse_td(
        str(args.td_file),
        [str(ip) for ip in args.include_paths],
    )
    emit_attrs_or_types(
        args.kind,
        defs_rk,
        args.output_dir,
        args.output_prefix,
        include=args.include,
        exclude=args.exclude,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("td_file", type=Path)
    args.add_argument("-o", "--output-dir", type=Path, required=True)
    args.add_argument("-k", "--kind", type=CClassKind, action=EnumAction, required=True)
    args.add_argument("-I", "--include-paths", nargs="+", type=Path, required=True)
    args.add_argument("--exclude", nargs="*")
    args.add_argument("--include", nargs="*")

    args = args.parse_args()
    if args.include:
        args.include = set(args.include)
    else:
        args.include = None
    if args.exclude:
        args.exclude = set(args.exclude)
    else:
        args.exclude = None
    args.output_prefix = Path(args.td_file).stem

    main(args)
