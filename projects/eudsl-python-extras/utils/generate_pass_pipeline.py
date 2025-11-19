# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
import glob
import json
import keyword
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE
from textwrap import dedent, indent


def dump_json(td_path: Path, root_include_path: Path):
    llvm_tblgen_name = "llvm-tblgen"
    if platform.system() == "Windows":
        llvm_tblgen_name += ".exe"

    # try from mlir-native-tools
    llvm_tblgen_path = Path(sys.prefix) / "bin" / llvm_tblgen_name
    # try to find using which
    if not llvm_tblgen_path.exists():
        llvm_tblgen_path = shutil.which(llvm_tblgen_name)
    assert Path(llvm_tblgen_path).exists() is not None, "couldn't find llvm-tblgen"

    args = [f"-I{root_include_path}", f"-I{td_path.parent}", str(td_path), "-dump-json"]
    res = subprocess.run(
        [llvm_tblgen_path] + args,
        cwd=Path(".").cwd(),
        check=True,
        stdout=PIPE,
        stderr=subprocess.DEVNULL,
    )
    res_json = json.loads(res.stdout.decode("utf-8"))

    return res_json


@dataclass
class Option:
    argument: str
    description: str
    type: str
    additional_opt_flags: str
    default_value: str
    list_option: bool = False


@dataclass
class Pass:
    name: str
    argument: str
    options: list[Option]
    description: str
    summary: str


TYPE_MAP = {
    "::mlir::gpu::amd::Runtime": '"gpu::amd::Runtime"',
    "OpPassManager": '"OpPassManager"',
    "bool": "bool",
    "double": "float",
    "enum FusionMode": '"FusionMode"',
    "int": "int",
    "int32_t": "int",
    "int64_t": "int",
    "mlir::SparseParallelizationStrategy": "SparseParallelizationStrategy",
    "mlir::arm_sme::ArmStreaming": '"arm_sme::ArmStreaming"',
    "std::string": "str",
    "uint64_t": "int",
    "unsigned": "int",
    "mlir::GreedySimplifyRegionLevel": "GreedySimplifyRegionLevel",
}


def print_options_doc_string(pass_, ident, output_file):
    print(
        indent(
            f'"""{pass_.summary}',
            prefix=" " * ident * 2,
        ),
        file=output_file,
    )
    if pass_.description:
        for l in pass_.description.split("\n"):
            print(
                indent(
                    f"{l}",
                    prefix=" " * ident,
                ),
                file=output_file,
            )
    if pass_.options:
        print(
            indent(
                f"Args:",
                prefix=" " * ident * 2,
            ),
            file=output_file,
        )
        for o in pass_.options:
            print(
                indent(
                    f"{o.argument}: {o.description}",
                    prefix=" " * ident * 3,
                ),
                file=output_file,
            )
    print(
        indent(
            f'"""',
            prefix=" " * ident * 2,
        ),
        file=output_file,
    )


def generate_pass_method(pass_: Pass, output_file):
    ident = 4
    py_args = []
    for o in pass_.options:
        argument = o.argument.replace("-", "_")
        if keyword.iskeyword(argument):
            argument += "_"
        type = TYPE_MAP.get(o.type, f"'{o.type}'")
        if o.list_option:
            type = f"List[{type}]"
        py_args.append((argument, type))

    pass_name = pass_.argument
    if py_args:
        py_args_str = ", ".join([f"{n}: {t} = None" for n, t in py_args])
        print(
            indent(
                f"def {pass_name.replace('-', '_')}(self, {py_args_str}):",
                prefix=" " * ident,
            ),
            file=output_file,
        )
        print_options_doc_string(pass_, ident, output_file)

        mlir_args = []
        for n, t in py_args:
            if "list" in t:
                print(
                    indent(
                        f"if {n} is not None and isinstance({n}, (list, tuple)):",
                        prefix=" " * ident * 2,
                    ),
                    file=output_file,
                )
                print(
                    indent(f"{n} = ','.join(map(str, {n}))", prefix=" " * ident * 3),
                    file=output_file,
                )
            mlir_args.append(f"{n}={n}")
        print(
            indent(
                dedent(
                    f"""\
                        self.add_pass("{pass_name}", {", ".join(mlir_args)})
                        return self
                    """
                ),
                prefix=" " * ident * 2,
            ),
            file=output_file,
        )

    else:
        print(
            indent(
                dedent(
                    f"""\
                        def {pass_name.replace("-", "_")}(self):"""
                ),
                prefix=" " * ident,
            ),
            file=output_file,
        )
        print_options_doc_string(pass_, ident, output_file)
        print(
            indent(
                dedent(
                    f"""\
                    self.add_pass("{pass_name}")
                    return self
                    """
                ),
                prefix=" " * ident * 2,
            ),
            file=output_file,
        )


def gather_passes_from_td_json(j):
    passes = []
    for pass_ in j["!instanceof"]["Pass"] + j["!instanceof"]["InterfacePass"]:
        pass_ = j[pass_]
        options = []
        for o in pass_["options"]:
            option = j[o["def"]]
            option = Option(
                argument=option["argument"],
                description=option["description"],
                type=option["type"],
                additional_opt_flags=option["additionalOptFlags"],
                default_value=option["defaultValue"],
                list_option="ListOption" in option["!superclasses"],
            )
            options.append(option)
        pass_ = Pass(
            name=pass_["!name"],
            argument=pass_["argument"],
            options=options,
            description=pass_["description"],
            summary=pass_["summary"],
        )
        passes.append(pass_)

    return passes


HEADER = """\
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

####################################################
# NOTE: This file is auto-generated using utils/generate_pass_pipeline.py. 
# DO NOT add functionality here (instead add to _passes_base.py)
####################################################

from ._passes_base import *

class Pipeline(Pipeline):
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_include_path", type=Path)
    parser.add_argument("-o", "--output", type=argparse.FileType("w"), default=None)
    args = parser.parse_args()
    if args.output is not None:
        output_file = args.output
    else:
        output_file = sys.stdout

    passes = []
    for td in glob.glob(str(args.root_include_path / "**" / "*.td"), recursive=True):
        try:
            j = dump_json(Path(td), args.root_include_path)
            if j["!instanceof"]["Pass"]:
                passes.extend(gather_passes_from_td_json(j))
        except Exception as e:
            print(f"Error parsing {td}: {e}")
            continue

    output_file.write(HEADER)
    for p in sorted(passes, key=lambda p: p.argument):
        generate_pass_method(p, output_file)
