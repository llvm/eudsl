# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from pip._internal.req import parse_requirements
from setuptools import find_namespace_packages, find_packages
from setuptools import setup

# By setting this env variable during install, this package (a namespace package)
# can "squat" in some host bindings namespace.
HOST_MLIR_PYTHON_PACKAGE_PREFIX = os.environ.get(
    "HOST_MLIR_PYTHON_PACKAGE_PREFIX", "mlir"
)


def load_requirements(fname):
    reqs = parse_requirements(fname, session="hack")
    return [str(ir.requirement) for ir in reqs]


packages = (
    [HOST_MLIR_PYTHON_PACKAGE_PREFIX]
    + [f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras"]
    + [
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras.{p}"
        for p in find_namespace_packages(where="mlir/extras")
        + find_packages(where="mlir/extras")
    ]
)

setup(
    name="eudsl-python-extras",
    version="0.1.0",
    description="The missing pieces (as far as boilerplate reduction goes) of the upstream MLIR python bindings.",
    license="LICENSE",
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "test": ["pytest", "mlir-native-tools", "astpretty"],
        "mlir": ["mlir-python-bindings"],
    },
    python_requires=">=3.8",
    packages=packages,  # lhs is package namespace, rhs is path (relative to this setup.py)
    package_dir={
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}": "mlir",
        f"{HOST_MLIR_PYTHON_PACKAGE_PREFIX}.extras": "mlir/extras",
    },
)
