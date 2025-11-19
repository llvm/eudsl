# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from datetime import datetime
from pathlib import Path

from pip._internal.req import parse_requirements
from setuptools import find_namespace_packages, find_packages
from setuptools import setup

# By setting this env variable during install, this package (a namespace package)
# can "squat" in some host bindings namespace. Note, this DOES work even for nested
# packages like EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX=foo.bar (extras will be installed into foo/bar/extras).
EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX = os.environ.get(
    "EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX", "mlir"
)

HERE = Path(__file__).parent


def load_requirements(fname):
    reqs = parse_requirements(fname, session="hack")
    return [str(ir.requirement) for ir in reqs]


packages = (
    [EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX]
    + [f"{EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX}.extras"]
    + [
        f"{EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX}.extras.{p}"
        for p in find_namespace_packages(where="mlir/extras")
        + find_packages(where="mlir/extras")
    ]
)

version = "0.1.0"
WHEEL_VERSION = os.getenv("WHEEL_VERSION", "XXXWHEEL_VERSIONXXX")
if WHEEL_VERSION is not None and not WHEEL_VERSION.startswith("XXX"):
    version += "." + WHEEL_VERSION

setup(
    name="eudsl-python-extras",
    version=version,
    description="The missing pieces (as far as boilerplate reduction goes) of the upstream MLIR python bindings.",
    license="LICENSE",
    install_requires=load_requirements(str(HERE / "requirements.txt")),
    extras_require={
        "test": [
            "pytest",
            "mlir-native-tools",
            "astpretty",
            "black",
            "pre-commit",
            "pre-commit-hooks",
        ],
        "mlir": ["mlir-python-bindings"],
    },
    python_requires=">=3.8",
    include_package_data=True,
    packages=packages,
    # lhs is package namespace, rhs is path (relative to this setup.py)
    package_dir={
        f"{EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX}": "mlir",
        f"{EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX}.extras": "mlir/extras",
    },
)
