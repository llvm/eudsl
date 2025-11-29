#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os

from setuptools import setup


version = os.environ.get("WHEEL_VERSION", "0.0.0+DEADBEEF").replace("_", "+")

setup(
    version=version,
    name="mlir_wheel",
    include_package_data=True,
)
