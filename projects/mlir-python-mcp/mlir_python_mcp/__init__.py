# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio

from .server import main as _main


def main():  # pragma: no cover
    asyncio.run(_main())


__all__ = ["main"]
