#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2025.

from .eudslllvm_ext import *
from .eudslllvm_ext import __doc__

from .dsl.values import install_value_casters as _install_value_casters

_install_value_casters()

from .dsl.func import function
from .dsl.cf import yield_, range_
