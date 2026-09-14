#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from ..eudslllvm_ext.mir import *  # noqa: F401,F403
from ..eudslllvm_ext.mir import __doc__  # noqa: F401

from .strategies import ReadyQueueStrategy, BasicRegAlloc  # noqa: F401
from .greedy import RAGreedy  # noqa: F401
from . import ilp  # noqa: F401
