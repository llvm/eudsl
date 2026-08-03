# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from mlir_python_mcp.server import MLIRMCPServer
from mlir_python_mcp.session import MLIRSession


@pytest.fixture
def session():
    s = MLIRSession("test")
    yield s
    s.destroy()


@pytest.fixture
def server():
    return MLIRMCPServer()


SIMPLE_MODULE = """\
func.func @add(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
"""

TWO_FUNC_MODULE = """\
func.func @foo(%a: i32) -> i32 {
  return %a : i32
}
func.func @bar(%x: f32) -> f32 {
  return %x : f32
}
"""
