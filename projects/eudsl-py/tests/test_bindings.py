from pathlib import Path

import pytest
from eudsl import MLIRContext, Threading


def test_mlir_context():
    m = MLIRContext(Threading.DISABLED)
    print(m)
    m = MLIRContext(Threading.ENABLED)
    print(m)
