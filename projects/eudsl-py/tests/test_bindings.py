from pathlib import Path

import pytest
import eudsl
from eudsl import VectorOfType, ArrayRefOfType
from eudsl.ir import (
    MLIRContext,
    Threading,
    Location,
    UnknownLoc,
    ModuleOp,
    Builder,
    DialectRegistry,
    OpBuilder,
    OperationState,
    FloatAttr,
    Float32Type,
    NamedAttribute,
    StringAttr,
    Dialect,
)
from eudsl.dialects.arith import ArithDialect, ConstantOp
from eudsl.dialects.cf import SwitchOp, ControlFlowDialect
from eudsl.dialects.scf import ConditionOp, SCFDialect


def test_arith_dialect():
    ctx = MLIRContext(Threading.DISABLED)
    ArithDialect.insert_into_registry(ctx.dialect_registry)
    ControlFlowDialect.insert_into_registry(ctx.dialect_registry)
    SCFDialect.insert_into_registry(ctx.dialect_registry)
    ctx.load_all_available_dialects()
    l = OpBuilder.Listener()
    b = OpBuilder(ctx, l)
    mod1 = ModuleOp.create(b.unknown_loc, "foo")
    b.set_insertion_point_to_start(mod1.body_region.blocks[0])
    op_state = OperationState(b.unknown_loc, ConstantOp.operation_name())
    f32_ty = Float32Type.get(ctx)
    f32_attr = FloatAttr.get(f32_ty, 1.0)
    str_attr = StringAttr.get(ctx, "value")
    op_state.add_attribute(str_attr, f32_attr)
    v = VectorOfType([f32_ty])
    op_state.add_types(v)
    op = b.create(op_state)
    mod1.operation.dump()
    assert mod1.verify()
