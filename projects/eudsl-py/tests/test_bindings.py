from eudsl import ArrayRef, SmallVector
from eudsl.dialects.arith import ArithDialect, ConstantOp
from eudsl.dialects.cf import ControlFlowDialect
from eudsl.dialects.scf import SCFDialect
from eudsl.ir import (
    MLIRContext,
    Threading,
    ModuleOp,
    OpBuilder,
    OperationState,
    FloatAttr,
    Float32Type,
    StringAttr,
)
from eudsl.ir import Type


def test_array_ref():
    tys = [Type(), Type(), Type()]
    print(ArrayRef[Type])
    arr_ref = ArrayRef[Type](tys)
    print(arr_ref)
    v = SmallVector[int]([666, 2, 1])
    for vv in v:
        print(vv)
    tys = ArrayRef[int](v)
    for t in tys:
        print(t)
    ctx = MLIRContext(Threading.DISABLED)
    f32_ty = Float32Type.get(ctx)
    v = SmallVector[Type]([f32_ty])
    tys = ArrayRef[Type](v)
    for t in tys:
        t.dump()


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
    f32_ty = Float32Type.get(ctx)
    f32_attr = FloatAttr.get(f32_ty, 1.0)
    str_attr = StringAttr.get(ctx, "value")

    op_state = OperationState(b.unknown_loc, ConstantOp.operation_name())
    op_state.add_attribute(str_attr, f32_attr)
    v = SmallVector[Type]([f32_ty])
    tys = ArrayRef[Type](v)
    op_state.add_types(tys)
    op = b.create(op_state)

    op_state = OperationState(b.unknown_loc, ConstantOp.operation_name())
    op_state.add_attribute(str_attr, f32_attr)
    v = SmallVector[Type]([f32_ty])
    op_state.add_types(v)
    op = b.create(op_state)

    mod1.operation.dump()
    assert mod1.verify()
