import numpy as np
from eudsl import ArrayRef, SmallVector
from eudsl import dialects
from eudsl import ir


def test_array_ref():
    v = SmallVector[int]([666, 2, 1])
    for vv in v:
        print(vv)
    tys = ArrayRef(v)
    for t in tys:
        print(t)

    v = SmallVector[float]([666.0, 2.0, 1.0])
    for vv in v:
        print(vv)
    tys = ArrayRef(v)
    for t in tys:
        print(t)

    v = SmallVector[bool]([True, False, True])
    for vv in v:
        print(vv)
    tys = ArrayRef(v)
    for t in tys:
        print(t)

    print(SmallVector[np.int16])
    print(SmallVector[np.int32])
    print(SmallVector[np.int64])

    ctx = ir.MLIRContext(ir.Threading.DISABLED)
    f32_ty = ir.Float32Type.get(ctx)
    v = SmallVector[ir.Type]([f32_ty])
    tys = ArrayRef(v)
    for t in tys:
        t.dump()

    attrs = [ir.Attribute(), ir.Attribute(), ir.Attribute()]
    v = SmallVector[ir.Attribute](attrs)
    tys = ArrayRef(v)
    for t in tys:
        t.dump()


def test_arith_dialect():
    ctx = ir.MLIRContext(ir.Threading.DISABLED)
    dialects.arith.ArithDialect.insert_into_registry(ctx.dialect_registry)
    ctx.load_all_available_dialects()
    l = ir.OpBuilder.Listener()
    b = ir.OpBuilder(ctx, l)
    mod1 = ir.ModuleOp.create(b.unknown_loc, "foo")
    b.set_insertion_point_to_start(mod1.body_region.blocks[0])
    f32_ty = ir.Float32Type.get(ctx)
    f32_attr = ir.FloatAttr.get(f32_ty, 1.0)
    str_attr = ir.StringAttr.get(ctx, "value")

    op_state = ir.OperationState(
        b.unknown_loc, dialects.arith.ConstantOp.get_operation_name()
    )
    op_state.add_attribute(str_attr, f32_attr)
    v = SmallVector[ir.Type]([f32_ty])
    tys = ArrayRef(v)
    op_state.add_types(tys)
    op = b.create(op_state)

    op_state = ir.OperationState(
        b.unknown_loc, dialects.arith.ConstantOp.get_operation_name()
    )
    op_state.add_attribute(str_attr, f32_attr)
    v = SmallVector[ir.Type]([f32_ty])
    op_state.add_types(v)
    op = b.create(op_state)

    mod1.operation.dump()
    assert mod1.verify()


if __name__ == "__main__":
    test_array_ref()
    test_arith_dialect()
