#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2024.

from pathlib import Path

import pytest
from eudsl_tblgen import (
    RecordKeeper,
    get_requested_op_definitions,
    get_all_type_constraints,
    collect_all_defs,
)


@pytest.fixture(scope="function")
def json_record_keeper():
    return RecordKeeper().parse_td(str(Path(__file__).parent / "td" / "JSON.td"))


def test_json_record_keeper(json_record_keeper):
    assert json_record_keeper.get_input_filename() == str(
        Path(__file__).parent / "td" / "JSON.td"
    )

    assert set(json_record_keeper.get_classes()) == {
        "Base",
        "Derived",
        "Intermediate",
        "Variables",
    }

    assert set(json_record_keeper.get_defs().keys()) == {
        "D",
        "ExampleDagOp",
        "FieldKeywordTest",
        "Named",
        "VarNull",
        "VarObj",
        "VarPrim",
        "anonymous_0",
    }

    assert len(json_record_keeper.get_all_derived_definitions("Base")) == 1
    assert len(json_record_keeper.get_all_derived_definitions("Intermediate")) == 1
    assert len(json_record_keeper.get_all_derived_definitions("Derived")) == 0

    assert json_record_keeper.get_all_derived_definitions("Base")[0].get_name() == "D"
    assert (
        json_record_keeper.get_all_derived_definitions("Intermediate")[0].get_name()
        == "D"
    )


def test_record(json_record_keeper):
    assert json_record_keeper.get_classes()["Base"]
    assert json_record_keeper.get_classes()["Intermediate"]
    assert json_record_keeper.get_classes()["Derived"]
    assert json_record_keeper.get_classes()["Variables"]

    base_cl = json_record_keeper.get_classes()["Base"]
    interm_cl = json_record_keeper.get_classes()["Intermediate"]
    deriv_cl = json_record_keeper.get_classes()["Derived"]
    variab_cl = json_record_keeper.get_classes()["Variables"]

    assert len(base_cl.get_direct_super_classes()) == 0
    assert len(interm_cl.get_direct_super_classes()) == 1
    assert len(deriv_cl.get_direct_super_classes()) == 1
    assert len(variab_cl.get_direct_super_classes()) == 0

    assert interm_cl.get_direct_super_classes()[0].get_name() == "Base"
    assert deriv_cl.get_direct_super_classes()[0].get_name() == "Intermediate"

    assert base_cl.get_name() == "Base"
    assert base_cl.get_name_init_as_string() == "Base"
    assert base_cl.get_records() is json_record_keeper
    assert base_cl.get_type()

    assert repr(base_cl.get_values()) == "RecordValues()"
    assert (
        repr(variab_cl.get_values())
        == "RecordValues(i=?, s=?, b=?, bs={ ?, ?, ?, ?, ?, ?, ?, ? }, c=?, li=?, base=?, d=?)"
    )

    assert interm_cl.has_direct_super_class(interm_cl.get_direct_super_classes()[0])
    assert interm_cl.has_direct_super_class(base_cl)

    assert base_cl.is_anonymous() is False
    assert base_cl.is_class() is True
    assert base_cl.is_multi_class() is False

    assert interm_cl.is_sub_class_of(base_cl)
    assert not interm_cl.is_sub_class_of(variab_cl)
    assert not interm_cl.is_sub_class_of("Variables")


def test_record_val_classes(json_record_keeper):
    variab_cl = json_record_keeper.get_classes()["Variables"]
    assert variab_cl.get_value("i")
    i_val = variab_cl.get_value("i")
    assert i_val.get_name() == "i"
    assert i_val.get_name_init_as_string() == "i"
    assert i_val.get_print_type() == "int"
    assert i_val.get_record_keeper() is json_record_keeper
    assert i_val.is_nonconcrete_ok() is False
    assert i_val.is_template_arg() is False
    assert i_val.is_used() is False


def test_record_val_defs(json_record_keeper):
    var_prim_def = json_record_keeper.get_defs()["VarPrim"]
    assert var_prim_def.get_value_as_int("i") == 3
    assert var_prim_def.get_value_as_int("enormous_pos") == 9123456789123456789
    assert var_prim_def.get_value_as_int("enormous_neg") == -9123456789123456789
    assert var_prim_def.get_value_as_string("s") == "hello, world"
    assert var_prim_def.get_value_as_bit("b") is False
    assert var_prim_def.get_value_as_string("c") == " void  "
    assert var_prim_def.get_value_as_list_of_ints("li") == [1, 2, 3, 4]


def test_init(json_record_keeper):
    variab_cl = json_record_keeper.get_classes()["Variables"]
    assert variab_cl.get_value("i")
    assert variab_cl.get_value("i").get_value()
    i_val_init = variab_cl.get_value("i").get_value()
    assert str(i_val_init) == "?"
    assert i_val_init.get_as_string() == "?"
    assert i_val_init.is_complete() is False
    assert i_val_init.is_concrete() is True


def test_record_rec_ty(json_record_keeper):
    base_cl = json_record_keeper.get_classes()["Base"]
    interm_cl = json_record_keeper.get_classes()["Intermediate"]
    deriv_cl = json_record_keeper.get_classes()["Derived"]

    assert not base_cl.get_type().get_classes()
    assert interm_cl.get_type().get_classes()
    assert deriv_cl.get_type().get_classes()
    assert len(interm_cl.get_type().get_classes()) == 1
    assert len(deriv_cl.get_type().get_classes()) == 1
    assert interm_cl.get_type().get_classes()[0].get_name() == "Base"
    assert deriv_cl.get_type().get_classes()[0].get_name() == "Intermediate"

    assert interm_cl.get_type().is_sub_class_of(base_cl)
    assert deriv_cl.get_type().is_sub_class_of(interm_cl)


@pytest.fixture(scope="function")
def record_keeper_test_dialect():
    here = Path(__file__).parent
    return RecordKeeper().parse_td(
        str(here / "td" / "TestDialect.td"), [str(here / "td")]
    )


def test_init_complex(record_keeper_test_dialect):
    op = record_keeper_test_dialect.get_defs()["Test_TypesOp"]
    assert str(op.get_values().opName) == "types"
    assert str(op.get_values().cppNamespace) == "test"
    assert str(op.get_values().opDocGroup) == "?"
    assert str(op.get_values().results) == "(outs)"
    assert str(op.get_values().regions) == "(region)"
    assert str(op.get_values().successors) == "(successor)"
    assert str(op.get_values().builders) == "?"
    assert bool(op.get_values().skipDefaultBuilders.get_value()) is False
    assert str(op.get_values().assemblyFormat) == "?"
    assert bool(op.get_values().hasCustomAssemblyFormat.get_value()) is False
    assert bool(op.get_values().hasVerifier.get_value()) is False
    assert bool(op.get_values().hasRegionVerifier.get_value()) is False
    assert bool(op.get_values().hasCanonicalizer.get_value()) is False
    assert bool(op.get_values().hasCanonicalizeMethod.get_value()) is False
    assert bool(op.get_values().hasFolder.get_value()) is False
    assert bool(op.get_values().useCustomPropertiesEncoding.get_value()) is False
    assert len(op.get_values().traits.get_value()) == 0
    assert str(op.get_values().extraClassDeclaration) == "?"
    assert str(op.get_values().extraClassDefinition) == "?"

    assert (
        repr(op.get_values())
        == "RecordValues(opDialect=Test_Dialect, opName=types, cppNamespace=test, summary=, description=, opDocGroup=?, arguments=(ins I32:$a, SI64:$b, UI8:$c, Index:$d, F32:$e, NoneType:$f, anonymous_348), results=(outs), regions=(region), successors=(successor), builders=?, skipDefaultBuilders=0, assemblyFormat=?, hasCustomAssemblyFormat=0, hasVerifier=0, hasRegionVerifier=0, hasCanonicalizer=0, hasCanonicalizeMethod=0, hasFolder=0, useCustomPropertiesEncoding=0, traits=[], extraClassDeclaration=?, extraClassDefinition=?)"
    )

    arguments = op.get_values().arguments
    assert arguments.get_value().get_arg_name_str(0) == "a"
    assert arguments.get_value().get_arg_name_str(1) == "b"
    assert arguments.get_value().get_arg_name_str(2) == "c"
    assert arguments.get_value().get_arg_name_str(3) == "d"
    assert arguments.get_value().get_arg_name_str(4) == "e"
    assert arguments.get_value().get_arg_name_str(5) == "f"

    assert str(arguments.get_value()[0]) == "I32"
    assert str(arguments.get_value()[1]) == "SI64"
    assert str(arguments.get_value()[2]) == "UI8"
    assert str(arguments.get_value()[3]) == "Index"
    assert str(arguments.get_value()[4]) == "F32"
    assert str(arguments.get_value()[5]) == "NoneType"

    attr = record_keeper_test_dialect.get_defs()["Test_TestAttr"]
    assert str(attr.get_values().predicate) == "anonymous_335"
    assert str(attr.get_values().storageType) == "test::TestAttr"
    assert str(attr.get_values().returnType) == "test::TestAttr"
    assert (
        str(attr.get_values().convertFromStorage.get_value())
        == "::llvm::cast<test::TestAttr>($_self)"
    )
    assert str(attr.get_values().constBuilderCall) == "?"
    assert str(attr.get_values().defaultValue) == "?"
    assert str(attr.get_values().valueType) == "?"
    assert bool(attr.get_values().isOptional.get_value()) is False
    assert str(attr.get_values().baseAttr) == "?"
    assert str(attr.get_values().cppNamespace) == "test"
    assert str(attr.get_values().dialect) == "Test_Dialect"
    assert str(attr.get_values().cppBaseClassName.get_value()) == "::mlir::Attribute"
    assert str(attr.get_values().storageClass) == "TestAttrStorage"
    assert str(attr.get_values().storageNamespace) == "detail"
    assert bool(attr.get_values().genStorageClass.get_value()) is True
    assert bool(attr.get_values().hasStorageCustomConstructor.get_value()) is False
    assert str(attr.get_values().parameters) == "(ins)"
    assert str(attr.get_values().builders) == "?"
    assert len(attr.get_values().traits.get_value()) == 0
    assert str(attr.get_values().mnemonic) == "test"
    assert str(attr.get_values().assemblyFormat) == "?"
    assert bool(attr.get_values().hasCustomAssemblyFormat.get_value()) is False
    assert bool(attr.get_values().genAccessors.get_value()) is True
    assert bool(attr.get_values().skipDefaultBuilders.get_value()) is False
    assert bool(attr.get_values().genVerifyDecl.get_value()) is False
    assert str(attr.get_values().cppClassName) == "TestAttr"
    assert str(attr.get_values().cppType) == "test::TestAttr"
    assert str(attr.get_values().attrName) == "test.test"


def test_mlir_tblgen(record_keeper_test_dialect):
    for op in get_requested_op_definitions(record_keeper_test_dialect):
        print(op.get_name())
    for constraint in get_all_type_constraints(record_keeper_test_dialect):
        print(constraint.get_def_name())
        print(constraint.get_summary())

    all_defs = collect_all_defs(record_keeper_test_dialect)
    for d in all_defs:
        print(d.get_name())


def test_cta_layout(record_keeper_test_dialect):
    all_defs = collect_all_defs(record_keeper_test_dialect)
    for d in all_defs:
        if d.get_name() == "CTALayoutAttr":
            for p in d.get_parameters():
                print(p.get_cpp_type())
