#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2024.

from pathlib import Path

import pytest
from eudsl_tblgen import RecordKeeper


@pytest.fixture(scope="function")
def json_record_keeper():
    return RecordKeeper().parse_td(str(Path(__file__).parent / "td" / "JSON.td"))


def test_json_record_keeper(json_record_keeper):
    assert json_record_keeper.input_filename == str(
        Path(__file__).parent / "td" / "JSON.td"
    )

    assert set(json_record_keeper.classes) == {
        "Base",
        "Derived",
        "Intermediate",
        "Variables",
    }

    assert set(json_record_keeper.defs.keys()) == {
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

    assert json_record_keeper.get_all_derived_definitions("Base")[0].name == "D"
    assert json_record_keeper.get_all_derived_definitions("Intermediate")[0].name == "D"


def test_record(json_record_keeper):
    assert json_record_keeper.classes["Base"]
    assert json_record_keeper.classes["Intermediate"]
    assert json_record_keeper.classes["Derived"]
    assert json_record_keeper.classes["Variables"]

    base_cl = json_record_keeper.classes["Base"]
    interm_cl = json_record_keeper.classes["Intermediate"]
    deriv_cl = json_record_keeper.classes["Derived"]
    variab_cl = json_record_keeper.classes["Variables"]

    assert len(base_cl.direct_super_classes) == 0
    assert len(interm_cl.direct_super_classes) == 1
    assert len(deriv_cl.direct_super_classes) == 1
    assert len(variab_cl.direct_super_classes) == 0

    assert interm_cl.direct_super_classes[0].name == "Base"
    assert deriv_cl.direct_super_classes[0].name == "Intermediate"

    assert base_cl.name == "Base"
    assert base_cl.name_init_as_string == "Base"
    assert base_cl.records is json_record_keeper
    assert base_cl.type

    assert repr(base_cl.values) == "RecordValues()"
    assert (
        repr(variab_cl.values)
        == "RecordValues(i=?, s=?, b=?, bs={ ?, ?, ?, ?, ?, ?, ?, ? }, c=?, li=?, base=?, d=?)"
    )

    assert interm_cl.has_direct_super_class(interm_cl.direct_super_classes[0])
    assert interm_cl.has_direct_super_class(base_cl)

    assert base_cl.is_anonymous is False
    assert base_cl.is_class is True
    assert base_cl.is_multi_class is False

    assert interm_cl.is_sub_class_of(base_cl)
    assert not interm_cl.is_sub_class_of(variab_cl)
    assert not interm_cl.is_sub_class_of("Variables")


def test_record_val_classes(json_record_keeper):
    variab_cl = json_record_keeper.classes["Variables"]
    assert variab_cl.get_value("i")
    i_val = variab_cl.get_value("i")
    assert i_val.name == "i"
    assert i_val.name_init_as_string == "i"
    assert i_val.print_type == "int"
    assert i_val.record_keeper is json_record_keeper
    assert i_val.is_nonconcrete_ok is False
    assert i_val.is_template_arg is False
    assert i_val.is_used is False


def test_record_val_defs(json_record_keeper):
    var_prim_def = json_record_keeper.defs["VarPrim"]
    assert var_prim_def.get_value_as_int("i") == 3
    assert var_prim_def.get_value_as_int("enormous_pos") == 9123456789123456789
    assert var_prim_def.get_value_as_int("enormous_neg") == -9123456789123456789
    assert var_prim_def.get_value_as_string("s") == "hello, world"
    assert var_prim_def.get_value_as_bit("b") is False
    assert var_prim_def.get_value_as_string("c") == " void  "
    assert var_prim_def.get_value_as_list_of_ints("li") == [1, 2, 3, 4]


def test_init(json_record_keeper):
    variab_cl = json_record_keeper.classes["Variables"]
    assert variab_cl.get_value("i")
    assert variab_cl.get_value("i").value
    i_val_init = variab_cl.get_value("i").value
    assert str(i_val_init) == "?"
    assert i_val_init.as_string == "?"
    assert i_val_init.is_complete() is False
    assert i_val_init.is_concrete() is True


def test_record_rec_ty(json_record_keeper):
    base_cl = json_record_keeper.classes["Base"]
    interm_cl = json_record_keeper.classes["Intermediate"]
    deriv_cl = json_record_keeper.classes["Derived"]

    assert not base_cl.type.classes
    assert interm_cl.type.classes
    assert deriv_cl.type.classes
    assert len(interm_cl.type.classes) == 1
    assert len(deriv_cl.type.classes) == 1
    assert interm_cl.type.classes[0].name == "Base"
    assert deriv_cl.type.classes[0].name == "Intermediate"

    assert interm_cl.type.is_sub_class_of(base_cl)
    assert deriv_cl.type.is_sub_class_of(interm_cl)


@pytest.fixture(scope="function")
def record_keeper_test_dialect():
    here = Path(__file__).parent
    return RecordKeeper().parse_td(
        str(here / "td" / "TestDialect.td"), [str(here / "td")]
    )


def test_init_complex(record_keeper_test_dialect):
    op = record_keeper_test_dialect.defs["Test_TypesOp"]
    assert str(op.values.opName) == "types"
    assert str(op.values.cppNamespace) == "test"
    assert str(op.values.opDocGroup) == "?"
    assert str(op.values.results) == "(outs)"
    assert str(op.values.regions) == "(region)"
    assert str(op.values.successors) == "(successor)"
    assert str(op.values.builders) == "?"
    assert bool(op.values.skipDefaultBuilders.value) is False
    assert str(op.values.assemblyFormat) == "?"
    assert bool(op.values.hasCustomAssemblyFormat.value) is False
    assert bool(op.values.hasVerifier.value) is False
    assert bool(op.values.hasRegionVerifier.value) is False
    assert bool(op.values.hasCanonicalizer.value) is False
    assert bool(op.values.hasCanonicalizeMethod.value) is False
    assert bool(op.values.hasFolder.value) is False
    assert bool(op.values.useCustomPropertiesEncoding.value) is False
    assert len(op.values.traits.value) == 0
    assert str(op.values.extraClassDeclaration) == "?"
    assert str(op.values.extraClassDefinition) == "?"

    assert (
        repr(op.values)
        == "RecordValues(opDialect=Test_Dialect, opName=types, cppNamespace=test, summary=, description=, opDocGroup=?, arguments=(ins I32:$a, SI64:$b, UI8:$c, Index:$d, F32:$e, NoneType:$f, anonymous_347), results=(outs), regions=(region), successors=(successor), builders=?, skipDefaultBuilders=0, assemblyFormat=?, hasCustomAssemblyFormat=0, hasVerifier=0, hasRegionVerifier=0, hasCanonicalizer=0, hasCanonicalizeMethod=0, hasFolder=0, useCustomPropertiesEncoding=0, traits=[], extraClassDeclaration=?, extraClassDefinition=?)"
    )

    arguments = op.values.arguments
    assert arguments.value.get_arg_name_str(0) == "a"
    assert arguments.value.get_arg_name_str(1) == "b"
    assert arguments.value.get_arg_name_str(2) == "c"
    assert arguments.value.get_arg_name_str(3) == "d"
    assert arguments.value.get_arg_name_str(4) == "e"
    assert arguments.value.get_arg_name_str(5) == "f"

    assert str(arguments.value[0]) == "I32"
    assert str(arguments.value[1]) == "SI64"
    assert str(arguments.value[2]) == "UI8"
    assert str(arguments.value[3]) == "Index"
    assert str(arguments.value[4]) == "F32"
    assert str(arguments.value[5]) == "NoneType"

    attr = record_keeper_test_dialect.defs["Test_TestAttr"]
    assert str(attr.values.predicate) == "anonymous_334"
    assert str(attr.values.storageType) == "test::TestAttr"
    assert str(attr.values.returnType) == "test::TestAttr"
    assert (
        str(attr.values.convertFromStorage.value)
        == "::llvm::cast<test::TestAttr>($_self)"
    )
    assert str(attr.values.constBuilderCall) == "?"
    assert str(attr.values.defaultValue) == "?"
    assert str(attr.values.valueType) == "?"
    assert bool(attr.values.isOptional.value) is False
    assert str(attr.values.baseAttr) == "?"
    assert str(attr.values.cppNamespace) == "test"
    assert str(attr.values.dialect) == "Test_Dialect"
    assert str(attr.values.cppBaseClassName.value) == "::mlir::Attribute"
    assert str(attr.values.storageClass) == "TestAttrStorage"
    assert str(attr.values.storageNamespace) == "detail"
    assert bool(attr.values.genStorageClass.value) is True
    assert bool(attr.values.hasStorageCustomConstructor.value) is False
    assert str(attr.values.parameters) == "(ins)"
    assert str(attr.values.builders) == "?"
    assert len(attr.values.traits.value) == 0
    assert str(attr.values.mnemonic) == "test"
    assert str(attr.values.assemblyFormat) == "?"
    assert bool(attr.values.hasCustomAssemblyFormat.value) is False
    assert bool(attr.values.genAccessors.value) is True
    assert bool(attr.values.skipDefaultBuilders.value) is False
    assert bool(attr.values.genVerifyDecl.value) is False
    assert str(attr.values.cppClassName) == "TestAttr"
    assert str(attr.values.cppType) == "test::TestAttr"
    assert str(attr.values.attrName) == "test.test"
