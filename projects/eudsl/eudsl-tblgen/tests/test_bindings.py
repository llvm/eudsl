from pathlib import Path

import pytest
from eudsl_tblgen import RecordKeeper


@pytest.fixture(scope="function")
def json_record_keeper():
    return RecordKeeper().parse_td(str(Path(__file__).parent / "JSON.td"))


def test_json_record_keeper(json_record_keeper):
    assert json_record_keeper.input_filename == str(Path(__file__).parent / "JSON.td")

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

    assert len(base_cl.values) == 0
    assert len(variab_cl.values) == 8

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


# @pytest.fixture(scope="function")
# def constraints_record_keeper():
#     return RecordKeeper().parse_td(
#         str(Path(__file__).parent / "CommonTypeConstraints.td")
#     )
#
#
# def test_init_complex(constraints_record_keeper):
#     print(list(constraints_record_keeper.classes.keys()))
#     print(list(constraints_record_keeper.defs.keys()))
#     print(list(constraints_record_keeper.globals.keys()))
