#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2024.
from typing import List, Optional

from .eudsl_tblgen_ext import *


import re


def get_operation_name(def_record):
    prefix = def_record.get_value_as_def("opDialect").get_value_as_string("name")
    op_name = def_record.get_value_as_string("opName")

    if not prefix:
        return op_name
    return f"{prefix}.{op_name}"


def get_requested_op_definitions(records, op_inc_filter=None, op_exc_filter=None):
    class_def = records.get_class("Op")
    if not class_def:
        raise RuntimeError("ERROR: Couldn't find the 'Op' class!")

    if op_inc_filter:
        include_regex = re.compile(op_inc_filter)
    if op_exc_filter:
        exclude_regex = re.compile(op_exc_filter)
    defs = []

    for def_name in records.get_defs():
        def_record = records.get_defs()[def_name]
        if not def_record.is_sub_class_of(class_def):
            continue
        # Include if no include filter or include filter matches.
        if op_inc_filter and not include_regex.match(get_operation_name(def_record)):
            continue
        # Unless there is an exclude filter and it matches.
        if op_exc_filter and exclude_regex.match(get_operation_name(def_record)):
            continue
        defs.append(def_record)

    return defs


def collect_all_defs(
    record_keeper: RecordKeeper,
    selected_dialect: Optional[str] = None,
) -> List[Record]:
    records = record_keeper.get_defs()
    records = [records[d] for d in records]
    # Nothing to do if no defs were found.
    if not records:
        return []

    defs = [rec for rec in records if rec.get_value("dialect")]
    result_defs = []

    if not selected_dialect:
        # If a dialect was not specified, ensure that all found defs belong to the same dialect.
        dialects = {d.get_value("dialect").get_value().get_as_string() for d in defs}
        if len(dialects) > 1:
            raise RuntimeError(
                "Defs belong to more than one dialect. Must select one via '--(attr|type)defs-dialect'"
            )
        result_defs.extend(defs)
    else:
        # Otherwise, generate the defs that belong to the selected dialect.
        dialect_defs = [
            d
            for d in defs
            if d.get_value("dialect").get_value().get_as_string() == selected_dialect
        ]
        result_defs.extend(dialect_defs)

    return result_defs


def collect_all_attr_or_type_defs(records):
    return [
        AttrOrTypeDef(rec)
        for rec in records
        if rec.get_value("builders") and rec.get_value("parameters")
    ]


def get_all_type_constraints(records: RecordKeeper) -> List[Constraint]:
    result = []
    for record in records.get_all_derived_definitions_if_defined("TypeConstraint"):
        # Ignore constraints defined outside of the top-level file.
        constr = Constraint(record)
        # Generate C++ function only if "cppFunctionName" is set.
        if not constr.get_cpp_function_name():
            continue
        result.append(constr)
    return result
