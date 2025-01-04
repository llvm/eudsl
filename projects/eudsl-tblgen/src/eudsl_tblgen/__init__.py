#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Copyright (c) 2024.

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
        def_record.dump()
        defs.append(def_record)

    return defs
