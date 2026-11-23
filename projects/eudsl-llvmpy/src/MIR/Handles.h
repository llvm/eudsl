// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "IR/Common.h"

namespace eudsl {

nb::object trackMachineHandles(nb::handle value, nb::handle parent = {});

struct TrackMachineHandles {
  static void precall(PyObject **, size_t, nb::detail::cleanup_list *) {}
  static void postcall(PyObject **args, size_t, PyObject *&result) {
    trackMachineHandles(args[0]);
    nb::object value = trackMachineHandles(result, args[0]);
    nb::handle(result).dec_ref();
    result = value.release().ptr();
  }
};

} // namespace eudsl
