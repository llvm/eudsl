// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nanobind/nanobind.h>

#include <stdexcept>

namespace eudsl {

// C++ exception types raised from binding code and mapped to Python exceptions
// registered in Errors.cpp.
struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct VerifyError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void registerExceptions(nanobind::module_ &m);

} // namespace eudsl
