// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

#include <nanobind/nanobind.h>

#include <cstdint>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace eudsl {

/// Render anything with a `print(raw_ostream&)` method to a std::string.
template <typename T> std::string toString(const T &t) {
  std::string s;
  llvm::raw_string_ostream os(s);
  t.print(os);
  return s;
}

/// Unwrap an llvm::Expected, raising RuntimeError on failure. Callers that
/// need a more specific Python exception catch and re-raise.
template <typename T> T unwrap(llvm::Expected<T> &&e) {
  if (!e)
    throw std::runtime_error(llvm::toString(e.takeError()));
  return std::move(*e);
}

/// Unwrap an llvm::Error, raising RuntimeError on failure.
inline void unwrap(llvm::Error &&e) {
  if (e)
    throw std::runtime_error(llvm::toString(std::move(e)));
}

} // namespace eudsl
