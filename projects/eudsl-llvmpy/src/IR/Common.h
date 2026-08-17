// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/IR/CallingConv.h>
#include "IR/Ownership.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
    throw std::runtime_error(llvm::toString(std::move(e))); // LCOV_EXCL_LINE
}

/// Python __getitem__ helper: negative-index aware, raises IndexError (not a
/// C++ crash) past the ends. Used by the iterable container bindings.
template <typename T>
T *nthOrThrow(const std::vector<T *> &items, Py_ssize_t i) {
  Py_ssize_t n = static_cast<Py_ssize_t>(items.size());
  if (i < 0)
    i += n;
  if (i < 0 || i >= n)
    throw nb::index_error("index out of range");
  return items[i];
}

/// Resolve a factory's optional `context` argument: use the passed Context, or
/// fall back to the thread-local current one (set by `with Context():`). Raises
/// if neither is available, mirroring MLIR's implicit-context factories.
inline Context &currentOr(Context *context) {
  if (context)
    return *context;
  Context *cur = Context::current();
  if (!cur) {
    throw std::runtime_error(
        "no context given and no current Context; pass context= or enter a "
        "'with Context():' block");
  }
  return *cur;
}

} // namespace eudsl

enum class CallingConvEnum : unsigned {
  C = llvm::CallingConv::C,
  FAST = llvm::CallingConv::Fast,
  COLD = llvm::CallingConv::Cold,
};

/// MLIR-style checked-downcast constructor, chained onto an nb::class_:
///   nb::class_<llvm::IntegerType, llvm::Type>(m, "IntegerType")
///       .EUDSL_CAST_CTOR(llvm::IntegerType, llvm::Type)
/// makes `IntegerType(v)` re-type v when isa<IntegerType>(v), else raise
/// ValueError -- the parity analogue of MLIR's `IntegerType(t)`. The result
/// borrows v's non-owning pointer; keep_alive<0,1> keeps the source (and thus
/// the owning Context/Module) alive for the new wrapper's lifetime, and
/// rv_policy::reference stops nanobind from trying to delete a context-owned
/// object.
#define EUDSL_CAST_CTOR(Derived, Base)                                         \
  def(nb::new_([](Base *v) -> Derived * {                                      \
        if (auto *d = llvm::dyn_cast_or_null<Derived>(v))                      \
          return d;                                                            \
        throw nb::value_error("value is not a " #Derived);                     \
      }),                                                                      \
      "value"_a.none(), nb::rv_policy::reference, nb::keep_alive<0, 1>())

// Pulled in here so every translation unit that returns an llvm::Type* or
// llvm::Value* sees the downcasting type_hook specializations. Without this a
// TU would silently fall back to base-class conversion.
#include "IR/Kinds.h"
