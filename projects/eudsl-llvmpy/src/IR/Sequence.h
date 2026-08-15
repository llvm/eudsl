// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "IR/Common.h"

#include <cstddef>
#include <functional>

namespace eudsl {

/// A lazy, sliceable sequence view over an LLVM container. It holds closures
/// that compute the length and the i-th element on demand, so iterating or
/// indexing does not materialize a list. The accessor sets `owner` to a strong
/// reference to the Python object that owns the elements' storage (the Module),
/// so a held view -- and every element it yields -- keeps that storage alive
/// and never dangles past it. `owner` may be none for context-owned values
/// (e.g. constants) that have no owning module.
///
/// __len__ and integer __getitem__ (negative-aware, IndexError past the ends)
/// are lazy; a slice materializes just the requested window into a list.
/// Iteration uses Python's sequence protocol (__getitem__ until IndexError).
template <typename T> struct Sequence {
  std::function<std::size_t()> length;
  std::function<T *(std::size_t)> at;
  nb::object owner;
};

template <typename T> void bindSequence(nb::module_ &m, const char *name) {
  nb::class_<Sequence<T>>(m, name)
      .def("__len__",
           [](Sequence<T> &self) {
             return static_cast<Py_ssize_t>(self.length());
           })
      .def("__getitem__", [](Sequence<T> &self, nb::handle index) -> nb::object {
        // Pin each yielded element to the storage owner so it survives even if
        // the view itself is dropped -- mirroring MLIR's owner-reference chain.
        auto element = [&](std::size_t k) -> nb::object {
          nb::object obj = nb::cast(self.at(k), nb::rv_policy::reference);
          if (self.owner.is_valid() && !self.owner.is_none())
            nb::detail::keep_alive(obj.ptr(), self.owner.ptr());
          return obj;
        };
        Py_ssize_t n = static_cast<Py_ssize_t>(self.length());
        if (nb::isinstance<nb::slice>(index)) {
          auto [start, stop, step, count] =
              nb::cast<nb::slice>(index).compute(static_cast<size_t>(n));
          nb::list out;
          Py_ssize_t idx = start;
          for (size_t k = 0; k < count; ++k) {
            out.append(element(static_cast<std::size_t>(idx)));
            idx += step;
          }
          return out;
        }
        if (!nb::isinstance<nb::int_>(index))
          throw nb::type_error("sequence indices must be integers or slices");
        Py_ssize_t i = nb::cast<Py_ssize_t>(index);
        if (i < 0)
          i += n;
        if (i < 0 || i >= n)
          throw nb::index_error("index out of range");
        return element(static_cast<std::size_t>(i));
      });
}

} // namespace eudsl
