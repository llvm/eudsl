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
/// indexing does not materialize a list.
///
/// Lifetime: two independent anchors keep storage alive. (1) The accessor sets
/// `owner` to a strong reference to the Python object that owns the *container's*
/// storage (usually the Module); holding the view -- which holds `owner` -- keeps
/// that storage alive, and `owner` is also pinned onto every yielded element.
/// (2) `ownerOf`, when set, resolves an *individual element's* own storage owner,
/// which is pinned onto that element too -- so a module-owned element yielded by
/// a none-owner container (e.g. a GlobalVariable via `ConstantExpr.operands`,
/// whose ConstantExpr has no module) is anchored to its own module rather than
/// left dangling. Either may resolve to none (a context-owned constant, or no
/// live module wrapper); the accessors additionally `keep_alive<0,1>` the view to
/// its immediate parent.
///
/// __len__ and integer __getitem__ (negative-aware, IndexError past the ends)
/// are lazy; a slice materializes just the requested window into a list.
/// Iteration uses Python's sequence protocol (__getitem__ until IndexError).
template <typename T> struct Sequence {
  std::function<std::size_t()> length;
  std::function<T *(std::size_t)> at;
  nb::object owner;
  std::function<nb::object(T *)> ownerOf;
};

template <typename T> void bindSequence(nb::module_ &m, const char *name) {
  nb::class_<Sequence<T>>(m, name)
      .def("__len__",
           [](Sequence<T> &self) {
             return static_cast<Py_ssize_t>(self.length());
           })
      .def("__getitem__", [](Sequence<T> &self, nb::handle index) -> nb::object {
        // Pin each yielded element to its storage owner(s) so it survives even
        // if the view itself is dropped -- mirroring MLIR's owner-reference
        // chain. Pin to both the container's owner and (when known) the
        // element's own owner: the latter anchors a module-owned element yielded
        // by a none-owner container to its own module.
        auto element = [&](std::size_t k) -> nb::object {
          T *ptr = self.at(k);
          nb::object obj = nb::cast(ptr, nb::rv_policy::reference);
          auto pin = [&](const nb::object &o) {
            // nb::detail::keep_alive is the dynamic (runtime nurse/patient) form
            // behind rv_policy::reference_internal; it is NB_CORE-exported and
            // has no public wrapper, so the detail:: reach-in is intentional.
            if (o.is_valid() && !o.is_none())
              nb::detail::keep_alive(obj.ptr(), o.ptr());
          };
          pin(self.owner);
          if (self.ownerOf)
            pin(self.ownerOf(ptr));
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
