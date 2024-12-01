/*
    nanobind/stl/bind_array_ref.h: Automatic creation of bindings for
   vector-style containers

    Copyright (c) 2022 Wenzel Jakob

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "llvm/ADT/ArrayRef.h"

#include <algorithm>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/detail/traits.h>
#include <vector>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

NAMESPACE_END(detail)

template <typename Vector, rv_policy Policy = rv_policy::automatic_reference,
          typename... Args>
class_<Vector> bind_vec_like(handle scope, const char *name, Args &&...args) {
  using ValueRef =
      typename detail::iterator_access<typename Vector::iterator>::result_type;
  using Value = std::decay_t<ValueRef>;

  handle cl_cur = type<Vector>();
  if (cl_cur.is_valid()) {
    // Binding already exists, don't re-create
    return borrow<class_<Vector>>(cl_cur);
  }

  static_assert(
      !detail::is_base_caster_v<detail::make_caster<Value>> ||
          detail::is_copy_constructible_v<Value> ||
          (Policy != rv_policy::automatic_reference &&
           Policy != rv_policy::copy),
      "bind_vector(): the generated __getitem__ would copy elements, so the "
      "element type must be copy-constructible");

  auto cl = class_<Vector>(scope, name, std::forward<Args>(args)...)
                .def(init<>(), "Default constructor")
                .def("__len__", [](const Vector &v) { return v.size(); })
                .def(
                    "__bool__", [](const Vector &v) { return !v.empty(); },
                    "Check whether the vector is nonempty")
                .def("__repr__",
                     [](handle_t<Vector> h) {
                       return steal<str>(detail::repr_list(h.ptr()));
                     })
                .def(
                    "__iter__",
                    [](Vector &v) {
                      return make_iterator<Policy>(type<Vector>(), "Iterator",
                                                   v.begin(), v.end());
                    },
                    keep_alive<0, 1>())
                .def(
                    "__getitem__",
                    [](Vector &v, Py_ssize_t i) -> ValueRef {
                      return v[detail::wrap(i, v.size())];
                    },
                    Policy);

  // if constexpr (detail::is_copy_constructible_v<Value>) {
  //   cl.def(init<const Vector &>(), "Copy constructor");
  //   cl.def(
  //       "__init__",
  //       [](Vector *v, typed<iterable, Value> seq) {
  //         new (v) Vector();
  //         v->reserve(len_hint(seq));
  //         for (handle h : seq)
  //           v->push_back(cast<Value>(h));
  //       },
  //       "Construct from an iterable object");
  //
  //   implicitly_convertible<iterable, Vector>();
  // }

  if constexpr (detail::is_equality_comparable_v<Value>) {
    cl.def(self == self, sig("def __eq__(self, arg: object, /) -> bool"))
        .def(self != self, sig("def __ne__(self, arg: object, /) -> bool"))

        .def("__contains__",
             [](const Vector &v, const Value &x) {
               return std::find(v.begin(), v.end(), x) != v.end();
             })

        .def("__contains__", // fallback for incompatible types
             [](const Vector &, handle) { return false; })

        .def(
            "count",
            [](const Vector &v, const Value &x) {
              return std::count(v.begin(), v.end(), x);
            },
            "Return number of occurrences of `arg`.");
  }

  return cl;
}

template <typename Vector, rv_policy Policy = rv_policy::automatic_reference,
          typename... Args>
class_<Vector> bind_iter_like(handle scope, const char *name, Args &&...args) {
  handle cl_cur = type<Vector>();
  if (cl_cur.is_valid()) {
    // Binding already exists, don't re-create
    return borrow<class_<Vector>>(cl_cur);
  }

  auto cl = class_<Vector>(scope, name, std::forward<Args>(args)...)
                .def(init<>(), "Default constructor")
                .def("__len__", [](const Vector &v) -> int { return v.size(); })
                .def(
                    "__bool__", [](const Vector &v) { return !v.empty(); },
                    "Check whether the vector is nonempty")
                .def("__repr__",
                     [](handle_t<Vector> h) {
                       return steal<str>(detail::repr_list(h.ptr()));
                     })
                .def(
                    "__iter__",
                    [](const Vector &v) {
                      return make_iterator<Policy>(type<Vector>(), "Iterator",
                                                   v.begin(), v.end());
                    },
                    keep_alive<0, 1>())
                .def(
                    "__getitem__",
                    [](Vector &v, Py_ssize_t i) {
                      int ii = nanobind::detail::wrap(i, v.size());
                      int iii = 0;
                      for (auto it = v.begin(); it != v.end(); ++it) {
                        if (iii == ii)
                          return &*it;
                      }
                      throw nanobind::index_error("");
                    },
                    Policy);

  using ValueRef =
      typename detail::iterator_access<typename Vector::iterator>::result_type;
  using Value = std::decay_t<ValueRef>;

  if constexpr (detail::is_equality_comparable_v<Value>) {
    cl.def(self == self, sig("def __eq__(self, arg: object, /) -> bool"))
        .def(self != self, sig("def __ne__(self, arg: object, /) -> bool"))
        .def("__contains__",
             [](const Vector &v, const Value &x) {
               return std::find(v.begin(), v.end(), x) != v.end();
             })
        .def("__contains__", // fallback for incompatible types
             [](const Vector &, handle) { return false; })
        .def(
            "count",
            [](const Vector &v, const Value &x) {
              return std::count(v.begin(), v.end(), x);
            },
            "Return number of occurrences of `arg`.");
  }

  return cl;
}

template <typename Vector, typename ValueRef,
          rv_policy Policy = rv_policy::automatic_reference, typename... Args>
class_<Vector> bind_iter_range(handle scope, const char *name, Args &&...args) {
  handle cl_cur = type<Vector>();
  if (cl_cur.is_valid()) {
    // Binding already exists, don't re-create
    return borrow<class_<Vector>>(cl_cur);
  }

  auto cl = class_<Vector>(scope, name, std::forward<Args>(args)...)
                .def("__len__", [](const Vector &v) -> int { return v.size(); })
                .def(
                    "__bool__", [](const Vector &v) { return !v.empty(); },
                    "Check whether the vector is nonempty")
                .def("__repr__",
                     [](handle_t<Vector> h) {
                       return steal<str>(detail::repr_list(h.ptr()));
                     })
                .def(
                    "__iter__",
                    [](const Vector &v) {
                      return make_iterator<Policy>(type<Vector>(), "Iterator",
                                                   v.begin(), v.end());
                    },
                    keep_alive<0, 1>())
                .def(
                    "__getitem__",
                    [](Vector &v, Py_ssize_t i) -> ValueRef {
                      int ii = nanobind::detail::wrap(i, v.size());
                      int iii = 0;
                      for (auto it = v.begin(); it != v.end(); ++it) {
                        if (iii == ii)
                          return *it;
                      }
                      throw nanobind::index_error("");
                    },
                    Policy);

  using Value = std::decay_t<ValueRef>;

  if constexpr (detail::is_equality_comparable_v<Value>) {
    cl.def(self == self, sig("def __eq__(self, arg: object, /) -> bool"))
        .def(self != self, sig("def __ne__(self, arg: object, /) -> bool"))
        .def("__contains__",
             [](const Vector &v, const Value &x) {
               return std::find(v.begin(), v.end(), x) != v.end();
             })
        .def("__contains__", // fallback for incompatible types
             [](const Vector &, handle) { return false; })
        .def(
            "count",
            [](const Vector &v, const Value &x) {
              return std::count(v.begin(), v.end(), x);
            },
            "Return number of occurrences of `arg`.");
  }

  return cl;
}

NAMESPACE_END(NB_NAMESPACE)
