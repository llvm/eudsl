#pragma once

#include "llvm/ADT/ArrayRef.h"

#include <algorithm>
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/detail/traits.h>

struct _ArrayRef {};
struct _MutableArrayRef {};
struct _SmallVector {};

extern nanobind::class_<_SmallVector> smallVector;
extern nanobind::class_<_ArrayRef> arrayRef;
extern nanobind::class_<_MutableArrayRef> mutableArrayRef;

template <typename Element, typename... Args>
std::tuple<nanobind::class_<llvm::SmallVector<Element>>,
           nanobind::class_<llvm::ArrayRef<Element>>,
           nanobind::class_<llvm::MutableArrayRef<Element>>>
bind_array_ref(nanobind::handle scope, Args &&...args) {
  using ArrayRef = llvm::ArrayRef<Element>;
  using ValueRef = Element &;

  auto vecClName = "SmallVector[" + std::string(typeid(Element).name()) + "]";
  auto _smallVectorOfElement =
      nanobind::bind_vector<llvm::SmallVector<Element>>(scope,
                                                        vecClName.c_str());

  smallVector.def_static(
      "__class_getitem__",
      [_smallVectorOfElement](nanobind::type_object_t<Element>) {
        return _smallVectorOfElement;
      });

  auto arrClName = "ArrayRef[" + std::string(typeid(Element).name()) + "]";
  auto cl =
      nanobind::class_<ArrayRef>(scope, arrClName.c_str(),
                                 std::forward<Args>(args)...)
          .def(nanobind::init<const llvm::SmallVector<Element> &>())
          .def(nanobind::init_implicit<llvm::SmallVector<Element>>())
          .def("__len__", [](const ArrayRef &v) { return v.size(); })
          .def("__bool__", [](const ArrayRef &v) { return !v.empty(); })
          .def("__repr__",
               [](nanobind::handle_t<ArrayRef> h) {
                 return nanobind::steal<nanobind::str>(
                     nanobind::detail::repr_list(h.ptr()));
               })
          .def(
              "__iter__",
              [](ArrayRef &v) {
                return nanobind::make_iterator<nanobind::rv_policy::reference>(
                    nanobind::type<ArrayRef>(), "Iterator", v.begin(), v.end());
              },
              nanobind::keep_alive<0, 1>())
          .def(
              "__getitem__",
              [](ArrayRef &v, Py_ssize_t i) -> ValueRef {
                return const_cast<Element &>(
                    v[nanobind::detail::wrap(i, v.size())]);
              },
              nanobind::rv_policy::reference);

  arrayRef.def_static("__class_getitem__",
                      [cl](nanobind::type_object_t<Element>) { return cl; });
  arrayRef.def(nanobind::new_([](const llvm::SmallVector<Element> &sv) {
    return llvm::ArrayRef<Element>(sv);
  }));

  if constexpr (nanobind::detail::is_equality_comparable_v<Element>) {
    cl.def(nanobind::self == nanobind::self,
           nanobind::sig("def __eq__(self, arg: object, /) -> bool"))
        .def(nanobind::self != nanobind::self,
             nanobind::sig("def __ne__(self, arg: object, /) -> bool"))

        .def("__contains__",
             [](const ArrayRef &v, const Element &x) {
               return std::find(v.begin(), v.end(), x) != v.end();
             })

        .def("__contains__", // fallback for incompatible types
             [](const ArrayRef &, nanobind::handle) { return false; })

        .def(
            "count",
            [](const ArrayRef &v, const Element &x) {
              return std::count(v.begin(), v.end(), x);
            },
            "Return number of occurrences of `arg`.");
  }

  using MutableArrayRef = llvm::MutableArrayRef<Element>;
  auto mutableArrClName =
      "MutableArrayRef[" + std::string(typeid(Element).name()) + "]";
  auto mutableCl =
      nanobind::class_<MutableArrayRef>(scope, arrClName.c_str(),
                                        std::forward<Args>(args)...)
          .def(nanobind::init<llvm::SmallVector<Element> &>())
          // .def(nanobind::init_implicit<llvm::SmallVector<Element>>())
          .def("__len__", [](const MutableArrayRef &v) { return v.size(); })
          .def("__bool__", [](const MutableArrayRef &v) { return !v.empty(); })
          .def("__repr__",
               [](nanobind::handle_t<MutableArrayRef> h) {
                 return nanobind::steal<nanobind::str>(
                     nanobind::detail::repr_list(h.ptr()));
               })
          .def(
              "__iter__",
              [](MutableArrayRef &v) {
                return nanobind::make_iterator<nanobind::rv_policy::reference>(
                    nanobind::type<MutableArrayRef>(), "Iterator", v.begin(),
                    v.end());
              },
              nanobind::keep_alive<0, 1>())
          .def(
              "__getitem__",
              [](MutableArrayRef &v, Py_ssize_t i) -> ValueRef {
                return v[nanobind::detail::wrap(i, v.size())];
              },
              nanobind::rv_policy::reference);

  mutableArrayRef.def_static(
      "__class_getitem__",
      [cl](nanobind::class_<MutableArrayRef>) { return cl; });

  if constexpr (nanobind::detail::is_equality_comparable_v<Element>) {
    mutableCl
        .def(nanobind::self == nanobind::self,
             nanobind::sig("def __eq__(self, arg: object, /) -> bool"))
        .def(nanobind::self != nanobind::self,
             nanobind::sig("def __ne__(self, arg: object, /) -> bool"))

        .def("__contains__",
             [](const MutableArrayRef &v, const Element &x) {
               return std::find(v.begin(), v.end(), x) != v.end();
             })

        .def("__contains__", // fallback for incompatible types
             [](const MutableArrayRef &, nanobind::handle) { return false; })

        .def(
            "count",
            [](const MutableArrayRef &v, const Element &x) {
              return std::count(v.begin(), v.end(), x);
            },
            "Return number of occurrences of `arg`.");
  }

  return {_smallVectorOfElement, cl, mutableCl};
}

template <typename Vector,
          nanobind::rv_policy Policy = nanobind::rv_policy::automatic_reference,
          typename... Args>
nanobind::class_<Vector> bind_iter_like(nanobind::handle scope,
                                        const char *name, Args &&...args) {
  nanobind::handle cl_cur = nanobind::type<Vector>();
  if (cl_cur.is_valid()) {
    // Binding already exists, don't re-create
    return nanobind::borrow<nanobind::class_<Vector>>(cl_cur);
  }

  auto cl =
      nanobind::class_<Vector>(scope, name, std::forward<Args>(args)...)
          .def("__len__", [](const Vector &v) -> int { return v.size(); })
          .def(
              "__bool__", [](const Vector &v) { return !v.empty(); },
              "Check whether the vector is nonempty")
          .def("__repr__",
               [](nanobind::handle_t<Vector> h) {
                 return nanobind::steal<nanobind::str>(
                     nanobind::detail::repr_list(h.ptr()));
               })
          .def(
              "__iter__",
              [](const Vector &v) {
                return nanobind::make_iterator<Policy>(
                    nanobind::type<Vector>(), "Iterator", v.begin(), v.end());
              },
              nanobind::keep_alive<0, 1>())
          .def(
              "__getitem__",
              [](Vector &v, Py_ssize_t i) {
                int ii;
                ii = nanobind::detail::wrap(i, v.size());
                int iii = 0;
                for (auto it = v.begin(); it != v.end(); ++it) {
                  if (iii == ii)
                    return &*it;
                }
                throw nanobind::index_error("");
              },
              Policy);

  using ValueRef = typename nanobind::detail::iterator_access<
      typename Vector::iterator>::result_type;
  using Value = std::decay_t<ValueRef>;

  if constexpr (nanobind::detail::is_equality_comparable_v<Value>) {
    cl.def(nanobind::self == nanobind::self,
           nanobind::sig("def __eq__(self, arg: object, /) -> bool"))
        .def(nanobind::self != nanobind::self,
             nanobind::sig("def __ne__(self, arg: object, /) -> bool"))
        .def("__contains__",
             [](const Vector &v, const Value &x) {
               return std::find(v.begin(), v.end(), x) != v.end();
             })
        .def("__contains__", // fallback for incompatible types
             [](const Vector &, nanobind::handle) { return false; })
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
          nanobind::rv_policy Policy = nanobind::rv_policy::automatic_reference,
          typename... Args>
nanobind::class_<Vector> bind_iter_range(nanobind::handle scope,
                                         const char *name, Args &&...args) {
  nanobind::handle cl_cur = nanobind::type<Vector>();
  if (cl_cur.is_valid()) {
    // Binding already exists, don't re-create
    return nanobind::borrow<nanobind::class_<Vector>>(cl_cur);
  }

  auto cl =
      nanobind::class_<Vector>(scope, name, std::forward<Args>(args)...)
          .def("__len__", [](const Vector &v) -> int { return v.size(); })
          .def(
              "__bool__", [](const Vector &v) { return !v.empty(); },
              "Check whether the vector is nonempty")
          .def("__repr__",
               [](nanobind::handle_t<Vector> h) {
                 return nanobind::steal<nanobind::str>(
                     nanobind::detail::repr_list(h.ptr()));
               })
          .def(
              "__iter__",
              [](const Vector &v) {
                return nanobind::make_iterator<Policy>(
                    nanobind::type<Vector>(), "Iterator", v.begin(), v.end());
              },
              nanobind::keep_alive<0, 1>())
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

  if constexpr (nanobind::detail::is_equality_comparable_v<Value>) {
    cl.def(nanobind::self == nanobind::self,
           nanobind::sig("def __eq__(self, arg: object, /) -> bool"))
        .def(nanobind::self != nanobind::self,
             nanobind::sig("def __ne__(self, arg: object, /) -> bool"))
        .def("__contains__",
             [](const Vector &v, const Value &x) {
               return std::find(v.begin(), v.end(), x) != v.end();
             })
        .def("__contains__", // fallback for incompatible types
             [](const Vector &, nanobind::handle) { return false; })
        .def(
            "count",
            [](const Vector &v, const Value &x) {
              return std::count(v.begin(), v.end(), x);
            },
            "Return number of occurrences of `arg`.");
  }

  return cl;
}
