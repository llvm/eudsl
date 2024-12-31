// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2024.

#ifndef TYPE_CASTERS_H
#define TYPE_CASTERS_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>

template <>
struct nanobind::detail::type_caster<llvm::StringRef> {
  NB_TYPE_CASTER(llvm::StringRef, const_name("str"))

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    Py_ssize_t size;
    const char *str = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
    if (!str) {
      PyErr_Clear();
      return false;
    }
    value = llvm::StringRef(str, (size_t)size);
    return true;
  }

  static handle from_cpp(llvm::StringRef value, rv_policy,
                         cleanup_list *) noexcept {
    return PyUnicode_FromStringAndSize(value.data(), value.size());
  }
};

template <>
struct nanobind::detail::type_caster<llvm::StringLiteral> {
  NB_TYPE_CASTER(llvm::StringLiteral, const_name("str"))

  static handle from_cpp(llvm::StringLiteral value, rv_policy,
                         cleanup_list *) noexcept {
    return PyUnicode_FromStringAndSize(value.data(), value.size());
  }
};

template <>
struct nanobind::detail::type_caster<llvm::Twine> {
  using Value = llvm::Twine;
  static constexpr auto Name = const_name("str");
  template <typename T_>
  using Cast = movable_cast_t<T_>;

  template <typename T_>
  static constexpr bool can_cast() {
    return true;
  }

  template <typename T_,
            enable_if_t<std::is_same_v<std::remove_cv_t<T_>, Value>> = 0>
  static handle from_cpp(T_ *p, rv_policy policy, cleanup_list *list) {
    if (!p)
      return none().release();
    return from_cpp(*p, policy, list);
  }

  explicit operator Value *() { return &*value; }
  explicit operator Value &() { return (Value &)*value; }
  explicit operator Value &&() { return (Value &&)*value; }

  // hack because Twine::operator= is deleted
  std::optional<Value> value;

  bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
    Py_ssize_t size;
    const char *str = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
    if (!str) {
      PyErr_Clear();
      return false;
    }
    std::string_view s{str, (size_t)size};
    value.emplace(s);
    return true;
  }

  static handle from_cpp(llvm::Twine value, rv_policy,
                         cleanup_list *) noexcept {
    llvm::StringRef s = value.getSingleStringRef();
    return PyUnicode_FromStringAndSize(s.data(), s.size());
  }
};

#endif // TYPE_CASTERS_H
