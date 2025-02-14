// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2025.

#ifndef HELPERS_H
#define HELPERS_H

#include <nanobind/nanobind.h>

template <typename T, typename... Ts>
struct non_copying_non_moving_class_ : nanobind::class_<T, Ts...> {
  template <typename... Extra>
  NB_INLINE non_copying_non_moving_class_(nanobind::handle scope,
                                          const char *name,
                                          const Extra &...extra) {
    nanobind::detail::type_init_data d;

    d.flags = 0;
    d.align = (uint8_t)alignof(typename nanobind::class_<T, Ts...>::Alias);
    d.size = (uint32_t)sizeof(typename nanobind::class_<T, Ts...>::Alias);
    d.name = name;
    d.scope = scope.ptr();
    d.type = &typeid(T);

    if constexpr (!std::is_same_v<typename nanobind::class_<T, Ts...>::Base,
                                  T>) {
      d.base = &typeid(typename nanobind::class_<T, Ts...>::Base);
      d.flags |= (uint32_t)nanobind::detail::type_init_flags::has_base;
    }

    if constexpr (std::is_destructible_v<T>) {
      d.flags |= (uint32_t)nanobind::detail::type_flags::is_destructible;

      if constexpr (!std::is_trivially_destructible_v<T>) {
        d.flags |= (uint32_t)nanobind::detail::type_flags::has_destruct;
        d.destruct = nanobind::detail::wrap_destruct<T>;
      }
    }

    if constexpr (nanobind::detail::has_shared_from_this_v<T>) {
      d.flags |= (uint32_t)nanobind::detail::type_flags::has_shared_from_this;
      d.keep_shared_from_this_alive = [](PyObject *self) noexcept {
        if (auto sp = nanobind::inst_ptr<T>(self)->weak_from_this().lock()) {
          nanobind::detail::keep_alive(
              self, new auto(std::move(sp)),
              [](void *p) noexcept { delete (decltype(sp) *)p; });
          return true;
        }
        return false;
      };
    }

    (nanobind::detail::type_extra_apply(d, extra), ...);

    this->m_ptr = nanobind::detail::nb_type_new(&d);
  }
};

#endif // HELPERS_H
