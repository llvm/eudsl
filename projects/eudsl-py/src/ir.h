//
// Created by mlevental on 12/1/24.
//

#ifndef IR_H
#define IR_H

#include "mlir/IR/Action.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Unit.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/InterfaceSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/SourceMgr.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/optional.h>
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

void populateIRModule(nanobind::module_ &m);

void populateArithModule(nanobind::module_ &m);

#endif //IR_H
