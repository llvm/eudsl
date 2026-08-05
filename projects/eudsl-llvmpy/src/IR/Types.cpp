// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Type.h>

#include <nanobind/stl/string.h>

void populate_types(nb::module_ &m) {
  nb::class_<llvm::Type>(m, "Type")
      .def_prop_ro("is_void", &llvm::Type::isVoidTy)
      .def_prop_ro("is_label", &llvm::Type::isLabelTy)
      .def_prop_ro("is_integer",
                   [](llvm::Type &self) { return self.isIntegerTy(); })
      .def_prop_ro("is_floating_point", &llvm::Type::isFloatingPointTy)
      .def_prop_ro("is_pointer", &llvm::Type::isPointerTy)
      .def_prop_ro("is_sized", [](llvm::Type &self) { return self.isSized(); })
      .def("__str__", [](llvm::Type &self) { return eudsl::toString(self); })
      .def("__eq__",
           [](llvm::Type &self, nb::handle other) {
             llvm::Type *o;
             if (!nb::try_cast<llvm::Type *>(other, o))
               return false;
             return &self == o;
           })
      .def("__hash__", [](llvm::Type &self) {
        return static_cast<Py_ssize_t>(
            reinterpret_cast<std::uintptr_t>(&self));
      });

  // Primitive type factories. Each takes the owning context and returns an
  // interned Type*, non-owning, kept alive by the context (keep_alive<0,1>:
  // the returned type keeps its context argument alive). reference_internal is
  // not usable here because these are free functions with no bound self.
#define EUDSL_PRIMITIVE_TYPE(pyName, getter)                                   \
  m.def(                                                                       \
      pyName,                                                                  \
      [](eudsl::Context &ctx) -> llvm::Type * {                                \
        return llvm::Type::getter(ctx.get());                                  \
      },                                                                       \
      "context"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>())

  EUDSL_PRIMITIVE_TYPE("void_t", getVoidTy);
  EUDSL_PRIMITIVE_TYPE("label_t", getLabelTy);
  EUDSL_PRIMITIVE_TYPE("i1", getInt1Ty);
  EUDSL_PRIMITIVE_TYPE("i8", getInt8Ty);
  EUDSL_PRIMITIVE_TYPE("i16", getInt16Ty);
  EUDSL_PRIMITIVE_TYPE("i32", getInt32Ty);
  EUDSL_PRIMITIVE_TYPE("i64", getInt64Ty);
  EUDSL_PRIMITIVE_TYPE("f16", getHalfTy);
  EUDSL_PRIMITIVE_TYPE("f32", getFloatTy);
  EUDSL_PRIMITIVE_TYPE("f64", getDoubleTy);
#undef EUDSL_PRIMITIVE_TYPE
}
