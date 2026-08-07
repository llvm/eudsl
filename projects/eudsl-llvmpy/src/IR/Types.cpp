// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <vector>

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

  EUDSL_PRIMITIVE_TYPE("void", getVoidTy);
  EUDSL_PRIMITIVE_TYPE("label", getLabelTy);
  EUDSL_PRIMITIVE_TYPE("i1", getInt1Ty);
  EUDSL_PRIMITIVE_TYPE("i8", getInt8Ty);
  EUDSL_PRIMITIVE_TYPE("i16", getInt16Ty);
  EUDSL_PRIMITIVE_TYPE("i32", getInt32Ty);
  EUDSL_PRIMITIVE_TYPE("i64", getInt64Ty);
  EUDSL_PRIMITIVE_TYPE("f16", getHalfTy);
  EUDSL_PRIMITIVE_TYPE("f32", getFloatTy);
  EUDSL_PRIMITIVE_TYPE("f64", getDoubleTy);
#undef EUDSL_PRIMITIVE_TYPE

  nb::class_<llvm::IntegerType, llvm::Type>(m, "IntegerType")
      .def_prop_ro("bit_width", &llvm::IntegerType::getBitWidth);

  nb::class_<llvm::PointerType, llvm::Type>(m, "PointerType")
      .def_prop_ro("address_space", &llvm::PointerType::getAddressSpace);

  nb::class_<llvm::StructType, llvm::Type>(m, "StructType")
      .def_prop_ro("name",
                   [](llvm::StructType &self) -> std::optional<std::string> {
                     if (!self.hasName())
                       return std::nullopt;
                     return self.getName().str();
                   })
      .def_prop_ro("num_elements", &llvm::StructType::getNumElements)
      .def("element_type", &llvm::StructType::getElementType, "index"_a,
           nb::rv_policy::reference)
      .def_prop_ro("is_packed", &llvm::StructType::isPacked)
      .def_prop_ro("is_opaque", &llvm::StructType::isOpaque)
      .def(
          "set_body",
          [](llvm::StructType &self, std::vector<llvm::Type *> elts,
             bool packed) { self.setBody(elts, packed); },
          "element_types"_a, "packed"_a = false);

  nb::class_<llvm::ArrayType, llvm::Type>(m, "ArrayType")
      .def_prop_ro("num_elements", &llvm::ArrayType::getNumElements)
      .def_prop_ro("element_type", &llvm::ArrayType::getElementType,
                   nb::rv_policy::reference);

  nb::class_<llvm::VectorType, llvm::Type>(m, "VectorType")
      .def_prop_ro("min_num_elements",
                   [](llvm::VectorType &self) {
                     return self.getElementCount().getKnownMinValue();
                   })
      .def_prop_ro("is_scalable",
                   [](llvm::VectorType &self) {
                     return self.getElementCount().isScalable();
                   })
      .def_prop_ro("element_type", &llvm::VectorType::getElementType,
                   nb::rv_policy::reference);

  nb::class_<llvm::FixedVectorType, llvm::VectorType>(m, "FixedVectorType")
      .def_prop_ro("num_elements", &llvm::FixedVectorType::getNumElements);

  nb::class_<llvm::ScalableVectorType, llvm::VectorType>(m, "ScalableVectorType")
      .def_prop_ro("min_num_elements",
                   &llvm::ScalableVectorType::getMinNumElements);

  nb::class_<llvm::FunctionType, llvm::Type>(m, "FunctionType")
      .def_prop_ro("return_type", &llvm::FunctionType::getReturnType,
                   nb::rv_policy::reference)
      .def_prop_ro("num_params", &llvm::FunctionType::getNumParams)
      .def("param_type", &llvm::FunctionType::getParamType, "index"_a,
           nb::rv_policy::reference)
      .def_prop_ro("params",
                   [](llvm::FunctionType &self) {
                     return std::vector<llvm::Type *>(self.param_begin(),
                                                      self.param_end());
                   })
      .def_prop_ro("is_var_arg", &llvm::FunctionType::isVarArg);

  m.def(
      "int",
      [](eudsl::Context &ctx, unsigned bits) -> llvm::Type * {
        return llvm::IntegerType::get(ctx.get(), bits);
      },
      "context"_a, "bits"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "ptr",
      [](eudsl::Context &ctx, unsigned addressSpace) -> llvm::Type * {
        return llvm::PointerType::get(ctx.get(), addressSpace);
      },
      "context"_a, "address_space"_a = 0, nb::rv_policy::reference,
      nb::keep_alive<0, 1>());
  m.def(
      "struct",
      [](eudsl::Context &ctx, std::vector<llvm::Type *> elts,
         bool packed) -> llvm::Type * {
        return llvm::StructType::get(ctx.get(), elts, packed);
      },
      "context"_a, "element_types"_a, "packed"_a = false,
      nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "named_struct",
      [](eudsl::Context &ctx, const std::string &name) -> llvm::Type * {
        return llvm::StructType::create(ctx.get(), name);
      },
      "context"_a, "name"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "array",
      [](llvm::Type *elt, uint64_t n) -> llvm::Type * {
        return llvm::ArrayType::get(elt, n);
      },
      "element_type"_a, "num_elements"_a, nb::rv_policy::reference,
      nb::keep_alive<0, 1>());
  m.def(
      "vector",
      [](llvm::Type *elt, unsigned n, bool scalable) -> llvm::Type * {
        return llvm::VectorType::get(elt, n, scalable);
      },
      "element_type"_a, "num_elements"_a, "scalable"_a = false,
      nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "function",
      [](llvm::Type *ret, std::vector<llvm::Type *> params,
         bool varArg) -> llvm::Type * {
        return llvm::FunctionType::get(ret, params, varArg);
      },
      "return_type"_a, "params"_a, "var_arg"_a = false,
      nb::rv_policy::reference, nb::keep_alive<0, 1>());
}

