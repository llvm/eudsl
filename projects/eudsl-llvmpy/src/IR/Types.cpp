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
  nb::enum_<llvm::Type::TypeID>(m, "TypeID")
      .value("Half", llvm::Type::HalfTyID)
      .value("BFloat", llvm::Type::BFloatTyID)
      .value("Float", llvm::Type::FloatTyID)
      .value("Double", llvm::Type::DoubleTyID)
      .value("X86_FP80", llvm::Type::X86_FP80TyID)
      .value("FP128", llvm::Type::FP128TyID)
      .value("PPC_FP128", llvm::Type::PPC_FP128TyID)
      .value("Void", llvm::Type::VoidTyID)
      .value("Label", llvm::Type::LabelTyID)
      .value("Metadata", llvm::Type::MetadataTyID)
      .value("Token", llvm::Type::TokenTyID)
      .value("Integer", llvm::Type::IntegerTyID)
      .value("Function", llvm::Type::FunctionTyID)
      .value("Pointer", llvm::Type::PointerTyID)
      .value("Struct", llvm::Type::StructTyID)
      .value("Array", llvm::Type::ArrayTyID)
      .value("FixedVector", llvm::Type::FixedVectorTyID)
      .value("ScalableVector", llvm::Type::ScalableVectorTyID);

  nb::class_<llvm::Type>(m, "Type")
      .def_prop_ro("is_void", &llvm::Type::isVoidTy)
      .def_prop_ro("is_label", &llvm::Type::isLabelTy)
      .def_prop_ro("is_integer",
                   [](llvm::Type &self) { return self.isIntegerTy(); })
      .def_prop_ro("is_floating_point", &llvm::Type::isFloatingPointTy)
      .def_prop_ro("is_pointer", &llvm::Type::isPointerTy)
      .def_prop_ro("is_sized", [](llvm::Type &self) { return self.isSized(); })
      .def_prop_ro("type_id", [](llvm::Type &self) { return self.getTypeID(); })
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

  // Primitive type factories. The context is optional: when omitted, the
  // thread-local current context (from `with Context():`) is used. Returns an
  // interned Type*, non-owning, kept alive by the context argument when given.
#define EUDSL_PRIMITIVE_TYPE(pyName, getter)                                   \
  m.def(                                                                       \
      pyName,                                                                  \
      [](eudsl::Context *context) -> llvm::Type * {                            \
        return llvm::Type::getter(eudsl::currentOr(context).get());            \
      },                                                                       \
      "context"_a.none() = nb::none(), nb::rv_policy::reference,               \
      nb::keep_alive<0, 1>())

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

  // Parametric type classes carry nb::is_generic() so `IntegerType[32]`,
  // `PointerType[0]`, `FunctionType[ret, [args]]`, etc. produce an unevaluated
  // types.GenericAlias -- a type annotation that needs no live context. Each
  // class also gets a `.get(...)` classmethod that the alias forwards to at
  // emit time. IntegerType/PointerType/StructType.get resolve the context via
  // currentOr, so it stays optional there. The element-deriving gets
  // (ArrayType/VectorType/FunctionType and the concrete vector subtypes) take
  // their context from the element or return type instead; they still accept
  // `context` for uniform dispatch and reject it only when it names a different
  // context.
  nb::class_<llvm::IntegerType, llvm::Type>(m, "IntegerType", nb::is_generic())
      .EUDSL_CAST_CTOR(llvm::IntegerType, llvm::Type)
      .def_static(
          "get",
          [](unsigned bits, eudsl::Context *context) -> llvm::IntegerType * {
            return llvm::IntegerType::get(eudsl::currentOr(context).get(), bits);
          },
          "bits"_a, "context"_a.none() = nb::none(), nb::rv_policy::reference,
          nb::keep_alive<0, 2>())
      .def_prop_ro("bit_width", &llvm::IntegerType::getBitWidth);

  nb::class_<llvm::PointerType, llvm::Type>(m, "PointerType", nb::is_generic())
      .EUDSL_CAST_CTOR(llvm::PointerType, llvm::Type)
      .def_static(
          "get",
          [](unsigned addressSpace,
             eudsl::Context *context) -> llvm::PointerType * {
            return llvm::PointerType::get(eudsl::currentOr(context).get(),
                                          addressSpace);
          },
          "address_space"_a = 0, "context"_a.none() = nb::none(),
          nb::rv_policy::reference, nb::keep_alive<0, 2>())
      .def_prop_ro("address_space", &llvm::PointerType::getAddressSpace);

  nb::class_<llvm::StructType, llvm::Type>(m, "StructType", nb::is_generic())
      .EUDSL_CAST_CTOR(llvm::StructType, llvm::Type)
      .def_static(
          "get",
          [](std::vector<llvm::Type *> elts, bool packed,
             eudsl::Context *context) -> llvm::StructType * {
            return llvm::StructType::get(eudsl::currentOr(context).get(), elts,
                                         packed);
          },
          "element_types"_a, "packed"_a = false,
          "context"_a.none() = nb::none(), nb::rv_policy::reference,
          nb::keep_alive<0, 3>())
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

  nb::class_<llvm::ArrayType, llvm::Type>(m, "ArrayType", nb::is_generic())
      .EUDSL_CAST_CTOR(llvm::ArrayType, llvm::Type)
      .def_static(
          "get",
          [](llvm::Type *elt, uint64_t n,
             eudsl::Context *context) -> llvm::ArrayType * {
            // The element type already pins the context; a passed `context` is
            // accepted for uniform dispatch but must not name a different one.
            if (context &&
                &elt->getContext() != &eudsl::currentOr(context).get()) {
              throw nb::value_error(
                  "element type belongs to a different context");
            }
            return llvm::ArrayType::get(elt, n);
          },
          "element_type"_a, "num_elements"_a, "context"_a.none() = nb::none(),
          nb::rv_policy::reference, nb::keep_alive<0, 1>())
      .def_prop_ro("num_elements", &llvm::ArrayType::getNumElements)
      .def_prop_ro("element_type", &llvm::ArrayType::getElementType,
                   nb::rv_policy::reference);

  nb::class_<llvm::VectorType, llvm::Type>(m, "VectorType", nb::is_generic())
      .EUDSL_CAST_CTOR(llvm::VectorType, llvm::Type)
      .def_static(
          "get",
          [](llvm::Type *elt, unsigned n, bool scalable,
             eudsl::Context *context) -> llvm::VectorType * {
            if (context &&
                &elt->getContext() != &eudsl::currentOr(context).get()) {
              throw nb::value_error(
                  "element type belongs to a different context");
            }
            return llvm::VectorType::get(elt, n, scalable);
          },
          "element_type"_a, "num_elements"_a, "scalable"_a = false,
          "context"_a.none() = nb::none(), nb::rv_policy::reference,
          nb::keep_alive<0, 1>())
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

  // FixedVectorType/ScalableVectorType inherit VectorType's __class_getitem__,
  // so they are subscriptable regardless; give each its own `.get` (returning
  // the concrete subtype) so `FixedVectorType[i32, 4]` resolves correctly
  // instead of hitting a missing classmethod.
  nb::class_<llvm::FixedVectorType, llvm::VectorType>(m, "FixedVectorType",
                                                      nb::is_generic())
      .def_static(
          "get",
          [](llvm::Type *elt, unsigned n,
             eudsl::Context *context) -> llvm::FixedVectorType * {
            if (context &&
                &elt->getContext() != &eudsl::currentOr(context).get()) {
              throw nb::value_error(
                  "element type belongs to a different context");
            }
            return llvm::FixedVectorType::get(elt, n);
          },
          "element_type"_a, "num_elements"_a, "context"_a.none() = nb::none(),
          nb::rv_policy::reference, nb::keep_alive<0, 1>())
      .def_prop_ro("num_elements", &llvm::FixedVectorType::getNumElements);

  nb::class_<llvm::ScalableVectorType, llvm::VectorType>(
      m, "ScalableVectorType", nb::is_generic())
      .def_static(
          "get",
          [](llvm::Type *elt, unsigned n,
             eudsl::Context *context) -> llvm::ScalableVectorType * {
            if (context &&
                &elt->getContext() != &eudsl::currentOr(context).get()) {
              throw nb::value_error(
                  "element type belongs to a different context");
            }
            return llvm::ScalableVectorType::get(elt, n);
          },
          "element_type"_a, "num_elements"_a, "context"_a.none() = nb::none(),
          nb::rv_policy::reference, nb::keep_alive<0, 1>())
      .def_prop_ro("min_num_elements",
                   &llvm::ScalableVectorType::getMinNumElements);

  nb::class_<llvm::FunctionType, llvm::Type>(m, "FunctionType",
                                             nb::is_generic())
      .EUDSL_CAST_CTOR(llvm::FunctionType, llvm::Type)
      .def_static(
          "get",
          [](llvm::Type *ret, std::vector<llvm::Type *> params, bool varArg,
             eudsl::Context *context) -> llvm::FunctionType * {
            if (context) {
              llvm::LLVMContext *c = &eudsl::currentOr(context).get();
              if (&ret->getContext() != c) {
                throw nb::value_error(
                    "return type belongs to a different context");
              }
              for (llvm::Type *p : params) {
                if (&p->getContext() != c) {
                  throw nb::value_error(
                      "parameter type belongs to a different context");
                }
              }
            }
            return llvm::FunctionType::get(ret, params, varArg);
          },
          "return_type"_a, "params"_a, "var_arg"_a = false,
          "context"_a.none() = nb::none(), nb::rv_policy::reference,
          nb::keep_alive<0, 1>())
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
      [](unsigned bits, eudsl::Context *context) -> llvm::Type * {
        return llvm::IntegerType::get(eudsl::currentOr(context).get(), bits);
      },
      "bits"_a, "context"_a.none() = nb::none(), nb::rv_policy::reference,
      nb::keep_alive<0, 2>());
  m.def(
      "ptr",
      [](unsigned addressSpace, eudsl::Context *context) -> llvm::Type * {
        return llvm::PointerType::get(eudsl::currentOr(context).get(),
                                      addressSpace);
      },
      "address_space"_a = 0, "context"_a.none() = nb::none(),
      nb::rv_policy::reference, nb::keep_alive<0, 2>());
  m.def(
      "struct",
      [](std::vector<llvm::Type *> elts, bool packed,
         eudsl::Context *context) -> llvm::Type * {
        return llvm::StructType::get(eudsl::currentOr(context).get(), elts,
                                     packed);
      },
      "element_types"_a, "packed"_a = false, "context"_a.none() = nb::none(),
      nb::rv_policy::reference, nb::keep_alive<0, 3>());
  m.def(
      "named_struct",
      [](const std::string &name, eudsl::Context *context) -> llvm::Type * {
        return llvm::StructType::create(eudsl::currentOr(context).get(), name);
      },
      "name"_a, "context"_a.none() = nb::none(), nb::rv_policy::reference,
      nb::keep_alive<0, 2>());
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

