// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/GlobalIFunc.h>
#include <llvm/IR/GlobalVariable.h>

void populate_constants(nb::module_ &m) {
  nb::class_<llvm::ConstantData, llvm::Constant>(m, "ConstantData");
  nb::class_<llvm::ConstantAggregate, llvm::Constant>(m, "ConstantAggregate");

  nb::class_<llvm::ConstantInt, llvm::ConstantData>(m, "ConstantInt")
      .EUDSL_CAST_CTOR(llvm::ConstantInt, llvm::Value)
      .def_prop_ro("value",
                   [](llvm::ConstantInt &self) {
                     return self.getValue().getSExtValue();
                   })
      .def_prop_ro("zext_value", [](llvm::ConstantInt &self) {
        return self.getValue().getZExtValue();
      });

  nb::class_<llvm::ConstantFP, llvm::ConstantData>(m, "ConstantFP")
      .EUDSL_CAST_CTOR(llvm::ConstantFP, llvm::Value)
      .def_prop_ro("double_value", [](llvm::ConstantFP &self) {
        return self.getValueAPF().convertToDouble();
      });

  nb::class_<llvm::UndefValue, llvm::ConstantData>(m, "UndefValue");
  nb::class_<llvm::PoisonValue, llvm::UndefValue>(m, "PoisonValue");
  nb::class_<llvm::ConstantPointerNull, llvm::ConstantData>(
      m, "ConstantPointerNull");
  nb::class_<llvm::ConstantAggregateZero, llvm::ConstantData>(
      m, "ConstantAggregateZero");
  nb::class_<llvm::ConstantTokenNone, llvm::ConstantData>(m,
                                                         "ConstantTokenNone");
  nb::class_<llvm::ConstantArray, llvm::ConstantAggregate>(m, "ConstantArray");
  nb::class_<llvm::ConstantStruct, llvm::ConstantAggregate>(m, "ConstantStruct");
  nb::class_<llvm::ConstantVector, llvm::ConstantAggregate>(m, "ConstantVector");
  nb::class_<llvm::ConstantDataSequential, llvm::ConstantData>(
      m, "ConstantDataSequential");
  nb::class_<llvm::ConstantDataArray, llvm::ConstantDataSequential>(
      m, "ConstantDataArray");
  nb::class_<llvm::ConstantDataVector, llvm::ConstantDataSequential>(
      m, "ConstantDataVector");
  nb::class_<llvm::ConstantExpr, llvm::Constant>(m, "ConstantExpr");
  nb::class_<llvm::BlockAddress, llvm::Constant>(m, "BlockAddress");

  nb::class_<llvm::GlobalVariable, llvm::GlobalObject>(m, "GlobalVariable")
      .EUDSL_CAST_CTOR(llvm::GlobalVariable, llvm::Value)
      .def_prop_ro("is_constant", &llvm::GlobalVariable::isConstant)
      .def_prop_ro(
          "initializer",
          [](llvm::GlobalVariable &self) -> llvm::Constant * {
            return self.hasInitializer() ? self.getInitializer() : nullptr;
          },
          nb::rv_policy::reference_internal);
  nb::class_<llvm::GlobalAlias, llvm::GlobalValue>(m, "GlobalAlias");
  nb::class_<llvm::GlobalIFunc, llvm::GlobalObject>(m, "GlobalIFunc");

  m.def(
      "const_int",
      [](llvm::Type *ty, int64_t value, bool isSigned) -> llvm::Constant * {
        auto *ity = llvm::cast<llvm::IntegerType>(ty);
        // A negative value must be built as signed: its uint64 bit pattern does
        // not fit the type width unsigned, and ConstantInt::get would trip an
        // APInt assertion (aborting the process) rather than raise. Forcing
        // signed for value < 0 keeps const_int(i32, -1) working with the
        // default signed=False.
        return llvm::ConstantInt::get(ity, static_cast<uint64_t>(value),
                                      isSigned || value < 0);
      },
      "type"_a, "value"_a, "signed"_a = false, nb::rv_policy::reference,
      nb::keep_alive<0, 1>());
  m.def(
      "const_bool",
      [](eudsl::Context &ctx, bool b) -> llvm::Constant * {
        return llvm::ConstantInt::getBool(ctx.get(), b);
      },
      "context"_a, "value"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "const_fp",
      [](llvm::Type *ty, double value) -> llvm::Constant * {
        return llvm::ConstantFP::get(ty, value);
      },
      "type"_a, "value"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "undef",
      [](llvm::Type *ty) -> llvm::Constant * {
        return llvm::UndefValue::get(ty);
      },
      "type"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "poison",
      [](llvm::Type *ty) -> llvm::Constant * {
        return llvm::PoisonValue::get(ty);
      },
      "type"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.def(
      "null",
      [](llvm::Type *ty) -> llvm::Constant * {
        return llvm::Constant::getNullValue(ty);
      },
      "type"_a, nb::rv_policy::reference, nb::keep_alive<0, 1>());
  m.attr("const_null") = m.attr("null");
}
