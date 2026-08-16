// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/IR/GlobalValue.h>

void populate_attributes(nb::module_ &m) {
  nb::enum_<llvm::GlobalValue::VisibilityTypes>(m, "Visibility")
      .value("DEFAULT", llvm::GlobalValue::VisibilityTypes::DefaultVisibility)
      .value("HIDDEN", llvm::GlobalValue::VisibilityTypes::HiddenVisibility)
      .value("PROTECTED", llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);

  nb::enum_<CallingConvEnum>(m, "CallingConv")
      .value("C", CallingConvEnum::C)
      .value("FAST", CallingConvEnum::FAST)
      .value("COLD", CallingConvEnum::COLD);
}
