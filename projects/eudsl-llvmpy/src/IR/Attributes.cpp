// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/IR/CallingConv.h>
#include <llvm/IR/GlobalValue.h>

void populate_attributes(nb::module_ &m) {
  // Linkage (GlobalValue::LinkageTypes) is bound with the GlobalValue hierarchy
  // in Values.cpp, next to the Function/global classes that use it.
  nb::enum_<llvm::GlobalValue::VisibilityTypes>(m, "Visibility")
      .value("DEFAULT", llvm::GlobalValue::DefaultVisibility)
      .value("HIDDEN", llvm::GlobalValue::HiddenVisibility)
      .value("PROTECTED", llvm::GlobalValue::ProtectedVisibility);

  // CallingConv::ID is a namespace of unsigned constants, not an enum class;
  // expose the common ones as module-level ints under a submodule.
  nb::module_ cc = m.def_submodule("CallingConv");
  cc.attr("C") = (unsigned)llvm::CallingConv::C;
  cc.attr("FAST") = (unsigned)llvm::CallingConv::Fast;
  cc.attr("COLD") = (unsigned)llvm::CallingConv::Cold;
}
