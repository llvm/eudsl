// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Driver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/InitLLVM.h"

int main(int Argc, const char **Argv) {
  llvm::InitLLVM X(Argc, Argv);
  return orc_pgo::driverMain(llvm::ArrayRef(Argv, Argc));
}
