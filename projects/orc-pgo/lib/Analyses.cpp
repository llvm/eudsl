// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Analyses.h"

#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

namespace orc_pgo {

PgoAnalysisManagers::PgoAnalysisManagers(const Module &M, TargetMachine *TM)
    : TLII(std::make_unique<TargetLibraryInfoImpl>(M.getTargetTriple())) {
  // Registered first, so PassBuilder's default TargetLibraryAnalysis is
  // ignored.
  FAM.registerPass([this] { return TargetLibraryAnalysis(*TLII); });
  PassBuilder PB(TM);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
}

} // namespace orc_pgo
