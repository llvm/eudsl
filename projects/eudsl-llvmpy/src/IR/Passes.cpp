// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>

void populate_passes(nb::module_ &m) {
  m.def(
      "run_passes",
      [](eudsl::Module &mod, const std::string &pipeline) {
        llvm::PassBuilder pb;
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm;
        if (llvm::Error err = pb.parsePassPipeline(mpm, pipeline))
          throw std::runtime_error(llvm::toString(std::move(err)));
        mpm.run(mod.get(), mam);
      },
      "module"_a, "pipeline"_a,
      "Parse and run a textual pass pipeline over the module in place.");
}
