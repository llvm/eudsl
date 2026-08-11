// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>

void populate_passes(nb::module_ &m) {
  nb::enum_<llvm::OptimizationLevel>(m, "OptLevel")
      .value("O0", llvm::OptimizationLevel::O0)
      .value("O1", llvm::OptimizationLevel::O1)
      .value("O2", llvm::OptimizationLevel::O2)
      .value("O3", llvm::OptimizationLevel::O3);

  nb::class_<llvm::PipelineTuningOptions>(m, "PipelineTuningOptions")
      .def(nb::init<>())
      .def_rw("loop_interleaving",
              &llvm::PipelineTuningOptions::LoopInterleaving)
      .def_rw("loop_vectorization",
              &llvm::PipelineTuningOptions::LoopVectorization)
      .def_rw("slp_vectorization",
              &llvm::PipelineTuningOptions::SLPVectorization)
      .def_rw("loop_unrolling", &llvm::PipelineTuningOptions::LoopUnrolling)
      .def_rw("merge_functions",
              &llvm::PipelineTuningOptions::MergeFunctions);

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

  m.def(
      "run_default_pipeline",
      [](eudsl::Module &mod, llvm::OptimizationLevel level,
         nb::object pto_obj) {
        llvm::PipelineTuningOptions pto;
        if (!pto_obj.is_none())
          pto = nb::cast<llvm::PipelineTuningOptions>(pto_obj);
        llvm::PassBuilder pb(nullptr, pto);
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm =
            (level == llvm::OptimizationLevel::O0)
                ? pb.buildO0DefaultPipeline(level)
                : pb.buildPerModuleDefaultPipeline(level);
        mpm.run(mod.get(), mam);
      },
      "module"_a, "level"_a, "tuning"_a = nb::none(),
      "Run the default optimization pipeline at the given level.");
}
