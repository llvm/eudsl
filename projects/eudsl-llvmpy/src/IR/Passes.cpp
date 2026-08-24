// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>

namespace {
// Bundles the PassBuilder with the four analysis managers and instrumentation,
// with the lifetimes ModulePassManager::run requires: the managers and the
// StandardInstrumentations must outlive the run that references them, and the
// PassBuilder holds a pointer to `pic`. run_passes and run_default_pipeline
// build this identically, so keeping it in one place stops the two copies from
// drifting. Member declaration order is the construction order: `pic` before
// `pb` (which captures &pic), `si` before its registerCallbacks(pic).
struct PassPipelineEnv {
  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si;
  llvm::PassBuilder pb;
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  PassPipelineEnv(llvm::LLVMContext &ctx,
                  const llvm::PipelineTuningOptions &opts, bool debug,
                  bool verifyEach)
      : si(ctx, debug, verifyEach), pb(nullptr, opts, std::nullopt, &pic) {
    si.registerCallbacks(pic);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
  }
};
} // namespace

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
      .def_rw("merge_functions", &llvm::PipelineTuningOptions::MergeFunctions);

  m.def(
      "run_passes",
      [](eudsl::Module &mod, const std::string &pipeline,
         std::optional<llvm::PipelineTuningOptions> pto, bool debug,
         bool verifyEach) {
        llvm::PipelineTuningOptions opts =
            pto.value_or(llvm::PipelineTuningOptions());
        PassPipelineEnv env(mod.get().getContext(), opts, debug, verifyEach);

        llvm::ModulePassManager mpm;
        if (llvm::Error err = env.pb.parsePassPipeline(mpm, pipeline))
          throw std::runtime_error(llvm::toString(std::move(err)));
        mpm.run(mod.get(), env.mam);
      },
      "module"_a, "pipeline"_a, "tuning"_a = nb::none(), "debug"_a = false,
      "verify_each"_a = false,
      "Parse and run a textual pass pipeline over the module in place.");

  m.def(
      "run_default_pipeline",
      [](eudsl::Module &mod, llvm::OptimizationLevel level,
         std::optional<llvm::PipelineTuningOptions> pto, bool debug,
         bool verifyEach) {
        llvm::PipelineTuningOptions opts =
            pto.value_or(llvm::PipelineTuningOptions());
        PassPipelineEnv env(mod.get().getContext(), opts, debug, verifyEach);

        llvm::ModulePassManager mpm =
            (level == llvm::OptimizationLevel::O0)
                ? env.pb.buildO0DefaultPipeline(level)
                : env.pb.buildPerModuleDefaultPipeline(level);
        mpm.run(mod.get(), env.mam);
      },
      "module"_a, "level"_a, "tuning"_a = nb::none(), "debug"_a = false,
      "verify_each"_a = false,
      "Run the default optimization pipeline at the given level.");
}
