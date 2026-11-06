// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

#include <memory>

namespace llvm {
class Module;
class TargetMachine;
} // namespace llvm

namespace orc_pgo {

/// Fresh analysis managers with every PassBuilder analysis registered and an
/// explicit TargetLibraryInfo for M's triple. With a TargetMachine, the
/// function analyses see its TargetTransformInfo, as under clang.
struct PgoAnalysisManagers {
  explicit PgoAnalysisManagers(const llvm::Module &M,
                               llvm::TargetMachine *TM = nullptr);
  PgoAnalysisManagers(const PgoAnalysisManagers &) = delete;
  PgoAnalysisManagers &operator=(const PgoAnalysisManagers &) = delete;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

private:
  std::unique_ptr<llvm::TargetLibraryInfoImpl> TLII;
};

/// The pipeline tuning options clang uses for -O<Level>
/// (EmitAssemblyHelper::RunOptimizationPipeline in clang's BackendUtil.cpp,
/// with the defaults of CompilerInvocation.cpp): loop unrolling and
/// interleaving, loop and SLP vectorization at O2 and above; loop interchange
/// always; no loop fusion and no function merging; the call graph profile.
/// PassBuilder's own defaults differ, e.g. for SLP vectorization.
llvm::PipelineTuningOptions
clangPipelineTuningOptions(llvm::OptimizationLevel Level);

} // namespace orc_pgo
