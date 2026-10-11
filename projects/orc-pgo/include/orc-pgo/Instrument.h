// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/Support/Error.h"

namespace llvm {
class GlobalVariable;
class Module;
class TargetMachine;
} // namespace llvm

namespace orc_pgo {

/// configurePgoOptions(), then LLVM's PGOInstrumentationGen (FDO) under
/// PgoAnalysisManagers. The llvm.instrprof.* intrinsics stay un-lowered.
llvm::Error runPgoGen(llvm::Module &M);

struct LoweredPgo {
  /// [N x OrcPgoFunctionRecord], named FunctionTableName.
  llvm::GlobalVariable *Functions;
  /// i32 N, named NumFunctionsName.
  llvm::GlobalVariable *NumFunctions;
  /// [2 x ptr], named VNodePoolName.
  llvm::GlobalVariable *VNodes;
};

/// What clang's -fprofile-generate pipeline runs right after Gen: loop
/// rotation, then LLVM's InstrProfilingLoweringPass with non-atomic counters
/// and counter promotion. Then the counter table. The module defines
/// __llvm_profile_runtime itself and calls __orc_pgo_instrument_target/_memop
/// instead of compiler-rt's value-profiling entry points. TM, if given, is the
/// target whose TargetTransformInfo loop rotation consults, as under clang.
llvm::Expected<LoweredPgo> lowerPgo(llvm::Module &M,
                                    llvm::TargetMachine *TM = nullptr);

} // namespace orc_pgo
