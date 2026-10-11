// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace llvm {
class Constant;
class Function;
} // namespace llvm

namespace orc_pgo {

/// Whether C still means the same value when rebuilt in a JIT'd module that
/// reaches the program's globals through their addresses. C must not refer to
/// a thread-local global, an ifunc or a block address, directly or through
/// constant expressions and aggregates. Global initializers are not looked
/// into.
bool isRecreatable(const llvm::Constant &C);

/// Whether F can be compiled again by the JIT and behave like the program's
/// copy: none of its blocks has its address taken, and every constant its
/// instructions use is recreatable.
bool hasRecreatableBody(const llvm::Function &F);

} // namespace orc_pgo
