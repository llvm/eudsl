// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "orc-pgo/Image.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>

namespace orc_pgo {

/// An IR-level, entry-first indexed profile of the functions' live counters
/// and value sites, as llvm-profdata would produce from a raw profile.
/// Indirect-call targets are mapped to their name's MD5 through each record's
/// Address.
llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
buildIndexedProfile(llvm::ArrayRef<OrcPgoFunctionRecord> Functions);

} // namespace orc_pgo
