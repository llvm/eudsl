// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ProfileData/InstrProf.h"

#include <cstdint>

namespace orc_pgo::prof {

using llvm::IPVK_Last;
using IntPtrT = void *;
struct ValueProfNode;
using PtrToNodeT = ValueProfNode *;

struct ValueProfNode {
#define INSTR_PROF_VALUE_NODE(Type, LLVMType, Name, Initializer) Type Name;
#include "llvm/ProfileData/InstrProfData.inc"
};

/// A __profd_<fn> record.
struct alignas(INSTR_PROF_DATA_ALIGNMENT) Data {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) Type Name;
#include "llvm/ProfileData/InstrProfData.inc"
};

} // namespace orc_pgo::prof
