// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "orc-pgo/Image.h"

#include "llvm/ProfileData/InstrProf.h"

#include <cstdint>
#include <vector>

namespace orc_pgo {

/// compiler-rt's INSTR_PROF_VNODE_POOL_SIZE: the nodes of the runtime pool
/// that serves values once the image's pool (about one node per value site)
/// is used up.
inline constexpr uint32_t RuntimeValueNodePoolSize = 1024;

/// Sets the value-node pool lowering emits (see VNodePoolName). Values are
/// recorded in the image's pool first, then in the runtime pool; with no image
/// pool only the latter is used. Values recorded before the call stay valid.
void setValueNodePool(void *Begin, void *End);

/// Forgets every value node handed out and the image pool, and clears the
/// exhaustion flag. Only for a process that starts a new program (tests): no
/// value list may be read or extended afterwards.
void resetValueNodePools();

/// Whether a new value was dropped for lack of nodes in both pools since the
/// pools were last reset.
bool valueNodePoolExhausted();

/// The values recorded at site Site of kind Kind, in insertion order.
std::vector<InstrProfValueData> readValueSite(const OrcPgoFunctionRecord &R,
                                              uint32_t Kind, uint32_t Site);

} // namespace orc_pgo

// compiler-rt's __llvm_profile_instrument_target / _memop under orc-pgo
// names; lowerPgo points lowered value-profiling calls at these.
extern "C" {
void __orc_pgo_instrument_target(uint64_t Value, void *Data,
                                 uint32_t CounterIndex);
void __orc_pgo_instrument_memop(uint64_t Value, void *Data,
                                uint32_t CounterIndex);
}
