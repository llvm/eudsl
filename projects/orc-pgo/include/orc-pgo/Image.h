// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/InstrProf.h"

#include <cstdint>

namespace orc_pgo {

inline constexpr llvm::StringLiteral FunctionTableName = "__orc_pgo_functions";
inline constexpr llvm::StringLiteral NumFunctionsName =
    "__orc_pgo_num_functions";
/// [2 x ptr] {begin, end} of the value-node pool lowering emits, or nulls.
inline constexpr llvm::StringLiteral VNodePoolName = "__orc_pgo_vnodes";
inline constexpr unsigned NumValueKinds = llvm::IPVK_Last + 1;

/// One instrumented function, as recorded by lowerPgo.
struct OrcPgoFunctionRecord {
  /// IR PGO name (getIRPGOObjectName), NUL-terminated.
  const char *Name;
  uint64_t CfgHash;
  /// Counters[0] is the entry count (-pgo-instrument-entry).
  uint64_t *Counters;
  /// The function's __profd record, as passed to the value-profiling callbacks.
  void *Data;
  /// The address indirect calls to the function observe, or null.
  const void *Address;
  uint32_t NumCounters;
  uint16_t NumValueSites[NumValueKinds];
};

static_assert(sizeof(OrcPgoFunctionRecord) == 56, "layout emitted by lowerPgo");

/// Bumped on any change to the structures below.
inline constexpr uint64_t ImageVersion = 1;
/// The private global holding a program's OrcPgoImage.
inline constexpr llvm::StringLiteral ImageName = "__orc_pgo_image";
inline constexpr llvm::StringLiteral RegisterImageName =
    "__orc_pgo_register_image";
inline constexpr llvm::StringLiteral StartName = "__orc_pgo_start";

/// One symbol table entry.
struct OrcPgoSymbol {
  /// IR name, NUL-terminated.
  const char *Name;
  /// Null for an extern_weak declaration that did not resolve.
  const void *Address;
};

/// What must match between the build tool and the runtime.
struct OrcPgoFingerprint {
  /// LLVM_VERSION_STRING of the build tool.
  const char *LLVMVersion;
  /// The value of __llvm_profile_raw_version: raw version and variant flags.
  uint64_t RawProfileVersion;
  /// pgoOptionsFingerprint() of the build tool.
  uint64_t OptionsHash;
};

/// Emitted once per program, IR type
/// { i64, { ptr, i64, i64 }, ptr, i64, ptr, i64, ptr, ptr, ptr, i64, ptr, i64
/// }.
struct OrcPgoImage {
  uint64_t Version;
  OrcPgoFingerprint Fingerprint;
  /// Bitcode of the merged module before instrumentation.
  const char *Snapshot;
  uint64_t SnapshotSize;
  const OrcPgoFunctionRecord *Functions;
  uint64_t NumFunctions;
  /// {begin, end} of the value-node pool (VNodePoolName).
  void *const *VNodePool;
  void **Dispatch;
  /// Index into Functions of each slot's function.
  const uint32_t *DispatchFunctions;
  uint64_t NumDispatch;
  const OrcPgoSymbol *Symbols;
  uint64_t NumSymbols;
};

static_assert(sizeof(OrcPgoSymbol) == 16, "layout emitted by emitSymbolTable");
static_assert(sizeof(OrcPgoImage) == 112, "layout emitted by buildAotModule");

} // namespace orc_pgo

extern "C" {
/// Called by the program's static constructor. Only records Image: it can run
/// before LLVM's own static initializers.
void __orc_pgo_register_image(const orc_pgo::OrcPgoImage *Image);
/// Called at the entry of main, after all static initialization. Validates
/// the recorded images and starts the runtime. Idempotent.
void __orc_pgo_start(void);
/// __orc_pgo_start, then one synchronous poll of every runtime.
void __orc_pgo_poll(void);
}
