//===- SPIRVToWGSLCAPI.h - SPIR-V to WGSL translation -----------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_TARGET_SPIRV_TO_WGSL_H
#define MLIR_C_TARGET_SPIRV_TO_WGSL_H

#include "mlir-c/Support.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  /// `*wgsl` holds the translated WGSL.
  MlirSPIRVToWGSLSuccess = 0,
  /// Tint rejected the module; `*wgsl` holds its diagnostics.
  MlirSPIRVToWGSLTranslationFailed = 1,
  /// Ran out of memory. `*wgsl` is NULL and any diagnostic is lost -- reporting
  /// this separately keeps it from masquerading as a translation failure with
  /// an empty message, which matters on wasm where the heap is small.
  MlirSPIRVToWGSLOutOfMemory = 2,
  /// `wgsl` was NULL, or `words` was NULL with a nonzero `wordCount`.
  MlirSPIRVToWGSLInvalidArgument = 3,
} MlirSPIRVToWGSLStatus;

/// Translates a SPIR-V binary to WGSL source.
///
/// `words`/`wordCount` are the SPIR-V module. On success `*wgsl` holds newly
/// allocated, NUL-terminated WGSL; on MlirSPIRVToWGSLTranslationFailed it holds
/// diagnostics instead. In both cases the caller owns the buffer and must
/// release it with `mlirSPIRVToWGSLFree`. Otherwise `*wgsl` is set to NULL.
MLIR_CAPI_EXPORTED MlirSPIRVToWGSLStatus
mlirSPIRVToWGSL(const uint32_t *words, size_t wordCount, char **wgsl);

MLIR_CAPI_EXPORTED void mlirSPIRVToWGSLFree(char *wgsl);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_TARGET_SPIRV_TO_WGSL_H
