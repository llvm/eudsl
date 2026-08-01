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

/// Translates a SPIR-V binary to WGSL source.
///
/// `words`/`wordCount` are the SPIR-V module. On success returns true and
/// stores newly allocated, NUL-terminated WGSL in `*wgsl`; on failure returns
/// false and stores a diagnostic there instead. Either way the caller owns the
/// buffer and must release it with `mlirSPIRVToWGSLFree`.
MLIR_CAPI_EXPORTED bool mlirSPIRVToWGSL(const uint32_t *words, size_t wordCount,
                                        char **wgsl);

MLIR_CAPI_EXPORTED void mlirSPIRVToWGSLFree(char *wgsl);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_TARGET_SPIRV_TO_WGSL_H
