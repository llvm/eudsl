#ifndef MLIR_C_TARGET_WASM_H
#define MLIR_C_TARGET_WASM_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef compileModule(MlirOperation module,
                                         MlirStringRef moduleName, int optLevel);
MLIR_CAPI_EXPORTED void linkLoadModule(MlirStringRef objectFileName,
                                 MlirStringRef binaryFileName);

MLIR_CAPI_EXPORTED void *getSymbolAddress(MlirStringRef name);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_TARGET_WASM_H
