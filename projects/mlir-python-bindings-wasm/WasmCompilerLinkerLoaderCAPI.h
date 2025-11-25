#ifndef MLIR_C_TARGET_WASM_H
#define MLIR_C_TARGET_WASM_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef compile(MlirOperation module,
                                         MlirStringRef moduleName);
MLIR_CAPI_EXPORTED void linkLoad(MlirStringRef objectFileName,
                                 MlirStringRef binaryFileName);

MLIR_CAPI_EXPORTED void *getSymbolAddress(MlirStringRef name);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_TARGET_WASM_H
