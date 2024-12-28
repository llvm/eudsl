
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatequantModule(nb::module_ &m) {
#include "EUDSLGenquant.cpp.inc"
}
