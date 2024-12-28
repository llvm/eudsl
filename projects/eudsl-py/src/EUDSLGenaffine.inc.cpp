
#include "mlir/InitAllDialects.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populateaffineModule(nb::module_ &m) {
#include "EUDSLGenaffine.cpp.inc"
}
