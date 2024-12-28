
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Polynomial/IR/PolynomialOps.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatepolynomialModule(nb::module_ &m) {
#include "EUDSLGenpolynomial.cpp.inc"
}
