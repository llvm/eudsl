
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatepdlModule(nb::module_ &m) {
#include "EUDSLGenpdl.cpp.inc"
}