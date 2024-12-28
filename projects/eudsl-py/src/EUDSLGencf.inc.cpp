
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatecfModule(nb::module_ &m) {
#include "EUDSLGencf.cpp.inc"
}
