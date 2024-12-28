
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populateindexModule(nb::module_ &m) {
#include "EUDSLGenindex.cpp.inc"
}
