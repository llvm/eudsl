
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatemeshModule(nb::module_ &m) {
#include "EUDSLGenmesh.cpp.inc"
}
