
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatespirvModule(nb::module_ &m) {
#include "EUDSLGenspirv.cpp.inc"
}
