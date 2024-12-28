
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatebufferizationModule(nb::module_ &m) {
#include "EUDSLGenbufferization.cpp.inc"
}
