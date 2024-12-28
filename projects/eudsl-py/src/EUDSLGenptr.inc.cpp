
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populateptrModule(nb::module_ &m) {
#include "EUDSLGenptr.cpp.inc"
}
