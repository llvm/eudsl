
#include "mlir/InitAllDialects.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatescfModule(nb::module_ &m) {
#include "EUDSLGenscf.cpp.inc"
}
