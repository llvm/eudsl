
#include "mlir/InitAllDialects.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatepdl_interpModule(nb::module_ &m) {
#include "EUDSLGenpdl_interp.cpp.inc"
}
