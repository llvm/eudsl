
#include "mlir/InitAllDialects.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populatex86vectorModule(nb::module_ &m) {
#include "EUDSLGenx86vector.cpp.inc"
}
