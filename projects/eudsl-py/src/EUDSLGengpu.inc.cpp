
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IRBuilder.h"
#include <nanobind/nanobind.h>
#include "type_casters.h"
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
void populategpuModule(nb::module_ &m) {
#include "EUDSLGengpu.cpp.inc"
}
