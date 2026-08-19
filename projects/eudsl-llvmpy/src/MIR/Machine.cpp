// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/CodeGenTypes/LowLevelType.h>

// LowLevelType (LLT) is the generic-MIR type: a target-independent "bag of
// bits" describing a scalar/pointer/vector operand, distinct from the uniqued
// llvm::Type hierarchy. It is a small value type (not context-owned and not
// polymorphic), so it is bound by value with no ownership/keep_alive plumbing.
// Binding it first also proves LLVMCodeGenTypes links into the extension.
void populate_mir(nb::module_ &m) {
  nb::class_<llvm::LLT>(m, "LLT")
      .def_static("scalar", &llvm::LLT::scalar, "size_in_bits"_a)
      .def_static("pointer", &llvm::LLT::pointer, "address_space"_a,
                  "size_in_bits"_a)
      .def_static(
          "fixed_vector",
          [](unsigned numElements, unsigned scalarSizeInBits) {
            return llvm::LLT::fixed_vector(numElements, scalarSizeInBits);
          },
          "num_elements"_a, "scalar_size_in_bits"_a)
      .def_prop_ro("size_in_bits",
                   [](const llvm::LLT &self) {
                     return self.getSizeInBits().getKnownMinValue();
                   })
      .def_prop_ro(
          "scalar_size_in_bits",
          [](const llvm::LLT &self) { return self.getScalarSizeInBits(); })
      .def_prop_ro("num_elements",
                   [](const llvm::LLT &self) { return self.getNumElements(); })
      .def_prop_ro("address_space",
                   [](const llvm::LLT &self) { return self.getAddressSpace(); })
      .def_prop_ro("is_scalar",
                   [](const llvm::LLT &self) { return self.isScalar(); })
      .def_prop_ro("is_pointer",
                   [](const llvm::LLT &self) { return self.isPointer(); })
      .def_prop_ro("is_vector",
                   [](const llvm::LLT &self) { return self.isVector(); })
      .def_prop_ro("is_integer",
                   [](const llvm::LLT &self) { return self.isInteger(); })
      .def_prop_ro("is_float",
                   [](const llvm::LLT &self) { return self.isFloat(); })
      .def_prop_ro("is_valid",
                   [](const llvm::LLT &self) { return self.isValid(); })
      .def("__eq__",
           [](const llvm::LLT &self, nb::handle other) {
             llvm::LLT o;
             if (!nb::try_cast<llvm::LLT>(other, o))
               return false;
             return self == o;
           })
      .def("__ne__",
           [](const llvm::LLT &self, nb::handle other) {
             llvm::LLT o;
             if (!nb::try_cast<llvm::LLT>(other, o))
               return true;
             return self != o;
           })
      .def("__hash__",
           [](const llvm::LLT &self) {
             return static_cast<Py_ssize_t>(self.getUniqueRAWLLTData());
           })
      .def("__str__",
           [](const llvm::LLT &self) { return eudsl::toString(self); });
}
