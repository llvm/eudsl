// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/Metadata.h>

#include <vector>

void populate_metadata(nb::module_ &m) {
  nb::class_<llvm::Metadata>(m, "Metadata")
      .def("__str__",
           [](llvm::Metadata &self) { return eudsl::toString(self); });

  nb::class_<llvm::MDString, llvm::Metadata>(m, "MDString")
      .def_prop_ro("string",
                   [](llvm::MDString &self) { return self.getString().str(); });

  nb::class_<llvm::MDNode, llvm::Metadata>(m, "MDNode")
      .def_prop_ro("num_operands", &llvm::MDNode::getNumOperands)
      .def(
          "operand",
          [](llvm::MDNode &self, unsigned i) -> llvm::Metadata * {
            return self.getOperand(i).get();
          },
          "index"_a, nb::rv_policy::reference_internal);

  m.def(
      "md_string",
      [](const std::string &s, eudsl::Context *context) -> llvm::MDString * {
        return llvm::MDString::get(eudsl::currentOr(context).get(), s);
      },
      "value"_a, "context"_a.none() = nb::none(), nb::rv_policy::reference,
      nb::keep_alive<0, 2>());
  m.def(
      "md_node",
      [](std::vector<llvm::Metadata *> ops,
         eudsl::Context *context) -> llvm::MDNode * {
        return llvm::MDNode::get(eudsl::currentOr(context).get(), ops);
      },
      "operands"_a, "context"_a.none() = nb::none(), nb::rv_policy::reference,
      nb::keep_alive<0, 2>());
}
