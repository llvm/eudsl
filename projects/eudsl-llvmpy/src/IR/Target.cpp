// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <memory>
#include <optional>

namespace {
std::string emit(llvm::TargetMachine &self, eudsl::Module &mod,
                 llvm::CodeGenFileType type) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  llvm::buffer_ostream bos(os);
  llvm::legacy::PassManager pm;
  if (self.addPassesToEmitFile(pm, bos, nullptr, type)) {
    throw std::runtime_error(                 // LCOV_EXCL_LINE
        "target cannot emit this file type"); // LCOV_EXCL_LINE
  } // LCOV_EXCL_LINE
  pm.run(mod.get());
  return buf;
}
} // namespace

void populate_target(nb::module_ &m) {
  m.def("host_triple", []() { return llvm::sys::getDefaultTargetTriple(); });

  nb::class_<llvm::TargetMachine>(m, "TargetMachine")
      .def(nb::new_([](std::optional<std::string> triple,
                       std::optional<std::string> cpu,
                       std::optional<std::vector<std::string>> features)
                        -> llvm::TargetMachine * {
             std::string tripleStr =
                 triple.value_or(llvm::sys::getDefaultTargetTriple());
             std::string cpuStr = cpu.value_or("");
             std::string featStr;
             if (features) {
               for (size_t i = 0; i < features->size(); ++i) {
                 if (i > 0)
                   featStr += ",";
                 featStr += (*features)[i];
               }
             }
             llvm::Triple tt(tripleStr);
             std::string err;
             const llvm::Target *target =
                 llvm::TargetRegistry::lookupTarget(tt, err);
             if (!target)
               throw std::runtime_error(err);
             llvm::TargetOptions opts;
             llvm::TargetMachine *tm = target->createTargetMachine(
                 tt, cpuStr, featStr, opts, std::nullopt);
             if (!tm) { // LCOV_EXCL_START
               throw std::runtime_error("could not create TargetMachine for " +
                                        tripleStr);
             } // LCOV_EXCL_STOP
             return tm;
           }),
           "triple"_a = nb::none(), "cpu"_a = nb::none(),
           "features"_a = nb::none())
      .def_prop_ro("triple",
                   [](llvm::TargetMachine &self) {
                     return self.getTargetTriple().str();
                   })
      .def_prop_ro("data_layout_str",
                   [](llvm::TargetMachine &self) {
                     return self.createDataLayout().getStringRepresentation();
                   })
      .def(
          "emit_assembly",
          [](llvm::TargetMachine &self, eudsl::Module &mod) {
            return emit(self, mod, llvm::CodeGenFileType::AssemblyFile);
          },
          "module"_a)
      .def(
          "emit_object",
          [](llvm::TargetMachine &self, eudsl::Module &mod) {
            std::string obj =
                emit(self, mod, llvm::CodeGenFileType::ObjectFile);
            return nb::bytes(obj.data(), obj.size());
          },
          "module"_a);

  m.def(
      "registered_targets",
      []() {
        std::vector<std::string> names;
        for (const llvm::Target &t : llvm::TargetRegistry::targets())
          names.emplace_back(t.getName());
        return names;
      },
      "Names of the LLVM targets linked into this extension.");
}
