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
struct TM {
  std::unique_ptr<llvm::TargetMachine> tm;
};

std::string emit(TM &self, eudsl::Module &mod, llvm::CodeGenFileType type) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  llvm::buffer_ostream bos(os);
  llvm::legacy::PassManager pm;
  if (self.tm->addPassesToEmitFile(pm, bos, nullptr, type))
    throw std::runtime_error(  // LCOV_EXCL_LINE
        "target cannot emit this file type");  // LCOV_EXCL_LINE
  pm.run(mod.get());
  return buf;
}
} // namespace

void populate_target(nb::module_ &m) {
  m.def("host_triple", []() { return llvm::sys::getDefaultTargetTriple(); });

  nb::class_<TM>(m, "TargetMachine")
      .def(
          "__init__",
          [](TM *self, const std::string &triple, const std::string &cpu,
             const std::string &features) {
            std::string tripleStr =
                triple.empty() ? llvm::sys::getDefaultTargetTriple() : triple;
            llvm::Triple tt(tripleStr);
            std::string err;
            const llvm::Target *target =
                llvm::TargetRegistry::lookupTarget(tt, err);
            if (!target)
              throw std::runtime_error(err);
            llvm::TargetOptions opts;
            llvm::TargetMachine *tm = target->createTargetMachine(
                tt, cpu, features, opts, std::nullopt);
            if (!tm)
              throw std::runtime_error(  // LCOV_EXCL_LINE
                  "could not create TargetMachine for " + tripleStr);  // LCOV_EXCL_LINE
            new (self) TM{std::unique_ptr<llvm::TargetMachine>(tm)};
          },
          "triple"_a = "", "cpu"_a = "", "features"_a = "")
      .def_prop_ro("triple",
                   [](TM &self) { return self.tm->getTargetTriple().str(); })
      .def_prop_ro("data_layout_str",
                   [](TM &self) {
                     return self.tm->createDataLayout()
                         .getStringRepresentation();
                   })
      .def(
          "emit_assembly",
          [](TM &self, eudsl::Module &mod) {
            return emit(self, mod, llvm::CodeGenFileType::AssemblyFile);
          },
          "module"_a)
      .def(
          "emit_object",
          [](TM &self, eudsl::Module &mod) {
            std::string obj = emit(self, mod, llvm::CodeGenFileType::ObjectFile);
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
