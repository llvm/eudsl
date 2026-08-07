// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"
#include "IR/TargetInit.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/SourceMgr.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

void populate_context(nb::module_ &m) {
  nb::class_<eudsl::Context>(m, "Context")
      .def(nb::init<>())
      .def(
          "__enter__",
          [](eudsl::Context &self) -> eudsl::Context & { return self; },
          nb::rv_policy::reference_internal)
      .def(
          "__exit__",
          [](eudsl::Context &self, nb::object, nb::object, nb::object) {
            self.release();
          },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none())
      .def_static("_get_live_count", &eudsl::Context::liveCount);

  nb::class_<eudsl::Module>(m, "Module")
      .def(nb::init<const std::string &, eudsl::Context &>(), "name"_a,
           "context"_a, nb::keep_alive<1, 3>())
      .def_prop_rw(
          "module_identifier",
          [](eudsl::Module &self) { return self.get().getModuleIdentifier(); },
          [](eudsl::Module &self, const std::string &id) {
            self.get().setModuleIdentifier(id);
          })
      .def_prop_rw(
          "source_filename",
          [](eudsl::Module &self) { return self.get().getSourceFileName(); },
          [](eudsl::Module &self, const std::string &name) {
            self.get().setSourceFileName(name);
          })
      .def_prop_ro("_is_consumed", &eudsl::Module::isConsumed)
      .def("_take",
           [](eudsl::Module &self) {
             // Test-only ownership sink: drops the module on the floor so
             // tests can observe the consumed state without a JIT.
             std::unique_ptr<llvm::Module> owned = self.take();
             owned.reset();
           })
      .def_prop_ro(
          "context",
          [](eudsl::Module &self) -> eudsl::Context & { return self.context(); },
          nb::rv_policy::reference_internal)
      .def("__str__", [](eudsl::Module &self) {
        std::string s;
        llvm::raw_string_ostream os(s);
        self.get().print(os, nullptr);
        return s;
      });

  m.def(
      "parse_assembly",
      [](const std::string &ir, eudsl::Context &ctx, const std::string &name) {
        llvm::SMDiagnostic err;
        std::unique_ptr<llvm::Module> mod =
            llvm::parseAssemblyString(ir, err, ctx.get());
        if (!mod) {
          std::string msg;
          llvm::raw_string_ostream os(msg);
          err.print(name.c_str(), os);
          throw std::runtime_error(msg);
        }
        mod->setModuleIdentifier(name);
        return new eudsl::Module(std::move(mod), ctx);
      },
      "ir"_a, "context"_a, "name"_a = "<string>", nb::keep_alive<0, 2>(),
      "Parse LLVM textual IR into a new Module.");

  eudsl::initializeTargets();

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
