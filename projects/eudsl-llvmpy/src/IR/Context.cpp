// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Errors.h"
#include "IR/Ownership.h"
#include "IR/Sequence.h"
#include "IR/TargetInit.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Target/TargetMachine.h>

#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

void populate_context(nb::module_ &m) {
  // GlobalValue linkage kinds (llvm::GlobalValue::LinkageTypes) and thread-local
  // modes. Registered here, ahead of every other populate_* call, so factories
  // such as Module.add_global and Function.create can take these as arguments
  // (rather than hardcoding one choice) with enum defaults.
  nb::enum_<llvm::GlobalValue::LinkageTypes>(m, "Linkage")
      .value("EXTERNAL", llvm::GlobalValue::LinkageTypes::ExternalLinkage)
      .value("INTERNAL", llvm::GlobalValue::LinkageTypes::InternalLinkage)
      .value("PRIVATE", llvm::GlobalValue::LinkageTypes::PrivateLinkage)
      .value("LINKONCE", llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage)
      .value("LINKONCE_ODR", llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage)
      .value("WEAK", llvm::GlobalValue::LinkageTypes::WeakAnyLinkage)
      .value("COMMON", llvm::GlobalValue::LinkageTypes::CommonLinkage)
      .value("APPENDING", llvm::GlobalValue::LinkageTypes::AppendingLinkage)
      .value("EXTERNAL_WEAK", llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage);

  nb::enum_<llvm::GlobalValue::ThreadLocalMode>(m, "ThreadLocalMode")
      .value("NOT_THREAD_LOCAL", llvm::GlobalValue::ThreadLocalMode::NotThreadLocal)
      .value("GENERAL_DYNAMIC", llvm::GlobalValue::ThreadLocalMode::GeneralDynamicTLSModel)
      .value("LOCAL_DYNAMIC", llvm::GlobalValue::ThreadLocalMode::LocalDynamicTLSModel)
      .value("INITIAL_EXEC", llvm::GlobalValue::ThreadLocalMode::InitialExecTLSModel)
      .value("LOCAL_EXEC", llvm::GlobalValue::ThreadLocalMode::LocalExecTLSModel);

  nb::class_<eudsl::Context>(m, "Context")
      .def(nb::init<>())
      .def(
          "__enter__",
          [](eudsl::Context &self) -> eudsl::Context & {
            self.pushCurrent();
            return self;
          },
          nb::rv_policy::reference_internal)
      .def(
          "__exit__",
          [](eudsl::Context &self, nb::object, nb::object, nb::object) {
            self.popCurrent();
            self.release();
          },
          nb::arg("exc_type").none(), nb::arg("exc_value").none(),
          nb::arg("traceback").none())
      .def_static(
          "current",
          []() -> eudsl::Context * { return eudsl::Context::current(); },
          nb::rv_policy::reference,
          "The innermost active `with Context():`, or None.")
      .def_static("_get_live_count", &eudsl::Context::liveCount)
      .def_static("_get_live_module_count", &eudsl::Module::liveCount);

  nb::class_<eudsl::Module>(m, "Module")
      .def(
          "__init__",
          [](eudsl::Module *self, const std::string &name, nb::handle context) {
            new (self) eudsl::Module(name, eudsl::currentOr(context));
          },
          "name"_a, "context"_a = nb::none(), nb::keep_alive<1, 3>())
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
      .def_prop_ro(
          "functions",
          [](eudsl::Module &self) {
            llvm::Module *mod = &self.get();
            eudsl::Sequence<llvm::Function> seq;
            seq.length = [mod] {
              return static_cast<std::size_t>(
                  std::distance(mod->begin(), mod->end()));
            };
            seq.at = [mod](std::size_t i) {
              auto it = mod->begin();
              std::advance(it, i);
              return &*it;
            };
            return seq;
          },
          nb::keep_alive<0, 1>())
      .def("__len__",
           [](eudsl::Module &self) {
             return static_cast<Py_ssize_t>(std::distance(
                 self.get().begin(), self.get().end()));
           })
      .def(
          "__getitem__",
          [](eudsl::Module &self, Py_ssize_t i) {
            std::vector<llvm::Function *> out;
            for (llvm::Function &f : self.get().functions())
              out.push_back(&f);
            return eudsl::nthOrThrow(out, i);
          },
          nb::rv_policy::reference_internal)
      .def(
          "__iter__",
          [](eudsl::Module &self) {
            return nb::make_iterator<nb::rv_policy::reference>(
                nb::type<eudsl::Module>(), "FunctionIterator",
                self.get().begin(), self.get().end());
          },
          nb::keep_alive<0, 1>())
      .def(
          "get_function",
          [](eudsl::Module &self, const std::string &name) {
            return self.get().getFunction(name);
          },
          "name"_a, nb::rv_policy::reference)
      .def(
          "get_global_variable",
          [](eudsl::Module &self, const std::string &name) {
            return self.get().getNamedGlobal(name);
          },
          "name"_a, nb::rv_policy::reference)
      .def(
          "add_named_metadata",
          [](eudsl::Module &self, const std::string &name, llvm::MDNode *node) {
            self.get().getOrInsertNamedMetadata(name)->addOperand(node);
          },
          "name"_a, "node"_a)
      .def(
          "named_metadata",
          [](eudsl::Module &self, const std::string &name) {
            std::vector<llvm::MDNode *> out;
            if (auto *nmd = self.get().getNamedMetadata(name))
              for (llvm::MDNode *op : nmd->operands())
                out.push_back(op);
            return out;
          },
          "name"_a)
      .def("__str__",
           [](eudsl::Module &self) {
             std::string s;
             llvm::raw_string_ostream os(s);
             self.get().print(os, nullptr);
             return s;
           })
      .def("verify",
           [](eudsl::Module &self) {
             std::string msg;
             llvm::raw_string_ostream os(msg);
             if (llvm::verifyModule(self.get(), &os))
               throw eudsl::VerifyError(msg);
           })
      .def("to_bitcode", [](eudsl::Module &self) {
        std::string buf;
        llvm::raw_string_ostream os(buf);
        llvm::WriteBitcodeToFile(self.get(), os);
        os.flush();
        return nb::bytes(buf.data(), buf.size());
      })
      .def(
          "set_data_layout_from",
          [](eudsl::Module &self, llvm::TargetMachine &tm) {
            self.get().setDataLayout(tm.createDataLayout());
          },
          "target_machine"_a)
      .def(
          "add_global",
          [](eudsl::Module &self, llvm::Type *ty, const std::string &name,
             llvm::Constant *init, bool isConstant,
             llvm::GlobalVariable* insertBefore,
             llvm::GlobalValue::LinkageTypes linkage,
             llvm::GlobalValue::ThreadLocalMode tlMode,
             unsigned addressSpace) -> llvm::GlobalVariable * {
            return new llvm::GlobalVariable(
                self.get(), ty, isConstant, linkage, init, name, insertBefore,
                tlMode, addressSpace);
          },
          "type"_a, "name"_a, "init"_a = nullptr, "constant"_a = false,
          "insert_before"_a = nullptr,
          "linkage"_a = llvm::GlobalValue::LinkageTypes::ExternalLinkage,
          "thread_local_mode"_a = llvm::GlobalValue::ThreadLocalMode::NotThreadLocal,
          "address_space"_a = 0, nb::rv_policy::reference_internal)
      .def(
          "get_global",
          [](eudsl::Module &self, const std::string &name) {
            return self.get().getNamedGlobal(name);
          },
          "name"_a, nb::rv_policy::reference_internal)
      .def_prop_ro("globals", [](eudsl::Module &self) {
        std::vector<llvm::GlobalVariable *> out;
        for (llvm::GlobalVariable &g : self.get().globals())
          out.push_back(&g);
        return out;
      });

  m.def(
      "parse_assembly",
      [](const std::string &ir, nb::handle context,
         const std::string &module_identifier,
         const std::string &source_filename) {
        eudsl::Context &ctx = eudsl::currentOr(context);
        llvm::SMDiagnostic err;
        std::unique_ptr<llvm::Module> mod =
            llvm::parseAssemblyString(ir, err, ctx.get());
        if (!mod) {
          std::string msg;
          llvm::raw_string_ostream os(msg);
          err.print(module_identifier.c_str(), os);
          throw eudsl::ParseError(msg);
        }
        mod->setModuleIdentifier(module_identifier);
        mod->setSourceFileName(source_filename);
        return new eudsl::Module(std::move(mod), ctx);
      },
      "ir"_a, "context"_a = nb::none(), "module_identifier"_a = "<string>",
      "source_filename"_a = "", nb::keep_alive<0, 2>(),
      "Parse LLVM textual IR into a new Module.");

  m.def(
      "parse_bitcode",
      [](nb::bytes data, nb::handle context) {
        eudsl::Context &ctx = eudsl::currentOr(context);
        llvm::StringRef ref(data.c_str(), data.size());
        auto buf = llvm::MemoryBuffer::getMemBuffer(ref, "<bitcode>", false);
        llvm::Expected<std::unique_ptr<llvm::Module>> mod =
            llvm::parseBitcodeFile(buf->getMemBufferRef(), ctx.get());
        if (!mod)
          throw eudsl::ParseError(llvm::toString(mod.takeError()));
        return new eudsl::Module(std::move(*mod), ctx);
      },
      "data"_a, "context"_a = nb::none(), nb::keep_alive<0, 2>(),
      "Parse an LLVM bitcode buffer into a new Module.");

  eudsl::initializeTargets();
}
