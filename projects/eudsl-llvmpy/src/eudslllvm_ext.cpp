// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <llvm/Support/Debug.h>

#include "IR/Errors.h"

#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

void populate_context(nb::module_ &m);
void populate_sequences(nb::module_ &m);
void populate_casters(nb::module_ &m);
void populate_attributes(nb::module_ &m);
void populate_types(nb::module_ &m);
void populate_values(nb::module_ &m);
void populate_instructions(nb::module_ &m);
void populate_constants(nb::module_ &m);
void populate_metadata(nb::module_ &m);
void populate_builder(nb::module_ &m);
void populate_passes(nb::module_ &m);
void populate_target(nb::module_ &m);
void populate_linker(nb::module_ &m);
void populate_jit(nb::module_ &m);
void populate_intrinsics(nb::module_ &m);
void populate_mir(nb::module_ &m);

NB_MODULE(eudslllvm_ext, m) {
  m.doc() = "Hand-written nanobind bindings over the LLVM C++ IR API.";

  m.def(
      "enable_debug",
      [](const std::vector<std::string> &debug_types, bool enabled) {
        llvm::DebugFlag = enabled;
        if (enabled && !debug_types.empty()) {
          // setCurrentDebugTypes copies each string into its own storage
          // (CurrentDebugType is a std::vector<std::string>), so the c_str()
          // pointers only need to be valid for the call.
          std::vector<const char *> ptrs;
          for (const std::string &s : debug_types)
            ptrs.push_back(s.c_str());
          llvm::setCurrentDebugTypes(ptrs.data(), ptrs.size());
        }
      },
      "debug_types"_a = std::vector<std::string>(), "enabled"_a = true,
      "Toggle LLVM's LLVM_DEBUG tracing (to stderr), like -debug / "
      "-debug-only. Pass a list of DEBUG_TYPEs (e.g. [\"regalloc\"]) to scope "
      "it, or nothing for all; pass enabled=False to turn it back off. No "
      "effect unless LLVM was built with assertions.");

  // llvm.ir -- the IR-core submodule (Context/Module, the Value and Constant
  // hierarchies, instructions, metadata, IRBuilder, attribute enums, errors).
  nb::module_ ir = m.def_submodule("ir");
  eudsl::registerExceptions(ir);
  populate_context(ir);
  populate_sequences(ir);
  populate_casters(ir);
  populate_attributes(ir);
  populate_values(ir);
  populate_instructions(ir);
  populate_constants(ir);
  populate_metadata(ir);
  populate_builder(ir);

  // llvm.types -- the Type hierarchy and type factories.
  nb::module_ types = m.def_submodule("types");
  populate_types(types);

  // llvm.passmanager -- pass pipeline execution.
  nb::module_ passmanager = m.def_submodule("passmanager");
  populate_passes(passmanager);

  // llvm.jit -- codegen and execution: target machine, linker, ORC JIT.
  nb::module_ jit = m.def_submodule("jit");
  populate_target(jit);
  populate_linker(jit);
  populate_jit(jit);

  // llvm.intrinsics -- intrinsic lookup and declaration.
  nb::module_ intrinsics = m.def_submodule("intrinsics");
  populate_intrinsics(intrinsics);

  // llvm.mir -- Machine IR: the target-independent LowLevelType (LLT).
  nb::module_ mir = m.def_submodule("mir");
  populate_mir(mir);
}
