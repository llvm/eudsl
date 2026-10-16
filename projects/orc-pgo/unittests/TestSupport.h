// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <string>

namespace orc_pgo::test {

/// Aborts the test binary with the diagnostic if Text is invalid IR.
std::unique_ptr<llvm::Module> parseIR(llvm::LLVMContext &Ctx,
                                      llvm::StringRef Text);

/// parseIR, then sets the host triple and data layout, as instrumentation
/// needs the object format the JIT will use.
std::unique_ptr<llvm::Module> parseHostIR(llvm::LLVMContext &Ctx,
                                          llvm::StringRef Text);

/// parseIR, then sets TT and its default data layout.
std::unique_ptr<llvm::Module> parseIRFor(llvm::LLVMContext &Ctx,
                                         llvm::StringRef Text,
                                         const llvm::Triple &TT);

/// The host architecture on Linux: what CI's Linux runners JIT, as ELF, which
/// JITLink links (and on most hosts runs) on any host.
llvm::Triple hostElfTriple();

/// The message of a failed Error, or "" on success. Consumes the error.
std::string errorText(llvm::Error E);

template <typename T>
std::string errorText(llvm::Expected<T> &E) {
  return E ? std::string() : errorText(E.takeError());
}

/// A TargetMachine for the host, as the JIT compiles for.
std::unique_ptr<llvm::TargetMachine> makeHostTargetMachine();

std::unique_ptr<llvm::orc::LLJIT> makeJIT();

/// A JIT for TT, for the generic CPU, that finds process symbols by TT's
/// mangling.
std::unique_ptr<llvm::orc::LLJIT> makeJIT(const llvm::Triple &TT);

/// Adds M to a new host JIT that also provides orc-pgo's value-profiling
/// callbacks.
std::unique_ptr<llvm::orc::LLJIT>
jitModule(std::unique_ptr<llvm::Module> M,
          std::unique_ptr<llvm::LLVMContext> Ctx);

/// Adds M to J, which also gets orc-pgo's value-profiling callbacks. On ELF,
/// first makes M's external comdat any globals weak, which ORC can't JIT
/// consistently. (Every test JIT of instrumented code must come through here.)
std::unique_ptr<llvm::orc::LLJIT>
jitModule(std::unique_ptr<llvm::orc::LLJIT> J, std::unique_ptr<llvm::Module> M,
          std::unique_ptr<llvm::LLVMContext> Ctx);

template <typename T>
T *lookupSym(llvm::orc::LLJIT &J, llvm::StringRef Name) {
  return llvm::cantFail(J.lookup(Name)).toPtr<T *>();
}

/// While alive, the program has a required positional cl::opt, as many LLVM
/// tools do, so cl::ParseCommandLineOptions and with it configurePgoOptions
/// fail. Starts and ends with cl::ResetAllOptionOccurrences.
class RequiredPositionalOption {
public:
  /// configurePgoOptions's error meanwhile.
  static constexpr const char *ConfigureError =
      "cannot set orc-pgo's LLVM options: orc-pgo: Not enough positional "
      "command line arguments specified!\nMust specify at least 1 positional "
      "argument: See: orc-pgo --help";

  RequiredPositionalOption() { llvm::cl::ResetAllOptionOccurrences(); }
  ~RequiredPositionalOption() {
    Opt.removeArgument();
    llvm::cl::ResetAllOptionOccurrences();
  }

private:
  llvm::cl::opt<std::string> Opt{llvm::cl::Positional, llvm::cl::Required};
};

/// The default O2 pipeline, without PGO options.
void runO2(llvm::Module &M);

} // namespace orc_pgo::test
