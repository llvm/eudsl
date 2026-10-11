// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace orc_pgo::test {

/// Aborts the test binary with the diagnostic if Text is invalid IR.
std::unique_ptr<llvm::Module> parseIR(llvm::LLVMContext &Ctx,
                                      llvm::StringRef Text);

/// The message of a failed Error, or "" on success. Consumes the error.
std::string errorText(llvm::Error E);

template <typename T>
std::string errorText(llvm::Expected<T> &E) {
  return E ? std::string() : errorText(E.takeError());
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

} // namespace orc_pgo::test
