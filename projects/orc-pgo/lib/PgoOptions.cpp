// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/PgoOptions.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"

using namespace llvm;

namespace orc_pgo {

namespace {
struct RequiredOption {
  StringLiteral Name;
  bool Value;
};
} // namespace

static constexpr RequiredOption Required[] = {
    {"pgo-instrument-entry", true},
    {"pgo-instr-select", true},
    {"disable-vp", false},
    {"pgo-instr-memop", true},
};

// LLVM's instrumentation options are library options (InstrumentationOptions):
// only cl::ParseCommandLineOptions sets them, and nothing reads them back. This
// option is parsed along with them, so it records that they are set until
// cl::ResetAllOptionOccurrences restores both to their defaults.
static cl::opt<bool> Configured(StringRef(ReservedOptionName), cl::ReallyHidden,
                                cl::desc("orc-pgo has set its LLVM options"));

// The options are registered by a static initializer in the instrumentation
// library, next to the pass that reads them. Referencing the pass from this
// never-called function keeps both linked into static binaries.
// LCOV_EXCL_START never called
LLVM_ATTRIBUTE_USED static void forceLinkPgoPasses(Module &M,
                                                   ModuleAnalysisManager &MAM) {
  PGOInstrumentationGen().run(M, MAM);
}
// LCOV_EXCL_STOP

static std::string requiredArg(const RequiredOption &R) {
  return ("-" + R.Name + "=" + (R.Value ? "true" : "false")).str();
}

Error configurePgoOptions() {
  // Parsing again would write options that other threads' passes may be
  // reading.
  if (Configured)
    return Error::success();
  SmallVector<std::string, 4> Args;
  for (const RequiredOption &R : Required)
    Args.push_back(requiredArg(R));
  std::string Sentinel = ("-" + ReservedOptionName).str();
  SmallVector<const char *, 8> Argv{"orc-pgo"};
  for (const std::string &A : Args)
    Argv.push_back(A.c_str());
  Argv.push_back(Sentinel.c_str());
  std::string Message;
  raw_string_ostream OS(Message);
  if (!cl::ParseCommandLineOptions(Argv.size(), Argv.data(), "", &OS)) {
    // The arguments are parsed in order, so this may have been set.
    Configured.reset();
    return createStringError("cannot set orc-pgo's LLVM options: %s",
                             StringRef(Message).rtrim().str().c_str());
  }
  return Error::success();
}

Error checkPgoOptionConflicts(ArrayRef<std::string> LLVMArgs) {
  // The last occurrence of an option wins, as in cl::ParseCommandLineOptions.
  StringMap<StringRef> Given;
  for (StringRef Arg : LLVMArgs) {
    if (!Arg.consume_front("-"))
      continue;
    Arg.consume_front("-");
    auto [Name, Value] = Arg.split('=');
    if (Name == ReservedOptionName)
      return createStringError("LLVM option -%s is reserved for orc-pgo",
                               ReservedOptionName.data());
    // A bare bool option means true.
    Given[Name] = Arg.contains('=') ? Value : "true";
  }
  for (const RequiredOption &R : Required) {
    auto It = Given.find(R.Name);
    if (It == Given.end())
      continue;
    // The values parsed, so each is one of true, false, 1 and 0.
    bool Value = It->second == "true" || It->second == "1";
    if (Value != R.Value)
      return createStringError(
          "LLVM option -%s was set to %s, but orc-pgo requires %s",
          R.Name.data(), Value ? "true" : "false", R.Value ? "true" : "false");
  }
  return Error::success();
}

uint64_t pgoOptionsFingerprint() {
  std::string Text;
  for (const RequiredOption &R : Required)
    Text += requiredArg(R) + ";";
  return xxh3_64bits(Text);
}

} // namespace orc_pgo
