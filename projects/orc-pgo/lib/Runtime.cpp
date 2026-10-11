// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Runtime.h"
#include "orc-pgo/PgoOptions.h"
#include "orc-pgo/ProfileBuilder.h"
#include "orc-pgo/Snapshot.h"
#include "orc-pgo/TierUp.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/EHFrameRegistrationPlugin.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/UnwindInfoRegistrationPlugin.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

using namespace llvm;

namespace orc_pgo {

static std::optional<uint64_t> envNumber(const char *Name) {
  const char *Value = std::getenv(Name);
  uint64_t N;
  if (!Value || StringRef(Value).getAsInteger(10, N))
    return std::nullopt;
  return N;
}

RuntimeOptions RuntimeOptions::fromEnv() {
  RuntimeOptions O;
  O.Threshold = envNumber("ORC_PGO_THRESHOLD").value_or(O.Threshold);
  O.PollMs =
      static_cast<unsigned>(envNumber("ORC_PGO_POLL_MS").value_or(O.PollMs));
  O.InlineDepth = static_cast<unsigned>(
      envNumber("ORC_PGO_INLINE_DEPTH").value_or(O.InlineDepth));
  O.Verbose = envNumber("ORC_PGO_VERBOSE").value_or(0) != 0;
  O.Disable = envNumber("ORC_PGO_DISABLE").value_or(0) != 0;
  return O;
}

unsigned pollPeriodMs(unsigned PollMs, std::chrono::milliseconds Elapsed) {
  if (Elapsed.count() < FastStartWindowMs)
    return std::min(FastStartPollMs, PollMs);
  return PollMs;
}

/// What runPgoGen produces once configurePgoOptions has run.
static constexpr uint64_t ExpectedRawProfileVersion =
    INSTR_PROF_RAW_VERSION | VARIANT_MASK_IR_PROF | VARIANT_MASK_INSTR_ENTRY;

Error validateImage(const OrcPgoImage &Image) {
  if (Image.Version != ImageVersion) {
    return createStringError("image version " + Twine(Image.Version) +
                             ", runtime expects " + Twine(ImageVersion));
  }
  if (StringRef(Image.Fingerprint.LLVMVersion) != LLVM_VERSION_STRING) {
    return createStringError("image built with LLVM " +
                             Twine(Image.Fingerprint.LLVMVersion) +
                             ", runtime is LLVM " LLVM_VERSION_STRING);
  }
  if (Image.Fingerprint.RawProfileVersion != ExpectedRawProfileVersion) {
    return createStringError(
        "image raw profile version 0x" +
        Twine::utohexstr(Image.Fingerprint.RawProfileVersion) +
        ", runtime expects 0x" + Twine::utohexstr(ExpectedRawProfileVersion));
  }
  if (Image.Fingerprint.OptionsHash != pgoOptionsFingerprint())
    return createStringError("image PGO options differ from the runtime's");
  return Error::success();
}

namespace {
/// The diagnostics of one batch's module, whatever pass emits them: warnings
/// are reported only when verbose, and an error fails the batch, where the
/// default handler would print it and go on.
class BatchDiagnostics : public DiagnosticHandler {
public:
  struct State {
    std::mutex Mu;
    std::string FirstError;
  };

  BatchDiagnostics(std::shared_ptr<State> S, bool Verbose)
      : S(std::move(S)), Verbose(Verbose) {}

  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    std::string Text;
    raw_string_ostream OS(Text);
    DiagnosticPrinterRawOStream DP(OS);
    DI.print(DP);
    if (DI.getSeverity() == DS_Error) {
      std::lock_guard<std::mutex> Lock(S->Mu);
      if (S->FirstError.empty())
        S->FirstError = Text;
    } else if (Verbose && DI.getSeverity() == DS_Warning) {
      errs() << "orc-pgo: warning: " << Text << "\n";
    }
    return true;
  }

private:
  std::shared_ptr<State> S;
  bool Verbose;
};
} // namespace

/// LLJIT's generic IR platform defines __dso_handle, atexit and __cxa_atexit
/// for the code it runs, which would be the JIT's own. Tier-up code is part of
/// the program, so it must register its exit handlers with the program's: the
/// ones in the symbol table. This sets up what the generic platform does for
/// unwinding (compact unwind info on MachO, eh-frames elsewhere, as
/// setUpGenericLLVMIRPlatform) and no platform otherwise.
static Expected<orc::JITDylibSP> setUpProgramPlatform(orc::LLJIT &J) {
  auto *OLL = dyn_cast<orc::ObjectLinkingLayer>(&J.getObjLinkingLayer());
  // Unwind registration is a JITLink plugin, and every supported host has
  // JITLink.
  if (!OLL)
    return orc::setUpInactivePlatform(J); // LCOV_EXCL_LINE
  // LCOV_EXCL_START a host has either object format
  orc::ExecutionSession &ES = J.getExecutionSession();
  if (J.getTargetTriple().isOSBinFormatMachO()) {
    auto Plugin = orc::UnwindInfoRegistrationPlugin::Create(ES);
    if (!Plugin)
      return Plugin.takeError();
    OLL->addPlugin(std::move(*Plugin));
  } else {
    auto Plugin = orc::EHFrameRegistrationPlugin::Create(ES);
    if (!Plugin)
      return Plugin.takeError();
    OLL->addPlugin(std::move(*Plugin));
  }
  // LCOV_EXCL_STOP
  return orc::setUpInactivePlatform(J);
}

Runtime::Runtime(const OrcPgoImage &Image, RuntimeOptions Opts)
    : Image(Image), Opts(Opts), States(Image.NumDispatch, SlotState::Aot),
      Excluded(Image.NumFunctions, false) {}

Runtime::~Runtime() { stopPolling(); }

Expected<std::unique_ptr<Runtime>> Runtime::create(const OrcPgoImage &Image,
                                                   RuntimeOptions Opts) {
  if (Error E = validateImage(Image))
    return std::move(E);
  Expected<orc::JITTargetMachineBuilder> JTMB =
      orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB)
    return JTMB.takeError(); // LCOV_EXCL_LINE the host is a registered target
  JTMB->setCodeGenOptLevel(CodeGenOptLevel::Aggressive);
  Expected<std::unique_ptr<TargetMachine>> TM = JTMB->createTargetMachine();
  if (!TM)
    return TM.takeError(); // LCOV_EXCL_LINE as above
  Expected<std::unique_ptr<orc::LLJIT>> J =
      orc::LLJITBuilder()
          .setJITTargetMachineBuilder(*JTMB)
          .setPlatformSetUp(setUpProgramPlatform)
          .create();
  if (!J)
    return J.takeError(); // LCOV_EXCL_LINE as above

  std::unique_ptr<Runtime> R(new Runtime(Image, Opts));
  R->TM = std::move(*TM);
  R->J = std::move(*J);
  R->J->getExecutionSession().setErrorReporter(
      [Raw = R.get()](Error E) { Raw->report("JIT error", std::move(E)); });
  orc::SymbolMap Symbols;
  for (const OrcPgoSymbol &S : ArrayRef(Image.Symbols, Image.NumSymbols))
    if (S.Address)
      Symbols[R->J->mangleAndIntern(S.Name)] = {
          orc::ExecutorAddr::fromPtr(S.Address), JITSymbolFlags::Exported};
  // Nothing is predefined without a platform, and a symbol map has unique
  // names, so this cannot fail.
  if (Error E = R->J->getMainJITDylib().define(
          orc::absoluteSymbols(std::move(Symbols))))
    return std::move(E); // LCOV_EXCL_LINE
  return std::move(R);
}

const OrcPgoFunctionRecord &Runtime::record(unsigned Slot) const {
  return Image.Functions[Image.DispatchFunctions[Slot]];
}

void Runtime::report(const Twine &What, Error E) {
  if (Opts.Verbose)
    errs() << "orc-pgo: " << What << ": " << toString(std::move(E)) << "\n";
  else
    consumeError(std::move(E));
}

void Runtime::fail(unsigned Slot) {
  States[Slot] = SlotState::Failed;
  Excluded[Image.DispatchFunctions[Slot]] = true;
  ++Stats.Failed;
}

unsigned Runtime::pollOnce() {
  std::lock_guard<std::recursive_mutex> Lock(Mu);
  if (Disabled)
    return 0;
  SmallVector<unsigned> Hot;
  for (unsigned Slot = 0, E = States.size(); Slot != E; ++Slot) {
    if (States[Slot] == SlotState::Aot &&
        __atomic_load_n(&record(Slot).Counters[0], __ATOMIC_RELAXED) >=
            Opts.Threshold) {
      Hot.push_back(Slot);
    }
  }
  if (Hot.empty())
    return 0;
  ++Stats.Batches;
  return tierUp(Hot);
}

unsigned Runtime::tierUp(ArrayRef<unsigned> Slots) {
  SmallVector<StringRef> Names;
  for (unsigned Slot : Slots)
    Names.push_back(record(Slot).Name);
  Expected<unsigned> Swapped = tryTierUp(Slots);
  if (Swapped)
    return *Swapped;
  report("tier-up of " + join(Names, ", ") + " failed", Swapped.takeError());

  if (Slots.size() == 1 || Disabled) {
    for (unsigned Slot : Slots)
      fail(Slot);
    return 0;
  }
  // The error does not say which root is to blame, so try them one at a time.
  // Until a root has had its turn it stays out of the profile, so that its
  // record cannot fail the others'; a root that fails stays out for good.
  for (unsigned Slot : Slots)
    Excluded[Image.DispatchFunctions[Slot]] = true;
  unsigned Total = 0;
  for (unsigned Slot : Slots) {
    Excluded[Image.DispatchFunctions[Slot]] = false;
    Total += tierUp(Slot);
  }
  return Total;
}

Expected<unsigned> Runtime::tryTierUp(ArrayRef<unsigned> Slots) {
  auto Begin = std::chrono::steady_clock::now();
  SmallVector<StringRef> Roots;
  for (unsigned Slot : Slots)
    Roots.push_back(record(Slot).Name);

  std::vector<OrcPgoFunctionRecord> Records;
  for (uint64_t I = 0; I != Image.NumFunctions; ++I)
    if (!Excluded[I])
      Records.push_back(Image.Functions[I]);
  Expected<std::unique_ptr<MemoryBuffer>> Profile =
      buildIndexedProfile(Records);
  if (!Profile)
    return Profile.takeError();

  auto Ctx = std::make_unique<LLVMContext>();
  auto Diagnostics = std::make_shared<BatchDiagnostics::State>();
  Ctx->setDiagnosticHandler(
      std::make_unique<BatchDiagnostics>(Diagnostics, Opts.Verbose));
  Expected<std::unique_ptr<Module>> M =
      parseSnapshot(StringRef(Image.Snapshot, Image.SnapshotSize), *Ctx);
  if (!M) {
    Disabled = true;
    return M.takeError();
  }
  Expected<TierUpBatch> Batch =
      prepareTierUpModule(**M, **Profile, Roots, Opts.InlineDepth);
  if (!Batch)
    return Batch.takeError();
  optimizeTierUp(**M, *TM);
  // After optimization, so that a callee that is tiered up can still be
  // inlined.
  bindTieredCalls(**M, Tiered);
  // Compilation and linking happen at lookup and fail the whole module, so
  // look up every root before any slot changes, and on failure take the module
  // out again: the retry of its roots one at a time must not meet its
  // definitions.
  orc::JITDylib &JD = J->getMainJITDylib();
  orc::ResourceTrackerSP RT = JD.createResourceTracker();
  if (Error E = J->addIRModule(
          RT, orc::ThreadSafeModule(std::move(*M), std::move(Ctx))))
    return std::move(E);
  orc::SymbolLookupSet Symbols;
  for (const std::string &Name : Batch->Names)
    Symbols.add(J->mangleAndIntern(Name));
  Expected<orc::SymbolMap> Defs = J->getExecutionSession().lookup(
      {{&JD, orc::JITDylibLookupFlags::MatchAllSymbols}}, std::move(Symbols));
  std::string Text;
  {
    std::lock_guard<std::mutex> Lock(Diagnostics->Mu);
    Text = Diagnostics->FirstError;
  }
  if (!Defs || !Text.empty()) {
    // LCOV_EXCL_START removing a tracker does not fail here
    if (Error E = RT->remove())
      report("removing the failed module", std::move(E));
    // LCOV_EXCL_STOP
    if (!Defs)
      return Defs.takeError();
    return createStringError(Text);
  }
  Stats.PromotedIndirectCalls += Batch->PromotedIndirectCalls;

  unsigned Swapped = 0;
  for (auto [Slot, Name] : zip(Slots, Batch->Names)) {
    // Pairs with the thunks' acquire load.
    __atomic_store_n(
        &Image.Dispatch[Slot],
        (*Defs)[J->mangleAndIntern(Name)].getAddress().toPtr<void *>(),
        __ATOMIC_RELEASE);
    States[Slot] = SlotState::TieredUp;
    Tiered.insert(Name);
    ++Stats.TieredUp;
    ++Swapped;
  }
  if (Opts.Verbose) {
    std::chrono::duration<double, std::milli> Took =
        std::chrono::steady_clock::now() - Begin;
    errs() << "orc-pgo: batch of " << Slots.size() << " roots took "
           << format("%.2f", Took.count()) << " ms, " << Swapped
           << " tiered up: " << join(Roots, ", ") << "\n";
  }
  return Swapped;
}

void Runtime::startPolling() {
  if (Opts.PollMs == 0 || Poller.joinable())
    return;
  StopRequested = false;
  Poller = std::thread([this] {
    auto Started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> Lock(PollMu);
    while (!PollCv.wait_for(
        Lock,
        std::chrono::milliseconds(pollPeriodMs(
            Opts.PollMs, std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - Started))),
        [this] { return StopRequested; })) {
      Lock.unlock();
      pollOnce();
      Lock.lock();
    }
  });
}

void Runtime::stopPolling() {
  {
    std::lock_guard<std::mutex> Lock(PollMu);
    StopRequested = true;
  }
  PollCv.notify_all();
  if (!Poller.joinable())
    return;
  // exit() on the poll thread (a fatal error whose handler exits) runs the
  // atexit handler there; the thread cannot wait for itself.
  if (Poller.get_id() == std::this_thread::get_id())
    Poller.detach();
  else
    Poller.join();
}

SlotState Runtime::state(unsigned Slot) const {
  std::lock_guard<std::recursive_mutex> Lock(Mu);
  return States[Slot];
}

std::optional<unsigned> Runtime::slotOf(StringRef PgoName) const {
  for (unsigned Slot = 0, E = States.size(); Slot != E; ++Slot)
    if (PgoName == record(Slot).Name)
      return Slot;
  return std::nullopt;
}

bool Runtime::imageDisabled() const {
  std::lock_guard<std::recursive_mutex> Lock(Mu);
  return Disabled;
}

RuntimeStats Runtime::stats() const {
  std::lock_guard<std::recursive_mutex> Lock(Mu);
  return Stats;
}

void Runtime::printStats(raw_ostream &OS) const {
  RuntimeStats S = stats();
  OS << "orc-pgo: " << S.Batches << " batches, " << S.TieredUp << " tiered up, "
     << S.Failed << " failed, " << S.PromotedIndirectCalls
     << " indirect calls promoted\n";
}

} // namespace orc_pgo
