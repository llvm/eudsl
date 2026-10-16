// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "orc-pgo/Image.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace llvm {
class TargetMachine;
class raw_ostream;
namespace orc {
class LLJIT;
} // namespace orc
} // namespace llvm

namespace orc_pgo {

/// Images __orc_pgo_register_image records; later ones are ignored.
inline constexpr unsigned MaxImages = 16;

struct RuntimeOptions {
  /// Entry count at which a function is tiered up (ORC_PGO_THRESHOLD).
  uint64_t Threshold = 10000;
  /// Poll period in milliseconds; 0 means no poll thread (ORC_PGO_POLL_MS).
  unsigned PollMs = 10;
  /// How many calls away callees are copied into a batch
  /// (ORC_PGO_INLINE_DEPTH).
  unsigned InlineDepth = 2;
  /// Report failures, batches and statistics at exit on stderr
  /// (ORC_PGO_VERBOSE=1).
  bool Verbose = false;
  /// ORC_PGO_DISABLE=1 turns the runtime off.
  bool Disable = false;

  /// The defaults, overridden by the environment. Malformed values are ignored.
  static RuntimeOptions fromEnv();
};

/// Right after the runtime starts, the poll thread polls every FastStartPollMs
/// for FastStartWindowMs, so that the first batch does not wait for a whole
/// poll period.
inline constexpr unsigned FastStartPollMs = 1;
inline constexpr unsigned FastStartWindowMs = 100;

/// How long the poll thread waits between polls, Elapsed after it started:
/// FastStartPollMs during the first FastStartWindowMs (or PollMs if that is
/// shorter), PollMs after.
unsigned pollPeriodMs(unsigned PollMs, std::chrono::milliseconds Elapsed);

struct RuntimeStats {
  uint64_t TieredUp = 0;
  uint64_t Failed = 0;
  /// Indirect call sites PGOIndirectCallPromotion promoted in tier-up
  /// definitions: promotions, not how often the promoted path was taken.
  uint64_t PromotedIndirectCalls = 0;
  uint64_t Batches = 0;
};

enum class SlotState : uint8_t { Aot, TieredUp, Failed };

/// Checks that Image was built for this runtime: the image version, the LLVM
/// version, the raw profile version and the PGO option fingerprint.
llvm::Error validateImage(const OrcPgoImage &Image);

/// The sum of the stats of every runtime __orc_pgo_start created; all zero
/// when none is running.
RuntimeStats processRuntimeStats();

class Runtime {
public:
  /// validateImage, then an LLJIT in which Image's symbol table is defined as
  /// absolute symbols. Nothing else is done until the first batch.
  static llvm::Expected<std::unique_ptr<Runtime>>
  create(const OrcPgoImage &Image, RuntimeOptions Opts);
  ~Runtime();

  /// Tiers up, as one batch, every slot still on its AOT body whose entry
  /// count has reached the threshold. Returns the number of slots swapped.
  unsigned pollOnce();
  /// Calls pollOnce every Opts.PollMs (see pollPeriodMs) on a background
  /// thread. Does nothing if PollMs is 0 or the thread is already running.
  void startPolling();
  /// Stops the poll thread, waiting for a batch in flight (unless called on
  /// the poll thread, which only detaches). Idempotent.
  void stopPolling();

  SlotState state(unsigned Slot) const;
  /// The dispatch slot of the function with this IR PGO name.
  std::optional<unsigned> slotOf(llvm::StringRef PgoName) const;
  /// Whether the snapshot failed to parse, which stops all tier-up.
  bool imageDisabled() const;
  RuntimeStats stats() const;
  void printStats(llvm::raw_ostream &OS) const;

private:
  Runtime(const OrcPgoImage &Image, RuntimeOptions Opts);
  const OrcPgoFunctionRecord &record(unsigned Slot) const;
  /// Tiers up Slots as one batch. If that fails, a single slot is marked
  /// failed; with more, each is tried alone, so only the culprit fails.
  unsigned tierUp(llvm::ArrayRef<unsigned> Slots);
  /// One batch: the number of slots swapped, or the error that failed it (in
  /// annotating, compiling or linking), in which case no slot has changed
  /// state and the batch's module is gone from the JIT.
  llvm::Expected<unsigned> tryTierUp(llvm::ArrayRef<unsigned> Slots);
  void fail(unsigned Slot);
  void report(const llvm::Twine &What, llvm::Error E);

  OrcPgoImage Image;
  RuntimeOptions Opts;
  std::unique_ptr<llvm::TargetMachine> TM;
  std::unique_ptr<llvm::orc::LLJIT> J;
  std::vector<SlotState> States;
  /// Records left out of later profiles: those of failed slots.
  std::vector<bool> Excluded;
  /// The tier-up symbols of the slots that were tiered up (see
  /// bindTieredCalls).
  llvm::StringSet<> Tiered;
  RuntimeStats Stats;
  bool Disabled = false;
  /// Held for a whole batch. Recursive: exiting from the poll thread, in the
  /// middle of a batch, still reads the stats.
  mutable std::recursive_mutex Mu;

  std::mutex PollMu;
  std::condition_variable PollCv;
  bool StopRequested = false;
  std::thread Poller;
};

} // namespace orc_pgo
