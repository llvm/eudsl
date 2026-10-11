// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The C entry points linked into every instrumented executable, and the
// process-wide runtime state behind them.

#include "orc-pgo/Image.h"
#include "orc-pgo/PgoOptions.h"
#include "orc-pgo/Runtime.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <pthread.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

using namespace llvm;
using namespace orc_pgo;

namespace {

// __orc_pgo_register_image runs from static constructors, possibly before
// LLVM's, so it only touches these constant-initialized globals.
const OrcPgoImage *Pending[MaxImages];
unsigned NumPending = 0;
unsigned NumDropped = 0;

std::mutex StartMutex;
bool Started = false;
bool Verbose = false;
// Set in a forked child, which has none of the parent's threads: the runtimes
// are left as they are, and never waited for.
bool Abandoned = false;
// Leaked: JIT'd code may still run on other threads after main returns.
std::vector<Runtime *> *Runtimes = nullptr;

void report(const Twine &Message) {
  if (Verbose)
    errs() << "orc-pgo: " << Message << "\n";
}

void report(Error E) {
  if (Verbose)
    errs() << "orc-pgo: " << toString(std::move(E)) << "\n";
  else
    consumeError(std::move(E));
}

/// The runtimes __orc_pgo_start created, once it has finished: they are never
/// removed, so callers can use them without the lock.
std::vector<Runtime *> runtimes() {
  std::lock_guard<std::mutex> Lock(StartMutex);
  if (Abandoned || !Runtimes)
    return {};
  return *Runtimes;
}

void stopRuntimes() {
  // Without StartMutex: waiting for a batch in flight must not block the
  // threads that query or start the runtime.
  for (Runtime *R : runtimes()) {
    R->stopPolling();
    if (Verbose)
      R->printStats(errs());
  }
}

// fork() while the poll thread runs leaves the child with a Runtime that
// thinks it has a thread to join, and possibly mutexes locked by it. StartMutex
// is held across the fork so it is consistent in both processes; the child
// then never touches the runtimes again, and its exit does not wait for
// anything. Functions that were tiered up before the fork stay so.
void beforeFork() { StartMutex.lock(); }
void afterForkInParent() { StartMutex.unlock(); }
void afterForkInChild() {
  Abandoned = true;
  StartMutex.unlock();
}

/// Called with StartMutex held.
void start() {
  if (Started)
    return;
  Started = true;
  RuntimeOptions Opts = RuntimeOptions::fromEnv();
  Verbose = Opts.Verbose;
  if (Opts.Disable)
    return report("disabled by ORC_PGO_DISABLE");
  if (NumDropped)
    report(Twine(NumDropped) + " images beyond the first " + Twine(MaxImages) +
           " ignored");
  if (Error E = configurePgoOptions())
    return report(std::move(E));
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  Runtimes = new std::vector<Runtime *>();
  // Before any poll thread exists. Registered after LLVM's static
  // constructors ran, so it runs before LLVM's static destructors.
  std::atexit(stopRuntimes);
  pthread_atfork(beforeFork, afterForkInParent, afterForkInChild);
  for (const OrcPgoImage *Image : ArrayRef(Pending, NumPending)) {
    Expected<std::unique_ptr<Runtime>> R = Runtime::create(*Image, Opts);
    if (!R) {
      report(R.takeError());
      continue;
    }
    // One value-node pool per process, as in compiler-rt.
    if (Runtimes->empty())
      setValueNodePool(Image->VNodePool[0], Image->VNodePool[1]);
    (*R)->startPolling();
    Runtimes->push_back(R->release());
  }
}

} // namespace

RuntimeStats orc_pgo::processRuntimeStats() {
  RuntimeStats Total;
  for (Runtime *R : runtimes()) {
    RuntimeStats S = R->stats();
    Total.TieredUp += S.TieredUp;
    Total.Failed += S.Failed;
    Total.PromotedIndirectCalls += S.PromotedIndirectCalls;
    Total.Batches += S.Batches;
  }
  return Total;
}

extern "C" void __orc_pgo_register_image(const OrcPgoImage *Image) {
  std::lock_guard<std::mutex> Lock(StartMutex);
  // An image that arrives later (a dlopen'd library) gets no runtime: the
  // runtimes are created once, from the images registered by then.
  if (Started)
    return report("image registered after the runtime started is ignored");
  if (NumPending == MaxImages) {
    ++NumDropped;
    return;
  }
  Pending[NumPending++] = Image;
}

extern "C" void __orc_pgo_start(void) {
  std::lock_guard<std::mutex> Lock(StartMutex);
  start();
}

extern "C" void __orc_pgo_poll(void) {
  __orc_pgo_start();
  for (Runtime *R : runtimes())
    R->pollOnce();
}
