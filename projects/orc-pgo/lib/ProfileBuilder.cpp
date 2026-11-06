// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/ProfileBuilder.h"
#include "orc-pgo/ValueProfRuntime.h"

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;

namespace orc_pgo {

Expected<std::unique_ptr<MemoryBuffer>>
buildIndexedProfile(ArrayRef<OrcPgoFunctionRecord> Functions) {
  InstrProfWriter Writer;
  cantFail(
      Writer.mergeProfileKind(InstrProfKind::IRInstrumentation |
                              InstrProfKind::FunctionEntryInstrumentation));

  InstrProfSymtab Symtab;
  for (const OrcPgoFunctionRecord &R : Functions)
    if (R.Address)
      Symtab.mapAddress(reinterpret_cast<uintptr_t>(R.Address),
                        IndexedInstrProf::ComputeHash(R.Name));

  std::string Problem;
  for (const OrcPgoFunctionRecord &R : Functions) {
    // The program keeps counting while this runs; relaxed loads are what
    // compiler-rt's profile writer does too.
    std::vector<uint64_t> Counts(R.NumCounters);
    for (uint32_t I = 0; I < R.NumCounters; ++I)
      Counts[I] = __atomic_load_n(&R.Counters[I], __ATOMIC_RELAXED);
    NamedInstrProfRecord Record(R.Name, R.CfgHash, std::move(Counts));
    for (uint32_t Kind = 0; Kind < NumValueKinds; ++Kind) {
      Record.reserveSites(Kind, R.NumValueSites[Kind]);
      for (uint32_t Site = 0; Site < R.NumValueSites[Kind]; ++Site) {
        Record.addValueData(Kind, Site, readValueSite(R, Kind, Site),
                            Kind == IPVK_IndirectCallTarget ? &Symtab
                                                            : nullptr);
      }
    }
    Writer.addRecord(std::move(Record), [&](Error E) {
      std::string Message = toString(std::move(E));
      if (Problem.empty())
        Problem = R.Name + std::string(": ") + Message;
    });
  }
  if (!Problem.empty())
    return createStringError(Problem);

  // InstrProfWriter::writeBuffer drops the write error and returns null.
  std::string Data;
  raw_string_ostream OS(Data);
  // LCOV_EXCL_START writing to a string stream does not fail
  if (Error E = Writer.write(OS))
    return std::move(E);
  // LCOV_EXCL_STOP
  return MemoryBuffer::getMemBufferCopy(Data);
}

} // namespace orc_pgo
