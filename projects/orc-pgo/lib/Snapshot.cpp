// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "orc-pgo/Snapshot.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace orc_pgo {

std::string writeSnapshot(const Module &M) {
  std::string Bytes;
  raw_string_ostream OS(Bytes);
  WriteBitcodeToFile(M, OS, /*ShouldPreserveUseListOrder=*/true);
  return Bytes;
}

Expected<std::unique_ptr<Module>> parseSnapshot(StringRef Bytes,
                                                LLVMContext &Ctx) {
  return parseBitcodeFile(MemoryBufferRef(Bytes, "orc-pgo-snapshot"), Ctx);
}

} // namespace orc_pgo
