//===- SPIRVToWGSL.cpp - SPIR-V to WGSL via Tint ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ported from IREE's compiler/plugins/target/WebGPUSPIRV/SPIRVToWGSL.cpp.
//
// The three stages (read SPIR-V into Tint IR, lower IR to a WGSL program, print
// it) each fail in their own way, so the diagnostics are kept: a failed
// translation is otherwise very hard to debug from Python.
//
//===----------------------------------------------------------------------===//

#include "SPIRVToWGSLCAPI.h"

#include "src/tint/lang/core/ir/disassembler.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/program/program.h"
#include "src/tint/lang/wgsl/writer/writer.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

void printSPIRVModuleSummary(const std::vector<uint32_t> &spv,
                             std::ostringstream &os) {
  if (spv.size() < 5) {
    os << "  SPIR-V module is too short to have a header (" << spv.size()
       << " words)\n";
    return;
  }
  uint32_t version = spv[1];
  os << "  magic: 0x" << std::hex << spv[0] << std::dec << "\n"
     << "  version: " << ((version >> 16) & 0xFF) << "."
     << ((version >> 8) & 0xFF) << "\n"
     << "  generator: 0x" << std::hex << spv[2] << std::dec << "\n"
     << "  bound: " << spv[3] << "\n"
     << "  words: " << spv.size() << "\n";
}

char *dupString(const std::string &s) {
  char *out = static_cast<char *>(std::malloc(s.size() + 1));
  if (!out)
    return nullptr;
  std::memcpy(out, s.data(), s.size());
  out[s.size()] = '\0';
  return out;
}

} // namespace

extern "C" {

MlirSPIRVToWGSLStatus mlirSPIRVToWGSL(const uint32_t *words, size_t wordCount,
                                      char **wgsl) {
  if (!wgsl)
    return MlirSPIRVToWGSLInvalidArgument;
  *wgsl = nullptr;
  if (!words && wordCount)
    return MlirSPIRVToWGSLInvalidArgument;

  // Report an allocation failure as itself rather than letting it look like a
  // translation failure with an empty diagnostic.
  auto emit = [&](const std::string &s) {
    *wgsl = dupString(s);
    return *wgsl ? MlirSPIRVToWGSLTranslationFailed
                 : MlirSPIRVToWGSLOutOfMemory;
  };

  std::vector<uint32_t> spv(words, words + wordCount);

  auto irResult = tint::spirv::reader::ReadIR(spv);
  if (irResult != tint::Success) {
    std::ostringstream os;
    os << "failed to parse SPIR-V into Tint IR: " << irResult.Failure().reason
       << "\n";
    printSPIRVModuleSummary(spv, os);
    return emit(os.str());
  }
  tint::core::ir::Module irModule = irResult.Move();

  tint::wgsl::writer::Options writerOptions;
  auto programResult =
      tint::wgsl::writer::ProgramFromIR(irModule, writerOptions);
  if (programResult != tint::Success) {
    std::ostringstream os;
    os << "failed to lower Tint IR to a WGSL program: "
       << programResult.Failure().reason << "\n";
    printSPIRVModuleSummary(spv, os);
    os << "Tint IR at failure:\n"
       << tint::core::ir::Disassembler(irModule).Plain() << "\n";
    return emit(os.str());
  }

  auto wgslResult =
      tint::wgsl::writer::Generate(programResult.Get(), writerOptions);
  if (wgslResult != tint::Success) {
    std::ostringstream os;
    os << "failed to print WGSL: " << wgslResult.Failure().reason << "\n";
    printSPIRVModuleSummary(spv, os);
    os << "Tint IR at failure:\n"
       << tint::core::ir::Disassembler(irModule).Plain() << "\n";
    return emit(os.str());
  }

  *wgsl = dupString(wgslResult->wgsl);
  return *wgsl ? MlirSPIRVToWGSLSuccess : MlirSPIRVToWGSLOutOfMemory;
}

void mlirSPIRVToWGSLFree(char *wgsl) { std::free(wgsl); }

} // extern "C"
