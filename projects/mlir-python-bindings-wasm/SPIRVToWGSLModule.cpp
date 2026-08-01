//===- SPIRVToWGSLModule.cpp - python bindings ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVToWGSLCAPI.h"

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

// std::string return values need the STL caster.
#include <nanobind/stl/string.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_mlirSPIRVToWGSL, m) {
  m.doc() = "Translate a SPIR-V binary to WGSL source using Tint.";

  m.def(
      "spirv_to_wgsl",
      [](nb::bytes spv) -> std::string {
        if (spv.size() % 4 != 0)
          throw std::invalid_argument(
              "SPIR-V binary length must be a multiple of 4 bytes");

        std::vector<uint32_t> words(spv.size() / 4);
        std::memcpy(words.data(), spv.c_str(), spv.size());

        char *out = nullptr;
        bool ok = mlirSPIRVToWGSL(words.data(), words.size(), &out);
        std::string result(out ? out : "");
        mlirSPIRVToWGSLFree(out);
        if (!ok)
          throw std::runtime_error(result);
        return result;
      },
      "spirv"_a,
      "Translate a SPIR-V binary to WGSL. Raises RuntimeError with Tint's "
      "diagnostics (including the SPIR-V header summary and, where available, "
      "the Tint IR) if translation fails.");
}
