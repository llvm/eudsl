//===- RunnerUtils.cpp - Utils for MLIR exec on targets with a C++ runtime ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to debug structured MLIR types at
// runtime. Entities in this file may not be compatible with targets without a
// C++ runtime. These may be progressively migrated to CRunnerUtils.cpp over
// time.
//
//===----------------------------------------------------------------------===//

#include "32bCRunnerUtils.h"
#include <iostream>

// NOLINTBEGIN(*-identifier-naming)

template <typename T, typename StreamType>
void myPrintMemRefMetaData(StreamType &os, const DynamicMemRefType<T> &v) {
  // Make the printed pointer format platform independent by casting it to an
  // integer and manually formatting it to a hex with prefix as tests expect.
  os << "base@ = " << std::hex << std::showbase
     << reinterpret_cast<std::intptr_t>(v.data) << std::dec << std::noshowbase
     << " rank = " << v.rank << " offset = " << v.offset;
  auto print = [&](const int32_t *ptr) {
    if (v.rank == 0)
      return;
    os << ptr[0];
    for (int32_t i = 1; i < v.rank; ++i)
      os << ", " << ptr[i];
  };
  os << " sizes = [";
  print(v.sizes);
  os << "] strides = [";
  print(v.strides);
  os << "]";
}

extern "C" MLIR_CRUNNERUTILS_EXPORT [[maybe_unused]] void
_mlir_ciface_myPrintMemrefShapeF32(UnrankedMemRefType<float> *M) {
  std::cout << "Unranked Memref ";
  myPrintMemRefMetaData(std::cout, DynamicMemRefType<float>(*M));
  std::cout << "\n";
}
