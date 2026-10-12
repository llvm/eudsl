// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "stats.h"

#include <stdexcept>

static long weight(int Kind) {
  switch (Kind) {
  case 0:
    return 1;
  case 1:
    return 2;
  default:
    return 3;
  }
}

// noinline keeps the hot functions real calls through their dispatch slots;
// otherwise O2 would inline them into their only callers before thunking.
__attribute__((noinline)) long scaled(const Shape &S) {
  long A = S.area();
  if (A < 0)
    throw std::runtime_error("negative area");
  return A * weight(S.kind());
}

long accumulate(const std::vector<std::unique_ptr<Shape>> &V, int Iters) {
  long Sum = 0;
  for (int It = 0; It < Iters; ++It)
    for (const std::unique_ptr<Shape> &S : V)
      Sum += scaled(*S);
  return Sum;
}

__attribute__((noinline)) long mulDiv(long A, long B, long C) {
  return static_cast<long>(static_cast<__int128>(A) * B / C);
}
