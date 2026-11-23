// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "shapes.h"

/// area() * weight(kind()); throws std::runtime_error for a negative area.
long scaled(const Shape &S);
long accumulate(const std::vector<std::unique_ptr<Shape>> &V, int Iters);
/// A * B / C in 128-bit arithmetic (a __divti3 libcall).
long mulDiv(long A, long B, long C);
