// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An archive member nothing references. A static linker doesn't load it, so
// its initializer must not run (expected.txt has no such line).

#include <cstdio>

static int Linked = (std::puts("dead member linked"), 0);

int deadSymbol() { return Linked; }
