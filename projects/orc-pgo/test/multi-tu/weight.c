// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The same weights as stats.cpp's, in C: a static library mixing C++ and C
// sources must still have bitcode members only. (`new` is an identifier here,
// so this does not compile as C++.)

int cWeight(int Kind) {
  int new = 3;
  if (Kind == 0)
    new = 1;
  else if (Kind == 1)
    new = 2;
  return new;
}
