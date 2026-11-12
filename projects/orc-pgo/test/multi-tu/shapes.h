// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <memory>
#include <vector>

struct Shape {
  virtual ~Shape();
  virtual long area() const = 0;
  virtual int kind() const = 0;
};

std::vector<std::unique_ptr<Shape>> makeShapes(int N);
int createdCount();
