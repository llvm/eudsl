// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shapes.h"

namespace {

// Internal state, vtables and type info: JIT'd code reaches them through the
// image's symbol table.
int Created = 0;

struct Circle final : Shape {
  long R;
  explicit Circle(long R) : R(R) { ++Created; }
  long area() const override { return 3 * R * R; }
  int kind() const override { return 2; }
};

struct Square final : Shape {
  long S;
  explicit Square(long S) : S(S) { ++Created; }
  long area() const override { return S * S; }
  int kind() const override { return 0; }
};

struct Triangle final : Shape {
  long B, H;
  Triangle(long B, long H) : B(B), H(H) { ++Created; }
  long area() const override { return B * H / 2; }
  int kind() const override { return 1; }
};

} // namespace

Shape::~Shape() = default;

std::vector<std::unique_ptr<Shape>> makeShapes(int N) {
  std::vector<std::unique_ptr<Shape>> V;
  for (int I = 0; I < N; ++I) {
    switch (I % 10) {
    case 0:
      V.push_back(std::make_unique<Square>(I % 5 + 1));
      break;
    case 1:
      V.push_back(std::make_unique<Triangle>(I % 3 + 1, 2));
      break;
    default: // 80% circles: the indirect calls' dominant target
      V.push_back(std::make_unique<Circle>(I % 4 + 1));
      break;
    }
  }
  return V;
}

int createdCount() { return Created; }
