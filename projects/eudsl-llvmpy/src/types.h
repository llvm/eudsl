// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2025.

#pragma once

#include "pp/Core.h"
#include "pp/Types.h"

struct LLVMModuleFlagEntry {
  LLVMModuleFlagBehavior Behavior;
  const char *Key;
  size_t KeyLen;
  LLVMMetadataRef Metadata;
};

struct LLVMValueMetadataEntry {
  unsigned Kind;
  LLVMMetadataRef Metadata;
};
