// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace llvm {
class GlobalValue;
class Triple;
class GlobalVariable;
class Module;
} // namespace llvm

namespace orc_pgo {

inline constexpr llvm::StringLiteral SymbolTableName = "__orc_pgo_symbols";

/// The globals of M (a module identical to the snapshot) that tier-up code may
/// reference by name, in module order: functions, then variables, then
/// aliases. That is every named definition, plus declarations and
/// available_externally definitions that have uses. Intrinsics, other llvm.*
/// globals, thread-locals and ifuncs are left out.
std::vector<llvm::GlobalValue *> collectSymbols(llvm::Module &M);

/// Symbols that code generated for T may call with no declaration in the IR,
/// and that the platform's C library, math library or unwinder defines:
/// - the unwinder's _Unwind_Resume, which lowering `resume` calls;
/// - memcpy, memmove and memset, which lowering mem* intrinsics calls, plus
///   bzero and memset_pattern{4,8,16} on Darwin, and __stack_chk_fail (and
///   __stack_chk_guard on Darwin) for stack protectors;
/// - the libm functions lowering of LLVM's math intrinsics and the
///   floating-point rounding operations calls when the target has no
///   instruction for them: the double, float (f) and long double (l) forms of
///   sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, pow, exp, exp2,
///   log, log2, log10, fmod, ceil, floor, trunc, rint, nearbyint, round, fma,
///   fmin, fmax, ldexp, frexp and modf.
/// Only C99 names present on every supported platform are listed, so that
/// listing them cannot fail the native link. Personality routines are not
/// here: IR that uses one declares it. The compiler-rt builtins are not here
/// either; the driver passes them as ExtraSymbols.
std::vector<std::string> platformSymbols(const llvm::Triple &T);

/// Emits SymbolTableName, an [N x OrcPgoSymbol] of Symbols. The table is added
/// to llvm.compiler.used, so every entry stays defined, and keeps its address
/// taken, through optimization.
llvm::GlobalVariable *
emitSymbolTable(llvm::Module &M, llvm::ArrayRef<llvm::GlobalValue *> Symbols);

/// Replaces Table with a table of its entries followed by the Names not among
/// them, and returns the new one, which takes Table's place in
/// llvm.compiler.used. A name that M declares or defines (optimization may
/// have declared it as a library call) gets M's global; any other is declared
/// as a `void()` function, since only its address matters (memset_pattern16
/// gets the library's prototype, which code generation requires). This is
/// meant to run after optimization: a stub declaration of the wrong type would
/// keep optimization from emitting calls to that library function.
llvm::GlobalVariable *extendSymbolTable(llvm::Module &M,
                                        llvm::GlobalVariable &Table,
                                        llvm::ArrayRef<std::string> Names);

} // namespace orc_pgo
