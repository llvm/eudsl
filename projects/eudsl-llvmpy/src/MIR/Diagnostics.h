// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EUDSL_MIR_DIAGNOSTICS_H
#define EUDSL_MIR_DIAGNOSTICS_H

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <exception>
#include <string>

namespace eudsl {

// A Python exception raised inside a scheduler pickNode callback is stashed here
// instead of thrown across llvm::legacy::PassManager::run, which libLLVMCodeGen
// compiles with -fno-exceptions: unwinding through those frames skips their
// destructors and std::terminates on targets built without asynchronous unwind
// tables. The pickNode trampoline stashes into this slot and returns a legal
// ready node so the (isRequired, unskippable) codegen passes wind down normally;
// runCodegenPipeline then re-raises after the run returns, so the throw only
// crosses our own -fexceptions frames. Thread-local so concurrent pipelines
// don't clobber it. Mirrors the IR-pass pendingPassError mechanism. Declared
// extern and defined once (PythonCodegen.cpp) rather than inline: an inline
// thread_local emits a duplicate TLS initialization routine in every including
// TU, which the linker rejects.
extern thread_local std::exception_ptr pendingCodegenError;

// Run a codegen PassManager, then re-raise any Python exception a callback
// (e.g. the "python" scheduler's pickNode) stashed during the run. The rethrow
// happens before any post-run diagnostic check the caller performs, so a
// re-raised Python error takes precedence over a captured LLVM diagnostic.
inline void runCodegenPipeline(llvm::legacy::PassManager &pm, llvm::Module &m) {
  pendingCodegenError = nullptr;
  pm.run(m);
  if (pendingCodegenError) {
    std::exception_ptr e = pendingCodegenError;
    pendingCodegenError = nullptr;
    std::rethrow_exception(e);
  }
}

// MIR parsing and codegen report failures through LLVMContext::diagnose --
// their return values are only a bare success/failure bit, so the rich reason
// (line, column, message) would otherwise be lost to the default handler's
// stderr print, which is invisible under the Python/nanobind harness. This RAII
// guard installs a handler that captures error-severity diagnostics into `sink`
// for the duration of one parse/codegen call, restoring the previous handler on
// scope exit so the capture buffer (a caller stack local) never outlives the
// handler that points at it.
struct ScopedDiagnosticCapture {
  llvm::LLVMContext &ctx;
  llvm::DiagnosticHandler::DiagnosticHandlerTy prevHandler;
  void *prevContext;

  ScopedDiagnosticCapture(llvm::LLVMContext &ctx, std::string &sink)
      : ctx(ctx), prevHandler(ctx.getDiagnosticHandlerCallBack()),
        prevContext(ctx.getDiagnosticContext()) {
    ctx.setDiagnosticHandlerCallBack(
        [](const llvm::DiagnosticInfo *di, void *context) {
          if (di->getSeverity() == llvm::DS_Error) {
            auto *out = static_cast<std::string *>(context);
            llvm::raw_string_ostream os(*out);
            llvm::DiagnosticPrinterRawOStream printer(os);
            di->print(printer);
          }
        },
        &sink);
  }
  ~ScopedDiagnosticCapture() {
    ctx.setDiagnosticHandlerCallBack(prevHandler, prevContext);
  }
  ScopedDiagnosticCapture(const ScopedDiagnosticCapture &) = delete;
  ScopedDiagnosticCapture &operator=(const ScopedDiagnosticCapture &) = delete;
};

// "<base>" when no diagnostic was captured, else "<base>: <detail>".
inline std::string withDetail(llvm::StringRef base, const std::string &detail) {
  if (detail.empty())
    return base.str(); // LCOV_EXCL_LINE -- MIRParser always diagnoses on error
  return (base + ": " + detail).str();
}

} // namespace eudsl

#endif // EUDSL_MIR_DIAGNOSTICS_H
