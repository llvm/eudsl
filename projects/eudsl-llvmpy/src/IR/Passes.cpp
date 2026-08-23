// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "IR/Ownership.h"

#include <llvm/ADT/Any.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>

#include <exception>

namespace {
// A Python exception raised inside a pass callback is stashed here instead of
// thrown across llvm::ModulePassManager::run, which libLLVMCore compiles with
// -fno-exceptions (LLVM_ENABLE_EH OFF): unwinding through those frames skips
// their destructors (e.g. leaves the thread-local PrettyStackTraceHead dangling
// and leaks the run's PreservedAnalyses) and std::terminates on targets built
// without asynchronous unwind tables. runPipeline resets the slot, runs the
// pipeline, and re-raises afterward, so the throw only crosses our own
// -fexceptions frames. Thread-local so concurrent pipelines don't clobber it.
thread_local std::exception_ptr pendingPassError;

void runPipeline(llvm::ModulePassManager &mpm, llvm::Module &m,
                 llvm::ModuleAnalysisManager &mam) {
  pendingPassError = nullptr;
  mpm.run(m, mam);
  if (pendingPassError) {
    std::exception_ptr err = pendingPassError;
    pendingPassError = nullptr;
    std::rethrow_exception(err);
  }
}

// Bundles the PassBuilder with the four analysis managers and instrumentation,
// with the lifetimes ModulePassManager::run requires: the managers and the
// StandardInstrumentations must outlive the run that references them, and the
// PassBuilder holds a pointer to `pic`. run_passes and run_default_pipeline
// build this identically, so keeping it in one place stops the two copies from
// drifting. Member declaration order is the construction order: `pic` before
// `pb` (which captures &pic), `si` before its registerCallbacks(pic).
struct PassPipelineEnv {
  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si;
  llvm::PassBuilder pb;
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  PassPipelineEnv(llvm::LLVMContext &ctx,
                  const llvm::PipelineTuningOptions &opts, bool debug,
                  bool verifyEach)
      : si(ctx, debug, verifyEach), pb(nullptr, opts, std::nullopt, &pic) {
    si.registerCallbacks(pic);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    // Once a pass callback has stashed an error, skip every subsequent optional
    // pass so the pipeline winds down to runPipeline's re-raise instead of
    // running builtins on IR the failed pass may have left half-mutated. (The
    // manager still runs any isRequired() pass -- those cannot be skipped.)
    pic.registerShouldRunOptionalPassCallback(
        [](llvm::StringRef, llvm::Any) { return !pendingPassError; });
  }
};

// A new-PM module pass whose body is a Python callable. The new PassManager is
// concept-based, so a pass needs no LLVM base class beyond the PassInfoMixin
// that supplies name()/isRequired()/printPipeline -- any movable type with a
// run(Module&, ModuleAnalysisManager&) qualifies. We hold the owning
// eudsl::Module so the callback receives the same Python wrapper the caller
// passed (rv_policy::reference maps the pointer back through nanobind's
// instance registry), and the nb::callable so the callback outlives the move
// into PassModel. Everything runs synchronously on the calling (Python) thread,
// so the callable's refcounting stays correct; we still take the GIL in run()
// to stay correct under a free-threaded interpreter.
struct PyModulePass : llvm::PassInfoMixin<PyModulePass> {
  eudsl::Module *mod;
  nb::callable callback;

  PyModulePass(eudsl::Module *mod, nb::callable callback)
      : mod(mod), callback(std::move(callback)) {}

  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    // The GIL guard is intentionally outside the try: the catch stashes the
    // exception with std::current_exception(), which for an nb::python_error
    // touches Python refcounts and so must run while the GIL is held. Its
    // construction (PyGILState_Ensure) does not raise, so nothing is lost by
    // leaving it uncaught here.
    nb::gil_scoped_acquire gil;
    try {
      nb::object res = callback(nb::cast(mod, nb::rv_policy::reference));
      // Convention: None or any falsy return means "IR unchanged" (preserve all
      // analyses); any truthy return means the pass mutated the IR.
      int truthy = PyObject_IsTrue(res.ptr());
      if (truthy < 0) {
        // PyObject_IsTrue set a Python error (the return value's __bool__
        // raised). Wrap it in a message that says where it came from, keeping
        // the original as the exception's cause.
        nb::python_error err;
        nb::raise_from(
            err, PyExc_ValueError,
            "could not evaluate the truthiness of a Python module pass's "
            "return value");
      }
      return truthy ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all();
    } catch (...) {
      // Do not let the exception unwind through LLVM's -fno-exceptions frames;
      // stash it and report "nothing changed" so the pipeline winds down.
      pendingPassError = std::current_exception();
      return llvm::PreservedAnalyses::all();
    }
  }
};

// The per-function analogue of PyModulePass. Wrapped in a
// ModuleToFunctionPassAdaptor it runs once per defined function; the callback
// receives the llvm::Function (bound directly, so nanobind hands back its
// wrapper). Same truthy-return / GIL contract as PyModulePass.
struct PyFunctionPass : llvm::PassInfoMixin<PyFunctionPass> {
  nb::callable callback;

  explicit PyFunctionPass(nb::callable callback)
      : callback(std::move(callback)) {}

  llvm::PreservedAnalyses run(llvm::Function &f,
                              llvm::FunctionAnalysisManager &) {
    nb::gil_scoped_acquire gil;
    try {
      nb::object res = callback(nb::cast(&f, nb::rv_policy::reference));
      int truthy = PyObject_IsTrue(res.ptr());
      if (truthy < 0)
        throw nb::python_error();
      return truthy ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all();
    } catch (...) {
      pendingPassError = std::current_exception();
      return llvm::PreservedAnalyses::all();
    }
  }
};
} // namespace

void populate_passes(nb::module_ &m) {
  nb::enum_<llvm::OptimizationLevel>(m, "OptLevel")
      .value("O0", llvm::OptimizationLevel::O0)
      .value("O1", llvm::OptimizationLevel::O1)
      .value("O2", llvm::OptimizationLevel::O2)
      .value("O3", llvm::OptimizationLevel::O3);

  nb::class_<llvm::PipelineTuningOptions>(m, "PipelineTuningOptions")
      .def(nb::init<>())
      .def_rw("loop_interleaving",
              &llvm::PipelineTuningOptions::LoopInterleaving)
      .def_rw("loop_vectorization",
              &llvm::PipelineTuningOptions::LoopVectorization)
      .def_rw("slp_vectorization",
              &llvm::PipelineTuningOptions::SLPVectorization)
      .def_rw("loop_unrolling", &llvm::PipelineTuningOptions::LoopUnrolling)
      .def_rw("merge_functions", &llvm::PipelineTuningOptions::MergeFunctions);

  m.def(
      "run_passes",
      [](eudsl::Module &mod, const std::string &pipeline,
         std::optional<llvm::PipelineTuningOptions> pto, bool debug,
         bool verifyEach) {
        llvm::PipelineTuningOptions opts =
            pto.value_or(llvm::PipelineTuningOptions());
        PassPipelineEnv env(mod.get().getContext(), opts, debug, verifyEach);

        llvm::ModulePassManager mpm;
        if (llvm::Error err = env.pb.parsePassPipeline(mpm, pipeline))
          throw std::runtime_error(llvm::toString(std::move(err)));
        runPipeline(mpm, mod.get(), env.mam);
      },
      "module"_a, "pipeline"_a, "tuning"_a = nb::none(), "debug"_a = false,
      "verify_each"_a = false,
      "Parse and run a textual pass pipeline over the module in place.");

  m.def(
      "run_default_pipeline",
      [](eudsl::Module &mod, llvm::OptimizationLevel level,
         std::optional<llvm::PipelineTuningOptions> pto, bool debug,
         bool verifyEach) {
        llvm::PipelineTuningOptions opts =
            pto.value_or(llvm::PipelineTuningOptions());
        PassPipelineEnv env(mod.get().getContext(), opts, debug, verifyEach);

        llvm::ModulePassManager mpm =
            (level == llvm::OptimizationLevel::O0)
                ? env.pb.buildO0DefaultPipeline(level)
                : env.pb.buildPerModuleDefaultPipeline(level);
        runPipeline(mpm, mod.get(), env.mam);
      },
      "module"_a, "level"_a, "tuning"_a = nb::none(), "debug"_a = false,
      "verify_each"_a = false,
      "Run the default optimization pipeline at the given level.");

  m.def(
      "run_python_pass_on_module",
      [](eudsl::Module &mod, nb::callable callback,
         std::optional<llvm::PipelineTuningOptions> pto, bool debug,
         bool verifyEach) {
        llvm::PipelineTuningOptions opts =
            pto.value_or(llvm::PipelineTuningOptions());
        PassPipelineEnv env(mod.get().getContext(), opts, debug, verifyEach);
        llvm::ModulePassManager mpm;
        mpm.addPass(PyModulePass(&mod, std::move(callback)));
        runPipeline(mpm, mod.get(), env.mam);
      },
      "module"_a, "callback"_a, "tuning"_a = nb::none(), "debug"_a = false,
      "verify_each"_a = false,
      "Run a Python callable as a module pass over the module in place. The "
      "callable receives the Module; return a truthy value if it mutated the "
      "IR (so analyses are invalidated), None/falsy otherwise.");

  m.def(
      "run_python_pass_on_function",
      [](eudsl::Module &mod, nb::callable callback) {
        PassPipelineEnv env(mod.get().getContext(),
                            llvm::PipelineTuningOptions(), /*debug=*/false,
                            /*verifyEach=*/false);
        llvm::ModulePassManager mpm;
        mpm.addPass(llvm::createModuleToFunctionPassAdaptor(
            PyFunctionPass(std::move(callback))));
        runPipeline(mpm, mod.get(), env.mam);
      },
      "module"_a, "callback"_a,
      "Run a Python callable as a function pass over each defined function in "
      "the module in place. The callable receives a Function; return a truthy "
      "value if it mutated the function, None/falsy otherwise.");
}
