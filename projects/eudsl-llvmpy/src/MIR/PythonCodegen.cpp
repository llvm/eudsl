// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"
#include "MIR/Diagnostics.h"

#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/ScheduleDAG.h>
#include <llvm/CodeGen/ScheduleDAGMutation.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <deque>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace eudsl {
// The single definition of the codegen-error stash declared extern in
// Diagnostics.h; scheduler overrides in this TU stash into it.
thread_local std::exception_ptr pendingCodegenError;
} // namespace eudsl

namespace {

// The Python strategy class emit_object selected for the current run. Set
// (under the GIL) before the pipeline runs and cleared after, so the shared
// registry ctor knows which class to instantiate per MachineFunction.
// GIL-serialized, matching the process-global -misched handling in Machine.cpp.
thread_local nb::type_object activeSchedClass;

class PySchedStrategy : public llvm::MachineSchedStrategy {
public:
  NB_TRAMPOLINE(llvm::MachineSchedStrategy, 9);

  llvm::MachineSchedPolicy getPolicy() const override {
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return {};
    try {
      return nb::cast<llvm::MachineSchedPolicy>(
          nb_trampoline.base().attr("get_policy")());
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
      return {};
    }
  }

  bool shouldTrackPressure() const override {
    return getPolicy().ShouldTrackPressure;
  }

  void initialize(llvm::ScheduleDAGMI *dag) override {
    shadow.clear();
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("initialize")(
          nb::cast(dag, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void releaseTopNode(llvm::SUnit *su) override {
    shadow.push_back(su);
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("release_top_node")(
          nb::cast(su, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void releaseBottomNode(llvm::SUnit *su) override {
    shadow.push_back(su);
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("release_bottom_node")(
          nb::cast(su, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  llvm::SUnit *pickNode(bool &isTopNode) override {
    nb::gil_scoped_acquire gil;
    if (!eudsl::pendingCodegenError) {
      try {
        nb::object choice = nb_trampoline.base().attr("pick_node")();
        // None signals "nothing ready" -- end scheduling (LLVM's nullptr).
        if (choice.is_none())
          return nullptr;
        auto [su, isTop] = nb::cast<std::pair<llvm::SUnit *, bool>>(choice);
        isTopNode = isTop;
        return su;
      } catch (...) {
        eudsl::pendingCodegenError = std::current_exception();
      }
    }
    // Stash path (or draining after a prior stash): return a still-ready,
    // not-yet-scheduled node so the unskippable pipeline winds down to
    // runCodegenPipeline's re-raise. LLVM only releases ready nodes; skip any
    // it has since scheduled, and report the node's readiness direction.
    while (!shadow.empty()) {
      llvm::SUnit *su = shadow.front();
      shadow.erase(shadow.begin());
      if (!su->isScheduled) {
        isTopNode = su->isTopReady();
        return su;
      }
    }
    return nullptr;
  }

  void schedNode(llvm::SUnit *su, bool isTopNode) override {
    nb::gil_scoped_acquire gil;
    if (eudsl::pendingCodegenError)
      return;
    try {
      nb_trampoline.base().attr("sched_node")(
          nb::cast(su, nb::rv_policy::reference), isTopNode);
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  // Optional lifecycle hooks: forwarded only if the Python subclass defines
  // them, otherwise LLVM's no-op default stands. registerRoots fires once the
  // full initial ready set has been released; enterMBB/leaveMBB bracket each
  // block (which may hold several scheduling regions).
  void registerRoots() override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "register_roots"))
      return;
    try {
      self.attr("register_roots")();
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void enterMBB(llvm::MachineBasicBlock *mbb) override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "enter_mbb"))
      return;
    try {
      self.attr("enter_mbb")(nb::cast(mbb, nb::rv_policy::reference));
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

  void leaveMBB() override {
    nb::gil_scoped_acquire gil;
    nb::handle self = nb_trampoline.base();
    if (eudsl::pendingCodegenError || !nb::hasattr(self, "leave_mbb"))
      return;
    try {
      self.attr("leave_mbb")();
    } catch (...) {
      eudsl::pendingCodegenError = std::current_exception();
    }
  }

private:
  // Mirrors the ready nodes LLVM releases, recorded before the fallible Python
  // call. It is the pick_node fallback after a stash: LLVM only releases ready
  // nodes, so a still-unscheduled one is always a legal choice. The raw
  // contract puts the real ready set in Python, out of reach once it has failed
  // -- this is the safety net.
  std::vector<llvm::SUnit *> shadow;
};

// The strategy LLVM owns. nanobind refuses to move a Python-created instance
// into a default-deleter unique_ptr<MachineSchedStrategy> (its C++ subobject is
// inline in the PyObject, so a raw delete is UB) -- which is
// ScheduleDAGMILive's sink. So LLVM owns this plain heap object, which keeps
// the Python instance alive and forwards each virtual into its inline
// PySchedStrategy (which does the work and never throws). The dtor drops the
// Python reference under the GIL.
class OwningPyStrategy : public llvm::MachineSchedStrategy {
public:
  explicit OwningPyStrategy(nb::object pyStrategy)
      : pyStrategy(std::move(pyStrategy)),
        inner(nb::inst_ptr<llvm::MachineSchedStrategy>(this->pyStrategy)) {}

  ~OwningPyStrategy() override {
    nb::gil_scoped_acquire gil;
    pyStrategy.reset();
  }

  llvm::MachineSchedPolicy getPolicy() const override {
    return inner->getPolicy();
  }
  bool shouldTrackPressure() const override {
    return inner->shouldTrackPressure();
  }
  void initialize(llvm::ScheduleDAGMI *dag) override { inner->initialize(dag); }
  void releaseTopNode(llvm::SUnit *su) override { inner->releaseTopNode(su); }
  void releaseBottomNode(llvm::SUnit *su) override {
    inner->releaseBottomNode(su);
  }
  llvm::SUnit *pickNode(bool &isTopNode) override {
    return inner->pickNode(isTopNode);
  }
  void schedNode(llvm::SUnit *su, bool isTopNode) override {
    inner->schedNode(su, isTopNode);
  }
  void registerRoots() override { inner->registerRoots(); }
  void enterMBB(llvm::MachineBasicBlock *mbb) override { inner->enterMBB(mbb); }
  void leaveMBB() override { inner->leaveMBB(); }

private:
  nb::object pyStrategy;
  llvm::MachineSchedStrategy *inner;
};

// Stable storage for registered names: a leaked deque (pure C++, no Python
// refs) whose element c_str() the MachineSchedRegistry node borrows for process
// lifetime.
std::deque<std::string> &schedNames() {
  static auto *names = new std::deque<std::string>();
  return *names;
}

// name -> Python class, held in llvm.mir_strategies._scheduler_classes. Python
// owns it, so the classes are released at interpreter teardown (a C++-held
// nb::object static would pin the subclass types past nanobind's teardown and
// trip its leak checker).
nb::dict schedulerClasses() {
  return nb::cast<nb::dict>(
      nb::module_::import_("llvm.mir_strategies").attr("_scheduler_classes"));
}

// The live MachineSchedRegistry nodes, kept (leaked) so they stay registered
// for process lifetime.
std::vector<std::unique_ptr<llvm::MachineSchedRegistry>> &schedRegistryNodes() {
  static auto *nodes =
      new std::vector<std::unique_ptr<llvm::MachineSchedRegistry>>();
  return *nodes;
}

// Shared registry ctor for every registered name (the ctor is a non-capturing
// function pointer, so it cannot carry the class). emit_object set
// activeSchedClass; construct a fresh instance per MachineFunction and hand
// LLVM an OwningPyStrategy around it.
llvm::ScheduleDAGInstrs *
createRegisteredPyStrategy(llvm::MachineSchedContext *c) {
  nb::gil_scoped_acquire gil;
  // LCOV_EXCL_START -- emit_object always sets the active class before running
  if (!activeSchedClass.is_valid())
    return llvm::createSchedLive(c);
  // LCOV_EXCL_STOP
  try {
    auto strategy = std::make_unique<OwningPyStrategy>(activeSchedClass());
    auto *dag = new llvm::ScheduleDAGMILive(c, std::move(strategy));
    dag->addMutation(llvm::createCopyConstrainDAGMutation(dag->TII, dag->TRI));
    return dag;
  } catch (...) {
    // Constructing the strategy runs the subclass __init__, which can raise.
    // This ctor is called from inside pm.run (libLLVMCodeGen, -fno-exceptions),
    // so stash the error and wind down with the default DAG; runCodegenPipeline
    // re-raises after the run.
    eudsl::pendingCodegenError = std::current_exception();
    return llvm::createSchedLive(c);
  }
}

} // namespace

namespace eudsl {

// Validate that cls subclasses MachineSchedStrategy and defines the required
// methods, record it, and (if new) add a MachineSchedRegistry node so the
// pipeline can select it by name. Re-registering a name swaps the class.
void registerScheduler(const std::string &name, nb::type_object cls) {
  if (PyObject_IsSubclass(cls.ptr(),
                          nb::type<llvm::MachineSchedStrategy>().ptr()) != 1) {
    throw nb::type_error(
        "scheduler class must subclass mir.MachineSchedStrategy");
  }
  static const char *required[] = {"initialize",       "get_policy",
                                   "pick_node",        "sched_node",
                                   "release_top_node", "release_bottom_node"};
  for (const char *method : required) {
    if (!nb::hasattr(cls, method)) {
      throw nb::type_error(
          (std::string("scheduler class must define ") + method).c_str());
    }
  }
  nb::dict classes = schedulerClasses();
  if (!classes.contains(name.c_str())) {
    schedNames().push_back(name);
    const char *cname = schedNames().back().c_str();
    schedRegistryNodes().push_back(std::make_unique<llvm::MachineSchedRegistry>(
        cname, cname, createRegisteredPyStrategy));
  }
  classes[name.c_str()] = cls;
}

// The class registered under `name`, or an invalid object if `name` was not
// registered via register_scheduler.
nb::type_object schedulerClass(const std::string &name) {
  nb::dict classes = schedulerClasses();
  if (classes.contains(name.c_str()))
    return nb::borrow<nb::type_object>(classes[name.c_str()]);
  return nb::type_object();
}

// The ctor every register_scheduler name shares, so emit_object can point
// -misched at it without walking the registry.
llvm::MachineSchedRegistry::ScheduleDAGCtor registeredSchedCtor() {
  return createRegisteredPyStrategy;
}

void setActiveSchedClass(nb::type_object cls) {
  activeSchedClass = std::move(cls);
}
void clearActiveSchedClass() { activeSchedClass = nb::type_object(); }

} // namespace eudsl

void populate_python_codegen(nb::module_ &m) {
  // Per-region scheduling policy a strategy returns from get_policy(). Field
  // names mirror MachineSchedPolicy's members.
  nb::class_<llvm::MachineSchedPolicy>(m, "MachineSchedPolicy")
      .def(nb::init<>())
      .def_rw("should_track_pressure",
              &llvm::MachineSchedPolicy::ShouldTrackPressure)
      .def_rw("should_track_lane_masks",
              &llvm::MachineSchedPolicy::ShouldTrackLaneMasks)
      .def_rw("only_top_down", &llvm::MachineSchedPolicy::OnlyTopDown)
      .def_rw("only_bottom_up", &llvm::MachineSchedPolicy::OnlyBottomUp);

  // One scheduling unit (a MachineInstr plus its dependency edges). A strategy
  // receives these via release_top_node/release_bottom_node and returns one
  // from pick_node.
  nb::class_<llvm::SUnit>(m, "SUnit")
      .def_prop_ro(
          "node_num", [](llvm::SUnit &su) { return su.NodeNum; },
          "Entry number of this node in the DAG's node vector.")
      .def_prop_ro(
          "is_top_ready", [](llvm::SUnit &su) { return su.isTopReady(); },
          "All predecessors scheduled (ready top-down).")
      .def_prop_ro(
          "is_bottom_ready", [](llvm::SUnit &su) { return su.isBottomReady(); },
          "All successors scheduled (ready bottom-up).")
      .def_prop_ro(
          "instr", [](llvm::SUnit &su) { return su.getInstr(); },
          nb::rv_policy::reference_internal,
          "The representative MachineInstr this unit wraps.");

  // The scheduling DAG passed to initialize(dag). Opaque for now -- a strategy
  // receives its nodes via release_top_node/release_bottom_node.
  nb::class_<llvm::ScheduleDAGMI>(m, "ScheduleDAGMI");

  nb::class_<llvm::MachineSchedStrategy, PySchedStrategy>(
      m, "MachineSchedStrategy")
      .def(nb::init<>());

  m.def("register_scheduler", &eudsl::registerScheduler, "name"_a, "cls"_a,
        "Register a MachineSchedStrategy subclass under `name` so "
        "emit_object(scheduler=name) can select it. The class must define "
        "initialize, get_policy, pick_node, sched_node, release_top_node, and "
        "release_bottom_node; re-registering a name replaces it.");

  m.def(
      "registered_schedulers",
      []() {
        return std::vector<std::string>(schedNames().begin(),
                                        schedNames().end());
      },
      "Names registered via register_scheduler, selectable with "
      "emit_object(scheduler=...).");
}
