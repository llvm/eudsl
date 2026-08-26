// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Python-subclassable pre-RA MachineScheduler strategy.
// mir.MachineSchedStrategy binds llvm::MachineSchedStrategy via a trampoline;
// register_scheduler adds a MachineSchedRegistry node so
// emit_object(scheduler="name") can select it. LLVM owns an OwningPyStrategy
// adaptor that forwards into the Python instance, since nanobind won't move a
// Python-created instance into a default-deleter unique_ptr.

#include "IR/Common.h"
#include "MIR/Diagnostics.h"

#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineScheduler.h>
#include <llvm/CodeGen/ScheduleDAG.h>
#include <llvm/CodeGen/ScheduleDAGMutation.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include <algorithm>
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
thread_local nb::object activeSchedClass;

// C++ side of mir.MachineSchedStrategy. Each override forwards into the Python
// object (nb_trampoline.base()) by name with the GIL held outside the try; a
// Python exception is stashed in pendingCodegenError and a legal value
// returned, since LLVM's scheduler frames are -fno-exceptions.
// runCodegenPipeline re-raises after the run.
//
// shadowTop/shadowBottom mirror the ready nodes LLVM releases, recorded before
// the fallible Python call. They are the pick_node fallback after a stash: LLVM
// only releases ready nodes and we drop the picked one, so the shadow front is
// always a legal choice. The raw contract puts the real ready set in Python,
// out of reach once it has failed -- this is the safety net.
class PySchedStrategy : public llvm::MachineSchedStrategy {
public:
  NB_TRAMPOLINE(llvm::MachineSchedStrategy, 6);

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
    shadowTop.clear();
    shadowBottom.clear();
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
    shadowTop.push_back(su);
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
    shadowBottom.push_back(su);
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
    // runCodegenPipeline's re-raise. A node released both top- and bottom-ready
    // sits in both shadows, so skip any LLVM has already scheduled.
    if (llvm::SUnit *su = popUnscheduled(shadowTop)) {
      isTopNode = true;
      return su;
    }
    if (llvm::SUnit *su = popUnscheduled(shadowBottom)) {
      isTopNode = false;
      return su;
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

private:
  // Pop shadow entries until an unscheduled node is found (LLVM marks a node
  // scheduled once picked), or the shadow drains.
  static llvm::SUnit *popUnscheduled(std::vector<llvm::SUnit *> &shadow) {
    while (!shadow.empty()) {
      llvm::SUnit *su = shadow.front();
      shadow.erase(shadow.begin());
      if (!su->isScheduled)
        return su;
    }
    return nullptr;
  }

  std::vector<llvm::SUnit *> shadowTop;
  std::vector<llvm::SUnit *> shadowBottom;
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

private:
  nb::object pyStrategy;
  llvm::MachineSchedStrategy *inner;
};

// Registered name -> Python class. A leaked deque: deque never reallocates, so
// the name c_str() handed to MachineSchedRegistry stays valid, and leaking
// avoids dropping Python refs at interpreter shutdown.
std::deque<std::pair<std::string, nb::object>> &schedClasses() {
  static auto *classes = new std::deque<std::pair<std::string, nb::object>>();
  return *classes;
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
  auto strategy = std::make_unique<OwningPyStrategy>(activeSchedClass());
  auto *dag = new llvm::ScheduleDAGMILive(c, std::move(strategy));
  dag->addMutation(llvm::createCopyConstrainDAGMutation(dag->TII, dag->TRI));
  return dag;
}

} // namespace

namespace eudsl {

// Validate that cls defines the required methods, record it, and (if new) add a
// MachineSchedRegistry node so -misched / the pipeline can select it by name.
// Re-registering a name swaps the class.
void registerScheduler(const std::string &name, nb::object cls) {
  static const char *required[] = {"initialize",       "get_policy",
                                   "pick_node",        "sched_node",
                                   "release_top_node", "release_bottom_node"};
  for (const char *method : required) {
    if (!nb::hasattr(cls, method))
      throw nb::type_error(
          (std::string("scheduler class must define ") + method).c_str());
  }
  for (auto &entry : schedClasses()) {
    if (entry.first == name) {
      entry.second = std::move(cls);
      return;
    }
  }
  schedClasses().emplace_back(name, std::move(cls));
  const char *cname = schedClasses().back().first.c_str();
  schedRegistryNodes().push_back(std::make_unique<llvm::MachineSchedRegistry>(
      cname, cname, createRegisteredPyStrategy));
}

// The class registered under `name`, or an empty object.
nb::object schedulerClass(const std::string &name) {
  for (auto &entry : schedClasses()) {
    if (entry.first == name)
      return entry.second;
  }
  return nb::object();
}

void setActiveSchedClass(nb::object cls) { activeSchedClass = std::move(cls); }
void clearActiveSchedClass() { activeSchedClass = nb::object(); }

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

  // The pre-RA MachineScheduler strategy interface, subclassable from Python.
  // Override initialize(dag), get_policy() -> MachineSchedPolicy,
  // release_top_node(su) / release_bottom_node(su) (maintain your own ready
  // set), pick_node() -> (SUnit, is_top_node), and sched_node(su, is_top).
  // Register with register_scheduler and select via
  // emit_object(scheduler=name).
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
        std::vector<std::string> names;
        for (llvm::MachineSchedRegistry *node =
                 llvm::MachineSchedRegistry::getList();
             node; node = node->getNext())
          names.emplace_back(node->getName().str());
        return names;
      },
      "Names of the pre-RA MachineScheduler strategies registered in this "
      "extension, selectable via emit_object(scheduler=...).");
}
