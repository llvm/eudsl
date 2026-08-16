// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IR/Common.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>

void populate_instructions(nb::module_ &m) {
  // ICmp and FCmp share the C++ enum llvm::CmpInst::Predicate, and nanobind
  // keys enum registration by C++ type, so register it once and expose the two
  // Python names as aliases.
  nb::enum_<llvm::CmpInst::Predicate>(m, "CmpPredicate")
      .value("EQ", llvm::CmpInst::ICMP_EQ)
      .value("NE", llvm::CmpInst::ICMP_NE)
      .value("UGT", llvm::CmpInst::ICMP_UGT)
      .value("UGE", llvm::CmpInst::ICMP_UGE)
      .value("ULT", llvm::CmpInst::ICMP_ULT)
      .value("ULE", llvm::CmpInst::ICMP_ULE)
      .value("SGT", llvm::CmpInst::ICMP_SGT)
      .value("SGE", llvm::CmpInst::ICMP_SGE)
      .value("SLT", llvm::CmpInst::ICMP_SLT)
      .value("SLE", llvm::CmpInst::ICMP_SLE)
      .value("OEQ", llvm::CmpInst::FCMP_OEQ)
      .value("OGT", llvm::CmpInst::FCMP_OGT)
      .value("OGE", llvm::CmpInst::FCMP_OGE)
      .value("OLT", llvm::CmpInst::FCMP_OLT)
      .value("OLE", llvm::CmpInst::FCMP_OLE)
      .value("ONE", llvm::CmpInst::FCMP_ONE)
      .value("UEQ", llvm::CmpInst::FCMP_UEQ)
      .value("UNE", llvm::CmpInst::FCMP_UNE);
  m.attr("ICmpPredicate") = m.attr("CmpPredicate");
  m.attr("FCmpPredicate") = m.attr("CmpPredicate");

  // Memory-ordering + atomicrmw operation enums, used by the IRBuilder
  // fence/atomic_rmw/atomic_cmpxchg emitters.
  nb::enum_<llvm::AtomicOrdering>(m, "AtomicOrdering")
      .value("NotAtomic", llvm::AtomicOrdering::NotAtomic)
      .value("Unordered", llvm::AtomicOrdering::Unordered)
      .value("Monotonic", llvm::AtomicOrdering::Monotonic)
      .value("Acquire", llvm::AtomicOrdering::Acquire)
      .value("Release", llvm::AtomicOrdering::Release)
      .value("AcquireRelease", llvm::AtomicOrdering::AcquireRelease)
      .value("SequentiallyConsistent",
             llvm::AtomicOrdering::SequentiallyConsistent);
  nb::enum_<llvm::AtomicRMWInst::BinOp>(m, "AtomicRMWBinOp")
      .value("Xchg", llvm::AtomicRMWInst::Xchg)
      .value("Add", llvm::AtomicRMWInst::Add)
      .value("Sub", llvm::AtomicRMWInst::Sub)
      .value("And", llvm::AtomicRMWInst::And)
      .value("Nand", llvm::AtomicRMWInst::Nand)
      .value("Or", llvm::AtomicRMWInst::Or)
      .value("Xor", llvm::AtomicRMWInst::Xor)
      .value("Max", llvm::AtomicRMWInst::Max)
      .value("Min", llvm::AtomicRMWInst::Min)
      .value("UMax", llvm::AtomicRMWInst::UMax)
      .value("UMin", llvm::AtomicRMWInst::UMin)
      .value("FAdd", llvm::AtomicRMWInst::FAdd)
      .value("FSub", llvm::AtomicRMWInst::FSub)
      .value("FMax", llvm::AtomicRMWInst::FMax)
      .value("FMin", llvm::AtomicRMWInst::FMin)
      .value("FMaximum", llvm::AtomicRMWInst::FMaximum)
      .value("FMinimum", llvm::AtomicRMWInst::FMinimum)
      .value("FMaximumNum", llvm::AtomicRMWInst::FMaximumNum)
      .value("FMinimumNum", llvm::AtomicRMWInst::FMinimumNum)
      .value("UIncWrap", llvm::AtomicRMWInst::UIncWrap)
      .value("UDecWrap", llvm::AtomicRMWInst::UDecWrap)
      .value("USubCond", llvm::AtomicRMWInst::USubCond)
      .value("USubSat", llvm::AtomicRMWInst::USubSat);

  // Intermediate bases (the Value type_hook never names these, but leaf classes
  // need them registered as their nanobind base).
  nb::class_<llvm::UnaryInstruction, llvm::Instruction>(m, "UnaryInstruction");
  nb::class_<llvm::UnaryOperator, llvm::UnaryInstruction>(m, "UnaryOperator");
  nb::class_<llvm::BinaryOperator, llvm::Instruction>(m, "BinaryOperator");
  nb::class_<llvm::CastInst, llvm::UnaryInstruction>(m, "CastInst");
  nb::class_<llvm::FuncletPadInst, llvm::Instruction>(m, "FuncletPadInst");

  nb::class_<llvm::CmpInst, llvm::Instruction>(m, "CmpInst")
      .EUDSL_CAST_CTOR(llvm::CmpInst, llvm::Value)
      .def_prop_ro("predicate", &llvm::CmpInst::getPredicate);

  nb::class_<llvm::CallBase, llvm::Instruction>(m, "CallBase")
      .EUDSL_CAST_CTOR(llvm::CallBase, llvm::Value)
      .def_prop_ro("num_args", &llvm::CallBase::arg_size)
      .def(
          "arg_operand",
          [](llvm::CallBase &self, unsigned i) {
            if (i >= self.arg_size())
              throw nb::index_error("argument index out of range");
            return self.getArgOperand(i);
          },
          "index"_a, nb::rv_policy::reference_internal)
      .def_prop_ro(
          "called_operand",
          [](llvm::CallBase &self) { return self.getCalledOperand(); },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::ICmpInst, llvm::CmpInst>(m, "ICmpInst");
  nb::class_<llvm::FCmpInst, llvm::CmpInst>(m, "FCmpInst");
  nb::class_<llvm::CallInst, llvm::CallBase>(m, "CallInst");
  nb::class_<llvm::InvokeInst, llvm::CallBase>(m, "InvokeInst");
  nb::class_<llvm::CallBrInst, llvm::CallBase>(m, "CallBrInst");

  nb::class_<llvm::PHINode, llvm::Instruction>(m, "PHINode")
      .EUDSL_CAST_CTOR(llvm::PHINode, llvm::Value)
      .def_prop_ro("num_incoming", &llvm::PHINode::getNumIncomingValues)
      .def(
          "incoming_value",
          [](llvm::PHINode &self, unsigned i) {
            if (i >= self.getNumIncomingValues())
              throw nb::index_error("incoming index out of range");
            return self.getIncomingValue(i);
          },
          "index"_a, nb::rv_policy::reference_internal)
      .def(
          "incoming_block",
          [](llvm::PHINode &self, unsigned i) {
            if (i >= self.getNumIncomingValues())
              throw nb::index_error("incoming index out of range");
            return self.getIncomingBlock(i);
          },
          "index"_a, nb::rv_policy::reference_internal)
      .def("add_incoming", &llvm::PHINode::addIncoming, "value"_a, "block"_a);

  nb::class_<llvm::AllocaInst, llvm::UnaryInstruction>(m, "AllocaInst")
      .EUDSL_CAST_CTOR(llvm::AllocaInst, llvm::Value)
      .def_prop_ro(
          "allocated_type",
          [](llvm::AllocaInst &self) { return self.getAllocatedType(); },
          nb::rv_policy::reference_internal);
  nb::class_<llvm::LoadInst, llvm::UnaryInstruction>(m, "LoadInst")
      .EUDSL_CAST_CTOR(llvm::LoadInst, llvm::Value)
      .def_prop_ro(
          "pointer_operand",
          [](llvm::LoadInst &self) { return self.getPointerOperand(); },
          nb::rv_policy::reference_internal);
  nb::class_<llvm::StoreInst, llvm::Instruction>(m, "StoreInst")
      .EUDSL_CAST_CTOR(llvm::StoreInst, llvm::Value)
      .def_prop_ro(
          "pointer_operand",
          [](llvm::StoreInst &self) { return self.getPointerOperand(); },
          nb::rv_policy::reference_internal);
  nb::class_<llvm::GetElementPtrInst, llvm::Instruction>(m, "GetElementPtrInst")
      .EUDSL_CAST_CTOR(llvm::GetElementPtrInst, llvm::Value)
      .def_prop_ro(
          "source_element_type",
          [](llvm::GetElementPtrInst &self) {
            return self.getSourceElementType();
          },
          nb::rv_policy::reference_internal);

  nb::class_<llvm::ReturnInst, llvm::Instruction>(m, "ReturnInst")
      .EUDSL_CAST_CTOR(llvm::ReturnInst, llvm::Value)
      .def_prop_ro(
          "return_value",
          [](llvm::ReturnInst &self) { return self.getReturnValue(); },
          nb::rv_policy::reference_internal);
  nb::class_<llvm::UncondBrInst, llvm::Instruction>(m, "UncondBrInst")
      .EUDSL_CAST_CTOR(llvm::UncondBrInst, llvm::Value)
      .def_prop_ro("is_conditional", [](llvm::UncondBrInst &) { return false; });
  nb::class_<llvm::CondBrInst, llvm::Instruction>(m, "CondBrInst")
      .EUDSL_CAST_CTOR(llvm::CondBrInst, llvm::Value)
      .def_prop_ro("is_conditional", [](llvm::CondBrInst &) { return true; })
      .def_prop_ro(
          "condition",
          [](llvm::CondBrInst &self) { return self.getCondition(); },
          nb::rv_policy::reference_internal);
  nb::class_<llvm::SwitchInst, llvm::Instruction>(m, "SwitchInst")
      .def("add_case", &llvm::SwitchInst::addCase, "on_value"_a, "dest"_a);
  nb::class_<llvm::IndirectBrInst, llvm::Instruction>(m, "IndirectBrInst")
      .def("add_destination", &llvm::IndirectBrInst::addDestination, "dest"_a);
  nb::class_<llvm::ResumeInst, llvm::Instruction>(m, "ResumeInst");
  nb::class_<llvm::UnreachableInst, llvm::Instruction>(m, "UnreachableInst");
  nb::class_<llvm::SelectInst, llvm::Instruction>(m, "SelectInst");
  nb::class_<llvm::VAArgInst, llvm::UnaryInstruction>(m, "VAArgInst");
  nb::class_<llvm::ExtractElementInst, llvm::Instruction>(m,
                                                         "ExtractElementInst");
  nb::class_<llvm::InsertElementInst, llvm::Instruction>(m,
                                                        "InsertElementInst");
  nb::class_<llvm::ShuffleVectorInst, llvm::Instruction>(m,
                                                        "ShuffleVectorInst");
  nb::class_<llvm::ExtractValueInst, llvm::UnaryInstruction>(m,
                                                            "ExtractValueInst");
  nb::class_<llvm::InsertValueInst, llvm::Instruction>(m, "InsertValueInst");
  nb::class_<llvm::LandingPadInst, llvm::Instruction>(m, "LandingPadInst");
  nb::class_<llvm::FreezeInst, llvm::UnaryInstruction>(m, "FreezeInst");
  nb::class_<llvm::FenceInst, llvm::Instruction>(m, "FenceInst");
  nb::class_<llvm::AtomicCmpXchgInst, llvm::Instruction>(m,
                                                        "AtomicCmpXchgInst");
  nb::class_<llvm::AtomicRMWInst, llvm::Instruction>(m, "AtomicRMWInst");
  nb::class_<llvm::CleanupPadInst, llvm::FuncletPadInst>(m, "CleanupPadInst");
  nb::class_<llvm::CatchPadInst, llvm::FuncletPadInst>(m, "CatchPadInst");
  nb::class_<llvm::CatchReturnInst, llvm::Instruction>(m, "CatchReturnInst");
  nb::class_<llvm::CleanupReturnInst, llvm::Instruction>(m,
                                                        "CleanupReturnInst");
  nb::class_<llvm::CatchSwitchInst, llvm::Instruction>(m, "CatchSwitchInst");
  nb::class_<llvm::TruncInst, llvm::CastInst>(m, "TruncInst");
  nb::class_<llvm::ZExtInst, llvm::CastInst>(m, "ZExtInst");
  nb::class_<llvm::SExtInst, llvm::CastInst>(m, "SExtInst");
  nb::class_<llvm::FPToUIInst, llvm::CastInst>(m, "FPToUIInst");
  nb::class_<llvm::FPToSIInst, llvm::CastInst>(m, "FPToSIInst");
  nb::class_<llvm::UIToFPInst, llvm::CastInst>(m, "UIToFPInst");
  nb::class_<llvm::SIToFPInst, llvm::CastInst>(m, "SIToFPInst");
  nb::class_<llvm::FPTruncInst, llvm::CastInst>(m, "FPTruncInst");
  nb::class_<llvm::FPExtInst, llvm::CastInst>(m, "FPExtInst");
  nb::class_<llvm::PtrToIntInst, llvm::CastInst>(m, "PtrToIntInst");
  nb::class_<llvm::PtrToAddrInst, llvm::CastInst>(m, "PtrToAddrInst");
  nb::class_<llvm::IntToPtrInst, llvm::CastInst>(m, "IntToPtrInst");
  nb::class_<llvm::BitCastInst, llvm::CastInst>(m, "BitCastInst");
  nb::class_<llvm::AddrSpaceCastInst, llvm::CastInst>(m, "AddrSpaceCastInst");
  nb::class_<llvm::FPUnaryOperator, llvm::UnaryOperator>(m, "FPUnaryOperator");
  nb::class_<llvm::FPBinaryOperator, llvm::BinaryOperator>(m,
                                                          "FPBinaryOperator");
}
