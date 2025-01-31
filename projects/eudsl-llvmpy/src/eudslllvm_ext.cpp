// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2025.

#include "pp/Core.h"
#include "pp/IRReader.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;

NB_MODULE(eudslllvm_ext, m) {
  extern void populate_LLJIT(nb::module_ & m);
  populate_LLJIT(m);
  extern void populate_BitReader(nb::module_ & m);
  populate_BitReader(m);
  extern void populate_ErrorHandling(nb::module_ & m);
  populate_ErrorHandling(m);
  extern void populate_BitWriter(nb::module_ & m);
  populate_BitWriter(m);
  extern void populate_DisassemblerTypes(nb::module_ & m);
  populate_DisassemblerTypes(m);
  extern void populate_Orc(nb::module_ & m);
  populate_Orc(m);
  extern void populate_IRReader(nb::module_ & m);
  populate_IRReader(m);
  extern void populate_PassBuilder(nb::module_ & m);
  populate_PassBuilder(m);
  extern void populate_Types(nb::module_ & m);
  populate_Types(m);
  extern void populate_DebugInfo(nb::module_ & m);
  populate_DebugInfo(m);
  extern void populate_ExecutionEngine(nb::module_ & m);
  populate_ExecutionEngine(m);
  extern void populate_Object(nb::module_ & m);
  populate_Object(m);
  extern void populate_Comdat(nb::module_ & m);
  populate_Comdat(m);
  extern void populate_Analysis(nb::module_ & m);
  populate_Analysis(m);
  extern void populate_Support(nb::module_ & m);
  populate_Support(m);
  extern void populate_blake3(nb::module_ & m);
  populate_blake3(m);
  extern void populate_LLJITUtils(nb::module_ & m);
  populate_LLJITUtils(m);
  extern void populate_Linker(nb::module_ & m);
  populate_Linker(m);
  extern void populate_Remarks(nb::module_ & m);
  populate_Remarks(m);
  extern void populate_TargetMachine(nb::module_ & m);
  populate_TargetMachine(m);
  extern void populate_Error(nb::module_ & m);
  populate_Error(m);
  extern void populate_OrcEE(nb::module_ & m);
  populate_OrcEE(m);
  extern void populate_Core(nb::module_ & m);
  populate_Core(m);
  extern void populate_Disassembler(nb::module_ & m);
  populate_Disassembler(m);
  extern void populate_Target(nb::module_ & m);
  populate_Target(m);

  m.def("parse_ir_in_context",
        [](LLVMContextRef ContextRef, LLVMMemoryBufferRef MemBuf,
           LLVMModuleRef *OutM) {
          char *OutMessage;
          return LLVMParseIRInContext(ContextRef, MemBuf, OutM, &OutMessage);
        });

  m.def(
      "function_type",
      [](LLVMTypeRef ReturnType, std::vector<LLVMTypeRef> ParamTypes,
         LLVMBool IsVarArg) {
        return LLVMFunctionType(ReturnType, ParamTypes.data(),
                                ParamTypes.size(), IsVarArg);
      },
      nb::arg("return_type"), nb::arg("param_types"), nb::arg("is_var_arg"),
      " * Obtain a function type consisting of a specified signature.\n *\n "
      "* The function is defined as a tuple of a return Type, a list of\n * "
      "parameter types, and whether the function is variadic.");

  m.def(
      "get_param_types",
      [](LLVMTypeRef FunctionTy, std::vector<LLVMTypeRef> Dest) {
        LLVMGetParamTypes(FunctionTy, Dest.data());
      },
      nb::arg("function_ty"), nb::arg("dest"),
      " * Obtain the types of a function's parameters.\n *\n * The Dest "
      "parameter should point to a pre-allocated array of\n * LLVMTypeRef at "
      "least LLVMCountParamTypes() large. On return, the\n * first "
      "LLVMCountParamTypes() entries in the array will be populated\n * with "
      "LLVMTypeRef instances.\n *\n * @param FunctionTy The function type to "
      "operate on.\n * @param Dest Memory address of an array to be filled "
      "with result.");

  m.def(
      "struct_type_in_context",
      [](LLVMContextRef C, std::vector<LLVMTypeRef> ElementTypes,
         LLVMBool Packed) {
        return LLVMStructTypeInContext(C, ElementTypes.data(),
                                       ElementTypes.size(), Packed);
      },
      nb::arg("c"), nb::arg("element_types"), nb::arg("packed"),
      " * Create a new structure type in a context.\n *\n * A structure is "
      "specified by a list of inner elements/types and\n * whether these can "
      "be packed together.\n *\n * @see llvm::StructType::create()");

  m.def(
      "struct_type",
      [](std::vector<LLVMTypeRef> ElementTypes, LLVMBool Packed) {
        return LLVMStructType(ElementTypes.data(), ElementTypes.size(), Packed);
      },
      nb::arg("element_types"), nb::arg("packed"),
      " * Create a new structure type in the global context.\n *\n * @see "
      "llvm::StructType::create()");

  m.def(
      "struct_set_body",
      [](LLVMTypeRef StructTy, std::vector<LLVMTypeRef> ElementTypes,
         LLVMBool Packed) {
        LLVMStructSetBody(StructTy, ElementTypes.data(), ElementTypes.size(),
                          Packed);
      },
      nb::arg("struct_ty"), nb::arg("element_types"), nb::arg("packed"),
      " * Set the contents of a structure type.\n *\n * @see "
      "llvm::StructType::setBody()");

  m.def(
      "const_gep2",
      [](LLVMTypeRef Ty, LLVMValueRef ConstantVal,
         std::vector<LLVMValueRef> ConstantIndices) {
        return LLVMConstGEP2(Ty, ConstantVal, ConstantIndices.data(),
                             ConstantIndices.size());
      },
      nb::arg("ty"), nb::arg("constant_val"), nb::arg("constant_indices"));

  m.def(
      "const_in_bounds_gep2",
      [](LLVMTypeRef Ty, LLVMValueRef ConstantVal,
         std::vector<LLVMValueRef> ConstantIndices) {
        return LLVMConstInBoundsGEP2(Ty, ConstantVal, ConstantIndices.data(),
                                     ConstantIndices.size());
      },
      nb::arg("ty"), nb::arg("constant_val"), nb::arg("constant_indices"));

  m.def(
      "const_gep_with_no_wrap_flags",
      [](LLVMTypeRef Ty, LLVMValueRef ConstantVal,
         std::vector<LLVMValueRef> ConstantIndices,
         LLVMGEPNoWrapFlags NoWrapFlags) {
        return LLVMConstGEPWithNoWrapFlags(Ty, ConstantVal,
                                           ConstantIndices.data(),
                                           ConstantIndices.size(), NoWrapFlags);
      },
      nb::arg("ty"), nb::arg("constant_val"), nb::arg("constant_indices"),
      nb::arg("no_wrap_flags"));

  m.def(
      "get_intrinsic_declaration",
      [](LLVMModuleRef Mod, unsigned ID, std::vector<LLVMTypeRef> ParamTypes) {
        return LLVMGetIntrinsicDeclaration(Mod, ID, ParamTypes.data(),
                                           ParamTypes.size());
      },
      nb::arg("mod"), nb::arg("id"), nb::arg("param_types"),
      " * Get or insert the declaration of an intrinsic.  For overloaded "
      "intrinsics,\n * parameter types must be provided to uniquely identify "
      "an overload.\n *\n * @see llvm::Intrinsic::getOrInsertDeclaration()");

  m.def(
      "intrinsic_get_type",
      [](LLVMContextRef Ctx, unsigned ID, std::vector<LLVMTypeRef> ParamTypes) {
        return LLVMIntrinsicGetType(Ctx, ID, ParamTypes.data(),
                                    ParamTypes.size());
      },
      nb::arg("ctx"), nb::arg("id"), nb::arg("param_types"),
      " * Retrieves the type of an intrinsic.  For overloaded intrinsics, "
      "parameter\n * types must be provided to uniquely identify an "
      "overload.\n *\n * @see llvm::Intrinsic::getType()");

  m.def(
      "intrinsic_copy_overloaded_name",
      [](unsigned ID, std::vector<LLVMTypeRef> ParamTypes, size_t *NameLength) {
        return LLVMIntrinsicCopyOverloadedName(ID, ParamTypes.data(),
                                               ParamTypes.size(), NameLength);
      },
      nb::arg("id"), nb::arg("param_types"), nb::arg("name_length"),
      "Deprecated: Use LLVMIntrinsicCopyOverloadedName2 instead.");

  m.def(
      "intrinsic_copy_overloaded_name2",
      [](LLVMModuleRef Mod, unsigned ID, std::vector<LLVMTypeRef> ParamTypes,
         size_t *NameLength) {
        return LLVMIntrinsicCopyOverloadedName2(Mod, ID, ParamTypes.data(),
                                                ParamTypes.size(), NameLength);
      },
      nb::arg("mod"), nb::arg("id"), nb::arg("param_types"),
      nb::arg("name_length"),
      " * Copies the name of an overloaded intrinsic identified by a given "
      "list of\n * parameter types.\n *\n * Unlike LLVMIntrinsicGetName, the "
      "caller is responsible for freeing the\n * returned string.\n *\n * "
      "This version also supports unnamed types.\n *\n * @see "
      "llvm::Intrinsic::getName()");

  m.def(
      "build_call2",
      [](LLVMBuilderRef builder, LLVMTypeRef fn_type, LLVMValueRef Fn,
         std::vector<LLVMValueRef> Args, const char *Name) {
        return LLVMBuildCall2(builder, fn_type, Fn, Args.data(), Args.size(),
                              Name);
      },
      nb::arg("param_0"), nb::arg("fn_type"), nb::arg("fn"), nb::arg("args"),
      nb::arg("name"));
}
