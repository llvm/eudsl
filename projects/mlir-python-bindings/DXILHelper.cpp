//===----------------- DXILHelper.cpp ---------------------------*- C++ -*-===//
//
// Python (nanobind) bindings for a compact MLIR -> LLVM IR -> DXIL -> metallib
// pipeline. The exposed surface is:
//
//   translate_mlir_to_llvm(module, ctx) -> LLVMModule
//   add_dxil_module_metadata(module, sm_major, sm_minor, ...)
//   mark_as_dxil_compute_kernel(fn, x, y, z)
//   lower_mlir_to_dxil(module)
//   translate_llvm_to_dxil(module, triple) -> bytes
//   translate_dxil_to_metallib(dxil, stage, entry_point)
//       -> (bytes, [(type, space, slot, offset, size, name), ...])
//
//===----------------------------------------------------------------------===//

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "metal_irconverter/metal_irconverter.h"

#include <memory>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <tuple>

extern "C" void LLVMInitializeDirectXTargetInfo();
extern "C" void LLVMInitializeDirectXTarget();
extern "C" void LLVMInitializeDirectXTargetMC();
extern "C" void LLVMInitializeDirectXAsmPrinter();

namespace llvm {
void lowerMLIRToDXIL(llvm::Module &M);
} // namespace llvm

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_mlirDXILHelper, m) {

  nb::class_<llvm::LLVMContext>(m, "LLVMContext").def(nb::init<>());

  nb::class_<llvm::Function>(m, "LLVMFunction")
      .def_prop_ro("name",
                   [](const llvm::Function *fn) { return fn->getName().str(); })
      .def(
          "add_fn_attr",
          [](llvm::Function *fn, const std::string &name,
             const std::string &val) { fn->addFnAttr(name, val); },
          "name"_a, "value"_a);

  nb::class_<llvm::Module::FunctionListType>(m, "LLVMFunctionList")
      .def(
          "__iter__",
          [](llvm::Module::FunctionListType &s) {
            return nb::make_iterator<nb::rv_policy::reference_internal>(
                nb::type<llvm::Function>(), "iterator", s.begin(), s.end());
          },
          nb::keep_alive<0, 1>());

  nb::class_<llvm::Module>(m, "LLVMModule")
      .def("__str__",
           [](const llvm::Module *self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             os << *self;
             return os.str();
           })
      .def(
          "get_functions",
          [](llvm::Module *mod) -> llvm::Module::FunctionListType & {
            return mod->getFunctionList();
          },
          nb::rv_policy::reference_internal);

  m.def(
      "translate_mlir_to_llvm",
      [](MlirOperation module, llvm::LLVMContext &ctx) {
        mlir::Operation *op = unwrap(module);
        std::unique_ptr<llvm::Module> llvmModule =
            mlir::translateModuleToLLVMIR(op, ctx);
        if (!llvmModule)
          throw std::runtime_error("could not convert to LLVM IR");
        return llvmModule;
      },
      "module"_a, "ctx"_a, nb::keep_alive<0, 2>());

  m.def(
      "add_dxil_module_metadata",
      [](llvm::Module *mod, int sm_major, int sm_minor, int valver_major,
         int valver_minor, const std::string &triple_opt) {
        std::string triple =
            triple_opt.empty()
                ? ("dxil-pc-shadermodel" + std::to_string(sm_major) + "." +
                   std::to_string(sm_minor) + "-compute")
                : triple_opt;
        mod->setTargetTriple(llvm::Triple(triple));
        llvm::LLVMContext &ctx = mod->getContext();
        auto *i32 = llvm::Type::getInt32Ty(ctx);
        auto mkConstMd = [&](int val) {
          return llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(i32, val));
        };
        llvm::NamedMDNode *valver = mod->getOrInsertNamedMetadata("dx.valver");
        valver->addOperand(llvm::MDNode::get(
            ctx, {mkConstMd(valver_major), mkConstMd(valver_minor)}));
      },
      "module"_a, "sm_major"_a = 6, "sm_minor"_a = 0, "valver_major"_a = 1,
      "valver_minor"_a = 8, "triple"_a = "");

  m.def(
      "mark_as_dxil_compute_kernel",
      [](llvm::Function *fn, int x, int y, int z) {
        fn->addFnAttr("hlsl.shader", "compute");
        fn->addFnAttr("hlsl.numthreads", std::to_string(x) + "," +
                                             std::to_string(y) + "," +
                                             std::to_string(z));
      },
      "fn"_a, "x"_a = 1, "y"_a = 1, "z"_a = 1);

  m.def("lower_mlir_to_dxil",
        [](llvm::Module *mod) { llvm::lowerMLIRToDXIL(*mod); });

  m.def(
      "translate_llvm_to_dxil",
      [](llvm::Module *llvmModule, const std::string &triple) -> nb::bytes {
        LLVMInitializeDirectXTargetInfo();
        LLVMInitializeDirectXTarget();
        LLVMInitializeDirectXTargetMC();
        LLVMInitializeDirectXAsmPrinter();

        llvm::Triple targetTriple(triple);
        std::string error;
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(targetTriple, error);
        if (!target)
          throw std::runtime_error("DXIL target lookup error: " + error);

        llvm::TargetOptions opt;
        std::unique_ptr<llvm::TargetMachine> machine{
            target->createTargetMachine(targetTriple, "", "", opt,
                                        llvm::Reloc::PIC_, std::nullopt,
                                        llvm::CodeGenOptLevel::None)};
        if (!machine)
          throw std::runtime_error("could not create DXIL target machine");

        llvmModule->setTargetTriple(targetTriple);
        llvmModule->setDataLayout(machine->createDataLayout());

        llvm::SmallVector<char, 0> buffer;
        llvm::raw_svector_ostream ostream(buffer);
        llvm::legacy::PassManager pm;
        if (machine->addPassesToEmitFile(pm, ostream, nullptr,
                                         llvm::CodeGenFileType::ObjectFile))
          throw std::runtime_error("DXIL target cannot emit object file");
        pm.run(*llvmModule);

        return nb::bytes(buffer.data(), buffer.size());
      },
      "module"_a, "triple"_a = "dxil-pc-shadermodel6.0-compute");

  nb::enum_<IRShaderStage>(m, "IRShaderStage")
      .value("Invalid", IRShaderStageInvalid)
      .value("Vertex", IRShaderStageVertex)
      .value("Fragment", IRShaderStageFragment)
      .value("Hull", IRShaderStageHull)
      .value("Domain", IRShaderStageDomain)
      .value("Mesh", IRShaderStageMesh)
      .value("Amplification", IRShaderStageAmplification)
      .value("Geometry", IRShaderStageGeometry)
      .value("Compute", IRShaderStageCompute)
      .value("ClosestHit", IRShaderStageClosestHit)
      .value("Intersection", IRShaderStageIntersection)
      .value("AnyHit", IRShaderStageAnyHit)
      .value("Miss", IRShaderStageMiss)
      .value("RayGeneration", IRShaderStageRayGeneration)
      .value("Callable", IRShaderStageCallable);

  m.def(
      "translate_dxil_to_metallib",
      [](nb::bytes dxil, IRShaderStage stage, const std::string &entryPoint)
          -> std::tuple<
              nb::bytes,
              std::vector<std::tuple<int, unsigned, unsigned, unsigned,
                                     uint64_t, std::string>>> {
        IRCompiler *compiler = IRCompilerCreate();
        IRObject *input = IRObjectCreateFromDXIL(
            reinterpret_cast<const uint8_t *>(dxil.c_str()), dxil.size(),
            IRBytecodeOwnershipNone);

        IRError *error = nullptr;
        const char *epName = entryPoint.empty() ? nullptr : entryPoint.c_str();
        IRObject *out =
            IRCompilerAllocCompileAndLink(compiler, epName, input, &error);
        if (!out) {
          uint32_t code = error ? IRErrorGetCode(error) : 0;
          if (error)
            IRErrorDestroy(error);
          IRObjectDestroy(input);
          IRCompilerDestroy(compiler);
          throw std::runtime_error(
              "IRCompilerAllocCompileAndLink failed (IRErrorCode=" +
              std::to_string(code) + ")");
        }

        IRMetalLibBinary *lib = IRMetalLibBinaryCreate();
        if (!IRObjectGetMetalLibBinary(out, stage, lib)) {
          IRMetalLibBinaryDestroy(lib);
          IRObjectDestroy(out);
          IRObjectDestroy(input);
          IRCompilerDestroy(compiler);
          throw std::runtime_error(
              "IRObjectGetMetalLibBinary: no bytecode for requested stage");
        }

        size_t sz = IRMetalLibGetBytecodeSize(lib);
        std::vector<uint8_t> buf(sz);
        IRMetalLibGetBytecode(lib, buf.data());

        std::vector<std::tuple<int, unsigned, unsigned, unsigned, uint64_t,
                               std::string>>
            locations;
        IRShaderReflection *reflection = IRShaderReflectionCreate();
        if (IRObjectGetReflection(out, stage, reflection)) {
          size_t n = IRShaderReflectionGetResourceCount(reflection);
          std::vector<IRResourceLocation> rls(n);
          if (n > 0)
            IRShaderReflectionGetResourceLocations(reflection, rls.data());
          for (const IRResourceLocation &rl : rls)
            locations.emplace_back(
                static_cast<int>(rl.resourceType), rl.space, rl.slot,
                rl.topLevelOffset, rl.sizeBytes,
                rl.resourceName ? std::string(rl.resourceName) : std::string{});
        }
        IRShaderReflectionDestroy(reflection);

        IRMetalLibBinaryDestroy(lib);
        IRObjectDestroy(out);
        IRObjectDestroy(input);
        IRCompilerDestroy(compiler);

        return {
            nb::bytes(reinterpret_cast<const char *>(buf.data()), buf.size()),
            locations};
      },
      "dxil"_a, "stage"_a = IRShaderStageCompute, "entry_point"_a = "");
}
