// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (c) 2024.

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/LexicallyOrderedRecursiveASTVisitor.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

static llvm::cl::OptionCategory EUDSLPYGenCat("Options for eudslpy-gen");
static llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::Required,
                                                llvm::cl::cat(EUDSLPYGenCat));

static llvm::cl::list<std::string>
    IncludeDirs("I", llvm::cl::desc("Directory of include files"),
                llvm::cl::value_desc("directory"), llvm::cl::Prefix,
                llvm::cl::cat(EUDSLPYGenCat));

static llvm::cl::opt<std::string>
    OutputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"),
                   llvm::cl::cat(EUDSLPYGenCat));

static llvm::cl::list<std::string>
    Namespaces("namespaces", llvm::cl::desc("Namespaces to generate from"),
               llvm::cl::CommaSeparated, llvm::cl::cat(EUDSLPYGenCat));

static llvm::cl::list<std::string>
    Defines("D", llvm::cl::desc("Name of the macro to be defined"),
            llvm::cl::value_desc("macro name"), llvm::cl::Prefix,
            llvm::cl::cat(EUDSLPYGenCat));

static llvm::cl::opt<int> ShardSize("shard-size", llvm::cl::desc("Shard size"),
                                    llvm::cl::value_desc("shard size"),
                                    llvm::cl::cat(EUDSLPYGenCat),
                                    llvm::cl::init(10));

static llvm::cl::opt<bool> Shardify("shardify",
                                    llvm::cl::desc("Split into shards"),
                                    llvm::cl::value_desc("shardify"),
                                    llvm::cl::cat(EUDSLPYGenCat),
                                    llvm::cl::init(false));

static llvm::cl::opt<int>
    MaxNumShards("max-number-shards",
                 llvm::cl::desc("Maximum number of shards to split into"),
                 llvm::cl::value_desc("max number shards"),
                 llvm::cl::cat(EUDSLPYGenCat), llvm::cl::init(-1));

static llvm::cl::opt<std::string> ShardTarget(
    "shard-target", llvm::cl::desc("CMake target that is being sharded"),
    llvm::cl::value_desc("sharding target"), llvm::cl::cat(EUDSLPYGenCat));

static bool filterInNamespace(const std::string &s) {
  if (Namespaces.empty())
    return true;
  for (auto ns : Namespaces)
    if (ns == s || ("::" + ns == s))
      return true;
  return false;
}

static std::string
getNBBindClassName(const std::string &qualifiedNameAsString) {
  std::string s = qualifiedNameAsString;
  s = std::regex_replace(s, std::regex(R"(\s+)"), "");
  s = std::regex_replace(s, std::regex("::"), "_");
  s = std::regex_replace(s, std::regex("[<|>]"), "__");
  s = std::regex_replace(s, std::regex(R"(\*)"), "___");
  s = std::regex_replace(s, std::regex(","), "____");
  return s;
}

static std::string getPyClassName(const std::string &qualifiedNameAsString) {
  std::string s = qualifiedNameAsString;
  s = std::regex_replace(s, std::regex(R"(\s+)"), "");
  s = std::regex_replace(s, std::regex(R"(\*)"), "");
  s = std::regex_replace(s, std::regex("<"), "[");
  s = std::regex_replace(s, std::regex(">"), "]");
  s = std::regex_replace(s, std::regex("::"), ".");
  return s;
}

static std::string snakeCase(const std::string &name) {
  std::string s = name;
  s = std::regex_replace(s, std::regex(R"(([A-Z]+)([A-Z][a-z]))"), "$1_$2");
  s = std::regex_replace(s, std::regex(R"(([a-z\d])([A-Z]))"), "$1_$2");
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

static clang::PrintingPolicy getPrintingPolicy(bool canonical = true) {
  clang::LangOptions lo;
  lo.CPlusPlus = true;
  clang::PrintingPolicy p(lo);
  // TODO(max): none of this really does anything except PrintCanonical
  // keep namespaces
  p.FullyQualifiedName = true;
  p.SuppressScope = false;
  p.SuppressInlineNamespace = false;
  p.PrintAsCanonical = canonical;
  p.Bool = true;

  return p;
}

// stolen from
// https://github.com/llvm/llvm-project/blob/99dddef340e566e9d303010f1219f7d7d6d37a11/clang/lib/Sema/SemaChecking.cpp#L7055
// Determines if the specified is a C++ class or struct containing
// a member with the specified name and kind (e.g. a CXXMethodDecl named
// "c_str()").
template <typename T>
static llvm::SmallPtrSet<T *, 1> findOverloads(clang::FunctionDecl *decl,
                                               clang::Sema &s) {
  llvm::SmallPtrSet<T *, 1> results;
  clang::LookupResult r(s, &s.Context.Idents.get(decl->getNameAsString()),
                        decl->getLocation(), clang::Sema::LookupOrdinaryName);
  r.suppressDiagnostics();
  if (s.LookupQualifiedName(r, decl->getDeclContext()))
    for (clang::LookupResult::iterator i = r.begin(), e = r.end(); i != e;
         ++i) {
      clang::NamedDecl *namedDecl = (*i)->getUnderlyingDecl();
      if (T *fk = llvm::dyn_cast<T>(namedDecl))
        results.insert(fk);
    }
  return results;
}

static std::string sanitizeName(std::string name, int emptyIdx = 0) {
  if (name == "def")
    name = "def_";
  if (name == "from")
    name = "from_";
  else if (name == "except")
    name = "except_";
  else if (name == "")
    name = std::string(emptyIdx + 1, '_');
  return name;
}

static std::string sanitizeType(std::string type) {
  if (type.rfind("ArrayRef", 0) == 0)
    type = "llvm::" + type;
  if (std::regex_search(type, std::regex(R"(std::__1)")))
    type = std::regex_replace(type, std::regex("std::__1"), "std");
  if (std::regex_search(type, std::regex(R"(std::__cxx11)")))
    type = std::regex_replace(type, std::regex("std::__cxx11"), "std");
  return type;
}

// emit a lambda body to disambiguate/break ties amongst overloads
// TODO(max):: overloadimpl or whatever should work but it doesn't...
std::string emitNBLambdaBody(clang::FunctionDecl *decl,
                             llvm::SmallVector<std::string> paramNames,
                             llvm::SmallVector<std::string> paramTypes) {
  std::string typedParamsStr;
  if (decl->getNumParams()) {
    llvm::SmallVector<std::string> typedParams =
        llvm::to_vector(llvm::map_range(
            llvm::zip(paramTypes, paramNames),
            [](std::tuple<std::string, std::string> item) -> std::string {
              auto [t, n] = item;
              return llvm::formatv("{0} {1}", t, n);
            }));
    typedParamsStr = llvm::join(typedParams, ", ");
  }
  // since we're emitting a body, we need to do std::move for some
  // unique_ptrs
  llvm::SmallVector<std::string> newParamNames(paramNames);
  for (auto [idx, item] :
       llvm::enumerate(llvm::zip(paramTypes, newParamNames))) {
    // TODO(max): characterize this condition better...
    auto [t, n] = item;
    if ((t.rfind("std::unique_ptr") != std::string::npos && t.back() != '&') ||
        t.rfind("&&") != std::string::npos)
      n = llvm::formatv("std::move({0})", n);
  }
  std::string newParamNamesStr = llvm::join(newParamNames, ", ");
  std::string return_;
  auto returnTypeT = decl->getReturnType();
  if (!returnTypeT->isVoidType())
    return_ = "return";

  bool canonical = true;
  if (std::regex_search(returnTypeT.getAsString(), std::regex(R"(_t\b)")))
    canonical = false;
  std::string returnType =
      sanitizeType(returnTypeT.getAsString(getPrintingPolicy(canonical)));

  std::string funcRef;
  if (decl->isStatic() || !decl->isCXXClassMember()) {
    funcRef = llvm::formatv("\n  []({0}) -> {1} {{\n    {2} {3}({4});\n  }",
                            typedParamsStr, returnType, return_,
                            decl->getQualifiedNameAsString(), newParamNamesStr);
  } else {
    assert(decl->isCXXClassMember() && "expected class member");
    if (decl->getNumParams())
      typedParamsStr = llvm::formatv("self, {0}", typedParamsStr);
    else
      typedParamsStr = "self";
    std::string methName = decl->getNameAsString();
    if (llvm::isa<clang::CXXConversionDecl>(decl))
      methName = "operator " + returnType;
    const clang::CXXRecordDecl *parentRecord =
        llvm::cast<clang::CXXRecordDecl>(decl->getParent());
    funcRef =
        llvm::formatv("\n  []({0}& {1}) -> {2} {{\n    {3} self.{4}({5});\n  }",
                      parentRecord->getQualifiedNameAsString(), typedParamsStr,
                      returnType, return_, methName, newParamNamesStr);
  }
  return funcRef;
}

static std::string
processOperator(const std::string &fnName,
                const llvm::SmallVector<std::string> &argNames) {
  std::string newFnName = fnName;
  if (fnName == "operator!=") {
    newFnName = "__ne__";
  } else if (fnName == "operator==") {
    newFnName = "__eq__";
  } else if (fnName == "operator-") {
    newFnName = "__neg__";
  } else if (fnName == "operator[]") {
    newFnName = "__getitem__";
  } else if (fnName == "operator<") {
    newFnName = "__lt__";
  } else if (fnName == "operator<=") {
    newFnName = "__le__";
  } else if (fnName == "operator>") {
    newFnName = "__gt__";
  } else if (fnName == "operator>=") {
    newFnName = "__ge__";
  } else if (fnName == "operator%") {
    newFnName = "__mod__";
  } else if (fnName == "operator*") {
    if (!argNames.empty()) {
      newFnName = "__mul__";
    } else {
      // operator* not supported
    }
  } else if (fnName == "operator+" && !argNames.empty()) {
    newFnName = "__add__";
  } else if (fnName == "operator->") {
    // operator-> not supported
  } else if (fnName == "operator!") {
    // operator! not supported
  } else if (fnName == "operator<<") {
    // operator<< not supported
  }

  return newFnName;
}

static std::string getCppClass(const clang::CXXRecordDecl *decl) {
  std::string className;
  if (const clang::ClassTemplateSpecializationDecl *t =
          llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(decl)) {
    // TODO(max): this emits unnecessary default template args, like
    // mlir::detail::TypeIDResolver<void, void>
    // auto td = t->getTypeForDecl();
    className = t->getASTContext()
                    .getCanonicalTypeDeclType(decl)
                    .getTypePtr()
                    ->getCanonicalTypeInternal()
                    .getAsString(getPrintingPolicy());
  } else {
    className = decl->getQualifiedNameAsString();
  }

  return sanitizeType(className);
}

static bool
emitClassMethodOrFunction(clang::FunctionDecl *decl,
                          clang::CompilerInstance &ci,
                          std::shared_ptr<llvm::ToolOutputFile> outputFile) {
  llvm::SmallVector<std::string> paramNames;
  llvm::SmallVector<std::string> paramTypes;
  for (unsigned i = 0; i < decl->getNumParams(); ++i) {
    clang::ParmVarDecl *param = decl->getParamDecl(i);
    std::string name = param->getNameAsString();
    auto t = param->getType();
    bool canonical = true;
    // TODO(max): this is dumb... (maybe there's a way to check where the
    // typedef is defined...)
    // word boundary excludes x86_amx_tdpbf16ps
    if (std::regex_search(t.getAsString(), std::regex(R"(_t\b)")))
      canonical = false;
    std::string paramType = t.getAsString(getPrintingPolicy(canonical));
    paramTypes.push_back(sanitizeType(paramType));
    paramNames.push_back(sanitizeName(name, i));
  }

  llvm::SmallPtrSet<clang::FunctionDecl *, 1> funcOverloads =
      findOverloads<clang::FunctionDecl>(decl, ci.getSema());
  llvm::SmallPtrSet<clang::FunctionTemplateDecl *, 1> funcTemplOverloads =
      findOverloads<clang::FunctionTemplateDecl>(decl, ci.getSema());
  std::string funcRef, nbFnName;

  if (clang::CXXConstructorDecl *ctor =
          llvm::dyn_cast<clang::CXXConstructorDecl>(decl)) {
    if (ctor->isDeleted())
      return false;
    funcRef = llvm::formatv("nb::init<{0}>()", llvm::join(paramTypes, ", "));
  } else {
    if (funcOverloads.size() == 1 && funcTemplOverloads.empty())
      funcRef = llvm::formatv("&{0}", decl->getQualifiedNameAsString());
    else
      funcRef = emitNBLambdaBody(decl, paramNames, paramTypes);
    nbFnName = snakeCase(decl->getNameAsString());
    if (decl->isOverloadedOperator() ||
        llvm::isa<clang::CXXConversionDecl>(decl)) {
      nbFnName = processOperator(nbFnName, paramNames);
    } else if (decl->isStatic() && funcOverloads.size() > 1 &&
               llvm::any_of(funcOverloads, [](clang::FunctionDecl *m) {
                 return !m->isStatic();
               })) {
      // disambiguate static method with non-static overloads (nanobind doesn't
      // let you overload static with non-static) see mlir::ElementsAttr
      nbFnName += "_static";
    }
  }

  std::string paramNamesStr;
  if (decl->getNumParams()) {
    paramNames = llvm::to_vector(
        llvm::map_range(paramNames, [](std::string s) -> std::string {
          return llvm::formatv("\"{0}\"_a", snakeCase(s));
        }));
    paramNamesStr = ", " + llvm::join(paramNames, ", ");
  }

  std::string refInternal, defStr = "def";
  if (decl->isCXXClassMember()) {
    if (decl->isStatic())
      defStr = "def_static";
    if (decl->getReturnType()->hasPointerRepresentation())
      refInternal = ", nb::rv_policy::reference_internal";
  }

  std::string sig;
  if (!nbFnName.empty()) {
    // no clue why but nb has trouble inferring the signature
    // (and this causes and assert failure in nb_func_render_signature
    // https://github.com/wjakob/nanobind/blob/c2e394eee5d19816871151de43c29b4051fbf9ff/src/nb_func.cpp#L1020
    if (decl->isStatic() && !decl->getNumParams())
      sig =
          llvm::formatv(", nb::sig(\"def {0}(/) -> {1}\") ", nbFnName,
                        decl->getReturnType().getAsString(getPrintingPolicy()));
    // uncomment this to debug nb which doesn't tell where the method actually
    // is when it fails if (parentRecord) {
    //   nbFnName += parentRecord->getNameAsString();
    // }
    nbFnName = llvm::formatv("\"{0}\", ", nbFnName);
  }

  std::string scope = "m";
  if (decl->isCXXClassMember()) {
    const clang::CXXRecordDecl *parentRecord =
        llvm::cast<clang::CXXRecordDecl>(decl->getParent());
    scope = getNBBindClassName(getCppClass(parentRecord));
  }

  outputFile->os() << llvm::formatv("{0}.{1}({2}{3}{4}{5}{6});\n", scope,
                                    defStr, nbFnName, funcRef, paramNamesStr,
                                    refInternal, sig);

  return true;
}

static std::string getNBScope(clang::TagDecl *decl) {
  std::string scope = "m";
  const clang::DeclContext *declContext = decl->getDeclContext();
  if (const clang::CXXRecordDecl *ctx =
          llvm::dyn_cast<clang::CXXRecordDecl>(declContext)) {
    scope = getNBBindClassName(getCppClass(ctx));
  }
  return scope;
}

static bool emitClass(clang::CXXRecordDecl *decl, clang::CompilerInstance &ci,
                      std::shared_ptr<llvm::ToolOutputFile> outputFile) {
  if (decl->isTemplated()) {
    clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
        decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                 clang::DiagnosticsEngine::Note,
                                 "template classes not supported yet"));
    return false;
  }

  std::string additional = "";
  std::string cppClass = getCppClass(decl);
  std::string autoVar = llvm::formatv("auto {0}", getNBBindClassName(cppClass));
  if (decl->getNumBases() > 1) {
    clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
        decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                 clang::DiagnosticsEngine::Note,
                                 "multiple base classes not supported"));
  } else if (decl->getNumBases() == 1) {
    // handle some known bases that we've already found a wap to bind
    clang::CXXBaseSpecifier baseClass = *decl->bases_begin();
    clang::QualType baseType = baseClass.getType();
    std::string baseName = getCppClass(baseType->getAsCXXRecordDecl());
    // TODO(max): these could be lookups on the corresponding recorddecls using
    // sema...
    if (baseName.rfind("mlir::Op<", 0) == 0) {
      cppClass = llvm::formatv("{0}, mlir::OpState", cppClass);
    } else if (baseName.rfind("mlir::detail::StorageUserBase<", 0) == 0) {
      llvm::SmallVector<llvm::StringRef, 2> templParams;
      llvm::StringRef{baseName}.split(templParams, ",");
      // TODO(max): this needs to use getCppClass not templParams[1], which is a
      // string
      cppClass = llvm::formatv("{0}, {1}", cppClass, templParams[1]);
    } else if (baseName.rfind("mlir::Dialect", 0) == 0 &&
               cppClass.rfind("mlir::ExtensibleDialect") == std::string::npos) {
      // clang-format off
      additional += llvm::formatv("\n  .def_static(\"insert_into_registry\", [](mlir::DialectRegistry &registry) {{ registry.insert<{0}>(); })", cppClass);
      additional += llvm::formatv("\n  .def_static(\"load_into_context\", [](mlir::MLIRContext &context) {{ return context.getOrLoadDialect<{0}>(); })", cppClass);
      // clang-format on
    } else if (!llvm::isa<clang::ClassTemplateSpecializationDecl>(
                   baseClass.getType()->getAsCXXRecordDecl())) {
      cppClass = llvm::formatv("{0}, {1}", cppClass, baseName);
    } else {
      assert(llvm::isa<clang::ClassTemplateSpecializationDecl>(
                 baseClass.getType()->getAsCXXRecordDecl()) &&
             "expected class template specialization");
      clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
          baseClass.getBeginLoc(), ci.getDiagnostics().getCustomDiagID(
                                       clang::DiagnosticsEngine::Note,
                                       "unknown base templated base class: "));
      builder << baseName << "\n";
    }
  }

  std::string scope = getNBScope(decl);
  std::string pyClassName = getPyClassName(decl->getNameAsString());
  outputFile->os() << llvm::formatv(
      "\n{0} = nb::class_<{1}>({2}, \"{3}\"){4};\n", autoVar, cppClass, scope,
      pyClassName, additional);

  return true;
}

static bool emitEnum(clang::EnumDecl *decl, clang::CompilerInstance &ci,
                     std::shared_ptr<llvm::ToolOutputFile> outputFile) {
  outputFile->os() << llvm::formatv("nb::enum_<{0}>({1}, \"{2}\")\n",
                                    decl->getQualifiedNameAsString(),
                                    getNBScope(decl), decl->getNameAsString());

  int i = 0, nDecls = std::distance(decl->decls_begin(), decl->decls_end());
  for (clang::Decl *cst : decl->decls()) {
    clang::EnumConstantDecl *cstDecl = llvm::cast<clang::EnumConstantDecl>(cst);
    outputFile->os() << llvm::formatv("  .value(\"{0}\", {1})",
                                      cstDecl->getNameAsString(),
                                      cstDecl->getQualifiedNameAsString());
    if (i++ < nDecls - 1)
      outputFile->os() << "\n";
  }

  outputFile->os() << ";\n";
  return true;
}

static bool emitField(clang::DeclaratorDecl *field, clang::CompilerInstance &ci,
                      std::shared_ptr<llvm::ToolOutputFile> outputFile) {
  const clang::CXXRecordDecl *parentRecord =
      llvm::cast<clang::CXXRecordDecl>(field->getLexicalDeclContext());

  std::string defStr = "def_rw";
  if (clang::VarDecl *vard = llvm::dyn_cast<clang::VarDecl>(field)) {
    if (vard->isStaticDataMember()) {
      if (vard->isConstexpr())
        defStr = "def_ro_static";
      else
        defStr = "def_rw_static";
    }
  }

  std::string refInternal;
  if (field->getType()->hasPointerRepresentation())
    refInternal = ", nb::rv_policy::reference_internal";

  std::string scope = getNBBindClassName(getCppClass(parentRecord));
  std::string nbFnName =
      llvm::formatv("\"{0}\"", snakeCase(field->getNameAsString()));
  outputFile->os() << llvm::formatv("{0}.{1}({2}, &{3}{4});\n", scope, defStr,
                                    nbFnName, field->getQualifiedNameAsString(),
                                    refInternal);
  return true;
}

template <typename T>
static bool shouldSkip(T *decl, clang::CompilerInstance &ci) {
  auto *encl = llvm::dyn_cast<clang::NamespaceDecl>(
      decl->getEnclosingNamespaceContext());
  if (!encl)
    return true;
  // this isn't redundant - filter might be empty but we still don't want to
  // bind std::
  if (encl->isStdNamespace() || encl->isInStdNamespace())
    return true;
  if (ci.getSema().getSourceManager().isInSystemHeader(decl->getLocation()))
    return true;
  if (!filterInNamespace(encl->getQualifiedNameAsString()))
    return true;
  if constexpr (std::is_same_v<T, clang::CXXRecordDecl> ||
                std::is_same_v<T, clang::EnumDecl>) {
    if (!decl->isCompleteDefinition())
      return true;
  }
  if (decl->getAccess() == clang::AS_private ||
      decl->getAccess() == clang::AS_protected)
    return true;
  if (decl->isImplicit())
    return true;

  return false;
}

struct HackDeclContext : clang::DeclContext {
  bool islastDecl(clang::Decl *d) const { return d == LastDecl; }
  clang::Decl *getLastDecl() const { return LastDecl; }
};

struct BindingsVisitor
    : clang::LexicallyOrderedRecursiveASTVisitor<BindingsVisitor> {
  BindingsVisitor(clang::CompilerInstance &ci,
                  std::shared_ptr<llvm::ToolOutputFile> outputFile)
      : LexicallyOrderedRecursiveASTVisitor<BindingsVisitor>(
            ci.getSourceManager()),
        ci(ci), outputFile(outputFile),
        mc(clang::ItaniumMangleContext::create(ci.getASTContext(),
                                               ci.getDiagnostics())) {}

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *decl) {
    if (shouldSkip(decl, ci))
      return true;
    if (decl->isAbstract())
      return true;
    if (decl->isClass() || decl->isStruct()) {
      if (emitClass(decl, ci, outputFile))
        visitedRecords.insert(decl);
    }
    return true;
  }

  // clang-format off
  /* method (member function) -> CXXMethodDecl (isStatic() tells statis/non-static)
   * non-static data member -> FieldDecl
   * static data member -> VarDecl with isStaticDataMember() == true
   * static local variable -> VarDecl with isStaticLocal() == true
   * non-static local variable -> VarDecl with isLocalVarDecl() == true
   * there's also a isLocalVarDeclOrParm if you want local or parameter, etc
   */
  // clang-format on

  bool VisitFieldDecl(clang::FieldDecl *decl) {
    if (decl->isFunctionOrFunctionTemplate() || decl->isFunctionPointerType())
      return true;
    // fields can be methods???
    if (llvm::isa<clang::CXXMethodDecl>(decl))
      return true;
    if (!visitedRecords.contains(decl->getParent()))
      return true;
    if (decl->getAccess() == clang::AS_private ||
        decl->getAccess() == clang::AS_protected)
      return true;

    if (decl->isAnonymousStructOrUnion()) {
      clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
          decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                   clang::DiagnosticsEngine::Note,
                                   "anon structs/union fields not supported"));
      return true;
    }
    if (decl->isBitField())
      return true;

    emitField(decl, ci, outputFile);
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *decl) {
    if (decl->isFunctionOrMethodVarDecl())
      return true;
    if (llvm::isa<clang::CXXMethodDecl>(decl))
      return true;
    if (auto parent = llvm::dyn_cast<clang::CXXRecordDecl>(
            decl->getLexicalDeclContext())) {
      if (visitedRecords.contains(parent)) {
        if (decl->getAccess() == clang::AS_private ||
            decl->getAccess() == clang::AS_protected)
          return true;
        emitField(decl, ci, outputFile);
      }
    }

    return true;
  }

  // TODO(max): skip definitions somehow? like FloatType::getFloat4E2M1FN which
  // has both the decl and the impl in a header?
  bool VisitCXXMethodDecl(clang::CXXMethodDecl *decl) {
    if (shouldSkip(decl, ci) || llvm::isa<clang::CXXDestructorDecl>(decl) ||
        !visitedRecords.contains(decl->getParent()))
      return true;
    if (decl->isOverloadedOperator() &&
        decl->getNameAsString().rfind("operator=", 0) == 0)
      return true;
    if (decl->isTemplated() || decl->isTemplateDecl() ||
        decl->isTemplateInstantiation() ||
        decl->isFunctionTemplateSpecialization()) {
      clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
          decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                   clang::DiagnosticsEngine::Note,
                                   "template methods not supported yet"));
      return true;
    }
    if (decl->getFriendObjectKind()) {
      clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
          decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                   clang::DiagnosticsEngine::Note,
                                   "friend functions not supported"));
      return true;
    }
    if (decl->isCopyAssignmentOperator() || decl->isMoveAssignmentOperator())
      return true;
    if (decl->isDeleted())
      return true;

    emitClassMethodOrFunction(decl, ci, outputFile);

    return true;
  }

  bool VisitFunctionDecl(clang::FunctionDecl *decl) {
    if (shouldSkip(decl, ci) || decl->isCXXClassMember())
      return true;
    // clang-format off
    // this
    // template <typename EnumType> ::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);
    // is not a `TemplateDecl` but it is `Templated`...
    // on the other hand every method in a template class `isTemplated` even
    // if the template params don't play in the method decl (which is why this visitor can't be combined with VisitCXXMethodDecl)
    // clang-format on
    if (decl->isTemplated() || decl->isTemplateDecl() ||
        decl->isTemplateInstantiation() ||
        decl->isFunctionTemplateSpecialization()) {
      clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
          decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                   clang::DiagnosticsEngine::Note,
                                   "template functions not supported yet"));
      return true;
    }
    if (decl->getFriendObjectKind()) {
      clang::DiagnosticBuilder builder = ci.getDiagnostics().Report(
          decl->getLocation(), ci.getDiagnostics().getCustomDiagID(
                                   clang::DiagnosticsEngine::Note,
                                   "friend functions not supported"));
      return true;
    }
    emitClassMethodOrFunction(decl, ci, outputFile);
    return true;
  }

  bool VisitEnumDecl(clang::EnumDecl *decl) {
    if (shouldSkip(decl, ci))
      return true;
    if (decl->getQualifiedNameAsString().rfind("unnamed enum") !=
        std::string::npos)
      return true;
    emitEnum(llvm::cast<clang::EnumDecl>(decl), ci, outputFile);
    return true;
  }

  void maybeEmitShard(const clang::CXXRecordDecl *decl) {
    clang::CXXBaseSpecifier baseClass = *decl->bases_begin();
    std::string baseName = baseClass.getType().getAsString(getPrintingPolicy());
    if (baseName.rfind("mlir::Op<", 0) == 0) {
      ++opClassesSeen;
      if (opClassesSeen % ShardSize == 0)
        outputFile->os() << llvm::formatv("// eudslpy-gen-shard {}\n",
                                          int(opClassesSeen / ShardSize));
    }
  }

  // implicit methods are the "last decls"
  bool shouldVisitImplicitCode() const { return true; }

  // TODO(max): this is a hack and not stable
  bool VisitDecl(clang::Decl *decl) {
    if (ci.getSema().getSourceManager().isInSystemHeader(decl->getLocation()))
      return true;
    const clang::DeclContext *declContext = decl->getDeclContext();
    HackDeclContext *ctx =
        static_cast<HackDeclContext *>(decl->getDeclContext());
    if (declContext && declContext->isRecord()) {
      const clang::CXXRecordDecl *recordDecl =
          llvm::cast<clang::CXXRecordDecl>(declContext);
      if (visitedRecords.contains(recordDecl) && ctx->islastDecl(decl)) {
        outputFile->os() << "// " << recordDecl->getQualifiedNameAsString()
                         << "\n\n";
        if (recordDecl->getNumBases() == 1)
          maybeEmitShard(recordDecl);
      }
    }
    return true;
  }

  clang::CompilerInstance &ci;
  std::shared_ptr<llvm::ToolOutputFile> outputFile;
  std::unique_ptr<clang::ItaniumMangleContext> mc;
  llvm::DenseSet<clang::TagDecl *> visitedRecords;
  int opClassesSeen{0};
};

struct ClassStructEnumConsumer : clang::ASTConsumer {
  ClassStructEnumConsumer(
      clang::CompilerInstance &ci,
      const std::shared_ptr<llvm::ToolOutputFile> &outputFile)
      : visitor(ci, outputFile) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    if (!visitor.ci.hasSema())
      visitor.ci.createSema(clang::TU_Prefix,
                            /*CompletionConsumer*/ nullptr);
    visitor.TraverseDecl(context.getTranslationUnitDecl());
  }
  bool shouldSkipFunctionBody(clang::Decl *) override { return true; }
  BindingsVisitor visitor;
};

struct GenerateBindingsAction : clang::ASTFrontendAction {
  explicit GenerateBindingsAction(
      const std::shared_ptr<llvm::ToolOutputFile> &outputFile)
      : outputFile(outputFile) {}
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    llvm::StringRef inFile) override {
    compiler.getDiagnosticOpts().ShowLevel = true;
    return std::make_unique<ClassStructEnumConsumer>(compiler, outputFile);
  }

  // PARSE_INCOMPLETE
  clang::TranslationUnitKind getTranslationUnitKind() override {
    return clang::TU_Prefix;
  }

  std::shared_ptr<llvm::ToolOutputFile> outputFile;
};

static int generate() {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }
  llvm::StringRef buffer = fileOrErr.get()->getBuffer();

  if (OutputFilename.empty()) {
    llvm::errs() << "-o outputfile must be set.\n";
    return -1;
  }

  std::error_code error;
  auto outputFile = std::make_shared<llvm::ToolOutputFile>(
      OutputFilename, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "cannot open output file '" + OutputFilename +
                        "': " + error.message();
    return -1;
  }
  std::vector<std::string> args{
      "-E",
      "-xc++",
      "-std=c++17",
      "-fdirectives-only",
      "-fkeep-system-includes",
      "-fdelayed-template-parsing",
      "-Wno-unused-command-line-argument",
      // "-v",
      // annoyingly clang will insert -internal-isystem with relative paths and
      // those could hit on the build dir (which have headers and relationships
      // amongst them that won't necessarily be valid) more annoyingly different
      // toolchains decide differently which flag determines whether to include
      // the tell-tale sign is if you uncomment -v and see "ignoring
      // non-existant directory..." in the output
      "-nobuiltininc",
      "-nostdinc",
      "-nostdinc++",
      "-nostdlibinc",
  };
  int clangArgs = args.size();
  for (const auto &includeDir : IncludeDirs)
    args.emplace_back(llvm::formatv("-I{0}", includeDir));
  for (const auto &define : Defines)
    args.emplace_back(llvm::formatv("-D{0}", define));

  outputFile->os() << "// Generated with eudslpy-gen args:\n";
  outputFile->os() << "// " << InputFilename << " ";
  auto arg = args.begin();
  for (std::advance(arg, clangArgs); arg != args.end(); ++arg) {
    outputFile->os() << *arg << ' ';
  }
  for (auto ns : Namespaces)
    outputFile->os() << "-namespaces=" << ns << " ";
  outputFile->os() << "-o " << OutputFilename << "\n";
  outputFile->os() << "\n";

  if (!clang::tooling::runToolOnCodeWithArgs(
          std::make_unique<GenerateBindingsAction>(outputFile), buffer, args,
          InputFilename)) {
    llvm::errs() << "bindings generation failed.\n";
    return -1;
  }

  outputFile->keep();

  return 0;
}

// cannot fathom why but std::to_string is return ascii characters
template <typename T>
static std::string numberToString(T number) {
  std::ostringstream ss;
  ss << number;
  return ss.str();
}

static int makeSourceShards() {
  // Ensure the filename ends with ".cpp.gen"
  if (InputFilename.substr(InputFilename.size() - 8) != ".cpp.gen") {
    llvm::errs() << "expected .cpp.gen file\n";
    return -1;
  }

  std::ifstream file(InputFilename);
  if (!file.is_open()) {
    llvm::errs() << "Failed to open file\n";
    return -1;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string source = buffer.str();

  // Split source by regex pattern
  std::regex re("// eudslpy-gen-shard \\d+");
  std::sregex_token_iterator iter(source.begin(), source.end(), re, -1);
  std::sregex_token_iterator end;

  std::vector<std::string> shards;
  while (iter != end)
    shards.push_back(*iter++);

  std::string finalTarget =
      ShardTarget.empty()
          ? InputFilename.substr(0, InputFilename.find_last_of("."))
          : ShardTarget;

  // Generate shard files
  for (size_t i = 0; i < shards.size(); ++i) {
    std::ofstream shardFile(InputFilename.getValue() + ".shard." +
                            numberToString(i) + ".cpp");
    if (!shardFile.is_open()) {
      llvm::errs() << "Failed to create shard file";
      return -1;
    }

    if (!IncludeDirs.empty()) {
      for (const std::string &inc : IncludeDirs)
        shardFile << "#include \"" << inc << "\"" << std::endl;
    }

    // clang-format off
    shardFile << R"(
#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;
using namespace mlir;
using namespace llvm;
#include "eudsl/type_casters.h"

namespace eudsl {
void populate)" << finalTarget << i << R"(Module(nb::module_ &m) {
)";
    // clang-format on
    shardFile << shards[i] << std::endl;
    shardFile << "}" << std::endl;
    shardFile << "}" << std::endl;
    shardFile.flush();
    shardFile.close();
  }

  // Handle max number of shards
  if (MaxNumShards != -1 && shards.size() > static_cast<size_t>(MaxNumShards)) {
    llvm::errs() << "expected less than " +
                        numberToString(MaxNumShards.getValue()) + " shards";
    return -1;
  }
  if (MaxNumShards == -1)
    MaxNumShards = shards.size();
  else
    // _max_num_shards in cmake counts +1
    MaxNumShards.getValue() += 1;

  for (size_t i = shards.size(); i < static_cast<size_t>(MaxNumShards); ++i) {
    std::ofstream dummyShardFile(InputFilename.getValue() + ".shard." +
                                 numberToString(i) + ".cpp");
    if (!dummyShardFile.is_open()) {
      llvm::errs() << "Failed to create dummy shard file";
      return -1;
    }
    dummyShardFile << "// dummy shard " << i << std::endl;
    dummyShardFile.flush();
    dummyShardFile.close();
  }

  // Generate the final sharded cpp file
  std::ofstream finalShardedFile(InputFilename.getValue() + ".sharded.cpp");
  if (!finalShardedFile.is_open()) {
    llvm::errs() << "Failed to create final sharded file";
    return -1;
  }

  // clang-format off
  finalShardedFile << R"(
#include <nanobind/nanobind.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace eudsl {
void populate)" << finalTarget << R"(Module(nb::module_ &m) {
)";
  // clang-format on

  for (size_t i = 0; i < shards.size(); ++i)
    finalShardedFile << "extern void populate" << finalTarget << i
                     << "Module(nb::module_ &m);" << std::endl;

  for (size_t i = 0; i < shards.size(); ++i)
    finalShardedFile << "populate" << finalTarget << i << "Module(m);"
                     << std::endl;

  finalShardedFile << "}" << std::endl;
  finalShardedFile << "}" << std::endl;
  finalShardedFile.flush();
  finalShardedFile.close();

  return 0;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  if (Shardify)
    return makeSourceShards();
  return generate();

  return 0;
}
